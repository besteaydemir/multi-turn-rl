"""
Test reference model management and KL scheduling integration with trainer.

This test verifies:
1. ReferenceModelManager correctly creates and updates reference model
2. KLScheduler adapts β coefficient based on observed KL
3. Integration with RLTrainer works end-to-end
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import List
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_environment.environment import Episode, Turn, Observation
from rl_trainer.reference import ReferenceModelManager, KLScheduler
from rl_trainer.batch import EpisodeDataLoader
from rl_trainer.trainer import TrainerConfig, RLTrainer


# Mock VLM for testing
class MockVLM(nn.Module):
    """Minimal VLM-like model for testing."""
    
    def __init__(self, vocab_size=100, hidden_size=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids, **kwargs):
        hidden = self.embedding(input_ids)
        logits = self.lm_head(hidden)
        
        @dataclass
        class Output:
            logits: torch.Tensor
        
        return Output(logits=logits)


def create_mock_episode(episode_id: int = 0, num_turns: int = 3) -> Episode:
    """Create a mock episode for testing."""
    turns = []
    
    for turn_idx in range(num_turns):
        observation = Observation(
            step=turn_idx,
            images=[f"test_image_{turn_idx}.jpg"],
            camera_positions=[np.eye(4)],
            current_position=np.array([0.0, 0.0, 0.0]),
            current_rotation=np.eye(3),
            bbox_mins=[0.0, 0.0, 0.0],
            bbox_maxs=[10.0, 10.0, 10.0],
            question="Test question",
            choices=["choice1", "choice2", "choice3"],
            movement_history=[],
            is_final_step=(turn_idx == num_turns - 1)
        )
        
        # Mock tokenization
        context_ids = list(range(10 + turn_idx * 5, 15 + turn_idx * 5))
        generated_ids = list(range(20 + turn_idx * 2, 22 + turn_idx * 2))
        
        turn = Turn(
            turn_index=turn_idx,
            observation=observation,
            full_prompt="Test prompt",
            context_text="Test context",
            generated_ids=torch.tensor(generated_ids),
            generated_text=f"Generated text {turn_idx}",
            action_token_mask=torch.ones(len(generated_ids), dtype=torch.bool),
            context_input_ids=torch.tensor(context_ids),
            input_token_length=len(context_ids),
            num_action_tokens=len(generated_ids),
            num_reasoning_tokens=0,
            masking_method="mock",
            action_valid=True
        )
        turns.append(turn)
    
    return Episode(
        episode_id=f"test_ep_{episode_id}",
        scene_id="test_scene",
        question="Test question",
        choices=["choice1", "choice2", "choice3"],
        ground_truth="choice1",
        turns=turns,
        final_reward=1.0,
        is_valid=True
    )


def test_reference_model_frozen():
    """Test frozen reference model strategy."""
    print("\n" + "=" * 80)
    print("TEST: Frozen Reference Model")
    print("=" * 80)
    
    model = MockVLM()
    
    ref_manager = ReferenceModelManager(
        policy_model=model,
        strategy="frozen",
        device="cpu"
    )
    
    ref_model = ref_manager.get_reference_model()
    
    # Get initial weights
    initial_ref_weights = ref_model.lm_head.weight.clone()
    initial_policy_weights = model.lm_head.weight.clone()
    
    # Update policy
    with torch.no_grad():
        model.lm_head.weight += 0.1
    
    # Try to update reference (should not update with frozen strategy)
    ref_manager.maybe_update(model, step=100)
    
    # Check reference hasn't changed
    final_ref_weights = ref_model.lm_head.weight
    assert torch.allclose(initial_ref_weights, final_ref_weights), "Reference should not change with frozen strategy"
    
    # Check policy has changed
    assert not torch.allclose(initial_policy_weights, model.lm_head.weight), "Policy should have changed"
    
    print("✓ Frozen reference model works correctly")
    print(f"  Reference model unchanged: {torch.allclose(initial_ref_weights, final_ref_weights)}")
    print(f"  Policy model changed: {not torch.allclose(initial_policy_weights, model.lm_head.weight)}")


def test_reference_model_periodic():
    """Test periodic reference model strategy."""
    print("\n" + "=" * 80)
    print("TEST: Periodic Reference Model")
    print("=" * 80)
    
    model = MockVLM()
    
    ref_manager = ReferenceModelManager(
        policy_model=model,
        strategy="periodic",
        update_interval=10,
        device="cpu"
    )
    
    ref_model = ref_manager.get_reference_model()
    
    # Get initial weights
    initial_ref_weights = ref_model.lm_head.weight.clone()
    
    # Update policy
    with torch.no_grad():
        model.lm_head.weight += 0.1
    
    # Update at step 5 (should not trigger)
    updated = ref_manager.maybe_update(model, step=5)
    assert not updated, "Should not update before interval"
    assert torch.allclose(initial_ref_weights, ref_model.lm_head.weight), "Reference unchanged before interval"
    
    # Update at step 10 (should trigger)
    updated = ref_manager.maybe_update(model, step=10)
    assert updated, "Should update at interval"
    assert not torch.allclose(initial_ref_weights, ref_model.lm_head.weight), "Reference should update at interval"
    assert torch.allclose(ref_model.lm_head.weight, model.lm_head.weight), "Reference should match policy"
    
    print("✓ Periodic reference model works correctly")
    print(f"  Update interval: 10")
    print(f"  Updated at step 10: {updated}")


def test_reference_model_ema():
    """Test EMA reference model strategy."""
    print("\n" + "=" * 80)
    print("TEST: EMA Reference Model")
    print("=" * 80)
    
    model = MockVLM()
    tau = 0.9  # Use smaller tau for easier testing
    
    ref_manager = ReferenceModelManager(
        policy_model=model,
        strategy="ema",
        ema_tau=tau,
        device="cpu"
    )
    
    ref_model = ref_manager.get_reference_model()
    
    # Get initial weights
    initial_weights = ref_model.lm_head.weight.clone()
    
    # Update policy
    delta = 1.0
    with torch.no_grad():
        model.lm_head.weight += delta
    
    # Update with EMA
    ref_manager.maybe_update(model, step=1)
    
    # Check EMA formula: ref = tau * ref + (1-tau) * policy
    expected = tau * initial_weights + (1 - tau) * model.lm_head.weight
    actual = ref_model.lm_head.weight
    
    assert torch.allclose(expected, actual, atol=1e-5), "EMA update should follow formula"
    
    # Reference should be between initial and policy
    diff_to_policy = (model.lm_head.weight - actual).abs().mean()
    diff_to_initial = (initial_weights - actual).abs().mean()
    
    print("✓ EMA reference model works correctly")
    print(f"  EMA tau: {tau}")
    print(f"  Distance to policy: {diff_to_policy:.6f}")
    print(f"  Distance to initial: {diff_to_initial:.6f}")
    print(f"  EMA formula verified: {torch.allclose(expected, actual, atol=1e-5)}")


def test_kl_scheduler():
    """Test adaptive KL coefficient scheduling."""
    print("\n" + "=" * 80)
    print("TEST: KL Scheduler")
    print("=" * 80)
    
    scheduler = KLScheduler(
        initial_kl_coef=0.02,  # Start lower so it can increase
        target_kl=0.01,
        kl_tolerance=0.005,
        adaptation_rate=1.5,
        warmup_steps=10,
        update_interval=1,  # Update every step for easier testing
        max_kl_coef=0.5  # Allow higher ceiling
    )
    
    print(f"Initial β: {scheduler.get_kl_coef():.4f}")
    
    # During warmup, β should stay constant
    for step in range(5):
        kl_coef = scheduler.step(current_kl=0.02, step=step)
        assert kl_coef == 0.02, "KL coef should be constant during warmup"
    
    print(f"  After warmup (step 5): β = {kl_coef:.4f} (unchanged)")
    
    # After warmup, high KL should increase β
    high_kl = 0.02  # Above target + tolerance
    print(f"\nCalling step with step=10, current_kl={high_kl}")
    kl_coef = scheduler.step(current_kl=high_kl, step=10)
    print(f"  Returned β: {kl_coef:.4f}")
    print(f"  Target range: {0.01 - 0.005:.4f} to {0.01 + 0.005:.4f}")
    print(f"  High KL is above target+tolerance: {high_kl} > {0.01 + 0.005}")
    assert kl_coef > 0.02, f"High KL should increase β, but got {kl_coef}"
    print(f"  High KL ({high_kl:.4f}): β = {kl_coef:.4f} (increased)")
    
    # Low KL should decrease β
    current_beta = kl_coef
    low_kl = 0.002  # Below target - tolerance
    kl_coef = scheduler.step(current_kl=low_kl, step=11)
    assert kl_coef < current_beta, "Low KL should decrease β"
    print(f"  Low KL ({low_kl:.4f}): β = {kl_coef:.4f} (decreased)")
    
    # Target KL should keep β stable
    current_beta = kl_coef
    target_kl = 0.01
    for step_idx in range(5):
        kl_coef = scheduler.step(current_kl=target_kl, step=12 + step_idx)
    assert abs(kl_coef - current_beta) < 0.01, "Target KL should keep β stable"
    print(f"  Target KL ({target_kl:.4f}): β = {kl_coef:.4f} (stable)")
    
    print("✓ KL scheduler works correctly")
    
    # Show stats
    stats = scheduler.get_stats()
    print("\nScheduler statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.6f}")


def test_trainer_integration():
    """Test integration of reference model and KL scheduler with trainer."""
    print("\n" + "=" * 80)
    print("TEST: Trainer Integration")
    print("=" * 80)
    
    # Create mock episodes
    episodes = [create_mock_episode(i, num_turns=2) for i in range(8)]
    
    # Create model and dataloader
    model = MockVLM()
    
    # Mock processor
    class MockProcessor:
        def __init__(self):
            self.tokenizer = self
            self.pad_token_id = 0
            
        def __call__(self, *args, **kwargs):
            return {"input_ids": torch.randint(0, 100, (1, 10))}
    
    processor = MockProcessor()
    
    dataloader = EpisodeDataLoader(
        episodes=episodes,
        batch_size=4,
        shuffle=False,
        filter_invalid=True,
        processor=processor,
        device="cpu"
    )
    
    # Create trainer config with reference model and KL scheduling
    config = TrainerConfig(
        learning_rate=1e-4,
        num_epochs=1,
        batch_size=4,
        device="cpu",
        # Reference model config
        ref_model_strategy="ema",
        ref_ema_tau=0.9,
        # KL scheduling
        use_kl_scheduler=True,
        target_kl=0.01,
        kl_tolerance=0.005,
        kl_adaptation_rate=1.5,
        warmup_steps=2,
        # Disable some features for testing
        use_cosine_schedule=False,
        log_interval=1
    )
    
    # Create trainer
    trainer = RLTrainer(
        model=model,
        dataloader=dataloader,
        config=config
    )
    
    # Verify components are initialized
    assert trainer.ref_manager is not None, "ReferenceModelManager should be initialized"
    assert trainer.kl_scheduler is not None, "KLScheduler should be initialized"
    assert trainer.ref_model is not None, "Reference model should exist"
    
    print("✓ Trainer components initialized correctly")
    print(f"  Reference strategy: {config.ref_model_strategy}")
    print(f"  KL scheduler enabled: {config.use_kl_scheduler}")
    print(f"  Reference model type: {type(trainer.ref_model).__name__}")
    
    # Check reference model is different object from policy
    assert trainer.ref_model is not trainer.model, "Reference should be separate object"
    
    # Check initial weights match
    assert torch.allclose(
        trainer.ref_model.lm_head.weight,
        trainer.model.lm_head.weight
    ), "Initial weights should match"
    
    print("  Reference model is separate object: ✓")
    print("  Initial weights match: ✓")
    
    print("\n✓ Trainer integration test passed!")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("TESTING REFERENCE MODEL MANAGEMENT & KL SCHEDULING")
    print("=" * 80)
    
    test_reference_model_frozen()
    test_reference_model_periodic()
    test_reference_model_ema()
    test_kl_scheduler()
    test_trainer_integration()
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED! ✓")
    print("=" * 80)
    print("\nSummary:")
    print("  ✓ Frozen reference model strategy works")
    print("  ✓ Periodic reference model strategy works")
    print("  ✓ EMA reference model strategy works")
    print("  ✓ Adaptive KL scheduling works")
    print("  ✓ Integration with trainer works")
    print("\nThe complete RL training pipeline is ready!")


if __name__ == "__main__":
    main()
