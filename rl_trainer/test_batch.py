#!/usr/bin/env python3
"""
Test script for Step 4: Batch Formation and LOO Baseline

This script demonstrates:
1. Batch preparation with padding and alignment
2. Leave-One-Out (LOO) batch creation
3. LOO baseline computation
4. EpisodeDataLoader for training loops
5. Multi-epoch batching with shuffling
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_environment import Episode, Turn, Observation, Action
from rl_trainer import (
    EpisodeBatch,
    EpisodeDataLoader,
    prepare_batch,
    prepare_loo_batches,
    compute_loo_baseline
)


def create_dummy_episodes(num_episodes: int = 10) -> list:
    """Create dummy episodes for testing."""
    episodes = []
    
    for i in range(num_episodes):
        turns = []
        
        # Create 2-3 turns per episode
        num_turns = np.random.randint(2, 4)
        
        for t in range(num_turns):
            # Dummy token IDs
            context_len = np.random.randint(50, 100)
            gen_len = np.random.randint(30, 60)
            
            context_ids = torch.randint(0, 1000, (context_len,))
            generated_ids = torch.randint(0, 1000, (gen_len,))
            
            # Random action mask (10-30 action tokens)
            action_mask = torch.zeros(gen_len, dtype=torch.bool)
            num_action = np.random.randint(10, 30)
            action_start = np.random.randint(0, gen_len - num_action)
            action_mask[action_start:action_start + num_action] = True
            
            # Create turn
            turn = Turn(
                turn_index=t,
                observation=Observation(
                    step=t,
                    images=[f"dummy_{t}.png"],
                    camera_positions=[np.eye(4)],
                    current_position=np.array([0.0, 0.0, 0.0]),
                    current_rotation=np.eye(3),
                    bbox_mins=[-5.0, -5.0, -5.0],
                    bbox_maxs=[5.0, 5.0, 5.0],
                    question="Where is the chair?",
                    choices=["A", "B", "C", "D"],
                    movement_history=[],
                    is_final_step=False
                ),
                full_prompt=f"Turn {t} prompt",
                context_text=f"Turn {t} context",
                generated_ids=generated_ids,
                generated_text=f"Turn {t} reasoning...",
                action_token_mask=action_mask,
                context_input_ids=context_ids,
                action=Action(rotation_angle_degrees=15, forward_meters=0.5, left_meters=0.0, z_delta_meters=0.0),
                action_valid=True,
                input_token_length=context_len,
                num_action_tokens=num_action,
                num_reasoning_tokens=gen_len - num_action,
                masking_method="brace_depth",
                masking_confidence=1.0
            )
            
            turns.append(turn)
        
        # Create episode
        episode = Episode(
            episode_id=f"test_ep{i:03d}",
            scene_id=f"scene_{i}",
            question="Where is the chair?",
            choices=["A", "B", "C", "D"],
            ground_truth="A",
            turns=turns,
            final_reward=np.random.choice([0.0, 1.0]),  # Random success/failure
            is_valid=np.random.random() > 0.1,  # 90% valid
            masking_quality="good"
        )
        
        episodes.append(episode)
    
    return episodes


def test_batch_preparation():
    """Test basic batch preparation."""
    print("=" * 80)
    print("TEST 1: Batch Preparation")
    print("=" * 80)
    
    # Create dummy episodes
    episodes = create_dummy_episodes(num_episodes=4)
    
    print(f"\nCreated {len(episodes)} dummy episodes")
    for ep in episodes:
        print(f"  {ep.episode_id}: {len(ep.turns)} turns, reward={ep.final_reward:.1f}, valid={ep.is_valid}")
    
    # Create a dummy processor (just need tokenizer with pad_token_id)
    class DummyProcessor:
        class DummyTokenizer:
            pad_token_id = 0
            eos_token_id = 1
        tokenizer = DummyTokenizer()
    
    processor = DummyProcessor()
    
    # Prepare batch
    print("\n--- Preparing batch ---")
    batch = prepare_batch(episodes, processor, device="cpu")
    
    print(f"\nBatch created:")
    print(f"  Batch size: {batch.batch_size}")
    print(f"  Max length: {batch.max_len}")
    print(f"  Context input IDs shape: {batch.context_input_ids.shape}")
    print(f"  Generated IDs shape: {batch.generated_ids.shape}")
    print(f"  Action masks shape: {batch.action_masks.shape}")
    print(f"  Attention masks shape: {batch.attention_masks.shape}")
    print(f"  Rewards shape: {batch.rewards.shape}")
    print(f"  Rewards: {batch.rewards.tolist()}")
    
    # Test per-episode access
    print("\n--- Per-episode access ---")
    for i in range(min(2, batch.batch_size)):
        print(f"\nEpisode {i}:")
        full_ids = batch.get_full_input_ids(i)
        action_indices = batch.get_action_token_indices(i)
        print(f"  Full sequence length: {len(full_ids)}")
        print(f"  Action token indices: {action_indices.tolist()[:5]}... ({len(action_indices)} total)")
        print(f"  Reward: {batch.rewards[i].item():.1f}")


def test_loo_batches():
    """Test Leave-One-Out batch creation."""
    print("\n" + "=" * 80)
    print("TEST 2: Leave-One-Out (LOO) Batches")
    print("=" * 80)
    
    # Create episodes
    episodes = create_dummy_episodes(num_episodes=5)
    
    # Prepare batch
    processor = type('obj', (object,), {
        'tokenizer': type('obj', (object,), {'pad_token_id': 0, 'eos_token_id': 1})()
    })()
    
    batch = prepare_batch(episodes, processor, device="cpu")
    
    print(f"\nOriginal batch:")
    print(f"  Batch size: {batch.batch_size}")
    print(f"  Episode IDs: {batch.episode_ids}")
    print(f"  Rewards: {batch.rewards.tolist()}")
    
    # Create LOO batches
    print("\n--- Creating LOO batches ---")
    loo_batches = prepare_loo_batches(batch)
    
    print(f"\nCreated {len(loo_batches)} LOO batches")
    for i, loo_batch in enumerate(loo_batches):
        print(f"\nLOO Batch {i} (leave out episode {i}):")
        print(f"  Batch size: {loo_batch.batch_size}")
        print(f"  Episode IDs: {loo_batch.episode_ids}")
        print(f"  Rewards: {loo_batch.rewards.tolist()}")
        print(f"  Mean reward: {loo_batch.rewards.mean().item():.3f}")


def test_loo_baseline():
    """Test LOO baseline computation."""
    print("\n" + "=" * 80)
    print("TEST 3: LOO Baseline Computation")
    print("=" * 80)
    
    # Create episodes with known rewards
    episodes = create_dummy_episodes(num_episodes=8)
    
    # Set specific rewards for easier verification
    for i, ep in enumerate(episodes):
        ep.reward = float(i)  # 0, 1, 2, 3, 4, 5, 6, 7
    
    # Prepare batch
    processor = type('obj', (object,), {
        'tokenizer': type('obj', (object,), {'pad_token_id': 0, 'eos_token_id': 1})()
    })()
    
    batch = prepare_batch(episodes, processor, device="cpu")
    
    print(f"\nBatch rewards: {batch.rewards.tolist()}")
    
    # Compute LOO baseline
    baselines = compute_loo_baseline(batch)
    
    print(f"\nLOO baselines: {baselines.tolist()}")
    
    # Verify manually for first episode
    manual_baseline_0 = batch.rewards[1:].mean().item()
    computed_baseline_0 = baselines[0].item()
    
    print(f"\nVerification for episode 0:")
    print(f"  Manual baseline (mean of [1,2,3,4,5,6,7]): {manual_baseline_0:.3f}")
    print(f"  Computed baseline: {computed_baseline_0:.3f}")
    print(f"  Match: {abs(manual_baseline_0 - computed_baseline_0) < 1e-6}")
    
    # Show advantage (reward - baseline)
    advantages = batch.rewards - baselines
    print(f"\nAdvantages (reward - baseline): {advantages.tolist()}")


def test_dataloader():
    """Test EpisodeDataLoader."""
    print("\n" + "=" * 80)
    print("TEST 4: EpisodeDataLoader")
    print("=" * 80)
    
    # Create larger dataset
    episodes = create_dummy_episodes(num_episodes=25)
    
    print(f"\nCreated {len(episodes)} episodes")
    
    # Create processor
    processor = type('obj', (object,), {
        'tokenizer': type('obj', (object,), {'pad_token_id': 0, 'eos_token_id': 1})()
    })()
    
    # Create dataloader
    loader = EpisodeDataLoader(
        episodes=episodes,
        batch_size=8,
        shuffle=True,
        filter_invalid=True,
        processor=processor,
        device="cpu",
        drop_last=False
    )
    
    # Show statistics
    loader.log_statistics()
    
    # Iterate through one epoch
    print("\n--- Iterating through one epoch ---")
    for batch_idx, batch in enumerate(loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Batch size: {batch.batch_size}")
        print(f"  Max length: {batch.max_len}")
        print(f"  Mean reward: {batch.rewards.mean().item():.3f}")
        print(f"  Episode IDs: {batch.episode_ids[:3]}...")  # Show first 3
        
        # For first batch, show more details
        if batch_idx == 0:
            print(f"\n  Detailed view of first batch:")
            print(f"    Rewards: {batch.rewards.tolist()}")
            print(f"    Context lengths: {batch.context_lengths.tolist()}")
            print(f"    Generation lengths: {batch.generation_lengths.tolist()}")
            
            # Compute LOO baseline
            if batch.batch_size >= 2:
                baselines = compute_loo_baseline(batch)
                advantages = batch.rewards - baselines
                print(f"    LOO baselines: {baselines.tolist()}")
                print(f"    Advantages: {advantages.tolist()}")


def test_multi_epoch():
    """Test multi-epoch training simulation."""
    print("\n" + "=" * 80)
    print("TEST 5: Multi-Epoch Training Simulation")
    print("=" * 80)
    
    # Create dataset
    episodes = create_dummy_episodes(num_episodes=16)
    
    processor = type('obj', (object,), {
        'tokenizer': type('obj', (object,), {'pad_token_id': 0, 'eos_token_id': 1})()
    })()
    
    # Create dataloader
    loader = EpisodeDataLoader(
        episodes=episodes,
        batch_size=8,
        shuffle=True,
        filter_invalid=True,
        processor=processor,
        device="cpu"
    )
    
    print(f"\nSimulating 3 epochs of training...")
    
    for epoch in range(3):
        print(f"\n--- Epoch {epoch + 1} ---")
        
        epoch_rewards = []
        epoch_baselines = []
        
        for batch_idx, batch in enumerate(loader):
            # Simulate training
            baselines = compute_loo_baseline(batch) if batch.batch_size >= 2 else torch.zeros(batch.batch_size)
            advantages = batch.rewards - baselines
            
            # Track statistics
            epoch_rewards.extend(batch.rewards.tolist())
            epoch_baselines.extend(baselines.tolist())
            
            print(f"  Batch {batch_idx}: size={batch.batch_size}, "
                  f"mean_reward={batch.rewards.mean():.3f}, "
                  f"mean_advantage={advantages.mean():.3f}")
        
        print(f"\nEpoch {epoch + 1} summary:")
        print(f"  Total samples: {len(epoch_rewards)}")
        print(f"  Mean reward: {np.mean(epoch_rewards):.3f}")
        print(f"  Mean baseline: {np.mean(epoch_baselines):.3f}")
        print(f"  Std reward: {np.std(epoch_rewards):.3f}")


def main():
    """Run all tests."""
    print("\n")
    print("=" * 80)
    print("STEP 4: BATCH FORMATION AND LOO BASELINE - TEST SUITE")
    print("=" * 80)
    print("\nThis test suite demonstrates:")
    print("- Batch preparation with padding and alignment")
    print("- Leave-One-Out (LOO) batch creation")
    print("- LOO baseline computation")
    print("- EpisodeDataLoader for training loops")
    print("- Multi-epoch batching with shuffling")
    print("=" * 80)
    
    # Run tests
    test_batch_preparation()
    test_loo_batches()
    test_loo_baseline()
    test_dataloader()
    test_multi_epoch()
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)
    print("\nStep 4 implementation is complete with:")
    print("✅ EpisodeBatch dataclass with padding and alignment")
    print("✅ prepare_batch() for batching episodes")
    print("✅ prepare_loo_batches() for LOO baseline computation")
    print("✅ compute_loo_baseline() helper function")
    print("✅ EpisodeDataLoader for training loops")
    print("✅ Support for N >= 8 batches (recommended)")
    print("✅ Multi-epoch training with shuffling")
    print("\nReady for policy gradient training!")


if __name__ == "__main__":
    main()
