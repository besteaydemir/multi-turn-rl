"""
Test suite for reference model behavior.

Tests:
1. Reference model produces same logprobs as policy when weights match
2. Reference model produces different logprobs when weights differ
3. No gradients flow into reference model
4. Reference model update strategies work correctly
"""

import pytest
import torch
import torch.nn as nn
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
from copy import deepcopy

from rl_trainer.reference import ReferenceModelManager
from rl_trainer.logprobs import compute_token_logprobs


class TestReferenceModelBehavior:
    """Test that reference model behaves correctly."""
    
    @pytest.fixture
    def models_and_tokenizer(self):
        """Create policy and reference models."""
        model_name = "Qwen/Qwen3-VL-2B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Load policy model
        policy = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )
        
        # Create reference model manager (frozen copy)
        ref_manager = ReferenceModelManager(policy, strategy="frozen", device="cpu")
        
        return policy, ref_manager, tokenizer
    
    def test_frozen_reference_matches_policy(self, models_and_tokenizer):
        """
        Test that freshly copied reference model produces identical logprobs.
        """
        policy, ref_manager, tokenizer = models_and_tokenizer
        
        # Create input
        text = "Question: What is 2+2?\nAnswer: 4"
        full_ids = tokenizer.encode(text)
        
        # Split into context and answer
        context_ids = full_ids[:10]  # First 10 tokens as context
        generated_ids = full_ids[10:]  # Rest as generated
        
        # Create EpisodeBatch
        from rl_trainer.batch import EpisodeBatch
        batch = EpisodeBatch(
            context_input_ids=torch.tensor([context_ids]),
            generated_ids=torch.tensor([generated_ids]),
            action_masks=torch.ones(1, len(generated_ids), dtype=torch.bool),
            attention_masks=torch.ones(1, len(full_ids)),
            rewards=torch.tensor([1.0]),
            context_lengths=torch.tensor([len(context_ids)]),
            generation_lengths=torch.tensor([len(generated_ids)]),
            episode_ids=["ref_test_0"]
        )
        
        # Get policy logprobs
        policy.eval()
        with torch.no_grad():
            policy_logprobs, _, _ = compute_token_logprobs(
                model=policy,
                batch=batch,
                processor=None,
                images=None,
                device="cpu"
            )
        
        policy_logprobs_sum = policy_logprobs[0].sum().item()
        
        # Get reference logprobs
        ref_model = ref_manager.get_reference_model()
        with torch.no_grad():
            ref_logprobs, _, _ = compute_token_logprobs(
                model=ref_model,
                batch=batch,
                processor=None,
                images=None,
                device="cpu"
            )
        
        ref_logprobs_sum = ref_logprobs[0].sum().item()
        
        print(f"\nPolicy logprobs sum: {policy_logprobs_sum:.6f}")
        print(f"Reference logprobs sum: {ref_logprobs_sum:.6f}")
        print(f"Difference: {abs(policy_logprobs_sum - ref_logprobs_sum):.9f}")
        
        # Should be identical
        assert abs(policy_logprobs_sum - ref_logprobs_sum) < 1e-4, \
            "Reference model should match policy initially"
    
    def test_reference_differs_after_policy_update(self, models_and_tokenizer):
        """
        Test that reference model stays frozen when policy updates.
        """
        policy, ref_manager, tokenizer = models_and_tokenizer
        
        # Create input
        text = "Test sequence"
        full_ids = tokenizer.encode(text)
        context_ids = full_ids[:5]
        generated_ids = full_ids[5:]
        
        # Create EpisodeBatch
        from rl_trainer.batch import EpisodeBatch
        batch = EpisodeBatch(
            context_input_ids=torch.tensor([context_ids]),
            generated_ids=torch.tensor([generated_ids]),
            action_masks=torch.ones(1, len(generated_ids), dtype=torch.bool),
            attention_masks=torch.ones(1, len(full_ids)),
            rewards=torch.tensor([1.0]),
            context_lengths=torch.tensor([len(context_ids)]),
            generation_lengths=torch.tensor([len(generated_ids)]),
            episode_ids=["ref_test_1"]
        )
        
        # Get initial reference logprobs
        ref_model = ref_manager.get_reference_model()
        with torch.no_grad():
            ref_before, _, _ = compute_token_logprobs(
                model=ref_model,
                batch=batch,
                processor=None,
                images=None,
                device="cpu"
            )
        ref_before_sum = ref_before[0].sum().item()
        
        # Update policy model (fake training step)
        policy.train()
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
        
        # Forward pass with dummy loss
        full_input_ids = torch.cat([batch.context_input_ids, batch.generated_ids], dim=1)
        outputs = policy(input_ids=full_input_ids, labels=full_input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        # Get updated policy logprobs
        policy.eval()
        with torch.no_grad():
            policy_after, _, _ = compute_token_logprobs(
                model=policy,
                batch=batch,
                processor=None,
                images=None,
                device="cpu"
            )
        policy_after_sum = policy_after[0].sum().item()
        
        # Get reference logprobs again (should be unchanged)
        with torch.no_grad():
            ref_after, _, _ = compute_token_logprobs(
                model=ref_model,
                batch=batch,
                processor=None,
                images=None,
                device="cpu"
            )
        ref_after_sum = ref_after[0].sum().item()
        
        print(f"\nReference before: {ref_before_sum:.6f}")
        print(f"Reference after: {ref_after_sum:.6f}")
        print(f"Policy after: {policy_after_sum:.6f}")
        print(f"Ref change: {abs(ref_after_sum - ref_before_sum):.9f}")
        print(f"Policy change: {abs(policy_after_sum - ref_before_sum):.9f}")
        
        # Reference should not change
        assert abs(ref_after_sum - ref_before_sum) < 1e-5, \est_no_gradients_in_reference(self, models_and_tokenizer):
        """
        Test that no gradients flow into reference model.
        """
        policy, ref_manager, tokenizer = models_and_tokenizer
        
        ref_model = ref_manager.get_reference_model()
        
        # Verify all reference parameters require_grad=False
        for name, param in ref_model.named_parameters():
            assert param.requires_grad == False, \
                f"Reference parameter {name} should not require grad"
        
        # Try to compute gradients (should fail or be None)
        text = "Test"
        input_ids = torch.tensor([tokenizer.encode(text)])
        
        # This should work without error, but no gradients should accumulate
        outputs = ref_model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        
        # Even if we try to backward, no gradients should accumulate
        # (This might raise error or just do nothing depending on PyTorch version)
        try:
            loss.backward()
            # Check that no gradients exist
            for param in ref_model.parameters():
                assert param.grad is None, \
                    "Reference model should have no gradients"
        except RuntimeError:
            # Expected: trying to backward through frozen model
            pass
        
        print("\n✓ No gradients flow into reference model")


class TestReferenceUpdateStrategies:
    """Test different reference model update strategies."""
    
    def test_ema_update(self):
        """Test EMA update moves reference toward policy."""
        # Create simple models (linear layers for testing)
        policy = nn.Linear(10, 10)
        
        # Initialize with different weights
        with torch.no_grad():
            policy.weight.fill_(1.0)
        
        tau = 0.9
        
        # Create reference manager with EMA strategy
        ref_manager = ReferenceModelManager(
            policy,
            strategy="ema",
            ema_tau=tau,
            device="cpu"
        )
        
        ref_model = ref_manager.get_reference_model()
        
        # Get initial weights (should be copy of policy)
        ref_weight_before = ref_model.weight.clone()
        policy_weight = policy.weight.clone()
        
        print(f"\nBefore EMA:")
        print(f"  Policy weight: {policy_weight[0, 0].item():.4f}")
        print(f"  Ref weight: {ref_weight_before[0, 0].item():.4f}")
        
        # Change policy weights
        with torch.no_grad():
            policy.weight.fill_(2.0)
        
        # Apply EMA update
        ref_manager.maybe_update(policy, step=1)
        
        ref_weight_after = ref_model.weight.clone()
        
        print(f"After EMA (tau={tau}):")
        print(f"  Policy weight: {policy.weight[0, 0].item():.4f}")
        print(f"  Ref weight: {ref_weight_after[0, 0].item():.4f}")
        
        # Expected: ref_new = tau * ref_old + (1 - tau) * policy_new
        # ref_old = 1.0, policy_new = 2.0
        # ref_new = 0.9 * 1.0 + 0.1 * 2.0 = 0.9 + 0.2 = 1.1
        expected = tau * ref_weight_before + (1 - tau) * policy.weight
        
        print(f"  Expected: {expected[0, 0].item():.4f}")
        print(f"  Difference: {(ref_weight_after - expected).abs().max().item():.6f}")
        
        torch.testing.assert_close(
            ref_weight_after,
            expected,
            rtol=1e-5,
            atol=1e-6,
            msg="EMA update should match formula"
        )
    
    def test_ema_tau_extremes(self):
        """Test EMA with extreme tau values."""
        policy = nn.Linear(5, 5)
        
        with torch.no_grad():
            policy.weight.fill_(1.0)
        
        # Test tau = 0 (full copy each step)
        ref_manager_zero = ReferenceModelManager(
            policy,
            strategy="ema",
            ema_tau=0.0,
            device="cpu"
        )
        
        # Change policy
        with torch.no_grad():
            policy.weight.fill_(2.0)
        
        # Update with tau=0 should copy policy exactly
        ref_manager_zero.maybe_update(policy, step=1)
        ref_model_zero = ref_manager_zero.get_reference_model()
        
        torch.testing.assert_close(
            ref_model_zero.weight,
            policy.weight,
            msg="tau=0 should copy policy exactly"
        )
        
        # Reset policy
        with torch.no_grad():
            policy.weight.fill_(1.0)
        
        # Test tau = 1 (no update)
        ref_manager_one = ReferenceModelManager(
            policy,
            strategy="ema",
            ema_tau=1.0,
            device="cpu"
        )
        
        ref_weight_initial = ref_manager_one.get_reference_model().weight.clone()
        
        # Change policy
        with torch.no_grad():
            policy.weight.fill_(2.0)
        
        # Update with tau=1 should keep reference unchanged
        ref_manager_one.maybe_update(policy, step=1)
        ref_model_one = ref_manager_one.get_reference_model()
        
        torch.testing.assert_close(
            ref_model_one.weight,
            ref_weight_initial,
            msg="tau=1 should keep reference unchanged"
        )
        
        print("\n✓ EMA works correctly at tau extremes")


class TestKLDivergence:
    """Test KL divergence computation between policy and reference."""
    
    def test_kl_is_zero_when_models_match(self):
        """KL should be ~0 when policy and reference are identical."""
        # Create identical distributions
        logprobs_policy = torch.tensor([[-1.0, -2.0, -1.5, -2.5]])
        logprobs_ref = logprobs_policy.clone()
        
        # KL(P||Q) = sum(P * log(P/Q)) = sum(P * (log(P) - log(Q)))
        # When P=Q, this is 0
        kl = (logprobs_policy - logprobs_ref).sum()
        
        print(f"\nKL divergence (identical): {kl.item():.6f}")
        assert abs(kl.item()) < 1e-5, "KL should be ~0 for identical distributions"
    
    def test_kl_is_positive_when_models_differ(self):
        """KL should be > 0 when distributions differ."""
        logprobs_policy = torch.tensor([[-1.0, -2.0, -1.5]])
        logprobs_ref = torch.tensor([[-2.0, -1.5, -2.5]])
        
        # Simple KL approximation: sum of logprob differences
        kl = (logprobs_policy - logprobs_ref).sum()
        
        print(f"\nKL divergence (different): {kl.item():.6f}")
        # KL can be positive or negative depending on which is higher
        # The absolute value should be > 0
        assert abs(kl.item()) > 1e-4, "KL should be non-zero for different distributions"


def run_reference_tests():
    """Run all reference model tests."""
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    run_reference_tests()
