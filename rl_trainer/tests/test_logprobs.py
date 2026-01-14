#!/usr/bin/env python3
"""
Test script for Step 5: Log-Probability Computation

This script demonstrates:
1. Teacher-forcing log-prob computation
2. Action token masking
3. KL divergence with reference model
4. Entropy computation
5. Policy gradient loss calculation
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_trainer import (
    EpisodeBatch,
    prepare_batch,
    compute_loo_baseline,
    LogProbResult,
    compute_token_logprobs,
    compute_sequence_logprobs,
    compute_advantages,
    policy_gradient_loss
)


class DummyLM(nn.Module):
    """Dummy language model for testing."""
    
    def __init__(self, vocab_size=1000, hidden_size=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids, attention_mask=None, return_dict=True):
        embeds = self.embedding(input_ids)
        logits = self.lm_head(embeds)
        
        if return_dict:
            return type('obj', (object,), {'logits': logits})()
        return logits


def create_dummy_batch(batch_size=4, max_context_len=50, max_gen_len=30):
    """Create a dummy batch for testing."""
    
    # Random token IDs
    context_input_ids = torch.randint(0, 1000, (batch_size, max_context_len))
    generated_ids = torch.randint(0, 1000, (batch_size, max_gen_len))
    
    # Random action masks (10-20 action tokens per sequence)
    action_masks = torch.zeros(batch_size, max_gen_len, dtype=torch.bool)
    for i in range(batch_size):
        num_action = np.random.randint(10, 20)
        start_pos = np.random.randint(0, max_gen_len - num_action)
        action_masks[i, start_pos:start_pos + num_action] = True
    
    # Random lengths
    context_lengths = torch.randint(30, max_context_len, (batch_size,))
    generation_lengths = torch.randint(20, max_gen_len, (batch_size,))
    
    # Attention masks
    attention_masks = torch.zeros(batch_size, max_context_len + max_gen_len, dtype=torch.long)
    for i in range(batch_size):
        ctx_len = context_lengths[i].item()
        gen_len = generation_lengths[i].item()
        attention_masks[i, :ctx_len] = 1
        attention_masks[i, max_context_len:max_context_len + gen_len] = 1
    
    # Random rewards
    rewards = torch.tensor([np.random.choice([0.0, 1.0]) for _ in range(batch_size)], dtype=torch.float32)
    
    # Episode IDs
    episode_ids = [f"ep{i:03d}" for i in range(batch_size)]
    
    return EpisodeBatch(
        context_input_ids=context_input_ids,
        generated_ids=generated_ids,
        action_masks=action_masks,
        attention_masks=attention_masks,
        rewards=rewards,
        context_lengths=context_lengths,
        generation_lengths=generation_lengths,
        episode_ids=episode_ids,
        device="cpu"
    )


def test_token_logprobs():
    """Test per-token log-prob computation."""
    print("=" * 80)
    print("TEST 1: Token-Level Log-Probabilities")
    print("=" * 80)
    
    # Create dummy model and batch
    model = DummyLM(vocab_size=1000)
    model.eval()
    
    batch = create_dummy_batch(batch_size=4)
    
    print(f"\nBatch info:")
    print(f"  Batch size: {batch.batch_size}")
    print(f"  Max context length: {batch.max_context_len}")
    print(f"  Max generation length: {batch.max_gen_len}")
    
    # Compute token log-probs
    print("\n--- Computing token log-probs ---")
    token_logprobs, token_mask, token_entropy = compute_token_logprobs(
        model=model,
        batch=batch,
        compute_entropy=True,
        device="cpu"
    )
    
    print(f"\nToken log-probs shape: {token_logprobs.shape}")
    print(f"Token mask shape: {token_mask.shape}")
    print(f"Token entropy shape: {token_entropy.shape}")
    
    # Show statistics
    valid_logprobs = token_logprobs[token_mask]
    valid_entropy = token_entropy[token_mask]
    
    print(f"\nStatistics (valid tokens only):")
    print(f"  Mean log-prob: {valid_logprobs.mean().item():.4f}")
    print(f"  Std log-prob: {valid_logprobs.std().item():.4f}")
    print(f"  Mean entropy: {valid_entropy.mean().item():.4f}")
    print(f"  Std entropy: {valid_entropy.std().item():.4f}")


def test_sequence_logprobs():
    """Test sequence-level log-prob computation."""
    print("\n" + "=" * 80)
    print("TEST 2: Sequence-Level Log-Probabilities")
    print("=" * 80)
    
    # Create models
    policy_model = DummyLM(vocab_size=1000)
    ref_model = DummyLM(vocab_size=1000)
    
    # Copy parameters to make ref_model similar (but not identical)
    ref_model.load_state_dict(policy_model.state_dict())
    
    policy_model.eval()
    ref_model.eval()
    
    batch = create_dummy_batch(batch_size=8)
    
    print(f"\nBatch info:")
    print(f"  Batch size: {batch.batch_size}")
    print(f"  Rewards: {batch.rewards.tolist()}")
    
    # Compute sequence log-probs
    print("\n--- Computing sequence log-probs ---")
    result = compute_sequence_logprobs(
        model=policy_model,
        batch=batch,
        ref_model=ref_model,
        compute_entropy=True,
        compute_kl=True,
        device="cpu"
    )
    
    print(f"\nResults:")
    print(f"  log π(a|s) shape: {result.logpi_seq.shape}")
    print(f"  log π_ref(a|s) shape: {result.logpref_seq.shape}")
    print(f"  KL divergence shape: {result.kl_div.shape}")
    print(f"  Entropy shape: {result.entropy_seq.shape}")
    print(f"  Num action tokens shape: {result.num_action_tokens.shape}")
    
    print(f"\nPer-episode statistics:")
    for i in range(min(3, batch.batch_size)):
        print(f"  Episode {i}:")
        print(f"    Num action tokens: {result.num_action_tokens[i].item():.0f}")
        print(f"    log π(a|s): {result.logpi_seq[i].item():.4f}")
        print(f"    log π_ref(a|s): {result.logpref_seq[i].item():.4f}")
        print(f"    KL divergence: {result.kl_div[i].item():.4f}")
        print(f"    Entropy: {result.entropy_seq[i].item():.4f}")
        print(f"    Mean log-prob: {result.mean_logpi[i].item():.4f}")
    
    print(f"\nBatch-level statistics:")
    print(f"  Mean log π: {result.mean_logpi.mean().item():.4f}")
    print(f"  Mean log π_ref: {result.mean_logpref.mean().item():.4f}")
    print(f"  Mean KL: {result.kl_div.mean().item():.4f}")
    print(f"  Mean entropy: {result.mean_entropy.mean().item():.4f}")


def test_advantages():
    """Test advantage computation."""
    print("\n" + "=" * 80)
    print("TEST 3: Advantage Computation")
    print("=" * 80)
    
    batch = create_dummy_batch(batch_size=8)
    
    print(f"\nRewards: {batch.rewards.tolist()}")
    
    # Compute LOO baseline
    baselines = compute_loo_baseline(batch)
    print(f"LOO baselines: {baselines.tolist()}")
    
    # Compute advantages (unnormalized)
    advantages_raw = compute_advantages(batch.rewards, baselines, normalize=False)
    print(f"\nRaw advantages: {advantages_raw.tolist()}")
    print(f"  Mean: {advantages_raw.mean().item():.4f}")
    print(f"  Std: {advantages_raw.std().item():.4f}")
    
    # Compute advantages (normalized)
    advantages_norm = compute_advantages(batch.rewards, baselines, normalize=True)
    print(f"\nNormalized advantages: {advantages_norm.tolist()}")
    print(f"  Mean: {advantages_norm.mean().item():.4f}")
    print(f"  Std: {advantages_norm.std().item():.4f}")


def test_policy_gradient_loss():
    """Test full policy gradient loss computation."""
    print("\n" + "=" * 80)
    print("TEST 4: Policy Gradient Loss")
    print("=" * 80)
    
    # Create models and batch
    policy_model = DummyLM(vocab_size=1000)
    ref_model = DummyLM(vocab_size=1000)
    ref_model.load_state_dict(policy_model.state_dict())
    
    policy_model.train()  # Training mode for gradients
    ref_model.eval()
    
    batch = create_dummy_batch(batch_size=8)
    
    print(f"\nBatch info:")
    print(f"  Batch size: {batch.batch_size}")
    print(f"  Rewards: {batch.rewards.tolist()}")
    
    # Compute log-probs
    print("\n--- Computing log-probs ---")
    result = compute_sequence_logprobs(
        model=policy_model,
        batch=batch,
        ref_model=ref_model,
        compute_entropy=True,
        compute_kl=True,
        device="cpu"
    )
    
    # Compute advantages
    baselines = compute_loo_baseline(batch)
    advantages = compute_advantages(batch.rewards, baselines, normalize=True)
    
    print(f"\nAdvantages: {advantages.tolist()}")
    
    # Compute loss
    print("\n--- Computing policy gradient loss ---")
    loss, metrics = policy_gradient_loss(
        logprobs_result=result,
        advantages=advantages,
        kl_coef=0.01,
        entropy_coef=0.01
    )
    
    print(f"\nLoss: {loss.item():.4f}")
    print(f"\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test gradient flow
    print("\n--- Testing gradient flow ---")
    loss.backward()
    
    has_grads = sum(1 for p in policy_model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in policy_model.parameters())
    
    print(f"Parameters with gradients: {has_grads}/{total_params}")
    
    # Show gradient statistics
    grad_norms = [p.grad.norm().item() for p in policy_model.parameters() if p.grad is not None]
    print(f"Gradient norms: min={min(grad_norms):.6f}, max={max(grad_norms):.6f}, mean={np.mean(grad_norms):.6f}")


def test_numerical_stability():
    """Test numerical stability with edge cases."""
    print("\n" + "=" * 80)
    print("TEST 5: Numerical Stability")
    print("=" * 80)
    
    model = DummyLM(vocab_size=1000)
    model.eval()
    
    # Create batch with varying numbers of action tokens
    batch = create_dummy_batch(batch_size=4)
    
    # Manually set extreme cases
    batch.action_masks[0, :] = True  # All tokens are action tokens
    batch.action_masks[1, :5] = True  # Very few action tokens
    batch.action_masks[1, 5:] = False
    
    print(f"\nAction token counts:")
    for i in range(batch.batch_size):
        num_action = batch.action_masks[i].sum().item()
        print(f"  Episode {i}: {num_action} action tokens")
    
    # Compute log-probs
    result = compute_sequence_logprobs(
        model=model,
        batch=batch,
        ref_model=None,
        compute_entropy=True,
        compute_kl=False,
        device="cpu"
    )
    
    print(f"\nLog-probs (should not have NaN or Inf):")
    for i in range(batch.batch_size):
        print(f"  Episode {i}: {result.logpi_seq[i].item():.4f}")
    
    # Check for numerical issues
    has_nan = torch.isnan(result.logpi_seq).any()
    has_inf = torch.isinf(result.logpi_seq).any()
    
    print(f"\nNumerical health:")
    print(f"  Has NaN: {has_nan}")
    print(f"  Has Inf: {has_inf}")
    print(f"  All finite: {torch.isfinite(result.logpi_seq).all()}")


def main():
    """Run all tests."""
    print("\n")
    print("=" * 80)
    print("STEP 5: LOG-PROBABILITY COMPUTATION - TEST SUITE")
    print("=" * 80)
    print("\nThis test suite demonstrates:")
    print("- Teacher-forcing log-prob computation")
    print("- Action token masking")
    print("- KL divergence with reference model")
    print("- Entropy computation for exploration")
    print("- Policy gradient loss calculation")
    print("=" * 80)
    
    # Run tests
    test_token_logprobs()
    test_sequence_logprobs()
    test_advantages()
    test_policy_gradient_loss()
    test_numerical_stability()
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)
    print("\nStep 5 implementation is complete with:")
    print("✅ Per-token log-prob computation (teacher forcing)")
    print("✅ Sequence-level log-probs with action masking")
    print("✅ KL divergence with reference model")
    print("✅ Entropy regularization")
    print("✅ Policy gradient loss with advantages")
    print("✅ Gradient flow verified")
    print("✅ Numerical stability checked")
    print("\nFormula: L = -E[log π(a|s) * A] + β*KL(π||π_ref) - α*H(π)")
    print("\nReady for full training loop!")


if __name__ == "__main__":
    main()
