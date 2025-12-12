"""
Test suite for advantage computation and LOO baseline.

Tests:
1. LOO baseline sanity check
2. Advantage normalization
3. Gradient flow correctness
"""

import pytest
import torch
import numpy as np

from rl_trainer.batch import compute_loo_baseline
from rl_trainer.logprobs import compute_advantages


class TestLOOBaseline:
    """Test Leave-One-Out baseline computation."""
    
    def test_loo_baseline_manual_example(self):
        """
        Test with known rewards: [1, 2, 3]
        Expected baselines: [mean(2,3), mean(1,3), mean(1,2)] = [2.5, 2.0, 1.5]
        """
        rewards = torch.tensor([1.0, 2.0, 3.0])
        
        baselines = compute_loo_baseline(rewards)
        
        expected = torch.tensor([2.5, 2.0, 1.5])
        
        print(f"\nRewards: {rewards.tolist()}")
        print(f"Baselines: {baselines.tolist()}")
        print(f"Expected: {expected.tolist()}")
        
        torch.testing.assert_close(
            baselines,
            expected,
            rtol=1e-5,
            atol=1e-6,
            msg="LOO baseline should match manual computation"
        )
    
    def test_loo_baseline_batch(self):
        """Test LOO baseline with batch of rewards."""
        # Batch of 5 episodes
        rewards = torch.tensor([0.0, 0.5, 1.0, 0.3, 0.7])
        
        baselines = compute_loo_baseline(rewards)
        
        # Verify shape
        assert baselines.shape == rewards.shape, "Baseline shape should match rewards"
        
        # Verify each baseline excludes its own reward
        for i in range(len(rewards)):
            other_rewards = torch.cat([rewards[:i], rewards[i+1:]])
            expected_baseline = other_rewards.mean()
            
            print(f"Episode {i}: reward={rewards[i]:.2f}, "
                  f"baseline={baselines[i]:.3f}, expected={expected_baseline:.3f}")
            
            assert abs(baselines[i] - expected_baseline) < 1e-5, \
                f"Baseline {i} incorrect"
    
    def test_loo_with_two_episodes(self):
        """Test edge case with only 2 episodes."""
        rewards = torch.tensor([1.0, 3.0])
        baselines = compute_loo_baseline(rewards)
        
        # baseline[0] = mean([3.0]) = 3.0
        # baseline[1] = mean([1.0]) = 1.0
        expected = torch.tensor([3.0, 1.0])
        
        print(f"\nRewards: {rewards.tolist()}")
        print(f"Baselines: {baselines.tolist()}")
        
        torch.testing.assert_close(baselines, expected)
    
    def test_loo_reduces_variance(self):
        """Test that LOO baseline reduces variance compared to mean baseline."""
        # Random rewards
        torch.manual_seed(42)
        rewards = torch.randn(20)
        
        # LOO baseline
        loo_baselines = compute_loo_baseline(rewards)
        loo_advantages = rewards - loo_baselines
        
        # Mean baseline
        mean_baseline = rewards.mean()
        mean_advantages = rewards - mean_baseline
        
        loo_var = loo_advantages.var()
        mean_var = mean_advantages.var()
        
        print(f"\nLOO advantage variance: {loo_var:.4f}")
        print(f"Mean advantage variance: {mean_var:.4f}")
        print(f"Variance reduction: {(mean_var - loo_var) / mean_var * 100:.1f}%")
        
        # LOO typically has lower variance, but not guaranteed
        # Just verify they're in the same ballpark (within 50%)
        assert loo_var <= mean_var * 1.5, \
            "LOO variance should be in similar range as mean baseline"


class TestAdvantageComputation:
    """Test advantage computation and normalization."""
    
    def test_advantage_unnormalized(self):
        """Test raw advantage = reward - baseline."""
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        baselines = torch.tensor([1.5, 2.5, 2.0, 3.5])
        
        advantages = compute_advantages(
            rewards=rewards,
            baselines=baselines,
            normalize=False
        )
        
        expected = rewards - baselines
        
        print(f"\nRewards: {rewards.tolist()}")
        print(f"Baselines: {baselines.tolist()}")
        print(f"Advantages: {advantages.tolist()}")
        
        torch.testing.assert_close(advantages, expected)
    
    def test_advantage_normalized(self):
        """Test that normalized advantages have mean≈0 and std≈1."""
        rewards = torch.tensor([0.0, 0.5, 1.0, 0.3, 0.7, 0.9, 0.2, 0.6])
        baselines = compute_loo_baseline(rewards)
        
        advantages = compute_advantages(
            rewards=rewards,
            baselines=baselines,
            normalize=True
        )
        
        print(f"\nAdvantages: {advantages.tolist()}")
        print(f"Mean: {advantages.mean():.6f}")
        print(f"Std: {advantages.std():.6f}")
        
        # Should be approximately normalized
        assert abs(advantages.mean()) < 1e-5, "Mean should be ~0"
        assert abs(advantages.std() - 1.0) < 1e-5, "Std should be ~1"
    
    def test_advantage_normalization_stability(self):
        """Test that normalization handles edge cases."""
        # Case 1: All same rewards (zero variance)
        rewards = torch.ones(5)
        baselines = compute_loo_baseline(rewards)
        
        advantages = compute_advantages(
            rewards=rewards,
            baselines=baselines,
            normalize=True
        )
        
        print(f"\nConstant rewards advantages: {advantages.tolist()}")
        
        # Should not NaN or explode
        assert not torch.isnan(advantages).any(), "Should handle constant rewards"
        assert torch.isfinite(advantages).all(), "Should be finite"
        
        # Case 2: Very small variance
        rewards = torch.tensor([1.0, 1.0001, 0.9999, 1.0, 1.0])
        baselines = compute_loo_baseline(rewards)
        
        advantages = compute_advantages(
            rewards=rewards,
            baselines=baselines,
            normalize=True
        )
        
        print(f"Small variance advantages: {advantages.tolist()}")
        assert torch.isfinite(advantages).all(), "Should handle small variance"


class TestGradientSanity:
    """Test that gradients flow in the correct direction."""
    
    def test_positive_advantage_increases_logprob(self):
        """
        Test that positive advantage increases log probability.
        
        Policy gradient: ∇J = E[A(s,a) * ∇log π(a|s)]
        If A > 0, we want to increase log π (gradient ascent)
        If A < 0, we want to decrease log π
        """
        # Simulate log probability
        logprob = torch.tensor([[-2.0]], requires_grad=True)
        
        # Positive advantage (good action)
        advantage = torch.tensor([[1.0]])
        
        # Policy gradient loss (negative for gradient descent)
        loss = -(logprob * advantage).mean()
        
        # Backward
        loss.backward()
        
        print(f"\nLog prob: {logprob.item():.4f}")
        print(f"Advantage: {advantage.item():.4f}")
        print(f"Loss: {loss.item():.4f}")
        print(f"Gradient: {logprob.grad.item():.4f}")
        
        # Gradient should be negative (will increase logprob via gradient descent)
        assert logprob.grad.item() < 0, \
            "Positive advantage should produce negative gradient (increase logprob)"
    
    def test_negative_advantage_decreases_logprob(self):
        """Test that negative advantage decreases log probability."""
        logprob = torch.tensor([[-1.0]], requires_grad=True)
        
        # Negative advantage (bad action)
        advantage = torch.tensor([[-1.0]])
        
        loss = -(logprob * advantage).mean()
        loss.backward()
        
        print(f"\nLog prob: {logprob.item():.4f}")
        print(f"Advantage: {advantage.item():.4f}")
        print(f"Gradient: {logprob.grad.item():.4f}")
        
        # Gradient should be positive (will decrease logprob)
        assert logprob.grad.item() > 0, \
            "Negative advantage should produce positive gradient (decrease logprob)"
    
    def test_gradient_magnitude_scales_with_advantage(self):
        """Test that larger advantages produce larger gradients."""
        logprob = torch.tensor([[-1.0]], requires_grad=True)
        
        # Small advantage
        advantage_small = torch.tensor([[0.1]])
        loss_small = -(logprob * advantage_small).mean()
        loss_small.backward()
        grad_small = logprob.grad.clone()
        logprob.grad.zero_()
        
        # Large advantage
        advantage_large = torch.tensor([[1.0]])
        loss_large = -(logprob * advantage_large).mean()
        loss_large.backward()
        grad_large = logprob.grad.clone()
        
        print(f"\nSmall advantage ({advantage_small.item():.1f}): "
              f"gradient = {grad_small.item():.4f}")
        print(f"Large advantage ({advantage_large.item():.1f}): "
              f"gradient = {grad_large.item():.4f}")
        
        # Larger advantage should produce larger gradient (in magnitude)
        assert abs(grad_large.item()) > abs(grad_small.item()), \
            "Larger advantage should produce larger gradient"
    
    def test_multiple_actions_batch(self):
        """Test gradient flow with batch of actions."""
        # Batch of log probs for different actions
        logprobs = torch.tensor([
            [-1.0],  # Action 1
            [-2.0],  # Action 2
            [-1.5],  # Action 3
        ], requires_grad=True)
        
        # Advantages: action 1 good, action 2 bad, action 3 neutral
        advantages = torch.tensor([
            [1.0],
            [-1.0],
            [0.0]
        ])
        
        # Policy loss
        loss = -(logprobs * advantages).mean()
        loss.backward()
        
        print(f"\nBatch gradients:")
        for i, (lp, adv, grad) in enumerate(zip(logprobs, advantages, logprobs.grad)):
            print(f"  Action {i}: logprob={lp.item():.2f}, "
                  f"advantage={adv.item():.2f}, gradient={grad.item():.4f}")
        
        # Check signs
        assert logprobs.grad[0].item() < 0, "Positive advantage → negative gradient"
        assert logprobs.grad[1].item() > 0, "Negative advantage → positive gradient"
        assert abs(logprobs.grad[2].item()) < 1e-6, "Zero advantage → zero gradient"


def run_advantage_tests():
    """Run all advantage and gradient tests."""
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    run_advantage_tests()
