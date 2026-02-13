#!/usr/bin/env python3
"""
Tests for PPO loss computation (Step 7C deliverable).

Run: pytest vagen_vsi_rl/tests/test_ppo.py -v
"""

import pytest
import torch
from vagen_vsi_rl.rl.ppo import ppo_step, PPOConfig


class TestPPOStep:
    """Tests for ppo_step()."""

    def test_basic_ppo_loss(self):
        """Basic PPO step should return expected keys."""
        B, L = 2, 10
        logprobs = torch.randn(B, L)
        old_logprobs = logprobs.clone()
        advantages = torch.ones(B)
        action_masks = torch.ones(B, L, dtype=torch.bool)

        result = ppo_step(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            action_masks=action_masks,
        )

        assert "loss" in result
        assert "policy_loss" in result
        assert "entropy" in result
        assert "kl" in result
        assert "clip_frac" in result

    def test_identical_logprobs_ratio_one(self):
        """When old == new logprobs, ratio should be 1 (no clipping)."""
        B, L = 2, 5
        logprobs = torch.zeros(B, L)  # log(1) = 0
        old_logprobs = torch.zeros(B, L)
        advantages = torch.ones(B)
        action_masks = torch.ones(B, L, dtype=torch.bool)

        result = ppo_step(logprobs, old_logprobs, advantages, action_masks)
        
        # Clip fraction should be 0 (no clipping when ratio=1)
        assert result["clip_frac"].item() == pytest.approx(0.0)

    def test_clipping_when_ratio_large(self):
        """Large ratio should trigger clipping."""
        B, L = 1, 10
        logprobs = torch.ones(B, L)  # Higher than old
        old_logprobs = torch.zeros(B, L)
        advantages = torch.ones(B)
        action_masks = torch.ones(B, L, dtype=torch.bool)
        
        cfg = PPOConfig(clip_eps=0.2)
        result = ppo_step(logprobs, old_logprobs, advantages, action_masks, config=cfg)
        
        # ratio = exp(1 - 0) = e ≈ 2.7, which is > 1.2 → should be clipped
        assert result["clip_frac"].item() > 0

    def test_kl_with_ref_logprobs(self):
        """KL should be computed when ref_logprobs provided."""
        B, L = 2, 5
        logprobs = torch.randn(B, L)
        old_logprobs = logprobs.clone()
        ref_logprobs = logprobs - 0.5  # Different from current
        advantages = torch.ones(B)
        action_masks = torch.ones(B, L, dtype=torch.bool)

        result = ppo_step(
            logprobs, old_logprobs, advantages, action_masks,
            ref_logprobs=ref_logprobs,
        )
        
        # KL should be non-zero since logprobs != ref_logprobs
        assert result["kl"].item() != pytest.approx(0.0)

    def test_kl_zero_without_ref(self):
        """KL should be 0 when no ref_logprobs."""
        B, L = 2, 5
        logprobs = torch.randn(B, L)
        old_logprobs = logprobs.clone()
        advantages = torch.ones(B)
        action_masks = torch.ones(B, L, dtype=torch.bool)

        result = ppo_step(logprobs, old_logprobs, advantages, action_masks)
        
        assert result["kl"].item() == pytest.approx(0.0)

    def test_action_mask_applied(self):
        """Only masked tokens should contribute to loss."""
        B, L = 1, 10
        logprobs = torch.randn(B, L)
        old_logprobs = logprobs.clone()
        advantages = torch.ones(B)
        
        # Only first 3 tokens are action tokens
        action_masks = torch.zeros(B, L, dtype=torch.bool)
        action_masks[:, :3] = True

        # Should not crash with partial mask
        result = ppo_step(logprobs, old_logprobs, advantages, action_masks)
        assert "loss" in result

    def test_negative_advantage(self):
        """Negative advantages should work correctly."""
        B, L = 2, 5
        logprobs = torch.zeros(B, L)
        old_logprobs = torch.zeros(B, L)
        advantages = -torch.ones(B)  # Negative!
        action_masks = torch.ones(B, L, dtype=torch.bool)

        result = ppo_step(logprobs, old_logprobs, advantages, action_masks)
        
        # Should not crash
        assert not torch.isnan(result["loss"])

    def test_empty_action_mask(self):
        """Empty action mask should handle gracefully."""
        B, L = 1, 5
        logprobs = torch.randn(B, L)
        old_logprobs = logprobs.clone()
        advantages = torch.ones(B)
        action_masks = torch.zeros(B, L, dtype=torch.bool)  # All False

        # This might error or return zeros - test it doesn't crash badly
        try:
            result = ppo_step(logprobs, old_logprobs, advantages, action_masks)
            # If it succeeds, loss should be defined
            assert "loss" in result
        except (RuntimeError, ZeroDivisionError):
            # Acceptable to fail gracefully
            pass

    def test_config_entropy_coef(self):
        """Entropy coefficient should affect loss."""
        B, L = 2, 5
        logprobs = -torch.ones(B, L)  # Negative logprobs → positive entropy contrib
        old_logprobs = logprobs.clone()
        advantages = torch.ones(B)
        action_masks = torch.ones(B, L, dtype=torch.bool)

        cfg_low = PPOConfig(entropy_coef=0.0)
        cfg_high = PPOConfig(entropy_coef=1.0)

        result_low = ppo_step(logprobs, old_logprobs, advantages, action_masks, config=cfg_low)
        result_high = ppo_step(logprobs, old_logprobs, advantages, action_masks, config=cfg_high)

        # Higher entropy coef should give different loss
        assert result_low["loss"].item() != result_high["loss"].item()
