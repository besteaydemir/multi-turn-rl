#!/usr/bin/env python3
"""
PPO clipped-surrogate loss computation.

One call to ``ppo_step`` takes a **batch of turns** (from multiple
trajectories) and returns a scalar loss.  The caller is responsible
for ``.backward()`` and optimiser stepping.

⚠️  CRITICAL: Vision Token Masking
----------------------------------
The ``action_masks`` parameter MUST exclude observation tokens (prompt + images).
Only generated action tokens should contribute to the PPO loss.

This is a known failure mode from the VAGEN paper:
    "Observation tokens dominating the loss leads to degenerate policies."

Use ``vagen_vsi_rl.utils.token_utils.create_action_mask()`` to build proper masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class PPOConfig:
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    kl_coef: float = 0.05
    max_grad_norm: float = 1.0


def ppo_step(
    logprobs: torch.Tensor,          # (B, seq_len)  current policy log π_θ
    old_logprobs: torch.Tensor,      # (B, seq_len)  rollout policy log π_old
    advantages: torch.Tensor,        # (B,) or (B, seq_len)
    action_masks: torch.Tensor,      # (B, seq_len) bool — which tokens are action
    ref_logprobs: torch.Tensor | None = None,  # (B, seq_len) frozen reference
    config: PPOConfig | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute the PPO clipped surrogate loss.

    Returns a dict of scalar tensors::

        {
          "loss":        total loss (negate before .backward if maximising reward),
          "policy_loss": clipped surrogate,
          "entropy":     mean entropy bonus,
          "kl":          approx KL from ref (0 if ref_logprobs is None),
          "clip_frac":   fraction of ratios clipped,
        }
    """
    cfg = config or PPOConfig()

    # flatten to action tokens only
    mask = action_masks.bool()
    lp = logprobs[mask]           # (N,)
    old_lp = old_logprobs[mask]   # (N,)

    # advantages — broadcast if scalar per trajectory
    # Handle multiple cases:
    # - advantages shape (B,) → expand to (B, seq_len)
    # - advantages shape (B, 1) → expand to (B, seq_len) 
    # - advantages shape (B, seq_len) → use directly
    if advantages.numel() == logprobs.shape[0]:
        # One advantage per trajectory: expand to match logprobs shape
        adv_expanded = advantages.view(-1, 1).expand_as(logprobs)
        adv = adv_expanded[mask]
    elif advantages.shape == logprobs.shape:
        # Per-token advantages: use directly
        adv = advantages[mask]
    else:
        raise ValueError(f"Cannot broadcast advantages {advantages.shape} to logprobs {logprobs.shape}")

    # ratio  r = π_θ / π_old
    ratio = (lp - old_lp).exp()

    # clipped surrogate
    surr1 = ratio * adv
    surr2 = ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
    policy_loss = -torch.min(surr1, surr2).mean()

    # entropy bonus (approximate: −logprob is a proxy)
    entropy = -lp.mean()

    # KL from reference
    kl = torch.tensor(0.0, device=logprobs.device)
    if ref_logprobs is not None:
        ref_lp = ref_logprobs[mask]
        kl = (lp - ref_lp).mean()  # approx KL

    # total loss
    loss = policy_loss - cfg.entropy_coef * entropy + cfg.kl_coef * kl

    clip_frac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean()

    return {
        "loss": loss,
        "policy_loss": policy_loss,
        "entropy": entropy,
        "kl": kl,
        "clip_frac": clip_frac,
    }
