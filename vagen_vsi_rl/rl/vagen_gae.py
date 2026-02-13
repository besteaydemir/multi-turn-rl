#!/usr/bin/env python3
"""
VAGEN-style Token-Level and Bi-Level GAE with Critic Loss.

This module implements the exact formulation from the VAGEN paper:

Part I: VAGEN-Base (Token-Level GAE)
====================================
- Per-token KL penalty: r_i^KL = -β * KL(π_θ || π_ref)
- Task reward added only to final action token
- TD error across action tokens only (skip observation tokens)
- Token-level GAE backward recursion

Part II: VAGEN-Full (Bi-Level GAE)
===================================
- Turn-level values V_t = V_φ(τ_{≤a_t})
- Turn-level GAE for macro advantages
- Token-level GAE within each turn, initialized from turn advantage
- Final critic targets G_{t,i} = A_{t,i}^token + V_φ(τ_{<(t,i)})

All targets must be DETACHED from computation graph.
Observation tokens are masked everywhere via M_i^loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class VAGENConfig:
    """Configuration for VAGEN GAE computation."""
    
    # Discount factors
    gamma: float = 1.0
    gamma_turn: float = 1.0
    gamma_token: float = 1.0
    
    # GAE lambda values
    lambda_turn: float = 0.95
    lambda_token: float = 0.95
    
    # KL penalty coefficient (β in paper)
    kl_coef: float = 0.05
    
    # Normalization
    normalize_advantages: bool = True


@dataclass
class TokenData:
    """Per-token data for a single turn."""
    
    token_ids: torch.Tensor           # (seq_len,) generated token IDs
    logprobs: torch.Tensor            # (seq_len,) log π_θ(a_i | τ_{<i})
    ref_logprobs: torch.Tensor        # (seq_len,) log π_ref(a_i | τ_{<i})
    values: torch.Tensor              # (seq_len,) V_φ(τ_{<i}) for each position
    action_mask: torch.Tensor         # (seq_len,) bool - True for action tokens
    
    # Computed by GAE
    advantages: Optional[torch.Tensor] = None      # (seq_len,)
    returns: Optional[torch.Tensor] = None         # (seq_len,) G_i targets


@dataclass  
class TurnData:
    """Turn-level data for bi-level GAE."""
    
    turn_index: int
    token_data: TokenData
    
    # Turn-level labels
    turn_reward: float = 0.0          # r_t = r_t^reason + r_t^format + R(s_t, a_t)
    turn_value: float = 0.0           # V_t = V_φ(τ_{≤a_t})
    
    # Computed by bi-level GAE
    turn_advantage: float = 0.0       # A_t^turn


# ═══════════════════════════════════════════════════════════════════════
# Part I: VAGEN-Base Token-Level GAE
# ═══════════════════════════════════════════════════════════════════════

def compute_kl_rewards(
    logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    kl_coef: float = 0.05,
) -> torch.Tensor:
    """
    Compute per-token KL penalty as reward.
    
    r_i^KL = -β * KL(π_θ || π_ref) ≈ -β * (log π_θ - log π_ref)
    
    Returns
    -------
    kl_rewards : (seq_len,) per-token KL penalty rewards
    """
    # Approximate KL per token: log π_θ(a_i) - log π_ref(a_i)
    kl_per_token = logprobs - ref_logprobs
    return -kl_coef * kl_per_token


def compute_token_level_gae(
    token_data: TokenData,
    terminal_reward: float = 0.0,
    cfg: VAGENConfig = None,
) -> TokenData:
    """
    VAGEN-Base: Token-level GAE across generated action tokens.
    
    Algorithm:
    1. Initialize rewards: r_i = r_i^KL (per-token KL penalty)
    2. Add terminal reward to final action token: r_{i*} += R(τ)
    3. Compute TD error: δ_i = r_i + γ V(τ_{<j}) - V(τ_{<i})
       where j is the next action token index
    4. Backward GAE: A_i = δ_i + γλ A_j
    5. Return targets: G_i = A_i + V(τ_{<i})
    """
    cfg = cfg or VAGENConfig()
    device = token_data.logprobs.device
    
    # Get action token indices
    action_mask = token_data.action_mask.bool()
    action_indices = torch.where(action_mask)[0]
    
    if len(action_indices) == 0:
        # No action tokens - return zeros
        token_data.advantages = torch.zeros_like(token_data.logprobs)
        token_data.returns = torch.zeros_like(token_data.logprobs)
        return token_data
    
    # Step 1: Compute per-token KL rewards
    kl_rewards = compute_kl_rewards(
        token_data.logprobs,
        token_data.ref_logprobs,
        cfg.kl_coef,
    )  # (seq_len,)
    
    # Step 2: Create reward tensor, add terminal reward to final action token
    rewards = kl_rewards.clone()
    final_action_idx = action_indices[-1]
    rewards[final_action_idx] = rewards[final_action_idx] + terminal_reward
    
    # Step 3-4: Compute TD errors and GAE backward over action tokens
    num_action_tokens = len(action_indices)
    advantages = torch.zeros(token_data.logprobs.shape[0], device=device)
    
    gae = torch.tensor(0.0, device=device)
    
    for i in reversed(range(num_action_tokens)):
        idx = action_indices[i]
        
        # Current value
        V_i = token_data.values[idx]
        
        # Next action token value (or 0 if terminal)
        if i + 1 < num_action_tokens:
            next_idx = action_indices[i + 1]
            V_next = token_data.values[next_idx]
        else:
            V_next = torch.tensor(0.0, device=device)
        
        # TD error: δ_i = r_i + γ V_next - V_i
        delta = rewards[idx] + cfg.gamma * V_next - V_i
        
        # GAE: A_i = δ_i + γλ A_{i+1}
        gae = delta + cfg.gamma * cfg.lambda_token * gae
        advantages[idx] = gae
    
    # Step 5: Return targets G_i = A_i + V_i
    returns = advantages + token_data.values
    
    # Mask out observation tokens (set to 0)
    advantages = advantages * action_mask.float()
    returns = returns * action_mask.float()
    
    token_data.advantages = advantages
    token_data.returns = returns.detach()  # CRITICAL: detach targets
    
    return token_data


# ═══════════════════════════════════════════════════════════════════════
# Part II: VAGEN-Full Bi-Level GAE
# ═══════════════════════════════════════════════════════════════════════

def compute_turn_level_gae(
    turns: List[TurnData],
    cfg: VAGENConfig = None,
) -> List[TurnData]:
    """
    Stage 1 of Bi-Level GAE: Turn-level advantage computation.
    
    δ_t^turn = r_t + γ_turn V_{t+1} - V_t
    A_t^turn = δ_t^turn + γ_turn λ_turn A_{t+1}^turn
    """
    cfg = cfg or VAGENConfig()
    T = len(turns)
    
    if T == 0:
        return turns
    
    gae = 0.0
    for t in reversed(range(T)):
        r_t = turns[t].turn_reward
        V_t = turns[t].turn_value
        V_next = turns[t + 1].turn_value if t + 1 < T else 0.0
        
        delta = r_t + cfg.gamma_turn * V_next - V_t
        gae = delta + cfg.gamma_turn * cfg.lambda_turn * gae
        turns[t].turn_advantage = gae
    
    return turns


def compute_bilevel_gae_full(
    turns: List[TurnData],
    cfg: VAGENConfig = None,
) -> List[TurnData]:
    """
    VAGEN-Full: Bi-Level GAE with turn-level and token-level advantages.
    
    Stage 1: Turn-level GAE → A_t^turn for each turn
    Stage 2: Token-level GAE within each turn:
        - Final token initialized with A_t^turn
        - Backward recursion with γ_token, λ_token
    """
    cfg = cfg or VAGENConfig()
    
    # Stage 1: Turn-level GAE
    turns = compute_turn_level_gae(turns, cfg)
    
    # Stage 2: Token-level GAE within each turn
    for t, turn in enumerate(turns):
        token_data = turn.token_data
        device = token_data.logprobs.device
        
        action_mask = token_data.action_mask.bool()
        action_indices = torch.where(action_mask)[0]
        
        if len(action_indices) == 0:
            token_data.advantages = torch.zeros_like(token_data.logprobs)
            token_data.returns = torch.zeros_like(token_data.logprobs)
            continue
        
        # Per-token KL rewards
        kl_rewards = compute_kl_rewards(
            token_data.logprobs,
            token_data.ref_logprobs,
            cfg.kl_coef,
        )
        
        num_action = len(action_indices)
        advantages = torch.zeros_like(token_data.logprobs)
        
        # Final token (i = J-1): Initialize with turn advantage
        final_idx = action_indices[-1]
        
        # δ_{t,J-1} = r_{t,J-1}^KL + γ_token V(τ_{≤a_t}) - V(τ_{<(t,J-1)})
        # But V(τ_{≤a_t}) is the turn_value (value at end of turn)
        V_end_turn = torch.tensor(turn.turn_value, device=device)
        V_final = token_data.values[final_idx]
        
        delta_final = kl_rewards[final_idx] + cfg.gamma_token * V_end_turn - V_final
        
        # A_{t,J-1}^token = δ_{t,J-1}^token + A_t^turn
        gae = delta_final + turn.turn_advantage
        advantages[final_idx] = gae
        
        # Backward for i = J-2, ..., 0
        for i in reversed(range(num_action - 1)):
            idx = action_indices[i]
            next_idx = action_indices[i + 1]
            
            V_i = token_data.values[idx]
            V_next = token_data.values[next_idx]
            
            delta = kl_rewards[idx] + cfg.gamma_token * V_next - V_i
            gae = delta + cfg.gamma_token * cfg.lambda_token * gae
            advantages[idx] = gae
        
        # Compute return targets: G_{t,i} = A_{t,i}^token + V(τ_{<(t,i)})
        returns = advantages + token_data.values
        
        # Mask observation tokens
        advantages = advantages * action_mask.float()
        returns = returns * action_mask.float()
        
        token_data.advantages = advantages
        token_data.returns = returns.detach()  # CRITICAL: detach
    
    # Optional: Normalize advantages across all turns/tokens
    if cfg.normalize_advantages:
        all_advantages = []
        for turn in turns:
            mask = turn.token_data.action_mask.bool()
            all_advantages.append(turn.token_data.advantages[mask])
        
        if all_advantages:
            all_adv = torch.cat(all_advantages)
            if len(all_adv) > 1:
                mean = all_adv.mean()
                std = all_adv.std() + 1e-8
                for turn in turns:
                    turn.token_data.advantages = (turn.token_data.advantages - mean) / std
    
    return turns


# ═══════════════════════════════════════════════════════════════════════
# Critic Loss (MSE on value targets)
# ═══════════════════════════════════════════════════════════════════════

def compute_critic_loss(
    values: torch.Tensor,        # (B, seq_len) predicted V_φ(τ_{<i})
    returns: torch.Tensor,       # (B, seq_len) G_i targets (DETACHED)
    action_mask: torch.Tensor,   # (B, seq_len) bool mask
    clip_value: Optional[float] = None,
    old_values: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute critic loss with proper action masking.
    
    L_critic = (1/Σ M_i) Σ_i M_i * (V_φ(τ_{<i}) - G_i)²
    
    Optional value clipping for PPO-style training.
    
    Parameters
    ----------
    values : (B, seq_len)
        Predicted values from critic.
    returns : (B, seq_len)
        Target returns G_i (must be detached).
    action_mask : (B, seq_len)
        Boolean mask - True for action tokens, False for observation.
    clip_value : float, optional
        If provided, clip value update similar to PPO policy clipping.
    old_values : (B, seq_len), optional
        Required if clip_value is set.
    
    Returns
    -------
    loss : scalar tensor
    """
    mask = action_mask.bool()
    
    if clip_value is not None and old_values is not None:
        # Clipped value loss (PPO-style)
        values_clipped = old_values + torch.clamp(
            values - old_values, -clip_value, clip_value
        )
        loss_unclipped = (values - returns) ** 2
        loss_clipped = (values_clipped - returns) ** 2
        loss_per_token = torch.max(loss_unclipped, loss_clipped)
    else:
        loss_per_token = (values - returns) ** 2
    
    # Apply mask and compute mean
    masked_loss = loss_per_token * mask.float()
    num_action_tokens = mask.float().sum()
    
    if num_action_tokens > 0:
        loss = masked_loss.sum() / num_action_tokens
    else:
        loss = torch.tensor(0.0, device=values.device, requires_grad=True)
    
    return loss


# ═══════════════════════════════════════════════════════════════════════
# Combined PPO + Critic Step
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PPOCriticConfig:
    """Combined config for PPO actor and critic updates."""
    
    # PPO clipping
    clip_eps: float = 0.2
    clip_value: float = 0.2
    
    # Loss coefficients
    entropy_coef: float = 0.01
    kl_coef: float = 0.05
    value_coef: float = 0.5
    
    # GAE
    gamma: float = 1.0
    gae_lambda: float = 0.95
    
    # Gradient clipping
    max_grad_norm: float = 1.0


def ppo_critic_step(
    logprobs: torch.Tensor,           # (B, seq_len) current π_θ
    old_logprobs: torch.Tensor,       # (B, seq_len) rollout π_old
    values: torch.Tensor,             # (B, seq_len) current critic V_φ
    old_values: torch.Tensor,         # (B, seq_len) rollout critic
    returns: torch.Tensor,            # (B, seq_len) target G_i (detached)
    advantages: torch.Tensor,         # (B, seq_len) advantages
    action_mask: torch.Tensor,        # (B, seq_len) bool mask
    ref_logprobs: Optional[torch.Tensor] = None,
    config: PPOCriticConfig = None,
) -> dict:
    """
    Combined PPO policy + critic update step.
    
    Returns dict with:
    - total_loss: combined loss for backward
    - policy_loss, value_loss, entropy, kl, clip_frac
    """
    cfg = config or PPOCriticConfig()
    mask = action_mask.bool()
    
    # === Policy Loss ===
    lp = logprobs[mask]
    old_lp = old_logprobs[mask]
    adv = advantages[mask]
    
    # Normalize advantages
    if len(adv) > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    
    ratio = (lp - old_lp).exp()
    surr1 = ratio * adv
    surr2 = ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Entropy bonus
    entropy = -lp.mean()
    
    # KL penalty
    kl = torch.tensor(0.0, device=logprobs.device)
    if ref_logprobs is not None:
        ref_lp = ref_logprobs[mask]
        kl = (lp - ref_lp).mean()
    
    # Clipping fraction
    clip_frac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean()
    
    # === Value Loss ===
    value_loss = compute_critic_loss(
        values, returns, action_mask,
        clip_value=cfg.clip_value,
        old_values=old_values,
    )
    
    # === Total Loss ===
    total_loss = (
        policy_loss
        - cfg.entropy_coef * entropy
        + cfg.kl_coef * kl
        + cfg.value_coef * value_loss
    )
    
    return {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "kl": kl,
        "clip_frac": clip_frac,
    }
