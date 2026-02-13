#!/usr/bin/env python3
"""
Advantage estimation utilities.

* **Monte-Carlo returns** — simplest baseline; ``A = G − V`` or just ``G``
  when no critic is available.
* **GAE(λ)** — Generalized Advantage Estimation for lower-variance
  estimates when a value function is available.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from ..rollout.trajectory import Trajectory


# Global debug flag for bi-level GAE
_BILEVEL_DEBUG = True

def set_bilevel_debug(enabled: bool):
    """Enable/disable bi-level GAE debug logging."""
    global _BILEVEL_DEBUG
    _BILEVEL_DEBUG = enabled


# ─────────────────────────────────────────────────────────────────────
# Monte-Carlo
# ─────────────────────────────────────────────────────────────────────
def compute_monte_carlo_returns(
    trajectory: Trajectory,
    gamma: float = 1.0,
    normalise: bool = True,
) -> Trajectory:
    """
    Fill ``turn.returns`` and ``turn.advantage`` with discounted MC returns.

    When no critic is available, advantage = return (mean-normalised
    across the trajectory if *normalise* is True).
    """
    rets = trajectory.compute_returns(gamma)
    for turn, G in zip(trajectory.turns, rets):
        turn.returns = G

    if normalise and len(rets) > 1:
        mean = sum(rets) / len(rets)
        std = (sum((r - mean) ** 2 for r in rets) / len(rets)) ** 0.5 + 1e-8
        for turn in trajectory.turns:
            turn.advantage = (turn.returns - mean) / std
    else:
        for turn in trajectory.turns:
            turn.advantage = turn.returns

    return trajectory


# ─────────────────────────────────────────────────────────────────────
# GAE(λ)
# ─────────────────────────────────────────────────────────────────────
def compute_gae(
    trajectory: Trajectory,
    values: List[float],
    gamma: float = 1.0,
    lam: float = 0.95,
    normalise: bool = True,
) -> Trajectory:
    """
    Generalised Advantage Estimation.

    ``values`` should have the same length as ``trajectory.turns`` and
    contain the critic's state-value estimate for each turn.
    """
    T = len(trajectory.turns)
    assert len(values) == T, f"values length {len(values)} != turns {T}"

    advantages: List[float] = [0.0] * T
    gae = 0.0
    for t in reversed(range(T)):
        r_t = trajectory.turns[t].reward
        V_t = values[t]
        V_next = values[t + 1] if t + 1 < T else 0.0
        delta = r_t + gamma * V_next - V_t
        gae = delta + gamma * lam * gae
        advantages[t] = gae

    # Fill trajectory
    for t in range(T):
        trajectory.turns[t].advantage = advantages[t]
        trajectory.turns[t].value = values[t]
        trajectory.turns[t].returns = advantages[t] + values[t]

    if normalise and T > 1:
        mean = sum(advantages) / T
        std = (sum((a - mean) ** 2 for a in advantages) / T) ** 0.5 + 1e-8
        for turn in trajectory.turns:
            turn.advantage = (turn.advantage - mean) / std

    return trajectory


# ─────────────────────────────────────────────────────────────────────
# Bi-Level GAE (VAGEN Algorithm 2) — Step 9
# ─────────────────────────────────────────────────────────────────────
def compute_bilevel_gae(
    trajectory: Trajectory,
    turn_values: List[float],
    token_values: Optional[List[List[float]]] = None,
    gamma: float = 1.0,
    turn_lambda: float = 0.95,
    token_lambda: float = 0.95,
    kl_coef: float = 0.05,
    normalise: bool = True,
) -> Trajectory:
    """
    Bi-Level GAE (VAGEN Algorithm 2).
    
    Stage 1: Compute turn-level advantages A_turn[t] using rewards r_t
    Stage 2: Inside each turn, initialize last token advantage to A_turn[t],
             then propagate backward using per-token KL penalties.
    
    Parameters
    ----------
    trajectory : Trajectory
        The episode trajectory.
    turn_values : List[float]
        Critic value at each turn boundary (length = num_turns).
    token_values : List[List[float]], optional
        Per-token values within each turn. If None, uses constant turn_value
        for all tokens in that turn.
    gamma : float
        Discount factor.
    turn_lambda : float
        GAE lambda for turn-level computation.
    token_lambda : float
        GAE lambda for token-level computation within turns.
    kl_coef : float
        Coefficient for per-token KL penalty (β in paper).
    normalise : bool
        Whether to normalize advantages.
    """
    T = len(trajectory.turns)
    if T == 0:
        return trajectory
        
    assert len(turn_values) == T, f"turn_values length {len(turn_values)} != turns {T}"
    
    # ── Stage 1: Turn-level GAE ──
    turn_advantages: List[float] = [0.0] * T
    gae = 0.0
    for t in reversed(range(T)):
        r_t = trajectory.turns[t].reward
        V_t = turn_values[t]
        V_next = turn_values[t + 1] if t + 1 < T else 0.0
        delta = r_t + gamma * V_next - V_t
        gae = delta + gamma * turn_lambda * gae
        turn_advantages[t] = gae
    
    # ── Stage 2: Token-level advantage within each turn ──
    for t, turn in enumerate(trajectory.turns):
        A_turn = turn_advantages[t]
        
        if turn.logprobs is None or turn.ref_logprobs is None:
            # No token-level data; use turn-level advantage for all tokens
            turn.advantage = A_turn
            turn.value = turn_values[t]
            turn.returns = A_turn + turn_values[t]
            continue
        
        # Per-token KL: logp_actor - logp_ref (approximate)
        logprobs = turn.logprobs.cpu().numpy() if hasattr(turn.logprobs, 'cpu') else list(turn.logprobs)
        ref_logprobs = turn.ref_logprobs.cpu().numpy() if hasattr(turn.ref_logprobs, 'cpu') else list(turn.ref_logprobs)
        
        num_tokens = len(logprobs)
        if num_tokens == 0:
            turn.advantage = A_turn
            continue
        
        # Per-token reward = -beta * KL
        token_rewards = [-kl_coef * (lp - rlp) for lp, rlp in zip(logprobs, ref_logprobs)]
        
        # Token values (if provided, else use turn value)
        if token_values is not None and t < len(token_values):
            tv = token_values[t]
        else:
            tv = [turn_values[t]] * num_tokens
        
        # Backward pass: last token gets A_turn, others propagate
        token_advantages = [0.0] * num_tokens
        token_advantages[-1] = A_turn
        
        for i in reversed(range(num_tokens - 1)):
            r_i = token_rewards[i]
            V_i = tv[i] if i < len(tv) else 0.0
            V_next = tv[i + 1] if i + 1 < len(tv) else 0.0
            delta = r_i + gamma * V_next - V_i
            token_advantages[i] = delta + gamma * token_lambda * token_advantages[i + 1]
        
        # Store average token advantage as turn advantage
        turn.advantage = sum(token_advantages) / num_tokens
        turn.value = turn_values[t]
        turn.returns = turn.advantage + turn_values[t]
        
        # Log token-level details for first turn of first trajectory (debug)
        if t == 0 and _BILEVEL_DEBUG:
            avg_kl = sum([-r / kl_coef for r in token_rewards]) / max(len(token_rewards), 1)
            print(f"  [bi-level GAE] Turn {t}: {num_tokens} tokens, "
                  f"A_turn={A_turn:.4f}, avg_token_kl={avg_kl:.4f}, "
                  f"final_advantage={turn.advantage:.4f}")
    
    # Normalize across trajectory
    if normalise and T > 1:
        advantages = [turn.advantage for turn in trajectory.turns]
        mean = sum(advantages) / T
        std = (sum((a - mean) ** 2 for a in advantages) / T) ** 0.5 + 1e-8
        for turn in trajectory.turns:
            turn.advantage = (turn.advantage - mean) / std
    
    if _BILEVEL_DEBUG:
        final_advantages = [turn.advantage for turn in trajectory.turns]
        print(f"  [bi-level GAE] Final normalized advantages: {[f'{a:.4f}' for a in final_advantages]}")
    
    return trajectory


# ─────────────────────────────────────────────────────────────────────
# Unified interface with config toggle
# ─────────────────────────────────────────────────────────────────────
def compute_advantage(
    trajectory: Trajectory,
    values: List[float],
    method: str = "token_gae",
    gamma: float = 1.0,
    lam: float = 0.95,
    normalise: bool = True,
    **kwargs,
) -> Trajectory:
    """
    Unified advantage computation with method selection.
    
    Parameters
    ----------
    method : str
        One of: "mc" (Monte-Carlo), "token_gae" (standard GAE), "bilevel_gae"
    """
    if method == "mc":
        return compute_monte_carlo_returns(trajectory, gamma=gamma, normalise=normalise)
    elif method == "token_gae":
        return compute_gae(trajectory, values, gamma=gamma, lam=lam, normalise=normalise)
    elif method == "bilevel_gae":
        return compute_bilevel_gae(
            trajectory, 
            turn_values=values, 
            gamma=gamma, 
            turn_lambda=lam,
            token_lambda=kwargs.get("token_lambda", lam),
            kl_coef=kwargs.get("kl_coef", 0.05),
            normalise=normalise,
        )
    else:
        raise ValueError(f"Unknown advantage method: {method}")
    return trajectory
