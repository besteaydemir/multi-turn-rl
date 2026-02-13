#!/usr/bin/env python3
"""
Reward functions for the VSI-Bench RL environment.

Deliverable (Step 3): compute_turn_reward() and compute_terminal_reward()

Terminal reward:
    +R_correct   if the final answer matches ground truth.
    +R_wrong     otherwise (typically negative).
    +R_no_answer if the agent never produced an answer.

Optional per-step shaping:
    * format_reward   — +0.1 if JSON parses & fields valid; 0 otherwise.
    * step_penalty    — small negative cost per step (encourages efficiency).

MRA (Mean Relative Accuracy) for numerical questions:
    reward in [0,1] based on how close the prediction is to ground truth.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..rollout.trajectory import Trajectory, Turn

# Try importing calculate_mra; if not available, define a simple fallback
try:
    from utils.evaluation import calculate_mra
except ImportError:
    def calculate_mra(pred: float, gt: float) -> float:
        """Fallback MRA: 1 - |pred - gt| / max(|gt|, 1e-6), clipped to [0,1]."""
        diff = abs(pred - gt)
        denom = max(abs(gt), 1e-6)
        return max(0.0, min(1.0, 1.0 - diff / denom))


# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────
@dataclass
class RewardConfig:
    """Reward shaping parameters."""
    # Terminal rewards
    correct_answer: float = 1.0         # exact match for MCQ, or MRA >= 1 for numerical
    wrong_answer: float = -0.5          # incorrect answer
    no_answer: float = -1.0             # agent never answered
    
    # Per-turn shaping (optional, start disabled)
    format_reward: float = 0.1          # +0.1 if JSON parses correctly
    step_penalty: float = 0.0           # small negative per step (e.g., -0.02)
    
    # Whether to scale terminal reward (VAGEN uses 10 for some tasks)
    reward_scale: float = 1.0


# ─────────────────────────────────────────────────────────────────────
# Core reward functions (Deliverable Step 3)
# ─────────────────────────────────────────────────────────────────────
def compute_turn_reward(
    turn: Turn,
    config: RewardConfig | None = None,
) -> float:
    """
    Compute per-turn reward for shaping (not terminal).
    
    Components:
    - format_reward: +config.format_reward if action parsed successfully
    - step_penalty: config.step_penalty (small negative)
    
    Returns the reward value (also sets turn.reward in place).
    """
    cfg = config or RewardConfig()
    
    r = cfg.step_penalty
    
    # Format reward: JSON parsed successfully with valid action fields
    if turn.action_dict is not None or turn.done_flag:
        # Action was parseable
        r += cfg.format_reward
    # else: format_reward = 0 (no penalty, just no bonus)
    
    turn.reward = r
    return r


def compute_terminal_reward(
    final_answer: Optional[str],
    ground_truth: Optional[str],
    is_numerical: bool = False,
    config: RewardConfig | None = None,
) -> float:
    """
    Compute terminal task reward based on correctness.
    
    For MCQ: +correct_answer if exact match, else +wrong_answer
    For numerical: MRA-scaled reward in [wrong_answer, correct_answer]
    
    Returns scaled reward.
    """
    cfg = config or RewardConfig()
    
    if final_answer is None:
        return cfg.no_answer * cfg.reward_scale
    
    if is_numerical:
        try:
            pred = float(final_answer)
            gt = float(ground_truth)
            mra = calculate_mra(pred, gt)
            # Linear interpolation: wrong_answer at MRA=0, correct_answer at MRA=1
            base_reward = cfg.wrong_answer + (cfg.correct_answer - cfg.wrong_answer) * mra
            return base_reward * cfg.reward_scale
        except (ValueError, TypeError):
            return cfg.no_answer * cfg.reward_scale
    else:
        # Categorical / MCQ
        pred_norm = str(final_answer).strip().upper()
        gt_norm = str(ground_truth).strip().upper()
        if pred_norm == gt_norm:
            return cfg.correct_answer * cfg.reward_scale
        return cfg.wrong_answer * cfg.reward_scale


# ─────────────────────────────────────────────────────────────────────
# Trajectory-level convenience function
# ─────────────────────────────────────────────────────────────────────
def compute_rewards(
    trajectory: Trajectory,
    config: RewardConfig | None = None,
) -> Trajectory:
    """
    Fill ``turn.reward`` for every turn in *trajectory* **in-place**.
    
    - All turns get turn-level shaping (format + step penalty)
    - Last turn additionally gets the terminal task reward
    
    Returns the same trajectory for chaining.
    """
    cfg = config or RewardConfig()

    # 1. Per-turn shaping rewards
    for turn in trajectory.turns:
        compute_turn_reward(turn, cfg)

    # 2. Terminal reward (added to last turn only)
    if trajectory.turns:
        last = trajectory.turns[-1]
        terminal = compute_terminal_reward(
            final_answer=trajectory.final_answer,
            ground_truth=trajectory.ground_truth,
            is_numerical=trajectory.is_numerical,
            config=cfg,
        )
        last.reward += terminal
        trajectory.terminal_reward = terminal

    return trajectory


# ─────────────────────────────────────────────────────────────────────
# Helper: check if action JSON is valid
# ─────────────────────────────────────────────────────────────────────
def is_valid_action_json(text: str) -> bool:
    """
    Check if text contains valid action JSON with expected fields.
    
    Expected fields: rotation_angle_degrees, forward_meters, etc.
    """
    try:
        # Extract JSON from text (may be wrapped in markdown)
        json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if not json_match:
            return False
        data = json.loads(json_match.group())
        
        # Check for at least one movement field
        movement_fields = [
            "rotation_angle_degrees", "forward_meters", 
            "left_meters", "z_delta_meters"
        ]
        has_movement = any(f in data for f in movement_fields)
        
        # Or has answer/done
        has_terminal = "answer" in data or "done" in data
        
        return has_movement or has_terminal
    except (json.JSONDecodeError, AttributeError):
        return False
