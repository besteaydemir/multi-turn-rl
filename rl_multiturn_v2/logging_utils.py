#!/usr/bin/env python3
"""
Logging utilities for RL training.

Implements three levels of logging as per specification:
1. Per-Turn: Turn index, action parameters, token count
2. Per-Trajectory: Number of views, length of reasoning, final answer
3. Per-Update: Policy loss, value loss, KL divergence, entropy

All logs go to Weights & Biases.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
from pathlib import Path

from .data_structures import Trajectory, Turn


# ============================================================================
# TURN-LEVEL LOGGING
# ============================================================================

@dataclass
class TurnLog:
    """Log entry for a single turn."""
    turn_index: int
    num_tokens: int
    num_action_tokens: int
    num_reasoning_tokens: int
    action_json: Optional[Dict[str, Any]] = None
    generation_time_seconds: float = 0.0
    parse_success: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "num_tokens": self.num_tokens,
            "num_action_tokens": self.num_action_tokens,
            "num_reasoning_tokens": self.num_reasoning_tokens,
            "action_json": self.action_json,
            "generation_time_seconds": self.generation_time_seconds,
            "parse_success": self.parse_success,
        }


def log_turn(turn: Turn) -> TurnLog:
    """Create log entry for a turn."""
    num_tokens = len(turn.generated_ids) if turn.generated_ids is not None else 0
    num_action = turn.num_action_tokens()
    
    action_json = None
    if turn.action and turn.action.raw_json:
        action_json = turn.action.raw_json
    
    parse_success = True
    if turn.action:
        parse_success = turn.action.parse_success
    
    return TurnLog(
        turn_index=turn.turn_index,
        num_tokens=num_tokens,
        num_action_tokens=num_action,
        num_reasoning_tokens=num_tokens - num_action,
        action_json=action_json,
        generation_time_seconds=turn.generation_time_seconds,
        parse_success=parse_success,
    )


# ============================================================================
# TRAJECTORY-LEVEL LOGGING
# ============================================================================

@dataclass
class TrajectoryLog:
    """Log entry for a complete trajectory."""
    trajectory_id: str
    scene_id: str
    num_views: int
    total_tokens: int
    total_action_tokens: int
    total_reasoning_tokens: int
    final_answer: str
    is_correct: bool
    ground_truth: Optional[str] = None
    duration_seconds: float = 0.0
    
    # Turn-level logs
    turns: List[TurnLog] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "scene_id": self.scene_id,
            "num_views": self.num_views,
            "total_tokens": self.total_tokens,
            "total_action_tokens": self.total_action_tokens,
            "total_reasoning_tokens": self.total_reasoning_tokens,
            "final_answer": self.final_answer,
            "is_correct": self.is_correct,
            "ground_truth": self.ground_truth,
            "duration_seconds": self.duration_seconds,
            "turns": [t.to_dict() for t in self.turns],
        }


def log_trajectory(trajectory: Trajectory) -> TrajectoryLog:
    """Create log entry for a trajectory."""
    turn_logs = [log_turn(turn) for turn in trajectory.turns]
    
    total_tokens = sum(t.num_tokens for t in turn_logs)
    total_action = sum(t.num_action_tokens for t in turn_logs)
    
    return TrajectoryLog(
        trajectory_id=trajectory.trajectory_id,
        scene_id=trajectory.scene_id,
        num_views=trajectory.num_turns,
        total_tokens=total_tokens,
        total_action_tokens=total_action,
        total_reasoning_tokens=total_tokens - total_action,
        final_answer=trajectory.final_answer_text,
        is_correct=trajectory.is_correct,
        ground_truth=trajectory.ground_truth,
        duration_seconds=trajectory.total_duration_seconds,
        turns=turn_logs,
    )


# ============================================================================
# UPDATE-LEVEL LOGGING
# ============================================================================

@dataclass
class UpdateLog:
    """Log entry for a training update."""
    update_step: int
    policy_loss: float
    value_loss: float = 0.0
    kl_divergence: float = 0.0
    entropy: float = 0.0
    
    # Batch statistics
    num_trajectories: int = 0
    num_correct: int = 0
    accuracy: float = 0.0
    
    # Token counts
    total_action_tokens: int = 0
    
    # Timing
    rollout_time_seconds: float = 0.0
    training_time_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "update_step": self.update_step,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "kl_divergence": self.kl_divergence,
            "entropy": self.entropy,
            "num_trajectories": self.num_trajectories,
            "num_correct": self.num_correct,
            "accuracy": self.accuracy,
            "total_action_tokens": self.total_action_tokens,
            "rollout_time_seconds": self.rollout_time_seconds,
            "training_time_seconds": self.training_time_seconds,
        }


def log_update(
    update_step: int,
    metrics: Dict[str, Any],
    trajectories: List[Trajectory],
    rollout_time: float = 0.0,
    training_time: float = 0.0,
) -> UpdateLog:
    """Create log entry for a training update."""
    num_correct = sum(1 for t in trajectories if t.is_correct)
    accuracy = num_correct / len(trajectories) if trajectories else 0.0
    
    return UpdateLog(
        update_step=update_step,
        policy_loss=metrics.get("policy_loss", 0.0),
        value_loss=metrics.get("value_loss", 0.0),
        kl_divergence=metrics.get("kl_divergence", 0.0),
        entropy=metrics.get("entropy", 0.0),
        num_trajectories=len(trajectories),
        num_correct=num_correct,
        accuracy=accuracy,
        total_action_tokens=metrics.get("num_action_tokens", 0),
        rollout_time_seconds=rollout_time,
        training_time_seconds=training_time,
    )


# ============================================================================
# WANDB LOGGER
# ============================================================================

class WandBLogger:
    """
    Weights & Biases logger for RL training.
    
    Logs all three levels: per-turn, per-trajectory, per-update.
    """
    
    def __init__(
        self,
        project: str = "rl-view-selection",
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ):
        """
        Initialize W&B logger.
        
        Args:
            project: W&B project name
            entity: W&B entity (team/user)
            run_name: Run name (auto-generated if None)
            config: Configuration dict to log
            tags: Tags for the run
        """
        self.enabled = True
        
        try:
            import wandb
            self.wandb = wandb
            
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config=config or {},
                tags=tags or [],
            )
            
            print(f"[WandB] Initialized: {wandb.run.url}")
            
        except Exception as e:
            print(f"[WandB] Initialization failed: {e}")
            self.enabled = False
            self.run = None
    
    def log_update(self, update_log: UpdateLog):
        """Log a training update."""
        if not self.enabled:
            return
        
        self.wandb.log({
            "update/policy_loss": update_log.policy_loss,
            "update/value_loss": update_log.value_loss,
            "update/kl_divergence": update_log.kl_divergence,
            "update/entropy": update_log.entropy,
            "update/accuracy": update_log.accuracy,
            "update/num_trajectories": update_log.num_trajectories,
            "update/total_action_tokens": update_log.total_action_tokens,
            "timing/rollout_seconds": update_log.rollout_time_seconds,
            "timing/training_seconds": update_log.training_time_seconds,
        }, step=update_log.update_step)
    
    def log_trajectory(self, traj_log: TrajectoryLog, step: int):
        """Log a trajectory summary."""
        if not self.enabled:
            return
        
        self.wandb.log({
            "trajectory/num_views": traj_log.num_views,
            "trajectory/total_tokens": traj_log.total_tokens,
            "trajectory/action_tokens": traj_log.total_action_tokens,
            "trajectory/reasoning_tokens": traj_log.total_reasoning_tokens,
            "trajectory/is_correct": int(traj_log.is_correct),
            "trajectory/duration": traj_log.duration_seconds,
        }, step=step)
    
    def log_example(
        self,
        trajectory: Trajectory,
        step: int,
        prefix: str = "example",
    ):
        """Log an example trajectory for inspection."""
        if not self.enabled:
            return
        
        # Create text summary
        lines = [
            f"**Question:** {trajectory.question}",
            f"**Ground Truth:** {trajectory.ground_truth}",
            f"**Final Answer:** {trajectory.final_answer_text}",
            f"**Correct:** {trajectory.is_correct}",
            "",
        ]
        
        for turn in trajectory.turns:
            lines.append(f"### Turn {turn.turn_index + 1}")
            if turn.reasoning_blocks:
                for key, value in turn.reasoning_blocks.items():
                    lines.append(f"**{key.upper()}:** {value[:200]}...")
            if turn.action:
                lines.append(f"**Action:** {turn.action.raw_json}")
            lines.append("")
        
        self.wandb.log({
            f"{prefix}/text": self.wandb.Html(
                "<pre>" + "\n".join(lines) + "</pre>"
            )
        }, step=step)
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration update."""
        if not self.enabled:
            return
        
        self.wandb.config.update(config)
    
    def finish(self):
        """Finish the W&B run."""
        if self.enabled and self.run:
            self.wandb.finish()


# ============================================================================
# FILE LOGGER (LOCAL)
# ============================================================================

class FileLogger:
    """
    Local file logger as backup.
    
    Writes JSON logs to disk for reproducibility.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize file logger.
        
        Args:
            output_dir: Directory for log files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.update_log_file = self.output_dir / "update_logs.jsonl"
        self.trajectory_log_file = self.output_dir / "trajectory_logs.jsonl"
    
    def log_update(self, update_log: UpdateLog):
        """Append update log to file."""
        with open(self.update_log_file, "a") as f:
            f.write(json.dumps(update_log.to_dict()) + "\n")
    
    def log_trajectory(self, traj_log: TrajectoryLog):
        """Append trajectory log to file."""
        with open(self.trajectory_log_file, "a") as f:
            f.write(json.dumps(traj_log.to_dict()) + "\n")
    
    def save_trajectory(self, trajectory: Trajectory, subdir: str = "trajectories"):
        """Save full trajectory to disk."""
        traj_dir = self.output_dir / subdir / trajectory.trajectory_id
        trajectory.save(traj_dir)


# ============================================================================
# COMBINED LOGGER
# ============================================================================

class Logger:
    """
    Combined logger for both W&B and local files.
    """
    
    def __init__(
        self,
        output_dir: Path,
        use_wandb: bool = True,
        wandb_project: str = "rl-view-selection",
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize combined logger."""
        self.file_logger = FileLogger(output_dir)
        
        self.wandb_logger = None
        if use_wandb:
            self.wandb_logger = WandBLogger(
                project=wandb_project,
                entity=wandb_entity,
                run_name=wandb_run_name,
                config=config,
            )
    
    def log_update(
        self,
        update_step: int,
        metrics: Dict[str, Any],
        trajectories: List[Trajectory],
        rollout_time: float = 0.0,
        training_time: float = 0.0,
    ):
        """Log a training update."""
        update_log = log_update(
            update_step=update_step,
            metrics=metrics,
            trajectories=trajectories,
            rollout_time=rollout_time,
            training_time=training_time,
        )
        
        self.file_logger.log_update(update_log)
        
        if self.wandb_logger:
            self.wandb_logger.log_update(update_log)
        
        return update_log
    
    def log_trajectories(self, trajectories: List[Trajectory], step: int):
        """Log all trajectories from a batch."""
        for traj in trajectories:
            traj_log = log_trajectory(traj)
            self.file_logger.log_trajectory(traj_log)
            
            if self.wandb_logger:
                self.wandb_logger.log_trajectory(traj_log, step)
    
    def log_example(self, trajectory: Trajectory, step: int):
        """Log an example for inspection."""
        if self.wandb_logger:
            self.wandb_logger.log_example(trajectory, step)
        
        # Also save to file
        self.file_logger.save_trajectory(trajectory, subdir="examples")
    
    def finish(self):
        """Finish logging."""
        if self.wandb_logger:
            self.wandb_logger.finish()


# ============================================================================
# METRIC FORMATTERS
# ============================================================================

def format_update_summary(update_log: UpdateLog) -> str:
    """Format update log for console output."""
    lines = [
        f"Update {update_log.update_step}:",
        f"  Policy Loss: {update_log.policy_loss:.4f}",
        f"  KL Divergence: {update_log.kl_divergence:.4f}",
        f"  Entropy: {update_log.entropy:.4f}",
        f"  Accuracy: {update_log.accuracy:.2%} ({update_log.num_correct}/{update_log.num_trajectories})",
        f"  Action Tokens: {update_log.total_action_tokens}",
        f"  Timing: rollout={update_log.rollout_time_seconds:.1f}s, train={update_log.training_time_seconds:.1f}s",
    ]
    return "\n".join(lines)


def format_trajectory_summary(traj_log: TrajectoryLog) -> str:
    """Format trajectory log for console output."""
    status = "✓" if traj_log.is_correct else "✗"
    return (
        f"  {status} {traj_log.trajectory_id}: "
        f"{traj_log.num_views} views, "
        f"{traj_log.total_tokens} tokens, "
        f"answer={traj_log.final_answer} "
        f"({traj_log.duration_seconds:.1f}s)"
    )
