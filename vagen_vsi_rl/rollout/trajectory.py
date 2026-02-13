#!/usr/bin/env python3
"""
Lightweight Turn / Trajectory dataclasses for RL training.

These are intentionally simpler than ``rl_multiturn_v2.data_structures``
because the heavy lifting (mesh loading, rendering) lives in ``VSIEnv``.
Here we only store what PPO needs: tokens, log-probs, rewards, advantages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


@dataclass
class Turn:
    """One step inside an episode."""

    turn_index: int

    # ── observation that the model saw ──
    image_paths: List[str] = field(default_factory=list)
    prompt_text: str = ""

    # ── model output ──
    generated_text: str = ""
    generated_ids: Optional[torch.Tensor] = None   # (gen_len,) action tokens only
    logprobs: Optional[torch.Tensor] = None         # (gen_len,) per-action-token logprobs
    ref_logprobs: Optional[torch.Tensor] = None     # from frozen ref model
    prompt_token_ids: Optional[torch.Tensor] = None # (prompt_len,) full input context

    # ── token masks ──
    action_token_mask: Optional[torch.Tensor] = None  # bool (seq_len,)

    # ── parsed fields ──
    action_dict: Optional[Dict[str, Any]] = None  # raw JSON the model produced
    answer: Optional[str] = None
    done_flag: bool = False

    # ── RL labels (filled after rollout) ──
    reward: float = 0.0
    advantage: float = 0.0
    returns: float = 0.0
    value: float = 0.0          # critic estimate (if available)

    # ── metadata ──
    generation_time_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "image_paths": self.image_paths,
            "prompt_text": self.prompt_text,
            "generated_text": self.generated_text,
            "action_dict": self.action_dict,
            "answer": self.answer,
            "done_flag": self.done_flag,
            "reward": self.reward,
            "advantage": self.advantage,
            "returns": self.returns,
            "value": self.value,
            "generation_time_s": self.generation_time_s,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Turn:
        return cls(
            turn_index=d.get("turn_index", 0),
            image_paths=d.get("image_paths", []),
            prompt_text=d.get("prompt_text", ""),
            generated_text=d.get("generated_text", ""),
            action_dict=d.get("action_dict"),
            answer=d.get("answer"),
            done_flag=d.get("done_flag", False),
            reward=d.get("reward", 0.0),
            advantage=d.get("advantage", 0.0),
            returns=d.get("returns", 0.0),
            value=d.get("value", 0.0),
            generation_time_s=d.get("generation_time_s", 0.0),
        )


@dataclass
class Trajectory:
    """A complete episode: initial obs → T steps → terminal reward."""

    trajectory_id: str = ""
    question: str = ""
    choices: Any = None
    ground_truth: Optional[str] = None
    scene_id: str = ""
    dataset: str = ""
    question_type: str = "unknown"
    is_numerical: bool = False

    turns: List[Turn] = field(default_factory=list)

    # ── terminal outcome ──
    final_answer: Optional[str] = None
    is_correct: bool = False
    terminal_reward: float = 0.0

    # ── metadata ──
    num_steps: int = 0
    elapsed_time_s: float = 0.0

    def __len__(self) -> int:
        return len(self.turns)

    # ── returns / advantage helpers ──
    def compute_returns(self, gamma: float = 1.0) -> List[float]:
        """Discounted returns (back-propagated from terminal reward)."""
        G = self.terminal_reward
        returns: List[float] = []
        for turn in reversed(self.turns):
            G = turn.reward + gamma * G
            returns.append(G)
        returns.reverse()
        return returns

    def fill_returns(self, gamma: float = 1.0) -> None:
        """Write returns into each Turn in-place."""
        rets = self.compute_returns(gamma)
        for t, r in zip(self.turns, rets):
            t.returns = r

    # ── serialisation ──
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "question": self.question,
            "choices": self.choices,
            "ground_truth": self.ground_truth,
            "scene_id": self.scene_id,
            "dataset": self.dataset,
            "question_type": self.question_type,
            "is_numerical": self.is_numerical,
            "turns": [t.to_dict() for t in self.turns],
            "final_answer": self.final_answer,
            "is_correct": self.is_correct,
            "terminal_reward": self.terminal_reward,
            "num_steps": self.num_steps,
            "elapsed_time_s": self.elapsed_time_s,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Trajectory:
        traj = cls(
            trajectory_id=d.get("trajectory_id", ""),
            question=d.get("question", ""),
            choices=d.get("choices"),
            ground_truth=d.get("ground_truth"),
            scene_id=d.get("scene_id", ""),
            dataset=d.get("dataset", ""),
            question_type=d.get("question_type", "unknown"),
            is_numerical=d.get("is_numerical", False),
            final_answer=d.get("final_answer"),
            is_correct=d.get("is_correct", False),
            terminal_reward=d.get("terminal_reward", 0.0),
            num_steps=d.get("num_steps", 0),
            elapsed_time_s=d.get("elapsed_time_s", 0.0),
        )
        traj.turns = [Turn.from_dict(t) for t in d.get("turns", [])]
        return traj

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "trajectory.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        # save tensors
        for turn in self.turns:
            prefix = f"turn_{turn.turn_index:02d}"
            if turn.generated_ids is not None:
                torch.save(turn.generated_ids, path / f"{prefix}_ids.pt")
            if turn.logprobs is not None:
                torch.save(turn.logprobs, path / f"{prefix}_logprobs.pt")
            if turn.ref_logprobs is not None:
                torch.save(turn.ref_logprobs, path / f"{prefix}_ref_logprobs.pt")
            if turn.action_token_mask is not None:
                torch.save(turn.action_token_mask, path / f"{prefix}_mask.pt")

    @classmethod
    def load(cls, path: Path) -> Trajectory:
        path = Path(path)
        with open(path / "trajectory.json") as f:
            d = json.load(f)
        traj = cls.from_dict(d)
        for turn in traj.turns:
            prefix = f"turn_{turn.turn_index:02d}"
            for attr, suffix in [
                ("generated_ids", "_ids.pt"),
                ("logprobs", "_logprobs.pt"),
                ("ref_logprobs", "_ref_logprobs.pt"),
                ("action_token_mask", "_mask.pt"),
            ]:
                p = path / f"{prefix}{suffix}"
                if p.exists():
                    setattr(turn, attr, torch.load(p, weights_only=True))
        return traj
