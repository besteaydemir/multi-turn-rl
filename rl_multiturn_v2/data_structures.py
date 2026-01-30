#!/usr/bin/env python3
"""
Data structures for multi-turn RL view selection.

This module defines the core data structures for:
- Actions (camera poses)
- Turns (single step in an episode)
- Trajectories (complete episodes)

Designed for VAGEN-compatible turn-level reward assignment.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import torch
import numpy as np
import json
from pathlib import Path


@dataclass
class CameraPose:
    """
    Camera pose specification.
    
    The camera pose is defined as a 4x4 transformation matrix or
    equivalent position + rotation representation.
    """
    # Option 1: Full 4x4 transformation matrix
    transform_matrix: Optional[np.ndarray] = None  # Shape: (4, 4)
    
    # Option 2: Position + rotation (alternative representation)
    position: Optional[np.ndarray] = None      # Shape: (3,) - x, y, z
    rotation: Optional[np.ndarray] = None      # Shape: (3, 3) - rotation matrix
    
    # Field of view in degrees
    fov: float = 60.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result = {"fov": self.fov}
        if self.transform_matrix is not None:
            matrix = self.transform_matrix if isinstance(self.transform_matrix, list) else self.transform_matrix.tolist()
            result["transform_matrix"] = matrix
        if self.position is not None:
            pos = self.position if isinstance(self.position, list) else self.position.tolist()
            result["position"] = pos
        if self.rotation is not None:
            rot = self.rotation if isinstance(self.rotation, list) else self.rotation.tolist()
            result["rotation"] = rot
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CameraPose":
        """Create from dict."""
        transform_matrix = None
        position = None
        rotation = None
        
        if "transform_matrix" in data:
            transform_matrix = np.array(data["transform_matrix"])
        if "position" in data:
            position = np.array(data["position"])
        if "rotation" in data:
            rotation = np.array(data["rotation"])
            
        return cls(
            transform_matrix=transform_matrix,
            position=position,
            rotation=rotation,
            fov=data.get("fov", 60.0)
        )


@dataclass
class Action:
    """
    Parsed action from model output.
    
    Extracted from the [ACTION] block in the model's response.
    Contains only the structured camera pose parameters.
    """
    camera_pose: CameraPose
    raw_json: Dict[str, Any] = field(default_factory=dict)
    
    # Parsing metadata
    parse_success: bool = True
    parse_error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "camera_pose": self.camera_pose.to_dict(),
            "raw_json": self.raw_json,
            "parse_success": self.parse_success,
            "parse_error": self.parse_error,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        """Create from dict."""
        return cls(
            camera_pose=CameraPose.from_dict(data.get("camera_pose", {})),
            raw_json=data.get("raw_json", {}),
            parse_success=data.get("parse_success", True),
            parse_error=data.get("parse_error"),
        )


@dataclass
class FinalAnswer:
    """
    Final answer from the last turn.
    
    Extracted from the [FINAL_ANSWER] block.
    """
    answer_text: str
    raw_text: str  # The full text after [FINAL_ANSWER]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer_text": self.answer_text,
            "raw_text": self.raw_text,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FinalAnswer":
        return cls(
            answer_text=data.get("answer_text", ""),
            raw_text=data.get("raw_text", ""),
        )


@dataclass
class Turn:
    """
    Complete data for one turn in an episode.
    
    Contains:
    - Input: prompt, images, observation state
    - Output: generated tokens, parsed action
    - Token attribution: which tokens are action tokens
    
    This structure supports:
    - Turn-level rewards (VAGEN-compatible)
    - Action-only loss computation
    - Full trajectory reconstruction
    """
    turn_index: int
    
    # =========================================================================
    # INPUT STATE
    # =========================================================================
    
    # The complete prompt sent to the model
    prompt_text: str = ""
    
    # Image paths for this turn's context (all images so far)
    image_paths: List[str] = field(default_factory=list)
    
    # Question being answered
    question: str = ""
    
    # =========================================================================
    # MODEL OUTPUT
    # =========================================================================
    
    # Raw generated text (includes [STATE], [PLAN], [PREDICT], [ACTION])
    generated_text: str = ""
    
    # Token IDs of the generated sequence
    generated_ids: Optional[torch.Tensor] = None  # Shape: [seq_len]
    
    # Log probabilities for each generated token (from vLLM)
    logprobs: Optional[torch.Tensor] = None  # Shape: [seq_len]
    
    # =========================================================================
    # TOKEN ATTRIBUTION (Critical for RL)
    # =========================================================================
    
    # Boolean mask: True for action tokens (inside [ACTION] block JSON)
    action_token_mask: Optional[torch.Tensor] = None  # Shape: [seq_len], dtype=bool
    
    # Start/end indices of action tokens in generated_ids
    action_token_start: Optional[int] = None
    action_token_end: Optional[int] = None
    
    # =========================================================================
    # PARSED STRUCTURES
    # =========================================================================
    
    # Parsed action (from [ACTION] block)
    action: Optional[Action] = None
    
    # Final answer (only on last turn, from [FINAL_ANSWER] block)
    final_answer: Optional[FinalAnswer] = None
    
    # Reasoning text blocks (not supervised, but logged for debugging)
    reasoning_blocks: Dict[str, str] = field(default_factory=dict)
    # Keys: "state", "plan", "predict"
    
    # =========================================================================
    # EXECUTION RESULT
    # =========================================================================
    
    # Rendered image path after executing this action
    rendered_image_path: Optional[str] = None
    
    # =========================================================================
    # TURN-LEVEL REWARD (VAGEN-compatible, zero for now)
    # =========================================================================
    
    reward: float = 0.0  # Placeholder: always 0 in minimal implementation
    
    # =========================================================================
    # METADATA
    # =========================================================================
    
    timestamp: float = 0.0
    generation_time_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict (tensors saved separately)."""
        return {
            "turn_index": self.turn_index,
            "prompt_text": self.prompt_text,
            "image_paths": self.image_paths,
            "question": self.question,
            "generated_text": self.generated_text,
            "action_token_start": self.action_token_start,
            "action_token_end": self.action_token_end,
            "action": self.action.to_dict() if self.action else None,
            "final_answer": self.final_answer.to_dict() if self.final_answer else None,
            "reasoning_blocks": self.reasoning_blocks,
            "rendered_image_path": self.rendered_image_path,
            "reward": self.reward,
            "timestamp": self.timestamp,
            "generation_time_seconds": self.generation_time_seconds,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Turn":
        """Create from dict (tensors loaded separately)."""
        turn = cls(
            turn_index=data.get("turn_index", 0),
            prompt_text=data.get("prompt_text", ""),
            image_paths=data.get("image_paths", []),
            question=data.get("question", ""),
            generated_text=data.get("generated_text", ""),
            action_token_start=data.get("action_token_start"),
            action_token_end=data.get("action_token_end"),
            reasoning_blocks=data.get("reasoning_blocks", {}),
            rendered_image_path=data.get("rendered_image_path"),
            reward=data.get("reward", 0.0),
            timestamp=data.get("timestamp", 0.0),
            generation_time_seconds=data.get("generation_time_seconds", 0.0),
        )
        if data.get("action"):
            turn.action = Action.from_dict(data["action"])
        if data.get("final_answer"):
            turn.final_answer = FinalAnswer.from_dict(data["final_answer"])
        return turn
    
    def num_action_tokens(self) -> int:
        """Count of action tokens in this turn."""
        if self.action_token_mask is not None:
            return self.action_token_mask.sum().item()
        elif self.action_token_start is not None and self.action_token_end is not None:
            return self.action_token_end - self.action_token_start
        return 0
    
    def get_action_logprobs(self) -> Optional[torch.Tensor]:
        """Get log probabilities for action tokens only."""
        if self.logprobs is None:
            return None
        if self.action_token_mask is not None:
            return self.logprobs[self.action_token_mask]
        elif self.action_token_start is not None and self.action_token_end is not None:
            return self.logprobs[self.action_token_start:self.action_token_end]
        return None


@dataclass
class Trajectory:
    """
    Complete episode trajectory.
    
    Contains all turns and metadata for one episode.
    Designed for:
    - Trajectory storage and loading
    - Batch training with multiple trajectories
    - VAGEN-compatible turn-level reward assignment
    """
    trajectory_id: str
    
    # =========================================================================
    # TASK SPECIFICATION
    # =========================================================================
    
    question: str = ""
    choices: List[str] = field(default_factory=list)
    ground_truth: Optional[str] = None
    scene_id: str = ""
    
    # =========================================================================
    # TRAJECTORY DATA
    # =========================================================================
    
    turns: List[Turn] = field(default_factory=list)
    
    # =========================================================================
    # TERMINAL OUTCOME
    # =========================================================================
    
    # Final answer text (from last turn's [FINAL_ANSWER])
    final_answer_text: str = ""
    
    # Is the final answer correct?
    is_correct: bool = False
    
    # Terminal reward (0.0 in minimal implementation)
    terminal_reward: float = 0.0
    
    # =========================================================================
    # METADATA
    # =========================================================================
    
    max_turns: int = 5
    num_turns: int = 0
    
    start_time: float = 0.0
    end_time: float = 0.0
    total_duration_seconds: float = 0.0
    
    # Collection metadata
    model_id: str = ""
    device: str = "cuda"
    
    def __len__(self) -> int:
        """Return number of turns in trajectory."""
        return len(self.turns)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "trajectory_id": self.trajectory_id,
            "question": self.question,
            "choices": self.choices,
            "ground_truth": self.ground_truth,
            "scene_id": self.scene_id,
            "turns": [t.to_dict() for t in self.turns],
            "final_answer_text": self.final_answer_text,
            "is_correct": self.is_correct,
            "terminal_reward": self.terminal_reward,
            "max_turns": self.max_turns,
            "num_turns": self.num_turns,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration_seconds": self.total_duration_seconds,
            "model_id": self.model_id,
            "device": self.device,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trajectory":
        """Create from dict."""
        traj = cls(
            trajectory_id=data.get("trajectory_id", ""),
            question=data.get("question", ""),
            choices=data.get("choices", []),
            ground_truth=data.get("ground_truth"),
            scene_id=data.get("scene_id", ""),
            final_answer_text=data.get("final_answer_text", ""),
            is_correct=data.get("is_correct", False),
            terminal_reward=data.get("terminal_reward", 0.0),
            max_turns=data.get("max_turns", 5),
            num_turns=data.get("num_turns", 0),
            start_time=data.get("start_time", 0.0),
            end_time=data.get("end_time", 0.0),
            total_duration_seconds=data.get("total_duration_seconds", 0.0),
            model_id=data.get("model_id", ""),
            device=data.get("device", "cuda"),
        )
        traj.turns = [Turn.from_dict(t) for t in data.get("turns", [])]
        return traj
    
    def save(self, output_dir: Path):
        """
        Save trajectory to disk.
        
        Creates:
            output_dir/
            ├── trajectory.json          # Metadata and text
            ├── turn_00_generated_ids.pt # Tensor for turn 0
            ├── turn_00_logprobs.pt
            ├── turn_00_action_mask.pt
            └── ...
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main metadata
        with open(output_dir / "trajectory.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        # Save tensors for each turn
        for turn in self.turns:
            prefix = f"turn_{turn.turn_index:02d}"
            
            if turn.generated_ids is not None:
                torch.save(turn.generated_ids, output_dir / f"{prefix}_generated_ids.pt")
            
            if turn.logprobs is not None:
                torch.save(turn.logprobs, output_dir / f"{prefix}_logprobs.pt")
            
            if turn.action_token_mask is not None:
                torch.save(turn.action_token_mask, output_dir / f"{prefix}_action_mask.pt")
    
    @classmethod
    def load(cls, input_dir: Path) -> "Trajectory":
        """Load trajectory from disk."""
        input_dir = Path(input_dir)
        
        # Load metadata
        with open(input_dir / "trajectory.json", "r") as f:
            data = json.load(f)
        
        traj = cls.from_dict(data)
        
        # Load tensors for each turn
        for turn in traj.turns:
            prefix = f"turn_{turn.turn_index:02d}"
            
            ids_path = input_dir / f"{prefix}_generated_ids.pt"
            if ids_path.exists():
                turn.generated_ids = torch.load(ids_path)
            
            logprobs_path = input_dir / f"{prefix}_logprobs.pt"
            if logprobs_path.exists():
                turn.logprobs = torch.load(logprobs_path)
            
            mask_path = input_dir / f"{prefix}_action_mask.pt"
            if mask_path.exists():
                turn.action_token_mask = torch.load(mask_path)
        
        return traj
    
    # =========================================================================
    # RL UTILITIES
    # =========================================================================
    
    def get_all_action_logprobs(self) -> List[torch.Tensor]:
        """Get action log probabilities from all turns."""
        result = []
        for turn in self.turns:
            lp = turn.get_action_logprobs()
            if lp is not None:
                result.append(lp)
        return result
    
    def get_turn_boundaries(self) -> List[int]:
        """
        Get token indices where each turn ends.
        Useful for turn-level advantage estimation (VAGEN).
        """
        boundaries = []
        total_tokens = 0
        for turn in self.turns:
            if turn.generated_ids is not None:
                total_tokens += len(turn.generated_ids)
                boundaries.append(total_tokens)
        return boundaries
    
    def compute_returns(self, gamma: float = 1.0) -> List[float]:
        """
        Compute discounted returns for each turn.
        
        Currently: All zeros (no intermediate rewards, terminal=0).
        Extension point for VAGEN rewards.
        """
        returns = []
        running_return = self.terminal_reward
        
        # Traverse turns in reverse
        for turn in reversed(self.turns):
            running_return = turn.reward + gamma * running_return
            returns.append(running_return)
        
        return list(reversed(returns))
