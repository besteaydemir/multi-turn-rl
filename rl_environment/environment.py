#!/usr/bin/env python3
"""
RL Environment wrapper for the VSI-Bench navigation pipeline.
Provides clean interfaces for multi-turn reinforcement learning.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from PIL import Image
import json


@dataclass
class Observation:
    """
    Observation at timestep t containing all context for action generation.
    """
    step: int
    images: List[str]  # Paths to all images accumulated so far
    camera_positions: List[np.ndarray]  # 4x4 camera poses for each image
    current_position: np.ndarray  # Current camera position (x, y, z)
    current_rotation: np.ndarray  # Current camera rotation matrix (3x3)
    bbox_mins: List[float]  # Scene bounding box minimums [x, y, z]
    bbox_maxs: List[float]  # Scene bounding box maximums [x, y, z]
    question: str
    choices: List[str]
    movement_history: List[Dict[str, float]]  # Previous movements taken
    is_final_step: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert observation to JSON-serializable dict."""
        return {
            "step": self.step,
            "images": self.images,
            "camera_positions": [pos.tolist() for pos in self.camera_positions],
            "current_position": self.current_position.tolist(),
            "current_rotation": self.current_rotation.tolist(),
            "bbox_mins": self.bbox_mins,
            "bbox_maxs": self.bbox_maxs,
            "question": self.question,
            "choices": self.choices,
            "movement_history": self.movement_history,
            "is_final_step": self.is_final_step
        }


@dataclass
class Action:
    """
    Parsed action from model output (JSON only, no reasoning text).
    """
    rotation_angle_degrees: float
    forward_meters: float
    left_meters: float
    z_delta_meters: float
    answer: Optional[str] = None  # Only present if done=True
    done: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert action to JSON-serializable dict."""
        return {
            "rotation_angle_degrees": self.rotation_angle_degrees,
            "forward_meters": self.forward_meters,
            "left_meters": self.left_meters,
            "z_delta_meters": self.z_delta_meters,
            "answer": self.answer,
            "done": self.done
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Action':
        """Create Action from parsed JSON dict."""
        return cls(
            rotation_angle_degrees=float(data.get("rotation_angle_degrees", 0.0)),
            forward_meters=float(data.get("forward_meters", 0.0)),
            left_meters=float(data.get("left_meters", 0.0)),
            z_delta_meters=float(data.get("z_delta_meters", 0.0)),
            answer=data.get("answer"),
            done=bool(data.get("done", False))
        )
    
    def validate(self) -> Tuple[bool, str]:
        """
        Validate action constraints.
        Returns (is_valid, error_message).
        """
        # Check movement bounds
        if not (-0.5 <= self.forward_meters <= 0.5):
            return False, f"forward_meters {self.forward_meters} out of range [-0.5, 0.5]"
        if not (-0.5 <= self.left_meters <= 0.5):
            return False, f"left_meters {self.left_meters} out of range [-0.5, 0.5]"
        if not (-0.3 <= self.z_delta_meters <= 0.3):
            return False, f"z_delta_meters {self.z_delta_meters} out of range [-0.3, 0.3]"
        
        # Check answer consistency
        if self.done and self.answer is None:
            return False, "done=True but no answer provided"
        if not self.done and self.answer is not None:
            return False, "answer provided but done=False"
        
        # Check answer format if present
        if self.answer is not None:
            if not (isinstance(self.answer, str) and self.answer.upper() in "ABCDEFGHIJ"):
                return False, f"Invalid answer format: {self.answer}"
        
        return True, ""


@dataclass
class Turn:
    """
    Complete information for a single turn in an episode.
    """
    turn_index: int
    observation: Observation
    
    # Full context used for generation
    full_prompt: str
    context_text: str  # The instruction text portion
    
    # Generated outputs
    generated_ids: torch.Tensor  # Token IDs of entire generated sequence
    generated_text: str  # Human-readable generated text
    
    # Action token tracking (boolean mask preferred)
    action_token_mask: torch.Tensor  # Boolean mask: True for action tokens, False otherwise
    
    # Context tokens (for RL training)
    context_input_ids: Optional[torch.Tensor] = None  # Tokenized context (prompt + history)
    
    # Input offset (for aligning logits later)
    input_token_length: int = 0  # Length of input prompt in tokens
    
    # Alternative: action_token_start_index and action_token_end_index
    action_token_start_index: Optional[int] = None
    action_token_end_index: Optional[int] = None
    
    # Masking diagnostics (Step 3 requirement)
    num_action_tokens: int = 0  # Number of tokens marked as action
    num_reasoning_tokens: int = 0  # Number of non-action tokens
    masking_method: str = "unknown"  # How mask was computed: "brace_depth", "regex_fallback", "failed"
    masking_confidence: float = 1.0  # Confidence in mask quality (0.0-1.0)
    
    # Parsed action
    action: Optional[Action] = None
    action_valid: bool = False
    action_error: str = ""
    
    # Next observation (None if terminal)
    next_observation: Optional[Observation] = None
    
    # Metadata
    attempt_count: int = 1  # Number of generation attempts for this turn
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert turn to JSON-serializable dict (without tensors)."""
        return {
            "turn_index": self.turn_index,
            "observation": self.observation.to_dict(),
            "full_prompt": self.full_prompt,
            "context_text": self.context_text,
            "generated_text": self.generated_text,
            "generated_ids_length": len(self.generated_ids) if self.generated_ids is not None else 0,
            "input_token_length": self.input_token_length,
            "action_token_mask_length": len(self.action_token_mask) if self.action_token_mask is not None else 0,
            "action_token_start_index": self.action_token_start_index,
            "action_token_end_index": self.action_token_end_index,
            "num_action_tokens": self.num_action_tokens,
            "num_reasoning_tokens": self.num_reasoning_tokens,
            "masking_method": self.masking_method,
            "masking_confidence": self.masking_confidence,
            "action": self.action.to_dict() if self.action else None,
            "action_valid": self.action_valid,
            "action_error": self.action_error,
            "next_observation": self.next_observation.to_dict() if self.next_observation else None,
            "attempt_count": self.attempt_count,
            "timestamp": self.timestamp
        }


@dataclass
class Episode:
    """
    Complete episode data for a single question.
    """
    episode_id: str
    scene_id: str
    question: str
    choices: List[str]
    ground_truth: str
    
    turns: List[Turn] = field(default_factory=list)
    
    # Terminal reward
    final_reward: float = 0.0
    final_answer: Optional[str] = None
    is_correct: bool = False
    
    # Episode quality flags (Step 3 requirement)
    is_valid: bool = True  # False if episode should be dropped
    dropout_reason: Optional[str] = None  # Reason for dropout if invalid
    masking_quality: str = "good"  # "good", "low_confidence", "fallback_used"
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_turn(self, turn: Turn):
        """Add a turn to the episode."""
        self.turns.append(turn)
    
    def compute_final_reward(self) -> float:
        """
        Compute final reward based on answer correctness.
        Returns +1.0 for correct, 0.0 for incorrect.
        """
        if self.final_answer is None:
            self.final_reward = 0.0
        else:
            self.is_correct = (self.final_answer == self.ground_truth)
            self.final_reward = 1.0 if self.is_correct else 0.0
        return self.final_reward
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert episode to JSON-serializable dict (metadata only)."""
        return {
            "episode_id": self.episode_id,
            "scene_id": self.scene_id,
            "question": self.question,
            "choices": self.choices,
            "ground_truth": self.ground_truth,
            "num_turns": len(self.turns),
            "final_reward": self.final_reward,
            "final_answer": self.final_answer,
            "is_correct": self.is_correct,
            "is_valid": self.is_valid,
            "dropout_reason": self.dropout_reason,
            "masking_quality": self.masking_quality,
            "metadata": self.metadata,
            "turns_summary": [
                {
                    "turn": i,
                    "action_valid": t.action_valid,
                    "done": t.action.done if t.action else False,
                    "masking_method": t.masking_method,
                    "num_action_tokens": t.num_action_tokens
                }
                for i, t in enumerate(self.turns)
            ]
        }
    
    def save_full(self, output_dir: Path):
        """
        Save complete episode data including tensors.
        
        Args:
            output_dir: Directory to save episode data
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata as JSON
        with open(output_dir / "episode_metadata.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        # Save each turn
        for i, turn in enumerate(self.turns):
            turn_dir = output_dir / f"turn_{i:02d}"
            turn_dir.mkdir(exist_ok=True)
            
            # Save turn metadata
            with open(turn_dir / "turn_metadata.json", "w") as f:
                json.dump(turn.to_dict(), f, indent=2)
            
            # Save tensors
            if turn.generated_ids is not None:
                torch.save(turn.generated_ids, turn_dir / "generated_ids.pt")
            if turn.action_token_mask is not None:
                torch.save(turn.action_token_mask, turn_dir / "action_token_mask.pt")
            
            # Save text outputs
            with open(turn_dir / "full_prompt.txt", "w") as f:
                f.write(turn.full_prompt)
            with open(turn_dir / "generated_text.txt", "w") as f:
                f.write(turn.generated_text)
    
    def save_to_jsonl(self, jsonl_path: Path):
        """
        Save episode as a single line in JSONL format for batch processing.
        This is a compact format suitable for RL training data collection.
        
        Args:
            jsonl_path: Path to JSONL file (will append if exists)
        """
        # Build compact episode representation
        episode_data = {
            "episode_id": self.episode_id,
            "scene_id": self.scene_id,
            "question": self.question,
            "choices": self.choices,
            "ground_truth": self.ground_truth,
            "final_reward": self.final_reward,
            "final_answer": self.final_answer,
            "is_correct": self.is_correct,
            "metadata": self.metadata,
            "turns": []
        }
        
        # Add each turn (without full tensors - those are saved separately)
        for turn in self.turns:
            turn_data = {
                "turn_index": turn.turn_index,
                "generated_text": turn.generated_text,
                "generated_ids_length": len(turn.generated_ids) if turn.generated_ids is not None else 0,
                "action_token_start_index": turn.action_token_start_index,
                "action_token_end_index": turn.action_token_end_index,
                "action": turn.action.to_dict() if turn.action else None,
                "action_valid": turn.action_valid,
                "action_error": turn.action_error,
                "timestamp": turn.timestamp
            }
            episode_data["turns"].append(turn_data)
        
        # Append to JSONL file
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(episode_data) + "\n")


class NavigationEnvironment:
    """
    Environment wrapper for VSI-Bench navigation tasks.
    Manages state transitions and rendering.
    """
    
    def __init__(
        self,
        mesh,
        mesh_path: Path,
        scene_id: str,
        question: str,
        choices: List[str],
        ground_truth: str,
        max_steps: int = 10,
        image_wh: Tuple[int, int] = (1024, 768),
        default_fxfy: float = 300.0,
        cam_height: float = 1.6,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize environment.
        
        Args:
            mesh: Open3D TriangleMesh object
            mesh_path: Path to mesh file
            scene_id: Scene identifier
            question: Question text
            choices: List of answer choices
            ground_truth: Correct answer (A, B, C, D, etc.)
            max_steps: Maximum number of steps per episode
            image_wh: Image width and height
            default_fxfy: Default focal length
            cam_height: Camera height above floor
            output_dir: Directory to save rendered images
        """
        self.mesh = mesh
        self.mesh_path = mesh_path
        self.scene_id = scene_id
        self.question = question
        self.choices = choices
        self.ground_truth = ground_truth
        self.max_steps = max_steps
        self.image_wh = image_wh
        self.default_fxfy = default_fxfy
        self.cam_height = cam_height
        self.output_dir = output_dir or Path("env_temp")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Compute bounding box once
        vertices = np.asarray(mesh.vertices)
        self.bbox_mins = vertices.min(axis=0).tolist()
        self.bbox_maxs = vertices.max(axis=0).tolist()
        
        # Episode state
        self.current_step = 0
        self.image_history: List[str] = []
        self.cam_history: List[np.ndarray] = []
        self.movement_history: List[Dict[str, float]] = []
        self.episode_done = False
    
    def reset(self, initial_pose: np.ndarray) -> Observation:
        """
        Reset environment to initial state.
        
        Args:
            initial_pose: Initial 4x4 camera pose matrix
            
        Returns:
            Initial observation
        """
        self.current_step = 0
        self.image_history = []
        self.cam_history = []
        self.movement_history = []
        self.episode_done = False
        
        # Render initial view
        img_path = self.output_dir / "render_00.png"
        self._render_view(initial_pose, img_path)
        
        self.image_history.append(str(img_path))
        self.cam_history.append(initial_pose.copy())
        
        return self._build_observation()
    
    def step(self, action: Action) -> Tuple[Observation, bool]:
        """
        Execute action and return next observation.
        
        Args:
            action: Action object with movement parameters
            
        Returns:
            (next_observation, done)
        """
        if self.episode_done:
            raise RuntimeError("Episode already done. Call reset() first.")
        
        # Get current camera pose
        current_pose = self.cam_history[-1]
        R_current = current_pose[:3, :3]
        t_current = current_pose[:3, 3]
        
        # Apply action to compute next pose
        next_pose = self._apply_action(R_current, t_current, action)
        
        # Increment step counter
        self.current_step += 1
        
        # Render next view
        img_path = self.output_dir / f"render_{self.current_step:02d}.png"
        self._render_view(next_pose, img_path)
        
        # Update history
        self.image_history.append(str(img_path))
        self.cam_history.append(next_pose)
        self.movement_history.append({
            "rotation": action.rotation_angle_degrees,
            "forward": action.forward_meters,
            "left": action.left_meters,
            "z_delta": action.z_delta_meters
        })
        
        # Check terminal conditions
        done = action.done or (self.current_step >= self.max_steps)
        self.episode_done = done
        
        # Build next observation
        next_obs = self._build_observation()
        
        return next_obs, done
    
    def _build_observation(self) -> Observation:
        """Build observation from current state."""
        current_pose = self.cam_history[-1]
        R_current = current_pose[:3, :3]
        t_current = current_pose[:3, 3]
        
        return Observation(
            step=self.current_step,
            images=self.image_history.copy(),
            camera_positions=[pose.copy() for pose in self.cam_history],
            current_position=t_current.copy(),
            current_rotation=R_current.copy(),
            bbox_mins=self.bbox_mins,
            bbox_maxs=self.bbox_maxs,
            question=self.question,
            choices=self.choices,
            movement_history=self.movement_history.copy(),
            is_final_step=(self.current_step >= self.max_steps - 1)
        )
    
    def _apply_action(self, R_current: np.ndarray, t_current: np.ndarray, action: Action) -> np.ndarray:
        """
        Apply action to current pose to get next pose.
        Uses the same logic as the original pipeline.
        """
        from render_point_cloud_qwen_angle import parse_rotation_angle, apply_movement_in_camera_frame
        
        # Apply rotation
        R_new = parse_rotation_angle(action.rotation_angle_degrees, R_current)
        
        # Apply movement
        t_new = apply_movement_in_camera_frame(
            R_new, t_current,
            action.forward_meters,
            action.left_meters,
            action.z_delta_meters
        )
        
        # Build 4x4 matrix
        next_pose = np.eye(4, dtype=float)
        next_pose[:3, :3] = R_new
        next_pose[:3, 3] = t_new
        
        return next_pose
    
    def _render_view(self, cam_pose: np.ndarray, output_path: Path):
        """Render view from camera pose."""
        from render_point_cloud_qwen_angle import render_mesh_from_pose
        render_mesh_from_pose(self.mesh, cam_pose, output_path, fxfy=self.default_fxfy)
