#!/usr/bin/env python3
"""
POMDP environment that wraps the sequential.py question loop.

    obs          = env.reset(question_data)
    obs, r, d, i = env.step(action)

**State (hidden)**  — camera pose, mesh, ground-truth answer, full history.
**Observation**     — exactly what the model already receives: list of image
                      paths collected so far + the instruction / prompt text.

The rendering, movement, and prompt-building logic is *reused* from the
existing ``utils`` package so that episodes are pixel-identical to the
evaluation pipeline.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# ── reuse existing utils ──
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.camera import (
    apply_movement_in_camera_frame,
    look_at_camera_pose_center_from_forward,
    parse_rotation_angle,
    save_matrix,
)
from utils.mesh import find_mesh_file, get_mesh_bounds, load_mesh_cached
from utils.parsing import extract_first_json, parse_qwen_output_and_get_movement
from utils.rendering import render_mesh_from_pose, select_best_initial_view
from utils.evaluation import calculate_mra

# ---------------------------------------------------------------------------
# Constants — match evaluation/sequential.py defaults
# ---------------------------------------------------------------------------
DEFAULT_IMAGE_WH: Tuple[int, int] = (640, 480)
DEFAULT_FX_FY: float = 300.0
DEFAULT_CAM_HEIGHT: float = 1.6
DEFAULT_MAX_STEPS: int = 15

MESH_BASE_DIRS = {
    "arkitscenes": "/dss/mcmlscratch/06/di38riq/arkit_vsi/raw",
    "scannet": "/dss/mcmlscratch/06/di38riq/scans",
    "scannetpp": "/dss/mcmlscratch/06/di38riq/data",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class EnvConfig:
    """Tuneable environment knobs."""

    max_steps: int = DEFAULT_MAX_STEPS
    image_wh: Tuple[int, int] = DEFAULT_IMAGE_WH
    fx_fy: float = DEFAULT_FX_FY
    cam_height: float = DEFAULT_CAM_HEIGHT
    initial_view_selection: str = "visibility"  # "visibility" | "laplacian" | "random"
    mesh_base_dirs: Optional[Dict[str, str]] = None  # overrides per dataset

    def get_mesh_base_dir(self, dataset: str) -> str:
        dirs = self.mesh_base_dirs or MESH_BASE_DIRS
        return dirs.get(dataset, MESH_BASE_DIRS.get(dataset, ""))


# Observation dataclass shared across all env backends
from .common import Observation


# ---------------------------------------------------------------------------
# The environment
# ---------------------------------------------------------------------------
class VSIEnv:
    """
    Gym-style POMDP wrapper around the sequential exploration loop.

    Lifecycle::

        env = VSIEnv(output_dir="/tmp/test", config=EnvConfig())

        obs = env.reset(question_data)          # loads mesh, picks initial view
        while True:
            action = policy(obs)                # any dict or raw model text
            obs, reward, done, info = env.step(action)
            if done:
                break

    ``question_data`` is a dict with at least the keys produced by
    ``utils.data.load_vsi_bench_questions``:
        scene_name, question, choices, answer_id, question_type, is_numerical, dataset
    """

    def __init__(
        self,
        output_dir: str | Path,
        config: Optional[EnvConfig] = None,
        question_id: int = 0,
    ):
        self.output_dir = Path(output_dir)
        self.cfg = config or EnvConfig()
        self.question_id = question_id

        # ---------- state that lives across reset() calls ----------
        self._mesh = None
        self._mesh_path: Optional[Path] = None

        # ---------- per-episode hidden state ----------
        self._R: Optional[np.ndarray] = None        # 3×3 rotation
        self._t: Optional[np.ndarray] = None        # 3×1 translation
        self._cam_history: List[np.ndarray] = []     # list of 4×4 poses
        self._image_history: List[str] = []
        self._movement_history: List[Dict[str, Any]] = []
        self._bbox_mins: Optional[List[float]] = None
        self._bbox_maxs: Optional[List[float]] = None
        self._step: int = 0
        self._done: bool = True

        # question metadata (set in reset)
        self._question: str = ""
        self._choices: Any = None
        self._ground_truth: Any = None
        self._question_type: str = "unknown"
        self._is_numerical: bool = False
        self._scene_id: str = ""
        self._dataset: str = ""
        self._final_answer: Optional[str] = None
        self._episode_start: float = 0.0
        self._episode_dir: Optional[Path] = None

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self, question_data: Dict[str, Any]) -> Observation:
        """
        Load the mesh, render the initial view, return the first observation.

        Mirrors everything in ``run_single_question`` up to the point where
        ``image_history = [str(img0)]`` is set.
        """
        self._step = 0
        self._done = False
        self._final_answer = None
        self._episode_start = time.time()

        # ── unpack question ──
        self._scene_id = question_data["scene_name"]
        self._question = question_data["question"]
        self._choices = question_data.get("choices")
        self._ground_truth = question_data.get("answer_id")
        self._question_type = question_data.get("question_type", "unknown")
        self._is_numerical = question_data.get("is_numerical", False)
        self._dataset = question_data.get("dataset", "arkitscenes")

        # ── episode output dir ──
        self._episode_dir = self.output_dir / f"q{self.question_id:03d}"
        self._episode_dir.mkdir(parents=True, exist_ok=True)

        # ── load mesh ──
        mesh_base = self.cfg.get_mesh_base_dir(self._dataset)
        mesh_path = find_mesh_file(self._scene_id, mesh_base, dataset=self._dataset)
        if mesh_path is None:
            raise FileNotFoundError(
                f"No mesh for scene {self._scene_id} (dataset={self._dataset}) in {mesh_base}"
            )
        self._mesh_path = mesh_path
        self._mesh = load_mesh_cached(mesh_path)

        # ── bounding box ──
        self._bbox_mins, self._bbox_maxs = get_mesh_bounds(self._mesh, percentile_filter=True)

        # ── generate 4 candidate initial views & select best ──
        center_x = (self._bbox_mins[0] + self._bbox_maxs[0]) / 2.0
        center_y = (self._bbox_mins[1] + self._bbox_maxs[1]) / 2.0
        cam_z = self._bbox_mins[2] + self.cfg.cam_height
        eye = np.array([center_x, center_y, cam_z], dtype=float)

        view_images: Dict[int, np.ndarray] = {}
        view_poses: Dict[int, np.ndarray] = {}
        for angle_deg in [0, 90, 180, 270]:
            angle_rad = np.deg2rad(angle_deg)
            forward = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])
            pose = look_at_camera_pose_center_from_forward(
                eye, forward=forward, up=np.array([0.0, 0.0, -1.0])
            )
            view_poses[angle_deg] = pose

            img_path = self._episode_dir / f"render_candidate_{angle_deg}.png"
            render_mesh_from_pose(
                self._mesh, pose, img_path,
                fxfy=self.cfg.fx_fy, image_wh=self.cfg.image_wh,
            )
            img_pil = Image.open(img_path)
            view_images[angle_deg] = np.array(img_pil).astype(float) / 255.0

        # select (no Qwen call during training — use score-based metric)
        if self.cfg.initial_view_selection == "random":
            best_angle = int(np.random.choice([0, 90, 180, 270]))
        else:
            best_angle, _, _ = select_best_initial_view(
                view_images, metric=self.cfg.initial_view_selection
            )

        cam_pose = view_poses[best_angle]
        save_matrix(self._episode_dir / "cam_pose_00.npy", cam_pose)

        img0 = self._episode_dir / "render_00.png"
        Image.fromarray((view_images[best_angle] * 255).astype(np.uint8)).save(str(img0))

        # ── initialise per-episode state ──
        self._R = cam_pose[:3, :3].copy()
        self._t = cam_pose[:3, 3].copy()
        self._cam_history = [cam_pose.copy()]
        self._image_history = [str(img0)]
        self._movement_history = []

        # ── build the first observation ──
        return self._make_observation()

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(
        self, action: Dict[str, Any] | str
    ) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Execute one step.

        ``action`` can be:
        * a *dict* with ``rotation_angle_degrees``, ``forward_meters``,
          ``left_meters``, ``z_delta_meters``, ``answer``, ``done``.
        * raw model output text (will be parsed via
          ``parse_qwen_output_and_get_movement``).

        Returns ``(observation, reward, done, info)``.
        """
        if self._done:
            raise RuntimeError("Episode already finished; call reset() first.")

        info: Dict[str, Any] = {"parse_success": True, "format_penalty": 0.0}

        # ── parse action ──
        # VAGEN convention:
        #   a_t   = full model output text  (stored by collector as Turn.generated_text)
        #   a_t^e = executable movement dict (stored in info["executable_action"])
        if isinstance(action, str):
            rot, fwd, left, z_d, reasoning, raw_obj, done_flag = (
                parse_qwen_output_and_get_movement(action)
            )
            answer_value = None
            if raw_obj and isinstance(raw_obj, dict):
                answer_value = raw_obj.get("answer")
            # Debug: log parsing results on final step
            if self._step >= self.cfg.max_steps - 1:
                print(f"[VSIEnv] Final step parse: raw_obj={raw_obj}, answer={answer_value}, done={done_flag}")
                print(f"[VSIEnv] Raw action text (first 500 chars): {action[:500]}")
            info["raw_parsed"] = raw_obj
            info["action_is_text"] = True
        elif isinstance(action, dict):
            rot = action.get("rotation_angle_degrees", 0.0)
            fwd = action.get("forward_meters", 0.0)
            left = action.get("left_meters", 0.0)
            z_d = action.get("z_delta_meters", 0.0)
            answer_value = action.get("answer")
            done_flag = action.get("done", False)
            info["raw_parsed"] = action
            info["action_is_text"] = False
        else:
            raise TypeError(f"action must be dict or str, got {type(action)}")

        # ── check for valid parse ──
        movement_valid = all(v is not None for v in [rot, fwd, left, z_d])
        if not movement_valid and not done_flag:
            info["parse_success"] = False
            info["format_penalty"] = -0.1  # mild penalty for unparseable output

        # ── store executable action (a_t^e) ──
        info["executable_action"] = {
            "rotation_angle_degrees": rot,
            "forward_meters": fwd,
            "left_meters": left,
            "z_delta_meters": z_d,
            "answer": answer_value,
            "done": done_flag,
        }

        # ── handle answer ──
        has_answer = False
        if answer_value is not None:
            ans_str = str(answer_value).strip()
            if self._is_numerical:
                match = re.search(r"-?\d+\.?\d*", ans_str)
                if match:
                    has_answer = True
                    self._final_answer = match.group()
            else:
                if ans_str.upper() in "ABCDEFGHIJ":
                    has_answer = True
                    self._final_answer = ans_str.upper()

        is_final_step = self._step >= self.cfg.max_steps

        # ── decide if episode ends ──
        episode_done = is_final_step or (done_flag and has_answer)

        # ── apply movement (only if not done) ──
        if not episode_done and movement_valid:
            R_new = parse_rotation_angle(rot, self._R)
            t_new = apply_movement_in_camera_frame(
                R_new, self._t, fwd, left, z_d,
                bbox_mins=self._bbox_mins, bbox_maxs=self._bbox_maxs,
            )
            next_pose = np.eye(4, dtype=float)
            next_pose[:3, :3] = R_new
            next_pose[:3, 3] = t_new

            self._movement_history.append(
                {
                    "rotation": rot,
                    "forward": fwd,
                    "left": left,
                    "z_delta": z_d,
                    "position": f"X={t_new[0]:.2f}m, Y={t_new[1]:.2f}m, Z={t_new[2]:.2f}m",
                }
            )

            # render new image
            self._step += 1
            img_path = self._episode_dir / f"render_{self._step:02d}.png"
            render_mesh_from_pose(
                self._mesh, next_pose, img_path,
                fxfy=self.cfg.fx_fy, image_wh=self.cfg.image_wh,
            )
            self._image_history.append(str(img_path))
            self._cam_history.append(next_pose.copy())
            self._R = R_new
            self._t = t_new

            save_matrix(
                self._episode_dir / f"cam_pose_{self._step:02d}.npy", next_pose
            )
        elif not episode_done:
            # movement was invalid but episode isn't over — still advance step
            self._step += 1

        # ── re-check final step after advancing ──
        if self._step >= self.cfg.max_steps:
            episode_done = True

        # ── compute reward (only on done) ──
        reward = 0.0
        if episode_done:
            self._done = True
            reward = self._compute_reward()
            info["final_answer"] = self._final_answer
            info["ground_truth"] = self._ground_truth
            info["is_correct"] = self._check_correct()
            info["elapsed_time"] = time.time() - self._episode_start
            info["num_steps"] = self._step
            self._save_trajectory(info)

        obs = self._make_observation()
        return obs, reward, episode_done, info

    # ------------------------------------------------------------------
    # observation builder  (mirrors build_instruction_text in sequential.py)
    # ------------------------------------------------------------------
    def _make_observation(self) -> Observation:
        """Build the observation the policy should see at the current step."""
        # We import the prompt builder from sequential.py's inline code.
        # For decoupling we build a lightweight version here; the heavy
        # prompt is in evaluation/sequential.py::build_instruction_text
        # but for RL we keep the same contract.
        from evaluation.sequential import build_instruction_text

        # is_final = True when the NEXT action must include an answer
        # (i.e., this is the policy's last chance before forced termination)
        is_final = self._step >= self.cfg.max_steps - 1
        prompt = build_instruction_text(
            self._R,
            self._t,
            self._question,
            bbox=(self._bbox_mins, self._bbox_maxs),
            options=self._choices,
            is_final_step=is_final,
            movement_history=self._movement_history,
            step_num=self._step,
            question_type=self._question_type,
            is_numerical=self._is_numerical,
            max_steps=self.cfg.max_steps,
        )

        return Observation(
            image_paths=list(self._image_history),
            prompt_text=prompt,
            step=self._step,
            is_final_step=is_final,
            question=self._question,
            choices=self._choices,
            question_type=self._question_type,
            is_numerical=self._is_numerical,
            cam_position=self._t.copy() if self._t is not None else None,
            bbox=(self._bbox_mins, self._bbox_maxs),
        )

    # ------------------------------------------------------------------
    # reward
    # ------------------------------------------------------------------
    def _compute_reward(self) -> float:
        """
        Terminal reward.

        For MCQ:  +1 if correct letter, −0.5 otherwise, −1 if no answer.
        For numerical: MRA score (0‒1), −1 if no answer.
        """
        if self._final_answer is None:
            return -1.0

        if self._is_numerical:
            try:
                pred = float(self._final_answer)
                gt = float(self._ground_truth)
                mra = calculate_mra(pred, gt)
                # scale MRA (0-1) to reward in [-0.5, 1.0]
                return mra * 1.5 - 0.5
            except (ValueError, TypeError):
                return -1.0
        else:
            return 1.0 if self._check_correct() else -0.5

    def _check_correct(self) -> bool:
        if self._final_answer is None:
            return False
        if self._is_numerical:
            try:
                pred = float(self._final_answer)
                gt = float(self._ground_truth)
                return calculate_mra(pred, gt) > 0.5
            except (ValueError, TypeError):
                return False
        return str(self._final_answer).strip().upper() == str(self._ground_truth).strip().upper()

    # ------------------------------------------------------------------
    # trajectory saving
    # ------------------------------------------------------------------
    def _save_trajectory(self, info: Dict[str, Any]) -> None:
        """Persist episode data to disk (same format as sequential.py)."""
        traj = {
            "question_id": self.question_id,
            "scene_id": self._scene_id,
            "dataset": self._dataset,
            "question": self._question,
            "choices": self._choices,
            "question_type": self._question_type,
            "is_numerical": self._is_numerical,
            "final_answer": self._final_answer,
            "ground_truth": self._ground_truth,
            "is_correct": info.get("is_correct", False),
            "reward": self._compute_reward(),
            "elapsed_time": info.get("elapsed_time", 0.0),
            "num_steps": self._step,
            "num_images": len(self._image_history),
            "image_paths": self._image_history,
            "poses": [
                {
                    "index": i,
                    "position": p[:3, 3].tolist(),
                    "rotation": p[:3, :3].tolist(),
                }
                for i, p in enumerate(self._cam_history)
            ],
            "movement_history": self._movement_history,
        }
        with open(self._episode_dir / "trajectory.json", "w") as f:
            json.dump(traj, f, indent=2)

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------
    @property
    def ground_truth(self):
        return self._ground_truth

    @property
    def scene_id(self):
        return self._scene_id

    @property
    def episode_dir(self):
        return self._episode_dir

    @property
    def image_history(self):
        return list(self._image_history)

    @property
    def cam_history(self):
        return list(self._cam_history)

    def __repr__(self) -> str:
        return (
            f"VSIEnv(scene={self._scene_id!r}, step={self._step}, "
            f"done={self._done}, images={len(self._image_history)})"
        )
