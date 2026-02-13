#!/usr/bin/env python3
"""
POMDP environment that wraps **Habitat-Sim** for embodied 3D exploration.

Exposes exactly the same interface as ``VSIEnv`` so that rollout collectors,
actors, and training code can treat both backends interchangeably::

    obs          = env.reset(question_data)
    obs, r, d, i = env.step(action)

**State (hidden)** — agent pose inside the Habitat scene, ground-truth answer,
                     full observation history.
**Observation**    — list of RGB image paths collected so far + instruction /
                     prompt text (same ``Observation`` dataclass as ``VSIEnv``).

The key difference from ``VSIEnv``:
    * Rendering is done by habitat-sim's GPU renderer (fast, realistic lighting).
    * Movement is handled by habitat-sim's agent + navmesh (collision-aware).
    * Scenes are ``.glb`` files loaded natively by habitat-sim.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# ── Lazy import of habitat_sim — only required at runtime ──
# This lets the module be imported in envs without habitat-sim installed
# (e.g. the "env" conda environment) without crashing.
_habitat_sim = None


def _ensure_habitat():
    """Import habitat_sim lazily and cache the module."""
    global _habitat_sim
    if _habitat_sim is None:
        try:
            import habitat_sim as _hs

            _habitat_sim = _hs
        except ImportError as e:
            raise ImportError(
                "habitat-sim is not installed. "
                "Please run: conda install habitat-sim headless -c conda-forge -c aihabitat"
            ) from e
    return _habitat_sim


# ── Reuse parsing from existing utils ──
# Import directly from the file to avoid pulling in cv2/open3d
# via utils/__init__.py → utils.rendering
import sys
import importlib.util

_PARSING_PY = str(Path(__file__).parent.parent.parent / "utils" / "parsing.py")
_spec = importlib.util.spec_from_file_location("_habitat_parsing", _PARSING_PY)
_parsing_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_parsing_mod)
parse_qwen_output_and_get_movement = _parsing_mod.parse_qwen_output_and_get_movement

# ── Reuse the shared Observation dataclass ──
from .common import Observation


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_IMAGE_WH: Tuple[int, int] = (640, 480)
DEFAULT_SENSOR_HEIGHT: float = 1.5
DEFAULT_MAX_STEPS: int = 15

# Scene search directories (can be overridden via config)
DEFAULT_SCENE_DIRS: List[str] = [
    "/dss/mcmlscratch/06/di38riq/habitat",       # HM3D train (800 scenes)
    "/dss/mcmlscratch/06/di38riq/habitat_val",   # HM3D val   (100 scenes)
    "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/scene_datasets",
    "/dss/dsshome1/06/di38riq/habitat-sim/data/scene_datasets",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class HabitatEnvConfig:
    """Tuneable knobs for the Habitat environment."""

    max_steps: int = DEFAULT_MAX_STEPS
    image_wh: Tuple[int, int] = DEFAULT_IMAGE_WH  # (width, height)
    sensor_height: float = DEFAULT_SENSOR_HEIGHT
    hfov: float = 90.0  # horizontal field-of-view in degrees
    move_forward_amount: float = 0.25  # metres per "forward" step
    turn_amount: float = 10.0  # degrees per "turn" step
    enable_depth: bool = False
    enable_semantic: bool = False
    scene_dirs: Optional[List[str]] = None  # override default scene search
    gpu_device_id: int = 0
    # Path to NVIDIA EGL ICD JSON — fixes EGL device detection on DGX nodes.
    # Set to "" to skip.  Default searches habitat-sim/10_nvidia.json.
    egl_nvidia_json: str = ""


# ---------------------------------------------------------------------------
# Scene resolution helper
# ---------------------------------------------------------------------------
def _find_scene_glb(
    scene_name: str, scene_dirs: List[str]
) -> Optional[Path]:
    """
    Locate a ``.glb`` scene file.

    Supports (in order of priority):
    * Full path      → use directly if it exists
    * HM3D local     → ``{dir}/NNNNN-{name}/{name}.glb``  (our layout)
    * HM3D basis     → ``{dir}/hm3d/{name}/{name}.basis.glb``
    * habitat-test   → ``{dir}/habitat-test-scenes/{name}.glb``
    * MP3D           → ``{dir}/mp3d/{name}/{name}.glb``
    * flat           → ``{dir}/{name}.glb``
    """
    import glob as _glob

    # Full path already?
    p = Path(scene_name)
    if p.suffix == ".glb" and p.exists():
        return p

    for base in scene_dirs:
        base_p = Path(base)

        # HM3D local layout: NNNNN-{hash}/{hash}.glb
        # The dir name is e.g. "00042-kfPV7w3FaU5" and inside is "kfPV7w3FaU5.glb".
        # scene_name can be the hash ("kfPV7w3FaU5") or full dir ("00042-kfPV7w3FaU5").
        if "-" in scene_name and (base_p / scene_name).is_dir():
            # Full dir name given, e.g. "00042-kfPV7w3FaU5"
            hash_part = scene_name.split("-", 1)[1]
            candidate = base_p / scene_name / f"{hash_part}.glb"
            if candidate.exists():
                return candidate
        else:
            # Bare hash given — glob for NNNNN-{hash} directories
            matches = _glob.glob(str(base_p / f"*-{scene_name}"))
            for m in matches:
                candidate = Path(m) / f"{scene_name}.glb"
                if candidate.exists():
                    return candidate

        # habitat-test-scenes
        candidate = base_p / "habitat-test-scenes" / f"{scene_name}.glb"
        if candidate.exists():
            return candidate
        # HM3D nested layout (basis.glb)
        candidate = base_p / "hm3d" / scene_name / f"{scene_name}.basis.glb"
        if candidate.exists():
            return candidate
        # MP3D
        candidate = base_p / "mp3d" / scene_name / f"{scene_name}.glb"
        if candidate.exists():
            return candidate
        # direct child
        candidate = base_p / f"{scene_name}.glb"
        if candidate.exists():
            return candidate

    return None


# ---------------------------------------------------------------------------
# The environment
# ---------------------------------------------------------------------------
class HabitatEnv:
    """
    Gym-style POMDP wrapper around **Habitat-Sim**.

    Lifecycle (identical to ``VSIEnv``)::

        env = HabitatEnv(output_dir="/tmp/test", config=HabitatEnvConfig())

        obs = env.reset(question_data)
        while True:
            action = policy(obs)
            obs, reward, done, info = env.step(action)
            if done:
                break

    ``question_data`` is a dict with at least::

        scene_name   — scene identifier (name or path to .glb)
        question     — the question text
        choices      — (optional) list of MCQ choices
        answer_id    — ground-truth answer
        question_type, is_numerical, dataset  — metadata
    """

    def __init__(
        self,
        output_dir: str | Path,
        config: Optional[HabitatEnvConfig] = None,
        question_id: int = 0,
    ):
        self.output_dir = Path(output_dir)
        self.cfg = config or HabitatEnvConfig()
        self.question_id = question_id

        # Simulator state (created in reset, destroyed in close)
        self._sim = None
        self._agent = None

        # Per-episode hidden state
        self._image_history: List[str] = []
        self._movement_history: List[Dict[str, Any]] = []
        self._step: int = 0
        self._done: bool = True

        # Question metadata
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

        # Cached scene path (avoid re-creating sim for same scene)
        self._loaded_scene_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Habitat-Sim initialisation
    # ------------------------------------------------------------------
    def _make_sim(self, scene_path: Path) -> None:
        """Create (or re-create) the Habitat-Sim simulator for a scene."""
        import os

        # ── EGL device-detection fix for headless multi-GPU nodes ──
        # Without this, habitat-sim often fails with:
        #   "unable to find CUDA device 0 among N EGL devices"
        egl_key = "__EGL_VENDOR_LIBRARY_FILENAMES"
        if egl_key not in os.environ:
            egl_json = self.cfg.egl_nvidia_json
            if not egl_json:
                # Auto-detect: look next to habitat-sim install
                candidate = (
                    Path(__file__).parent.parent.parent.parent
                    / "habitat-sim" / "10_nvidia.json"
                )
                if candidate.exists():
                    egl_json = str(candidate)
            if egl_json and Path(egl_json).exists():
                os.environ[egl_key] = egl_json

        # ── GL runtime library fix ──
        # The habitat-sim binary must use conda-forge's libglvnd dispatch
        # (libEGL.so, libOpenGL.so, libGLdispatch.so) from $CONDA_PREFIX/lib.
        # If $CONDA_PREFIX/lib is NOT in LD_LIBRARY_PATH, the linker falls
        # back to system GLVND which is often ABI-incompatible and causes:
        #   "GL::Context: cannot retrieve OpenGL version"
        # Fix: put $CONDA_PREFIX/lib first so conda's GLVND is found.
        _conda_lib = os.path.join(os.environ.get("CONDA_PREFIX", ""), "lib")
        _ld = os.environ.get("LD_LIBRARY_PATH", "")
        if _conda_lib and _conda_lib not in _ld:
            os.environ["LD_LIBRARY_PATH"] = f"{_conda_lib}:{_ld}" if _ld else _conda_lib

        # Also ensure headless
        os.environ.pop("DISPLAY", None)

        habitat_sim = _ensure_habitat()

        # Close previous sim if scene changed
        if self._sim is not None:
            if self._loaded_scene_path == scene_path:
                # Same scene → just reset agent, no need to rebuild
                return
            self._sim.close()
            self._sim = None

        # ── Simulator configuration ──
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = str(scene_path)
        sim_cfg.gpu_device_id = self.cfg.gpu_device_id
        sim_cfg.enable_physics = False  # no physics needed for exploration

        # ── Sensor specs ──
        W, H = self.cfg.image_wh

        color_spec = habitat_sim.CameraSensorSpec()
        color_spec.uuid = "color_sensor"
        color_spec.sensor_type = habitat_sim.SensorType.COLOR
        color_spec.resolution = [H, W]  # [height, width]
        color_spec.position = [0.0, self.cfg.sensor_height, 0.0]
        color_spec.hfov = self.cfg.hfov

        sensor_specs = [color_spec]

        if self.cfg.enable_depth:
            depth_spec = habitat_sim.CameraSensorSpec()
            depth_spec.uuid = "depth_sensor"
            depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
            depth_spec.resolution = [H, W]
            depth_spec.position = [0.0, self.cfg.sensor_height, 0.0]
            depth_spec.hfov = self.cfg.hfov
            sensor_specs.append(depth_spec)

        if self.cfg.enable_semantic:
            sem_spec = habitat_sim.CameraSensorSpec()
            sem_spec.uuid = "semantic_sensor"
            sem_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
            sem_spec.resolution = [H, W]
            sem_spec.position = [0.0, self.cfg.sensor_height, 0.0]
            sem_spec.hfov = self.cfg.hfov
            sensor_specs.append(sem_spec)

        # ── Agent configuration ──
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        agent_cfg.action_space = {
            "move_forward": habitat_sim.ActionSpec(
                "move_forward",
                habitat_sim.ActuationSpec(amount=self.cfg.move_forward_amount),
            ),
            "turn_left": habitat_sim.ActionSpec(
                "turn_left",
                habitat_sim.ActuationSpec(amount=self.cfg.turn_amount),
            ),
            "turn_right": habitat_sim.ActionSpec(
                "turn_right",
                habitat_sim.ActuationSpec(amount=self.cfg.turn_amount),
            ),
        }

        # ── Build simulator ──
        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        self._sim = habitat_sim.Simulator(cfg)
        self._loaded_scene_path = scene_path

    def _init_agent(self) -> Dict[str, np.ndarray]:
        """Initialise agent at a random navigable position, return first obs."""
        habitat_sim = _ensure_habitat()
        self._agent = self._sim.initialize_agent(0)

        # Try to find a valid start on the ground floor
        state = self._agent.get_state()
        for _ in range(100):
            pos = self._sim.pathfinder.get_random_navigable_point()
            if pos is not None and np.isfinite(pos).all():
                state.position = pos
                self._agent.set_state(state)
                break

        return self._sim.get_sensor_observations()

    # ------------------------------------------------------------------
    # Observation rendering
    # ------------------------------------------------------------------
    def _save_observation(
        self, obs_dict: Dict[str, np.ndarray], step_idx: int
    ) -> str:
        """Save the RGB observation to disk, return the image path."""
        rgba = obs_dict["color_sensor"]  # (H, W, 4) uint8 RGBA
        rgb = rgba[:, :, :3]  # drop alpha
        img = Image.fromarray(rgb)
        img_path = self._episode_dir / f"render_{step_idx:02d}.png"
        img.save(str(img_path))
        return str(img_path)

    def _get_agent_position(self) -> Optional[np.ndarray]:
        """Return agent position as numpy array, or None."""
        if self._agent is None:
            return None
        return self._agent.get_state().position.copy()

    # ------------------------------------------------------------------
    # Action → Habitat actions mapping
    # ------------------------------------------------------------------
    @staticmethod
    def _rotation_to_turns(
        rotation_degrees: float, turn_amount: float
    ) -> List[str]:
        """
        Convert a rotation in degrees to a sequence of turn_left / turn_right
        actions, each of ``turn_amount`` degrees.
        """
        if rotation_degrees is None or rotation_degrees == 0:
            return []
        n_turns = int(round(abs(rotation_degrees) / turn_amount))
        n_turns = max(n_turns, 0)
        direction = "turn_left" if rotation_degrees > 0 else "turn_right"
        return [direction] * n_turns

    @staticmethod
    def _forward_to_steps(
        forward_metres: float, step_amount: float
    ) -> List[str]:
        """
        Convert forward distance to a sequence of move_forward actions.
        Negative values are ignored (habitat has no move_backward by default).
        """
        if forward_metres is None or forward_metres <= 0:
            return []
        n_steps = int(round(forward_metres / step_amount))
        return ["move_forward"] * max(n_steps, 0)

    def _execute_movement(
        self,
        rotation_deg: float,
        forward_m: float,
        left_m: float,
        z_delta_m: float,
    ) -> Dict[str, np.ndarray]:
        """
        Execute movement by converting to discrete Habitat actions.

        Returns the final sensor observations after all sub-steps.
        """
        actions: List[str] = []

        # 1. Rotate
        actions.extend(
            self._rotation_to_turns(rotation_deg, self.cfg.turn_amount)
        )

        # 2. Lateral movement (left) → rotate 90° left, move forward, rotate 90° right
        if left_m is not None and abs(left_m) > 0.01:
            if left_m > 0:
                actions.append("turn_left")
                actions.extend(
                    self._forward_to_steps(abs(left_m), self.cfg.move_forward_amount)
                )
                actions.append("turn_right")
            else:
                actions.append("turn_right")
                actions.extend(
                    self._forward_to_steps(abs(left_m), self.cfg.move_forward_amount)
                )
                actions.append("turn_left")

        # 3. Forward movement
        actions.extend(
            self._forward_to_steps(forward_m, self.cfg.move_forward_amount)
        )

        # Execute all sub-actions
        obs_dict = self._sim.get_sensor_observations()
        collided = False
        for act in actions:
            obs_dict = self._sim.step(act)
            if obs_dict.get("collided", False):
                collided = True

        return obs_dict

    # ------------------------------------------------------------------
    # Prompt builder (mirrors build_instruction_text from sequential.py)
    # ------------------------------------------------------------------
    def _build_prompt(self, is_final: bool) -> str:
        """
        Build the instruction prompt for the current step.

        Uses ``evaluation/sequential.py::build_instruction_text`` if
        available, otherwise falls back to a lightweight version.
        """
        pos = self._get_agent_position()

        try:
            from evaluation.sequential import build_instruction_text

            # build_instruction_text expects R, t — we synthesise them from
            # the agent's state.  For Habitat we pass identity R and the
            # agent position as t.
            R = np.eye(3)
            t = pos if pos is not None else np.zeros(3)

            return build_instruction_text(
                R,
                t,
                self._question,
                bbox=None,  # navmesh handles boundaries
                options=self._choices,
                is_final_step=is_final,
                movement_history=self._movement_history,
                step_num=self._step,
                question_type=self._question_type,
                is_numerical=self._is_numerical,
                max_steps=self.cfg.max_steps,
            )
        except ImportError:
            return self._build_prompt_fallback(is_final)

    def _build_prompt_fallback(self, is_final: bool) -> str:
        """Minimal prompt when sequential.py is not importable."""
        lines = [f"Question: {self._question}"]
        if self._choices:
            for i, c in enumerate(self._choices):
                lines.append(f"  ({chr(65 + i)}) {c}")

        pos = self._get_agent_position()
        if pos is not None:
            lines.append(
                f"\nCurrent position: X={pos[0]:.2f}m, Y={pos[1]:.2f}m, Z={pos[2]:.2f}m"
            )
        lines.append(f"\nStep {self._step}/{self.cfg.max_steps}")

        if is_final:
            lines.append(
                "\n**This is your FINAL step.** You MUST provide your answer now."
            )
        else:
            lines.append(
                "\nProvide movement as JSON: "
                '{"rotation_angle_degrees": ..., "forward_meters": ..., '
                '"left_meters": ..., "z_delta_meters": ...}'
                " or provide your answer."
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self, question_data: Dict[str, Any]) -> Observation:
        """
        Load the scene, initialise the agent, return the first observation.
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
        self._dataset = question_data.get("dataset", "habitat")

        # ── episode output dir ──
        self._episode_dir = self.output_dir / f"q{self.question_id:03d}"
        self._episode_dir.mkdir(parents=True, exist_ok=True)

        # ── find scene .glb ──
        scene_dirs = self.cfg.scene_dirs or DEFAULT_SCENE_DIRS
        scene_path = _find_scene_glb(self._scene_id, scene_dirs)
        if scene_path is None:
            raise FileNotFoundError(
                f"No .glb scene found for '{self._scene_id}' in {scene_dirs}"
            )

        # ── create / reuse simulator ──
        self._make_sim(scene_path)

        # ── initialise agent & capture first observation ──
        obs_dict = self._init_agent()
        img_path = self._save_observation(obs_dict, step_idx=0)

        self._image_history = [img_path]
        self._movement_history = []

        return self._make_observation()

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(
        self, action: Dict[str, Any] | str
    ) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Execute one step — same signature as ``VSIEnv.step()``.

        ``action`` can be:
        * a dict with movement keys + optional ``answer`` / ``done``
        * raw model output text (parsed via ``parse_qwen_output_and_get_movement``)

        Returns ``(observation, reward, done, info)``.
        """
        if self._done:
            raise RuntimeError("Episode already finished; call reset() first.")

        info: Dict[str, Any] = {"parse_success": True, "format_penalty": 0.0}

        # ── parse action ──
        if isinstance(action, str):
            rot, fwd, left, z_d, reasoning, raw_obj, done_flag = (
                parse_qwen_output_and_get_movement(action)
            )
            answer_value = None
            if raw_obj and isinstance(raw_obj, dict):
                answer_value = raw_obj.get("answer")
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
            info["format_penalty"] = -0.1

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
            obs_dict = self._execute_movement(rot, fwd, left, z_d)

            pos = self._get_agent_position()
            self._movement_history.append(
                {
                    "rotation": rot,
                    "forward": fwd,
                    "left": left,
                    "z_delta": z_d,
                    "position": (
                        f"X={pos[0]:.2f}m, Y={pos[1]:.2f}m, Z={pos[2]:.2f}m"
                        if pos is not None
                        else "unknown"
                    ),
                }
            )

            self._step += 1
            img_path = self._save_observation(obs_dict, step_idx=self._step)
            self._image_history.append(img_path)
        elif not episode_done:
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
    # observation builder
    # ------------------------------------------------------------------
    def _make_observation(self) -> Observation:
        """Build the observation the policy should see at the current step."""
        # is_final = True when the NEXT action must include an answer
        # (i.e., this is the policy's last chance before forced termination)
        is_final = self._step >= self.cfg.max_steps - 1
        prompt = self._build_prompt(is_final)

        return Observation(
            image_paths=list(self._image_history),
            prompt_text=prompt,
            step=self._step,
            is_final_step=is_final,
            question=self._question,
            choices=self._choices,
            question_type=self._question_type,
            is_numerical=self._is_numerical,
            cam_position=self._get_agent_position(),
            bbox=None,  # habitat uses navmesh, not explicit bbox
        )

    # ------------------------------------------------------------------
    # reward (same logic as VSIEnv)
    # ------------------------------------------------------------------
    def _compute_reward(self) -> float:
        if self._final_answer is None:
            return -1.0

        if self._is_numerical:
            try:
                from utils.evaluation import calculate_mra

                pred = float(self._final_answer)
                gt = float(self._ground_truth)
                mra = calculate_mra(pred, gt)
                return mra * 1.5 - 0.5
            except (ValueError, TypeError, ImportError):
                return -1.0
        else:
            return 1.0 if self._check_correct() else -0.5

    def _check_correct(self) -> bool:
        if self._final_answer is None:
            return False
        if self._is_numerical:
            try:
                from utils.evaluation import calculate_mra

                pred = float(self._final_answer)
                gt = float(self._ground_truth)
                return calculate_mra(pred, gt) > 0.5
            except (ValueError, TypeError, ImportError):
                return False
        return (
            str(self._final_answer).strip().upper()
            == str(self._ground_truth).strip().upper()
        )

    # ------------------------------------------------------------------
    # trajectory saving
    # ------------------------------------------------------------------
    def _save_trajectory(self, info: Dict[str, Any]) -> None:
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
            "movement_history": self._movement_history,
            "env_type": "habitat",
        }
        with open(self._episode_dir / "trajectory.json", "w") as f:
            json.dump(traj, f, indent=2)

    # ------------------------------------------------------------------
    # cleanup
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Release the Habitat-Sim simulator."""
        if self._sim is not None:
            self._sim.close()
            self._sim = None
            self._agent = None
            self._loaded_scene_path = None

    def __del__(self):
        self.close()

    # ------------------------------------------------------------------
    # convenience (match VSIEnv interface)
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
        """Not stored as 4×4 matrices in Habitat — return empty list."""
        return []

    def __repr__(self) -> str:
        return (
            f"HabitatEnv(scene={self._scene_id!r}, step={self._step}, "
            f"done={self._done}, images={len(self._image_history)})"
        )
