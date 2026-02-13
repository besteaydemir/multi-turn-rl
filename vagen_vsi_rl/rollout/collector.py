#!/usr/bin/env python3
"""
Rollout collector — drives a policy through ``VSIEnv`` and builds
``Trajectory`` objects ready for PPO training.

Three modes:
* **dummy** — uses a random action dict (for pipeline testing).
* **model (text-only)** — calls an actor's ``generate()`` → text, no logprobs.
* **model (with logprobs)** — calls ``ActorVLLM.generate_with_logprobs()``
  and stores token IDs + per-token log-probs in each ``Turn`` so that
  PPO can compute the ratio  π_θ / π_old  without retokenisation.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from ..env.common import Observation
from .trajectory import Trajectory, Turn

if TYPE_CHECKING:
    from ..models.actor_vllm import ActorVLLM, GenerateOutput
    from ..env.vsi_env import VSIEnv, EnvConfig
    from ..env.habitat_env import HabitatEnv, HabitatEnvConfig

# Union of supported env config types (for type hints)
EnvConfigType = Any  # EnvConfig | HabitatEnvConfig at runtime


PolicyFn = Callable[[Observation], str]  # obs → raw model text


# ─────────────────────────────────────────────────────────────────────
# Helpers to build vLLM-compatible messages from an Observation
# ─────────────────────────────────────────────────────────────────────
def _obs_to_messages(obs: Observation) -> List[Dict[str, Any]]:
    """
    Convert an ``Observation`` into the chat-style ``messages`` list that
    the Qwen3-VL model expects (same format as ``sequential.py``).

    Each accumulated image becomes a ``{"type": "image", "image": path}``
    entry, followed by the instruction prompt as text.
    """
    content: List[Dict[str, Any]] = []
    for img_idx, img_path in enumerate(obs.image_paths):
        if img_idx == 0:
            label = "\n**Image 0 (Initial view):**"
        else:
            label = f"\n**Image {img_idx} (After movement {img_idx}):**"
        content.append({"type": "text", "text": label})
        content.append({"type": "image", "image": img_path})

    content.append({"type": "text", "text": f"\n\n{obs.prompt_text}"})

    return [{"role": "user", "content": content}]


# ─────────────────────────────────────────────────────────────────────
# Built-in dummy policy
# ─────────────────────────────────────────────────────────────────────
def dummy_policy(obs: Observation) -> Dict[str, Any]:
    """
    Random walk policy for testing the environment.

    Produces a dict action (rotation + forward/left + maybe an answer on the
    last step).
    """
    if obs.is_final_step:
        # Must answer
        if obs.is_numerical:
            answer = str(random.randint(1, 20))
        else:
            answer = random.choice(["A", "B", "C", "D"])
        return {
            "rotation_angle_degrees": 0.0,
            "forward_meters": 0.0,
            "left_meters": 0.0,
            "z_delta_meters": 0.0,
            "answer": answer,
            "done": True,
        }

    # random exploration
    return {
        "rotation_angle_degrees": random.uniform(-90, 90),
        "forward_meters": random.uniform(-0.5, 0.8),
        "left_meters": random.uniform(-0.3, 0.3),
        "z_delta_meters": random.uniform(-0.1, 0.1),
        "answer": None,
        "done": False,
    }


# ─────────────────────────────────────────────────────────────────────
# Collector
# ─────────────────────────────────────────────────────────────────────
class RolloutCollector:
    """
    Collect trajectories by running a policy in an environment.

    Supports two backends:
    * ``"vsi"``     — ``VSIEnv`` (Open3D mesh rendering, the default)
    * ``"habitat"`` — ``HabitatEnv`` (Habitat-Sim GPU rendering)

    Usage (dummy)::

        collector = RolloutCollector(output_dir="runs/test", config=EnvConfig(max_steps=7))
        trajs = collector.collect(questions[:8], policy_fn=dummy_policy)

    Usage (Habitat backend)::

        collector = RolloutCollector(
            output_dir="runs/test",
            env_type="habitat",
            config=HabitatEnvConfig(max_steps=7),
        )
        trajs = collector.collect(questions[:8], policy_fn=dummy_policy)

    Usage (vLLM actor with logprobs)::

        actor = ActorVLLM(model_id="Qwen/Qwen3-VL-8B-Instruct")
        collector = RolloutCollector(output_dir="runs/train", config=EnvConfig(max_steps=7))
        trajs = collector.collect(questions[:8], actor=actor)
    """

    SUPPORTED_ENVS = {"vsi", "habitat"}

    def __init__(
        self,
        output_dir: str | Path,
        config: Optional["EnvConfigType"] = None,
        env_type: str = "vsi",
    ):
        if env_type not in self.SUPPORTED_ENVS:
            raise ValueError(
                f"Unknown env_type={env_type!r}. "
                f"Choose from {self.SUPPORTED_ENVS}"
            )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.env_type = env_type

        # Assign sensible default config for the chosen env
        if config is None:
            if env_type == "habitat":
                from ..env.habitat_env import HabitatEnvConfig

                self.config = HabitatEnvConfig()
            else:
                from ..env.vsi_env import EnvConfig

                self.config = EnvConfig()
        else:
            self.config = config

    # ─────────────────────── main entry point ────────────────────────
    def collect(
        self,
        questions: List[Dict[str, Any]],
        policy_fn: Optional[PolicyFn] = None,
        actor: Optional["ActorVLLM"] = None,
        save: bool = True,
    ) -> List[Trajectory]:
        """
        Run one episode per question.

        Exactly **one** of ``policy_fn`` or ``actor`` should be supplied.

        * ``policy_fn`` — simple callable, no logprobs.
        * ``actor``     — ``ActorVLLM`` instance; ``generate_with_logprobs``
          is called and token_ids + logprobs are stored in each Turn.

        If neither is given, falls back to ``dummy_policy``.
        """
        if actor is not None and policy_fn is not None:
            raise ValueError("Supply either `policy_fn` or `actor`, not both.")

        use_actor = actor is not None
        if policy_fn is None and actor is None:
            policy_fn = dummy_policy

        trajectories: List[Trajectory] = []

        for q_idx, q_data in enumerate(questions):
            print(f"\n[Collector] ── Episode {q_idx+1}/{len(questions)} "
                  f"scene={q_data['scene_name']} ──")
            t0 = time.time()

            if self.env_type == "habitat":
                from ..env.habitat_env import HabitatEnv

                env = HabitatEnv(
                    output_dir=self.output_dir,
                    config=self.config,
                    question_id=q_idx,
                )
            else:
                from ..env.vsi_env import VSIEnv

                env = VSIEnv(
                    output_dir=self.output_dir,
                    config=self.config,
                    question_id=q_idx,
                )

            try:
                obs = env.reset(q_data)
            except FileNotFoundError as e:
                print(f"[Collector] SKIP — {e}")
                continue

            traj = Trajectory(
                trajectory_id=f"traj_{q_idx:04d}",
                question=q_data["question"],
                choices=q_data.get("choices"),
                ground_truth=q_data.get("answer_id"),
                scene_id=q_data["scene_name"],
                dataset=q_data.get("dataset", "arkitscenes"),
                question_type=q_data.get("question_type", "unknown"),
                is_numerical=q_data.get("is_numerical", False),
            )

            step = 0
            done = False
            while not done:
                turn = Turn(
                    turn_index=step,
                    image_paths=list(obs.image_paths),
                    prompt_text=obs.prompt_text,
                )

                # ── generate action ──
                gen_t0 = time.time()
                if use_actor:
                    # Model-based policy with logprobs
                    messages = _obs_to_messages(obs)
                    gen_out = actor.generate_with_logprobs(messages)

                    action_text = gen_out.text
                    turn.generated_text = action_text
                    turn.generated_ids = gen_out.token_ids
                    turn.logprobs = gen_out.token_logprobs
                    turn.prompt_token_ids = gen_out.prompt_token_ids  # Full input context for PPO
                    # We pass the raw text to env.step() — it will parse
                    # movement + done/answer internally (Step 2 contract).
                    action = action_text
                else:
                    # Simple policy (dummy or custom callable)
                    action = policy_fn(obs)
                    if isinstance(action, str):
                        turn.generated_text = action
                    else:
                        turn.generated_text = json.dumps(action)
                        turn.action_dict = action

                turn.generation_time_s = time.time() - gen_t0

                # ── step environment ──
                obs, reward, done, info = env.step(action)

                turn.reward = reward
                turn.done_flag = done
                # Store the executable action parsed by the env
                turn.action_dict = info.get("executable_action", turn.action_dict)
                if done:
                    turn.answer = info.get("final_answer")

                traj.turns.append(turn)
                step += 1

            elapsed = time.time() - t0
            traj.final_answer = info.get("final_answer")
            traj.is_correct = info.get("is_correct", False)
            traj.terminal_reward = reward
            traj.num_steps = step
            traj.elapsed_time_s = elapsed

            correct_str = "✅" if traj.is_correct else "❌"
            print(
                f"[Collector] {correct_str} answer={traj.final_answer}  "
                f"gt={traj.ground_truth}  reward={reward:.2f}  "
                f"steps={step}  time={elapsed:.1f}s"
            )

            if save:
                traj.save(self.output_dir / f"traj_{q_idx:04d}")

            # Release GPU resources for Habitat envs
            if hasattr(env, "close"):
                env.close()

            trajectories.append(traj)

        # summary
        n_correct = sum(t.is_correct for t in trajectories)
        print(
            f"\n[Collector] Done — {len(trajectories)} episodes, "
            f"accuracy {n_correct}/{len(trajectories)}"
        )
        return trajectories
