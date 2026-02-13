"""
vagen_vsi_rl — Reinforcement Learning for VSI-Bench Multi-Turn Spatial Reasoning.

This package wraps the existing sequential evaluation pipeline as a POMDP
environment and provides PPO training infrastructure on top of it.

Main components
───────────────
env.vsi_env        Gym-like POMDP wrapper around the question loop.
rollout.trajectory  Serialisable trajectory / turn dataclasses.
rollout.collector   Collects trajectories by running the environment with a policy.
models.*            Thin wrappers around actor (vLLM / HF), critic, and reference models.
rl.rewards          Reward functions (correctness, format, exploration).
rl.advantage        GAE / Monte-Carlo return estimation.
rl.ppo              PPO loss computation (clipped surrogate + KL).
rl.sync             Weight synchronisation between HF ↔ vLLM copies.
scripts.train       End-to-end training entrypoint.
scripts.eval        Greedy evaluation entrypoint.
"""

__version__ = "0.1.0"
