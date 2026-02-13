#!/usr/bin/env python3
"""
Evaluation / smoke-test script.

Runs episodes through ``VSIEnv`` using either:
* ``--dummy``  — random walk policy (verifies env produces correct images).
* ``--backend vllm``  — real model inference (greedy evaluation).

The main deliverable of **Step 1** is::

    python -m vagen_vsi_rl.scripts.eval --dummy --max-questions 2

which must save rendered images and trajectory JSON that are identical
in structure to what ``evaluation/sequential.py`` produces.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vagen_vsi_rl.env import VSIEnv, EnvConfig
from vagen_vsi_rl.rollout import RolloutCollector
from vagen_vsi_rl.rollout.collector import dummy_policy

from utils.data import load_vsi_bench_questions, MCA_QUESTION_TYPES


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate VSIEnv")
    p.add_argument("--dummy", action="store_true",
                   help="Use random-walk policy (no model required)")
    p.add_argument("--backend", type=str, default="vllm", choices=["hf", "vllm"],
                   help="Inference backend (ignored when --dummy)")
    p.add_argument("--dataset", type=str, default="arkitscenes",
                   choices=["arkitscenes", "scannet", "scannetpp", "all"])
    p.add_argument("--max-questions", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=7,
                   help="Exploration steps per question (default 7 = 8 images)")
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"test_env_output/{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] Output directory: {output_dir.resolve()}")

    # Load a few questions
    questions = load_vsi_bench_questions(
        question_types=MCA_QUESTION_TYPES, dataset=args.dataset
    )
    questions = questions[: args.max_questions]
    print(f"[eval] Running {len(questions)} questions")

    env_cfg = EnvConfig(max_steps=args.max_steps)
    collector = RolloutCollector(output_dir=output_dir, config=env_cfg)

    if args.dummy:
        trajs = collector.collect(questions, policy_fn=dummy_policy, save=True)
    else:
        # Model-based rollout with logprobs
        from vagen_vsi_rl.models import ActorVLLM
        actor = ActorVLLM()
        trajs = collector.collect(questions, actor=actor, save=True)

    # Print summary
    print(f"\n{'='*60}")
    print(f"[eval] SUMMARY")
    print(f"{'='*60}")
    n_correct = sum(t.is_correct for t in trajs)
    print(f"  Episodes:  {len(trajs)}")
    print(f"  Correct:   {n_correct}/{len(trajs)}")
    print(f"  Output:    {output_dir.resolve()}")

    # list files produced
    print(f"\nFiles produced:")
    for p in sorted(output_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(output_dir)
            size_kb = p.stat().st_size / 1024
            print(f"  {rel}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
