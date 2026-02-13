#!/usr/bin/env python3
"""
End-to-end PPO training loop (Step 7 deliverable).

Outer loop:
  1. Collect rollout trajectories with vLLM actor + VSIEnv/HabitatEnv.
  2. Compute rewards and advantages.
  3. Re-compute log-probs with HF actor (gradient enabled).
  4. Run PPO update on HF actor + critic.
  5. Sync weights back to vLLM.
  6. Repeat.

Usage:
    python -m vagen_vsi_rl.scripts.train --dummy --episodes-per-update 4
    python -m vagen_vsi_rl.scripts.train --model Qwen/Qwen3-VL-4B-Instruct --env habitat
"""
from __future__ import annotations

# CRITICAL: Set multiprocessing start method to 'spawn' BEFORE importing torch/CUDA
# vLLM uses multiprocessing and will fail with "Cannot re-initialize CUDA in forked subprocess"
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

import argparse
import gc
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vagen_vsi_rl.env import VSIEnv, EnvConfig
from vagen_vsi_rl.rollout import RolloutCollector, Trajectory
from vagen_vsi_rl.rollout.collector import dummy_policy
from vagen_vsi_rl.rl.rewards import compute_rewards, RewardConfig
from vagen_vsi_rl.rl.advantage import compute_monte_carlo_returns, compute_gae, compute_bilevel_gae, set_bilevel_debug
from vagen_vsi_rl.rl.ppo import ppo_step, PPOConfig

from utils.data import load_vsi_bench_questions, MCA_QUESTION_TYPES


# ─────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="PPO training for VSI-Bench")
    p.add_argument("--config", type=str, default=None, help="YAML config file")
    
    # Training
    p.add_argument("--num-updates", type=int, default=100, help="Number of PPO updates")
    p.add_argument("--episodes-per-update", type=int, default=8, help="Rollout batch size")
    p.add_argument("--ppo-epochs", type=int, default=4, help="PPO update epochs per batch")
    p.add_argument("--lr", type=float, default=1e-5, help="Actor learning rate")
    p.add_argument("--critic-lr", type=float, default=1e-4, help="Critic learning rate")
    p.add_argument("--gamma", type=float, default=1.0, help="Discount factor")
    # GAE configuration
    p.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    p.add_argument("--bilevel-gae", action="store_true", default=True,
                   help="Use bi-level GAE (VAGEN Algorithm 2) with token-level KL penalty")
    p.add_argument("--token-kl-coef", type=float, default=0.05,
                   help="Token-level KL coefficient for bi-level GAE")
    
    # LR Scheduler
    p.add_argument("--lr-scheduler", type=str, default="cosine",
                   choices=["none", "cosine", "linear", "warmup_cosine"],
                   help="LR scheduler type")
    p.add_argument("--warmup-updates", type=int, default=10, help="Warmup updates for LR scheduler")
    p.add_argument("--min-lr", type=float, default=1e-7, help="Minimum LR for scheduler")
    
    # Resume
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint directory to resume from")
    p.add_argument("--resume-strict", action="store_true", help="Fail if checkpoint loading has issues")
    
    # Model
    p.add_argument("--model", type=str, default="Qwen/Qwen3-VL-4B-Instruct",
                   help="Model ID (vLLM + HF)")
    p.add_argument("--cache-dir", type=str, 
                   default="/dss/mcmlscratch/06/di38riq")
    
    # LoRA
    p.add_argument("--lora", action="store_true", help="Use LoRA for efficient fine-tuning")
    p.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    p.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    p.add_argument("--lora-target-modules", type=str, default="q_proj,k_proj,v_proj,o_proj",
                   help="Comma-separated list of target modules for LoRA")
    
    # Environment
    p.add_argument("--env", type=str, default="vsi", choices=["vsi", "habitat"])
    p.add_argument("--max-steps", type=int, default=7)
    p.add_argument("--dataset", type=str, default="arkitscenes",
                   choices=["arkitscenes", "scannet", "scannetpp", "all"])
    
    # Output
    p.add_argument("--output-dir", type=str, default="./checkpoints/vagen_vsi_rl")
    p.add_argument("--save-every", type=int, default=50, help="Save checkpoint every N updates")
    p.add_argument("--save-trajectories", action="store_true", help="Save trajectory JSON files")
    
    # Validation
    p.add_argument("--val-every", type=int, default=10, help="Run validation every N updates")
    p.add_argument("--val-episodes", type=int, default=8, help="Number of validation episodes")
    p.add_argument("--val-dataset", type=str, default=None, 
                   help="Validation dataset (defaults to same as training)")
    
    # Logging
    p.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    p.add_argument("--wandb-project", type=str, default="vagen-vsi-rl", help="Wandb project name")
    p.add_argument("--wandb-run-name", type=str, default=None, help="Wandb run name (auto-generated if not set)")
    p.add_argument("--wandb-entity", type=str, default=None, help="Wandb entity/team")
    
    # Debug modes
    p.add_argument("--dummy", action="store_true", help="Use dummy policy (no model)")
    p.add_argument("--no-critic", action="store_true", help="Skip critic, use MC returns")
    p.add_argument("--no-ref", action="store_true", help="Skip reference model (no KL penalty)")
    p.add_argument("--share-critic-backbone", action="store_true",
                   help="Share actor backbone with critic (saves ~8GB, but critic sees LoRA updates)")
    p.add_argument("--no-vllm", action="store_true", 
                   help="Use HF for inference too (no vLLM). Slower but fits on 40GB GPU.")
    p.add_argument("--sync-every", type=int, default=1, help="Sync vLLM weights every N updates")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.4,
                   help="vLLM GPU memory utilization (lower = more room for HF models)")
    p.add_argument("--max-model-len", type=int, default=8192,
                   help="vLLM max context length (lower = less KV cache memory)")
    
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Load config
    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    # Merge CLI overrides
    if args.env == "habitat":
        from vagen_vsi_rl.env.habitat_env import HabitatEnvConfig
        env_cfg = HabitatEnvConfig(max_steps=args.max_steps)
    else:
        env_cfg = EnvConfig(max_steps=args.max_steps)
        
    reward_cfg = RewardConfig(**(cfg.get("rewards", {})))
    ppo_cfg = PPOConfig(**(cfg.get("ppo", {})))

    # Directories
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] Output: {output_dir}")
    
    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Initialize wandb
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            run_name = args.wandb_run_name or f"ppo_{args.env}_{ts}"
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config=vars(args),
                dir=str(output_dir),
                reinit=True,
            )
            print(f"[train] Wandb: {wandb_run.url}")
        except ImportError:
            print("[train] wandb not installed, skipping wandb logging")
            args.wandb = False

    # Load questions
    qtypes = cfg.get("env", {}).get("question_types", "mcq")
    if qtypes == "mcq":
        question_types = MCA_QUESTION_TYPES
    else:
        from utils.data import ALL_SEQUENTIAL_QUESTION_TYPES
        question_types = ALL_SEQUENTIAL_QUESTION_TYPES

    questions = load_vsi_bench_questions(
        question_types=question_types, dataset=args.dataset
    )
    print(f"[train] Loaded {len(questions)} questions")

    # ── Initialize models (skip in dummy mode) ──
    actor_vllm = None
    actor_hf = None
    critic = None
    ref_model = None
    actor_optim = None
    critic_optim = None
    
    if not args.dummy:
        print(f"\n[train] Loading models: {args.model}")
        
        if args.no_vllm:
            # Use HF for both inference and training (fits on 40GB GPU)
            print("[train] Using HF for inference (--no-vllm mode, no vLLM)")
            from vagen_vsi_rl.models import ActorHF
            lora_targets = args.lora_target_modules.split(",") if args.lora else None
            actor_hf = ActorHF(
                model_id=args.model,
                cache_dir=args.cache_dir,
                device="cuda",
                use_lora=args.lora,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                lora_target_modules=lora_targets,
            )
            actor_hf.load()
            actor_optim = torch.optim.AdamW(
                [p for p in actor_hf.parameters() if p.requires_grad], 
                lr=args.lr
            )
            # No vLLM - we'll use actor_hf for inference too
            actor_vllm = None
        else:
            # Standard mode: vLLM for inference, HF for backprop
            # 1. vLLM actor for rollout (MUST initialize first to reserve GPU memory)
            from vagen_vsi_rl.models import ActorVLLM
            actor_vllm = ActorVLLM(
                model_id=args.model,
                cache_dir=args.cache_dir,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
            )
            # Initialize vLLM NOW before HF models claim GPU memory
            _ = actor_vllm.tokenizer  # This triggers lazy init
            
            # 2. HF actor for gradient updates
            from vagen_vsi_rl.models import ActorHF
            lora_targets = args.lora_target_modules.split(",") if args.lora else None
            actor_hf = ActorHF(
                model_id=args.model,
                cache_dir=args.cache_dir,
                device="cuda",
                use_lora=args.lora,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                lora_target_modules=lora_targets,
            )
            actor_hf.load()
            actor_optim = torch.optim.AdamW(
                [p for p in actor_hf.parameters() if p.requires_grad], 
                lr=args.lr
            )
        
        # 3. Critic (separate model, or sharing backbone with actor)
        if not args.no_critic:
            from vagen_vsi_rl.models import CriticHF
            critic = CriticHF(
                model_id=args.model,
                cache_dir=args.cache_dir,
                freeze_backbone=True,  # Only train value head
            )
            if args.share_critic_backbone:
                # Share backbone with actor - saves ~8GB but critic sees LoRA updates
                print("[train] Sharing actor backbone with critic (--share-critic-backbone)")
                critic.load(share_backbone=actor_hf.model)
            else:
                critic.load()  # Load separate backbone copy
            critic_optim = torch.optim.AdamW(critic.value_head.parameters(), lr=args.critic_lr)
        
        # 4. Reference model for KL (frozen) - optional
        if not args.no_ref:
            from vagen_vsi_rl.models import ReferenceModel
            ref_model = ReferenceModel(
                model_id=args.model,
                cache_dir=args.cache_dir,
            )
            ref_model.load()
        else:
            print("[train] Skipping reference model (--no-ref)")
        
        # 5. CRITICAL: Validate tokenizer alignment between vLLM and HF
        # Misaligned tokenizers will break PPO (log-probs won't match token IDs)
        # Skip validation if using HF for inference (--no-vllm, same model)
        if actor_vllm is not None:
            from vagen_vsi_rl.utils.token_utils import validate_tokenizer_alignment
            print("[train] Validating tokenizer alignment between vLLM and HF...")
            try:
                report = validate_tokenizer_alignment(
                    actor_vllm.tokenizer, 
                    actor_hf.tokenizer,
                    strict=True,
                )
                print("[train] ✓ Tokenizers aligned")
            except Exception as e:
                print(f"[train] ⚠ Tokenizer validation failed: {e}")
                print("[train] Proceeding anyway, but PPO may not work correctly!")
        else:
            print("[train] Using HF for inference (--no-vllm), skipping tokenizer validation")

    # ── LR Schedulers ──
    actor_scheduler = None
    critic_scheduler = None
    
    if not args.dummy and actor_optim is not None:
        actor_scheduler = _create_scheduler(
            actor_optim, args.lr_scheduler, args.num_updates,
            args.warmup_updates, args.min_lr
        )
        if critic_optim is not None:
            critic_scheduler = _create_scheduler(
                critic_optim, args.lr_scheduler, args.num_updates,
                args.warmup_updates, args.min_lr
            )

    # ── Resume from checkpoint ──
    start_update = 1
    best_acc = 0.0
    if args.resume:
        start_update, best_acc = _load_checkpoint(
            args.resume,
            actor_hf=actor_hf,
            actor_optim=actor_optim,
            actor_scheduler=actor_scheduler,
            critic=critic,
            critic_optim=critic_optim,
            critic_scheduler=critic_scheduler,
            actor_vllm=actor_vllm,
            strict=args.resume_strict,
        )
        print(f"[train] Resumed from update {start_update - 1}, best_acc={best_acc:.2%}")

    # Collector
    collector = RolloutCollector(
        output_dir=output_dir / "rollouts", 
        config=env_cfg,
        env_type=args.env,
    )

    # Set up policy
    if args.dummy:
        policy_fn = dummy_policy
        inference_actor = None
    else:
        policy_fn = None
        # Use actor_hf for inference if --no-vllm, otherwise actor_vllm
        inference_actor = actor_hf if args.no_vllm else actor_vllm

    # ── Main training loop ──
    
    for update in range(start_update, args.num_updates + 1):
        update_t0 = time.time()
        print(f"\n{'='*60}")
        print(f"[train] Update {update}/{args.num_updates}")
        print(f"{'='*60}")

        # 1. Sample a batch of questions
        batch = random.sample(questions, min(args.episodes_per_update, len(questions)))

        # 2. Collect trajectories (with logprobs if actor is provided)
        trajs = collector.collect(
            batch,
            policy_fn=policy_fn,
            actor=inference_actor,
            save=args.save_trajectories,
        )

        # 3. Compute rewards
        for traj in trajs:
            compute_rewards(traj, reward_cfg)

        # 3.5. Compute ref_logprobs for bi-level GAE (needed for token-level KL)
        if ref_model is not None and args.bilevel_gae:
            _compute_ref_logprobs(trajs, ref_model)

        # 4. Compute advantages
        if args.no_critic or critic is None:
            # Monte-Carlo returns (no value function)
            for traj in trajs:
                compute_monte_carlo_returns(traj, gamma=args.gamma)
        elif args.bilevel_gae and ref_model is not None:
            # Bi-level GAE (VAGEN Algorithm 2) with token-level KL
            for traj in trajs:
                values = _compute_critic_values(traj, critic)
                compute_bilevel_gae(
                    traj, values,
                    gamma=args.gamma,
                    turn_lambda=args.gae_lambda,
                    token_lambda=args.gae_lambda,
                    kl_coef=args.token_kl_coef,
                )
        else:
            # Standard GAE with critic values
            for traj in trajs:
                values = _compute_critic_values(traj, critic)
                compute_gae(traj, values, gamma=args.gamma, lam=args.gae_lambda)

        # 4.5. Save debug trajectory on first update
        if update == 1 and len(trajs) > 0:
            _save_debug_trajectory(trajs[0], output_dir / "debug_trajectory.json")
        
        # Disable verbose bi-level GAE logging after first 3 updates
        if update == 3:
            set_bilevel_debug(False)

        # 5. PPO update (skip in dummy mode)
        if not args.dummy and actor_hf is not None:
            ppo_stats = _ppo_update(
                trajs=trajs,
                actor_hf=actor_hf,
                actor_optim=actor_optim,
                critic=critic,
                critic_optim=critic_optim,
                ref_model=ref_model,
                ppo_cfg=ppo_cfg,
                num_epochs=args.ppo_epochs,
            )
            print(f"[train] PPO: loss={ppo_stats['loss']:.4f} "
                  f"policy={ppo_stats['policy_loss']:.4f} "
                  f"kl={ppo_stats['kl']:.4f} "
                  f"clip_frac={ppo_stats['clip_frac']:.2%}")
            
            # 6. Sync weights to vLLM
            if update % args.sync_every == 0 and actor_vllm is not None:
                from vagen_vsi_rl.rl.sync import sync_weights
                sync_ok = sync_weights(actor_hf, actor_vllm, verify=True)
                if not sync_ok:
                    print(f"[train] ⚠️  WARNING: Weight sync failed at update {update}!")
                    print(f"[train] ⚠️  vLLM will use stale weights until next successful sync.")
            
            # 7. Step LR schedulers
            if actor_scheduler is not None:
                actor_scheduler.step()
            if critic_scheduler is not None:
                critic_scheduler.step()

        # 7. Log metrics
        rewards = [t.terminal_reward for t in trajs]
        acc = sum(t.is_correct for t in trajs) / max(len(trajs), 1)
        avg_r = sum(rewards) / max(len(rewards), 1)
        avg_steps = sum(t.num_steps for t in trajs) / max(len(trajs), 1)
        
        update_time = time.time() - update_t0
        print(f"[train] acc={acc:.2%}  avg_reward={avg_r:.3f}  "
              f"avg_steps={avg_steps:.1f}  time={update_time:.1f}s")

        # Wandb logging
        if args.wandb and wandb_run is not None:
            log_dict = {
                "update": update,
                "accuracy": acc,
                "avg_reward": avg_r,
                "avg_steps": avg_steps,
                "num_episodes": len(trajs),
                "update_time_s": update_time,
            }
            if not args.dummy and 'ppo_stats' in dir():
                log_dict.update({
                    "ppo/loss": ppo_stats["loss"],
                    "ppo/policy_loss": ppo_stats["policy_loss"],
                    "ppo/kl": ppo_stats["kl"],
                    "ppo/entropy": ppo_stats.get("entropy", 0.0),
                    "ppo/clip_frac": ppo_stats["clip_frac"],
                })
                # Critic/value function metrics
                if ppo_stats.get("critic_loss", 0) > 0:
                    log_dict.update({
                        "critic/loss": ppo_stats["critic_loss"],
                        "critic/value_mean": ppo_stats["value_mean"],
                        "critic/value_std": ppo_stats["value_std"],
                        "critic/explained_variance": ppo_stats.get("explained_variance", 0.0),
                    })
            # Add LR to logs
            if actor_scheduler is not None:
                log_dict["lr/actor"] = actor_scheduler.get_last_lr()[0]
            if critic_scheduler is not None:
                log_dict["lr/critic"] = critic_scheduler.get_last_lr()[0]
            wandb.log(log_dict, step=update)

        # Save checkpoint
        ckpt = {
            "update": update,
            "accuracy": acc,
            "avg_reward": avg_r,
            "avg_steps": avg_steps,
            "num_episodes": len(trajs),
            "update_time_s": update_time,
        }
        if not args.dummy and 'ppo_stats' in dir():
            ckpt.update({k: float(v) for k, v in ppo_stats.items()})
            
        with open(output_dir / "log.jsonl", "a") as f:
            f.write(json.dumps(ckpt) + "\n")
            
        # Save model checkpoint
        if acc > best_acc and not args.dummy:
            best_acc = acc
            _save_checkpoint(
                output_dir / "best", actor_hf, critic, update, best_acc,
                actor_optim, actor_scheduler, critic_optim, critic_scheduler
            )
            
        if update % args.save_every == 0 and not args.dummy:
            _save_checkpoint(
                output_dir / f"ckpt_{update:04d}", actor_hf, critic, update, best_acc,
                actor_optim, actor_scheduler, critic_optim, critic_scheduler
            )
        
        # Run validation
        if update % args.val_every == 0 and not args.dummy:
            val_metrics = _run_validation(
                args=args,
                collector=collector,
                inference_actor=inference_actor,
                policy_fn=policy_fn,
                output_dir=output_dir,
                update=update,
            )
            print(f"[train] VAL: acc={val_metrics['val_accuracy']:.2%}  "
                  f"avg_reward={val_metrics['val_avg_reward']:.3f}")
            
            if args.wandb and wandb_run is not None:
                wandb.log(val_metrics, step=update)

    print(f"\n[train] Finished {args.num_updates} updates.  Output: {output_dir}")
    print(f"[train] Best accuracy: {best_acc:.2%}")
    
    # Finalize wandb
    if args.wandb and wandb_run is not None:
        wandb.log({"best_accuracy": best_acc})
        wandb.finish()


# ─────────────────────────────────────────────────────────────────────
# Helper: run validation
# ─────────────────────────────────────────────────────────────────────
def _run_validation(
    args,
    collector,
    inference_actor,
    policy_fn,
    output_dir: Path,
    update: int,
) -> Dict[str, float]:
    """
    Run validation on a held-out set of questions.
    
    Returns metrics dict for wandb logging.
    """
    from vagen_vsi_rl.rl.rewards import compute_rewards, RewardConfig
    
    # Determine validation dataset
    val_dataset = args.val_dataset or args.dataset
    
    # Load validation questions (use different random seed for variety)
    val_questions = load_vsi_bench_questions(
        question_types=MCA_QUESTION_TYPES, 
        dataset=val_dataset
    )
    
    # Sample validation batch (different from training by using offset)
    import random as val_random
    val_random.seed(update + 12345)  # Different seed per update
    val_batch = val_random.sample(
        val_questions, 
        min(args.val_episodes, len(val_questions))
    )
    
    print(f"\n[train] Running validation ({len(val_batch)} episodes)...")
    
    # Create separate collector for validation (no save)
    val_collector = RolloutCollector(
        output_dir=output_dir / "val_rollouts" / f"update_{update:04d}",
        config=collector.config,
        env_type=collector.env_type,
    )
    
    # Collect without saving (inference only)
    val_trajs = val_collector.collect(
        val_batch,
        policy_fn=policy_fn,
        actor=inference_actor,
        save=False,
    )
    
    # Compute rewards
    reward_cfg = RewardConfig()
    for traj in val_trajs:
        compute_rewards(traj, reward_cfg)
    
    # Compute metrics
    val_acc = sum(t.is_correct for t in val_trajs) / max(len(val_trajs), 1)
    val_rewards = [t.terminal_reward for t in val_trajs]
    val_avg_r = sum(val_rewards) / max(len(val_rewards), 1)
    val_avg_steps = sum(t.num_steps for t in val_trajs) / max(len(val_trajs), 1)
    
    return {
        "val_accuracy": val_acc,
        "val_avg_reward": val_avg_r,
        "val_avg_steps": val_avg_steps,
        "val_num_episodes": len(val_trajs),
    }


# ─────────────────────────────────────────────────────────────────────
# Helper: save debug trajectory
# ─────────────────────────────────────────────────────────────────────
def _save_debug_trajectory(traj: Trajectory, path: Path) -> None:
    """
    Save a detailed debug trajectory to inspect training data flow.
    
    This helps verify:
    - Images are accumulated correctly across steps
    - Prompt tokens are stored
    - Action tokens and logprobs are aligned
    - Advantages and returns are computed
    """
    debug_data = {
        "trajectory_id": traj.trajectory_id,
        "question": traj.question,
        "ground_truth": traj.ground_truth,
        "final_answer": traj.final_answer,
        "is_correct": traj.is_correct,
        "terminal_reward": traj.terminal_reward,
        "num_turns": len(traj.turns),
        "turns": [],
    }
    
    for turn in traj.turns:
        turn_data = {
            "turn_index": turn.turn_index,
            "num_images": len(turn.image_paths),
            "image_paths": turn.image_paths,
            "prompt_text_len": len(turn.prompt_text),
            "prompt_text_preview": turn.prompt_text[:500] if turn.prompt_text else "",
            "generated_text": turn.generated_text,
            "generated_text_len": len(turn.generated_text) if turn.generated_text else 0,
            # Token info
            "has_prompt_token_ids": turn.prompt_token_ids is not None,
            "prompt_token_ids_len": turn.prompt_token_ids.shape[0] if turn.prompt_token_ids is not None else 0,
            "has_generated_ids": turn.generated_ids is not None,
            "generated_ids_len": turn.generated_ids.shape[0] if turn.generated_ids is not None else 0,
            "has_logprobs": turn.logprobs is not None,
            "logprobs_sample": turn.logprobs[:10].tolist() if turn.logprobs is not None and len(turn.logprobs) > 0 else [],
            "has_ref_logprobs": turn.ref_logprobs is not None,
            "ref_logprobs_sample": turn.ref_logprobs[:10].tolist() if turn.ref_logprobs is not None and len(turn.ref_logprobs) > 0 else [],
            # RL labels
            "reward": turn.reward,
            "advantage": turn.advantage,
            "returns": turn.returns,
            "value": turn.value,
            # Flags
            "done_flag": turn.done_flag,
            "answer": turn.answer,
        }
        debug_data["turns"].append(turn_data)
    
    with open(path, "w") as f:
        json.dump(debug_data, f, indent=2)
    
    print(f"[train] Saved debug trajectory to {path}")
    print(f"[train] Debug trajectory summary:")
    print(f"  - Question: {traj.question[:100]}...")
    print(f"  - {len(traj.turns)} turns")
    for i, turn in enumerate(traj.turns):
        print(f"  - Turn {i}: {len(turn.image_paths)} images, "
              f"prompt_tokens={turn.prompt_token_ids.shape[0] if turn.prompt_token_ids is not None else 0}, "
              f"action_tokens={turn.generated_ids.shape[0] if turn.generated_ids is not None else 0}, "
              f"advantage={turn.advantage:.4f}")


# ─────────────────────────────────────────────────────────────────────
# Helper: compute critic values for a trajectory
# ─────────────────────────────────────────────────────────────────────
def _build_turn_messages(turn) -> List[Dict[str, Any]]:
    """Build multimodal message format from a Turn for critic processing."""
    content = []
    for img_idx, img_path in enumerate(turn.image_paths):
        if img_idx == 0:
            label = "\n**Image 0 (Initial view):**"
        else:
            label = f"\n**Image {img_idx} (After movement {img_idx}):**"
        content.append({"type": "text", "text": label})
        content.append({"type": "image", "image": img_path})
    
    content.append({"type": "text", "text": f"\n\n{turn.prompt_text}"})
    
    # Also include the model's generated response for critic to evaluate
    if turn.generated_text:
        content.append({"type": "text", "text": f"\n\nAssistant: {turn.generated_text}"})
    
    return [{"role": "user", "content": content}]


def _compute_critic_values(traj: Trajectory, critic) -> List[float]:
    """Get end-of-turn critic values for GAE using MULTIMODAL inputs (images + text)."""
    if critic is None or critic.backbone is None:
        return [0.0] * len(traj.turns)
    
    values = []
    for turn in traj.turns:
        if turn.generated_ids is None or len(turn.image_paths) == 0:
            values.append(0.0)
            continue
        
        # Build multimodal messages with images
        messages = _build_turn_messages(turn)
        
        with torch.no_grad():
            # Process images + text through critic's processor
            batch = critic.prepare_inputs([messages])
            
            # Get value at the last token position
            seq_len = batch["input_ids"].shape[1]
            end_idx = torch.tensor([seq_len - 1], device=critic._device)
            
            value = critic.forward_at_indices(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                end_indices=end_idx,
                pixel_values=batch.get("pixel_values"),
                image_grid_thw=batch.get("image_grid_thw"),
            )
            values.append(value.item())
    
    return values


# ─────────────────────────────────────────────────────────────────────
# Helper: compute reference log-probs for bi-level GAE
# ─────────────────────────────────────────────────────────────────────
def _compute_ref_logprobs(trajs: List[Trajectory], ref_model) -> None:
    """Compute and store ref_logprobs for each turn (needed for bi-level GAE)."""
    print("[train] Computing reference log-probs for bi-level GAE...")
    
    for traj in trajs:
        for turn in traj.turns:
            if turn.generated_ids is None:
                continue
            
            action_ids = turn.generated_ids.to(ref_model._device)
            if turn.prompt_token_ids is not None:
                prompt_ids = turn.prompt_token_ids.to(ref_model._device)
                full_ids = torch.cat([prompt_ids, action_ids], dim=0)
                prompt_len = prompt_ids.shape[0]
            else:
                full_ids = action_ids
                prompt_len = 0
            
            full_ids = full_ids.unsqueeze(0)
            
            with torch.no_grad():
                full_logprobs = ref_model.forward_logprobs(
                    input_ids=full_ids,
                    attention_mask=torch.ones_like(full_ids),
                    labels=full_ids,
                )
                # Extract only action token log-probs
                ref_lp = full_logprobs[0, prompt_len:].cpu()
                turn.ref_logprobs = ref_lp


# ─────────────────────────────────────────────────────────────────────
# Helper: PPO update step
# ─────────────────────────────────────────────────────────────────────
def _ppo_update(
    trajs: List[Trajectory],
    actor_hf,
    actor_optim,
    critic,
    critic_optim,
    ref_model,
    ppo_cfg: PPOConfig,
    num_epochs: int = 4,
) -> dict:
    """
    Run PPO update on collected trajectories.
    
    CRITICAL NOTES:
    1. Only train on ACTION tokens, NOT observation tokens (prompt + images).
       Observation tokens dominating loss is a known failure mode (VAGEN paper).
    2. Token IDs from vLLM must align with HF tokenizer. Use
       `validate_tokenizer_alignment()` before training.
    """
    from vagen_vsi_rl.utils.token_utils import create_action_mask
    
    # Aggregate statistics across epochs
    stats = {
        "loss": 0.0, "policy_loss": 0.0, "kl": 0.0, "entropy": 0.0, "clip_frac": 0.0,
        "critic_loss": 0.0, "value_mean": 0.0, "value_std": 0.0,
    }
    n_updates = 0
    n_critic_updates = 0
    all_values = []
    all_returns = []
    
    # Ensure model is in training mode (may have been set to eval during inference)
    actor_hf.model.train()
    
    # Sanity check: verify we have trainable parameters
    trainable = [p for p in actor_hf.parameters() if p.requires_grad]
    if len(trainable) == 0:
        raise RuntimeError("[PPO] No trainable parameters found in actor! Check LoRA configuration.")
    
    for epoch in range(num_epochs):
        for traj in trajs:
            for turn in traj.turns:
                if turn.generated_ids is None:
                    continue  # Skip turns without token data
                
                # ═══════════════════════════════════════════════════════════════
                # CRITICAL: Use FULL sequence (prompt + action) for forward pass
                # but only compute loss on ACTION tokens (masking out prompt).
                # This is how VAGEN works: the model sees the full context but
                # gradients only flow through action token predictions.
                # ═══════════════════════════════════════════════════════════════
                
                action_ids = turn.generated_ids.to(actor_hf._device)  # (action_len,)
                old_logprobs = turn.logprobs.to(actor_hf._device)      # (action_len,)
                
                # Build full sequence: prompt + action
                if turn.prompt_token_ids is not None:
                    prompt_ids = turn.prompt_token_ids.to(actor_hf._device)  # (prompt_len,)
                    full_ids = torch.cat([prompt_ids, action_ids], dim=0)    # (seq_len,)
                    prompt_len = prompt_ids.shape[0]
                else:
                    # Fallback: no prompt stored (shouldn't happen with vLLM)
                    full_ids = action_ids
                    prompt_len = 0
                
                full_ids = full_ids.unsqueeze(0)  # (1, seq_len)
                seq_len = full_ids.shape[1]
                
                # Create action mask: True for action tokens, False for prompt
                action_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=actor_hf._device)
                action_mask[0, prompt_len:] = True  # Only action tokens
                
                advantages = torch.tensor([[turn.advantage]], device=actor_hf._device)
                
                # Recompute current log-probs on FULL sequence
                with torch.enable_grad():
                    full_logprobs = actor_hf.forward_logprobs(
                        input_ids=full_ids,
                        attention_mask=torch.ones_like(full_ids),
                        labels=full_ids,
                    )  # (1, seq_len)
                    
                    # Extract only action token log-probs
                    current_logprobs = full_logprobs[0, prompt_len:].unsqueeze(0)  # (1, action_len)
                
                # Reference log-probs for KL
                ref_logprobs = None
                if ref_model is not None:
                    with torch.no_grad():
                        ref_full_logprobs = ref_model.forward_logprobs(
                            input_ids=full_ids,
                            attention_mask=torch.ones_like(full_ids),
                            labels=full_ids,
                        )
                        ref_logprobs = ref_full_logprobs[0, prompt_len:].unsqueeze(0)
                
                # Expand old_logprobs to match current shape
                old_logprobs = old_logprobs.unsqueeze(0)  # (1, action_len)
                
                # Action mask for action tokens only
                action_only_mask = torch.ones(1, old_logprobs.shape[1], dtype=torch.bool, device=actor_hf._device)
                
                # PPO step
                step_stats = ppo_step(
                    logprobs=current_logprobs,
                    old_logprobs=old_logprobs,
                    advantages=advantages,
                    action_masks=action_only_mask,
                    ref_logprobs=ref_logprobs,
                    config=ppo_cfg,
                )
                
                # Backward for policy
                actor_optim.zero_grad()
                step_stats["loss"].backward()
                # Only clip trainable parameters (all for full FT, LoRA adapters for LoRA)
                trainable_params = [p for p in actor_hf.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable_params, ppo_cfg.max_grad_norm)
                actor_optim.step()
                
                # ── Critic update ──
                if critic is not None and critic_optim is not None:
                    # Target return for value function (stored by compute_monte_carlo_returns or compute_gae)
                    # Match dtype to critic output (bfloat16 when sharing backbone)
                    target_return = torch.tensor(
                        [[turn.returns]], 
                        device=critic._device, 
                        dtype=critic.torch_dtype
                    )
                    
                    # Forward pass through critic using MULTIMODAL inputs (images + text)
                    # Build messages from turn data
                    messages = _build_turn_messages(turn)
                    batch = critic.prepare_inputs([messages])
                    
                    seq_len = batch["input_ids"].shape[1]
                    end_idx = torch.tensor([seq_len - 1], device=critic._device)
                    
                    pred_value = critic.forward_at_indices(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        end_indices=end_idx,
                        pixel_values=batch.get("pixel_values"),
                        image_grid_thw=batch.get("image_grid_thw"),
                    )  # (1,)
                    
                    # MSE value loss
                    value_loss = torch.nn.functional.mse_loss(pred_value, target_return.squeeze(-1))
                    
                    # Backward for critic
                    critic_optim.zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.value_head.parameters(), ppo_cfg.max_grad_norm)
                    critic_optim.step()
                    
                    stats["critic_loss"] += value_loss.item()
                    all_values.append(pred_value.item())
                    all_returns.append(target_return.item())
                    n_critic_updates += 1
                
                # Accumulate stats
                for k in ["loss", "policy_loss", "kl", "entropy", "clip_frac"]:
                    stats[k] += step_stats[k].item()
                n_updates += 1
    
    # Average stats
    if n_updates > 0:
        for k in ["loss", "policy_loss", "kl", "entropy", "clip_frac"]:
            stats[k] /= n_updates
    
    if n_critic_updates > 0:
        stats["critic_loss"] /= n_critic_updates
        stats["value_mean"] = sum(all_values) / len(all_values)
        stats["value_std"] = (sum((v - stats["value_mean"])**2 for v in all_values) / len(all_values)) ** 0.5
        
        # Explained variance: how well critic predicts returns
        if len(all_returns) > 1:
            ret_mean = sum(all_returns) / len(all_returns)
            ret_var = sum((r - ret_mean)**2 for r in all_returns) / len(all_returns)
            if ret_var > 1e-8:
                residuals = [(v - r)**2 for v, r in zip(all_values, all_returns)]
                unexplained_var = sum(residuals) / len(residuals)
                stats["explained_variance"] = 1.0 - (unexplained_var / ret_var)
            else:
                stats["explained_variance"] = 0.0
        else:
            stats["explained_variance"] = 0.0
            
    return stats


# ─────────────────────────────────────────────────────────────────────
# Helper: save checkpoint
# ─────────────────────────────────────────────────────────────────────
def _save_checkpoint(
    path: Path,
    actor_hf,
    critic,
    update: int,
    best_acc: float = 0.0,
    actor_optim=None,
    actor_scheduler=None,
    critic_optim=None,
    critic_scheduler=None,
) -> None:
    """Save model checkpoint with optimizer and scheduler states."""
    path.mkdir(parents=True, exist_ok=True)
    
    # Save actor model
    if actor_hf is not None and actor_hf.model is not None:
        actor_hf.model.save_pretrained(path / "actor")
        if actor_hf.tokenizer is not None:
            actor_hf.tokenizer.save_pretrained(path / "actor")
    
    # Save critic value head
    if critic is not None and critic.value_head is not None:
        torch.save(critic.value_head.state_dict(), path / "critic_head.pt")
    
    # Save optimizer states
    if actor_optim is not None:
        torch.save(actor_optim.state_dict(), path / "actor_optim.pt")
    if critic_optim is not None:
        torch.save(critic_optim.state_dict(), path / "critic_optim.pt")
    
    # Save scheduler states
    if actor_scheduler is not None:
        torch.save(actor_scheduler.state_dict(), path / "actor_scheduler.pt")
    if critic_scheduler is not None:
        torch.save(critic_scheduler.state_dict(), path / "critic_scheduler.pt")
    
    # Save metadata
    with open(path / "meta.json", "w") as f:
        json.dump({"update": update, "best_acc": best_acc}, f)
    
    print(f"[train] Saved checkpoint to {path}")


# ─────────────────────────────────────────────────────────────────────
# Helper: load checkpoint
# ─────────────────────────────────────────────────────────────────────
def _load_checkpoint(
    ckpt_path: str,
    actor_hf=None,
    actor_optim=None,
    actor_scheduler=None,
    critic=None,
    critic_optim=None,
    critic_scheduler=None,
    actor_vllm=None,
    strict: bool = False,
) -> tuple:
    """
    Load checkpoint and return (start_update, best_acc).
    
    Returns
    -------
    start_update : int
        Next update number to start from.
    best_acc : float
        Best accuracy so far.
    """
    ckpt_dir = Path(ckpt_path)
    if not ckpt_dir.exists():
        if strict:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir}")
        print(f"[train] Warning: Checkpoint directory not found: {ckpt_dir}")
        return 1, 0.0
    
    # Load metadata
    meta_path = ckpt_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        last_update = meta.get("update", 0)
        best_acc = meta.get("best_acc", 0.0)
    else:
        last_update = 0
        best_acc = 0.0
    
    # Load actor model
    actor_path = ckpt_dir / "actor"
    if actor_path.exists() and actor_hf is not None:
        print(f"[train] Loading actor from {actor_path}")
        
        # Check if this is a LoRA checkpoint (has adapter_config.json)
        is_lora_checkpoint = (actor_path / "adapter_config.json").exists()
        
        if is_lora_checkpoint and actor_hf._is_peft_model:
            # Load LoRA adapter into existing PEFT model
            from peft import PeftModel
            base_model = actor_hf.model.get_base_model()
            actor_hf.model = PeftModel.from_pretrained(
                base_model,
                actor_path,
                is_trainable=True,
            )
            print("[train] Loaded LoRA adapter from checkpoint")
        elif is_lora_checkpoint and not actor_hf._is_peft_model:
            print(f"[train] Warning: Checkpoint has LoRA but model not using LoRA. Skipping.")
        else:
            # Load full model weights
            from transformers import Qwen3VLForConditionalGeneration
            actor_hf.model = Qwen3VLForConditionalGeneration.from_pretrained(
                actor_path,
                torch_dtype=actor_hf.torch_dtype,
                device_map=actor_hf._device,
                trust_remote_code=True,
            )
            # Re-apply LoRA if needed
            if actor_hf.use_lora:
                actor_hf._apply_lora()
        
        actor_hf.model.train()
        
        # Sync to vLLM if available
        if actor_vllm is not None:
            try:
                from vagen_vsi_rl.rl.sync import sync_weights
                sync_weights(actor_hf, actor_vllm)
                print("[train] Synced actor weights to vLLM")
            except Exception as e:
                print(f"[train] Warning: Failed to sync to vLLM: {e}")
    
    # Load critic value head
    critic_path = ckpt_dir / "critic_head.pt"
    if critic_path.exists() and critic is not None:
        print(f"[train] Loading critic head from {critic_path}")
        critic.value_head.load_state_dict(torch.load(critic_path, map_location="cpu"))
        critic.value_head.to(critic._device)
    
    # Load optimizer states
    actor_optim_path = ckpt_dir / "actor_optim.pt"
    if actor_optim_path.exists() and actor_optim is not None:
        print("[train] Loading actor optimizer state")
        actor_optim.load_state_dict(torch.load(actor_optim_path, map_location="cpu"))
    
    critic_optim_path = ckpt_dir / "critic_optim.pt"
    if critic_optim_path.exists() and critic_optim is not None:
        print("[train] Loading critic optimizer state")
        critic_optim.load_state_dict(torch.load(critic_optim_path, map_location="cpu"))
    
    # Load scheduler states
    actor_sched_path = ckpt_dir / "actor_scheduler.pt"
    if actor_sched_path.exists() and actor_scheduler is not None:
        print("[train] Loading actor scheduler state")
        actor_scheduler.load_state_dict(torch.load(actor_sched_path, map_location="cpu"))
    
    critic_sched_path = ckpt_dir / "critic_scheduler.pt"
    if critic_sched_path.exists() and critic_scheduler is not None:
        print("[train] Loading critic scheduler state")
        critic_scheduler.load_state_dict(torch.load(critic_sched_path, map_location="cpu"))
    
    return last_update + 1, best_acc


# ─────────────────────────────────────────────────────────────────────
# Helper: create LR scheduler
# ─────────────────────────────────────────────────────────────────────
def _create_scheduler(
    optimizer,
    scheduler_type: str,
    num_updates: int,
    warmup_updates: int = 10,
    min_lr: float = 1e-7,
):
    """
    Create an LR scheduler.
    
    Supported types:
    - "none": No scheduler (constant LR)
    - "cosine": Cosine annealing to min_lr
    - "linear": Linear decay to min_lr
    - "warmup_cosine": Linear warmup + cosine decay
    """
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR,
        LinearLR,
        SequentialLR,
        LambdaLR,
    )
    
    if scheduler_type == "none":
        # Constant LR (no-op scheduler)
        return LambdaLR(optimizer, lambda _: 1.0)
    
    elif scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=num_updates,
            eta_min=min_lr,
        )
    
    elif scheduler_type == "linear":
        # Linear decay from initial LR to min_lr
        return LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=min_lr / optimizer.defaults["lr"],
            total_iters=num_updates,
        )
    
    elif scheduler_type == "warmup_cosine":
        # Linear warmup + cosine decay
        warmup = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_updates,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=num_updates - warmup_updates,
            eta_min=min_lr,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_updates],
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


if __name__ == "__main__":
    main()
