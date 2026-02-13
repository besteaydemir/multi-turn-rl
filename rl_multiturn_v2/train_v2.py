#!/usr/bin/env python3
"""
Main training script for multi-turn RL view selection.

This implements the complete training pipeline:
1. Load pretrained VLM
2. Initialize rollout engine (vLLM) and trainer
3. Load scenes (ScanNet, ScanNet++, ARKitScenes)
4. Collect trajectories with rendering
5. Train with policy gradient
6. Sync weights between trainer and rollout engine
7. Repeat

Usage:
    python train_v2.py --config configs/train_rl_test.yaml
    python train_v2.py --model_id Qwen/Qwen3-VL-4B-Instruct --num_updates 100
    python train_v2.py --mock  # For testing without GPU
"""

import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import torch
import numpy as np

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_multiturn_v2.load_config import load_yaml_config, merge_args_with_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train multi-turn RL for view selection"
    )
    
    # =========================================================================
    # CONFIG FILE
    # =========================================================================
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (CLI args override config)",
    )
    
    # =========================================================================
    # MODEL
    # =========================================================================
    
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Pretrained model ID",
    )
    
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir",
        help="Cache directory for models",
    )
    
    # =========================================================================
    # EPISODE SETTINGS
    # =========================================================================
    
    parser.add_argument(
        "--max_turns",
        type=int,
        default=5,
        help="Fixed number of views T per episode",
    )
    
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens per turn generation",
    )
    
    # =========================================================================
    # TRAINING
    # =========================================================================
    
    parser.add_argument(
        "--num_updates",
        type=int,
        default=100,
        help="Number of policy updates",
    )
    
    parser.add_argument(
        "--episodes_per_update",
        type=int,
        default=4,
        help="Trajectories to collect per update",
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for AdamW",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training",
    )
    
    # =========================================================================
    # ALGORITHM
    # =========================================================================
    
    parser.add_argument(
        "--use_ppo",
        action="store_true",
        help="Use PPO instead of REINFORCE",
    )
    
    parser.add_argument(
        "--kl_coef",
        type=float,
        default=0.0,
        help="KL penalty coefficient (0 = disabled)",
    )
    
    parser.add_argument(
        "--entropy_coef",
        type=float,
        default=0.0,
        help="Entropy bonus coefficient (0 = disabled)",
    )
    
    # =========================================================================
    # DATASET & SCENE LOADING
    # =========================================================================
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="combined",
        choices=["arkitscenes", "scannet", "scannetpp", "combined"],
        help="Dataset to use for training",
    )
    
    parser.add_argument(
        "--use_scene_loader",
        action="store_true",
        help="Enable scene loading and rendering",
    )
    
    parser.add_argument(
        "--render_output_dir",
        type=str,
        default=None,
        help="Directory for rendered images (uses temp if not set)",
    )
    
    # =========================================================================
    # WEIGHT SYNCHRONIZATION
    # =========================================================================
    
    parser.add_argument(
        "--weight_sync",
        type=str,
        default="checkpoint",
        choices=["checkpoint", "direct", "none"],
        help="Weight sync method (checkpoint=Option A, direct=Option B)",
    )
    
    parser.add_argument(
        "--weight_sync_interval",
        type=int,
        default=1,
        help="Sync weights every N updates",
    )
    
    # =========================================================================
    # CHECKPOINTING
    # =========================================================================
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint",
    )
    
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Specific checkpoint to resume from",
    )
    
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,
        help="Save checkpoint every N updates",
    )
    
    # =========================================================================
    # OUTPUT
    # =========================================================================
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/rl_multiturn_v2",
        help="Output directory for checkpoints and logs",
    )
    
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="rl-view-selection",
        help="W&B project name",
    )
    
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity (team/user)",
    )
    
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    
    # =========================================================================
    # DEVELOPMENT
    # =========================================================================
    
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock rollout engine (no GPU needed)",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (verbose logging)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    # Load config from YAML if provided
    if args.config:
        print(f"Loading config from: {args.config}")
        config = load_yaml_config(args.config)
        args = merge_args_with_config(args, config)
    
    # Set defaults for any missing args
    if args.model_id is None:
        args.model_id = "Qwen/Qwen3-VL-4B-Instruct"
    if not hasattr(args, 'dataset') or args.dataset is None:
        args.dataset = "combined"
    if not hasattr(args, 'weight_sync') or args.weight_sync is None:
        args.weight_sync = "checkpoint"
    if not hasattr(args, 'weight_sync_interval') or args.weight_sync_interval is None:
        args.weight_sync_interval = 1
    if not hasattr(args, 'use_scene_loader'):
        args.use_scene_loader = False
    if not hasattr(args, 'render_output_dir'):
        args.render_output_dir = None
    if not hasattr(args, 'resume'):
        args.resume = False
    if not hasattr(args, 'checkpoint_path'):
        args.checkpoint_path = None
    if not hasattr(args, 'save_interval'):
        args.save_interval = 100
    
    return args


def load_sample_questions() -> List[Dict[str, Any]]:
    """
    Load sample questions for testing (fallback if VSI-Bench not available).
    """
    # Sample questions for testing
    return [
        {
            "question": "Which object is closer to the camera?",
            "choices": ["The chair", "The table", "The lamp", "The sofa"],
            "scene_id": "sample_scene_001",
            "ground_truth": "A",
            "dataset": "scannetpp",
        },
        {
            "question": "What is the relative position of the door to the window?",
            "choices": ["Left", "Right", "Above", "Below"],
            "scene_id": "sample_scene_002",
            "ground_truth": "B",
            "dataset": "scannetpp",
        },
        {
            "question": "How many chairs are in the room?",
            "choices": ["1", "2", "3", "4"],
            "scene_id": "sample_scene_003",
            "ground_truth": "C",
            "dataset": "scannetpp",
        },
        {
            "question": "Which direction should you walk to reach the kitchen?",
            "choices": ["Forward", "Backward", "Left", "Right"],
            "scene_id": "sample_scene_004",
            "ground_truth": "A",
            "dataset": "scannetpp",
        },
    ] * 10  # Repeat for more training data


def load_training_questions(dataset: str = "combined") -> List[Dict[str, Any]]:
    """
    Load training questions from VSI-Bench.
    
    Args:
        dataset: Dataset to load ("arkitscenes", "scannet", "scannetpp", "combined")
        
    Returns:
        List of question dicts
    """
    try:
        from rl_multiturn_v2.scene_loader import load_vsi_bench_questions
        questions = load_vsi_bench_questions(dataset=dataset)
        if questions:
            return questions
    except Exception as e:
        print(f"[Main] Failed to load VSI-Bench: {e}")
    
    # Fallback to sample questions
    print("[Main] Using sample questions for testing")
    return load_sample_questions()


def main():
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Print configuration
    print("=" * 70)
    print("Multi-Turn RL View Selection - Training")
    print("=" * 70)
    print(f"Model: {args.model_id}")
    print(f"Max turns: {args.max_turns}")
    print(f"Updates: {args.num_updates}")
    print(f"Episodes/update: {args.episodes_per_update}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Dataset: {args.dataset}")
    print(f"Scene loader: {args.use_scene_loader}")
    print(f"Weight sync: {args.weight_sync} (interval: {args.weight_sync_interval})")
    print(f"Mock mode: {args.mock}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Import modules
    from rl_multiturn_v2.rollout import RolloutConfig, VLLMRolloutEngine, MockRolloutEngine
    from rl_multiturn_v2.trainer import TrainerConfig, RLTrainer, OnlineRLTrainer
    from rl_multiturn_v2.logging_utils import Logger, format_update_summary, log_trajectory
    
    # =========================================================================
    # INITIALIZE SCENE LOADER (if enabled)
    # =========================================================================
    
    scene_loader = None
    if args.use_scene_loader and not args.mock:
        print("[Main] Initializing scene loader...")
        from rl_multiturn_v2.scene_loader import SceneLoader
        
        render_dir = Path(args.render_output_dir) if args.render_output_dir else None
        scene_loader = SceneLoader(
            output_dir=render_dir,
            use_temp_dir=(render_dir is None),
        )
        print("[Main] Scene loader initialized")
    
    # =========================================================================
    # INITIALIZE ROLLOUT ENGINE
    # =========================================================================
    
    rollout_config = RolloutConfig(
        max_turns=args.max_turns,
        max_new_tokens=args.max_new_tokens,
        model_id=args.model_id,
        cache_dir=args.cache_dir,
    )
    
    if args.mock:
        print("[Main] Using mock rollout engine (no GPU)")
        rollout_engine = MockRolloutEngine(rollout_config)
    else:
        print("[Main] Initializing vLLM rollout engine...")
        rollout_engine = VLLMRolloutEngine(
            config=rollout_config,
            render_fn=None,  # Will be set per-scene
        )
    
    # =========================================================================
    # INITIALIZE TRAINER
    # =========================================================================
    
    trainer_config = TrainerConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        use_ppo=args.use_ppo,
        kl_coef=args.kl_coef,
        entropy_coef=args.entropy_coef,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=f"v2_{timestamp}",
        output_dir=str(run_dir / "checkpoints"),
        save_interval=args.save_interval,
    )
    
    if args.mock:
        # Mock trainer - use a simple model
        print("[Main] Creating mock trainer...")
        
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Linear(1, 1)
            
            def forward(self, x):
                return x
        
        model = MockModel()
        trainer = RLTrainer(model=model, config=trainer_config)
    else:
        # Real trainer with VLM
        print("[Main] Loading trainable model...")
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=args.cache_dir,
        )
        
        processor = AutoProcessor.from_pretrained(args.model_id, cache_dir=args.cache_dir)
        trainer = RLTrainer(
            model=model,
            config=trainer_config,
            tokenizer=processor.tokenizer,
            original_model_path=args.model_id,  # For copying processor files during weight sync
        )
    
    # =========================================================================
    # RESUME FROM CHECKPOINT
    # =========================================================================
    
    if args.resume:
        if args.checkpoint_path:
            print(f"[Main] Resuming from: {args.checkpoint_path}")
            trainer.load_checkpoint(Path(args.checkpoint_path))
        else:
            trainer.resume_from_latest()
    
    # =========================================================================
    # INITIALIZE LOGGER
    # =========================================================================
    
    logger = Logger(
        output_dir=run_dir / "logs",
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=f"v2_{timestamp}",
        config=vars(args),
    )
    
    # =========================================================================
    # LOAD TRAINING DATA
    # =========================================================================
    
    print("[Main] Loading training questions...")
    questions = load_training_questions(args.dataset)
    print(f"[Main] Loaded {len(questions)} questions")
    
    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("Starting Training Loop")
    print("=" * 70)
    
    online_trainer = OnlineRLTrainer(
        rollout_engine=rollout_engine,
        trainer=trainer,
        config=trainer_config,
        weight_sync_method=args.weight_sync,
        weight_sync_interval=args.weight_sync_interval,
    )
    
    try:
        results = online_trainer.train(
            questions=questions,
            num_updates=args.num_updates,
            episodes_per_update=args.episodes_per_update,
            scene_loader=scene_loader,
        )
        
        print("\n" + "=" * 70)
        print("Training Complete!")
        print("=" * 70)
        print(f"Total updates: {results['num_updates']}")
        print(f"Total trajectories: {results['total_trajectories']}")
        print(f"Final metrics: {results['final_metrics']}")
        
    except KeyboardInterrupt:
        print("\n[Main] Training interrupted by user")
    
    except Exception as e:
        print(f"\n[Main] Training failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Save final checkpoint (disabled for testing)
        # print("[Main] Saving final checkpoint...")
        # trainer.save_checkpoint(run_dir / "checkpoints" / "final")
        
        # Also save HF format for easy loading (disabled for testing)
        # print("[Main] Saving HuggingFace format checkpoint...")
        # trainer.save_hf_checkpoint(run_dir / "checkpoints" / "final_hf")
        
        # Cleanup scene loader
        if scene_loader is not None:
            scene_loader.cleanup()
        
        # Finish logging
        logger.finish()
        
        print(f"[Main] Run directory: {run_dir}")


if __name__ == "__main__":
    main()
