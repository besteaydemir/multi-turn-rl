#!/usr/bin/env python3
"""
Main training script for multi-turn RL view selection.

This implements the complete training pipeline:
1. Load pretrained VLM
2. Initialize rollout engine (vLLM) and trainer
3. Collect trajectories
4. Train with policy gradient
5. Repeat

Usage:
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train multi-turn RL for view selection"
    )
    
    # =========================================================================
    # MODEL
    # =========================================================================
    
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen3-VL-4B-Instruct",
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
    
    return parser.parse_args()


def load_sample_questions() -> List[Dict[str, Any]]:
    """
    Load sample questions for training.
    
    In production, this would load from VSI-Bench dataset.
    """
    # Sample questions for testing
    return [
        {
            "question": "Which object is closer to the camera?",
            "choices": ["The chair", "The table", "The lamp", "The sofa"],
            "scene_id": "sample_scene_001",
            "ground_truth": "A",
        },
        {
            "question": "What is the relative position of the door to the window?",
            "choices": ["Left", "Right", "Above", "Below"],
            "scene_id": "sample_scene_002",
            "ground_truth": "B",
        },
        {
            "question": "How many chairs are in the room?",
            "choices": ["1", "2", "3", "4"],
            "scene_id": "sample_scene_003",
            "ground_truth": "C",
        },
        {
            "question": "Which direction should you walk to reach the kitchen?",
            "choices": ["Forward", "Backward", "Left", "Right"],
            "scene_id": "sample_scene_004",
            "ground_truth": "A",
        },
    ] * 10  # Repeat for more training data


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
            render_fn=None,  # No rendering for now (images from dataset)
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
        )
    
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
    questions = load_sample_questions()
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
    )
    
    try:
        results = online_trainer.train(
            questions=questions,
            num_updates=args.num_updates,
            episodes_per_update=args.episodes_per_update,
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
        # Save final checkpoint
        print("[Main] Saving final checkpoint...")
        trainer.save_checkpoint(run_dir / "checkpoints" / "final")
        
        # Finish logging
        logger.finish()
        
        print(f"[Main] Run directory: {run_dir}")


if __name__ == "__main__":
    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent))
    main()
