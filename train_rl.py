#!/usr/bin/env python3
"""
End-to-end training script for VLM navigation with policy gradient RL.

This script demonstrates the complete pipeline:
1. Load pretrained model and create reference copy
2. Collect episodes using current policy
3. Train with REINFORCE + LOO baseline + KL penalty + entropy
4. Evaluate and save checkpoints
"""

import sys
import torch
import numpy as np
from pathlib import Path
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
import argparse
import open3d as o3d
from datasets import load_dataset
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_environment import NavigationEnvironment, EpisodeSimulator, EpisodeBatchCollector
from rl_trainer import TrainerConfig, RLTrainer, EpisodeDataLoader
from config import SimulatorConfig

# Import rendering utilities
from render_point_cloud_qwen_angle import (
    find_mesh_file,
    look_at_camera_pose_center_from_forward,
    sky_direction_to_up_vector,
    CAM_HEIGHT,
    get_sky_direction_for_scene
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train VLM with policy gradient RL")
    
    # Model
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-2B-Instruct",
                        help="Pretrained model ID")
    parser.add_argument("--cache_dir", type=str,
                        default="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir",
                        help="Cache directory for models")
    
    # Data
    parser.add_argument("--dataset_path", type=str,
                        default="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/raw",
                        help="Path to mesh files (raw directory)")
    parser.add_argument("--train_split", type=float, default=0.8,
                        help="Fraction of data to use for training (default: 0.8 for 80/20 split)")
    parser.add_argument("--num_episodes", type=int, default=2,
                        help="Number of episodes to collect for training")
    parser.add_argument("--episodes_per_scene", type=int, default=1,
                        help="Episodes to collect per scene")
    
    # Training
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Micro batch size (start small with 2-4)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps (effective_batch_size = batch_size * grad_accum)")
    parser.add_argument("--online_rl", action="store_true",
                        help="Enable online RL: collect new episodes after each model update")
    parser.add_argument("--num_updates", type=int, default=10,
                        help="Number of policy updates for online RL (replaces num_epochs)")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate (1e-5 to 5e-6 recommended)")
    parser.add_argument("--kl_coef", type=float, default=0.01,
                        help="KL penalty coefficient (β)")
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help="Entropy bonus coefficient (α)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Gradient clipping threshold")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./rl_checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--episode_storage_dir", type=str, default="./rl_episodes",
                        help="Directory to store collected episodes")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    
    return parser.parse_args()


def load_models(args):
    """Load policy model. Reference model will be created by trainer."""
    print("=" * 80)
    print("LOADING MODEL")
    print("=" * 80)
    
    # Load processor
    print(f"Loading processor: {args.model_id}")
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        trust_remote_code=True
    )
    
    # Load policy model (will be trained)
    print(f"Loading policy model: {args.model_id}")
    policy_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=args.device
    )
    
    # Enable gradient checkpointing to save memory
    policy_model.gradient_checkpointing_enable()
    print("[Memory] Gradient checkpointing enabled")
    
    print(f"Model loaded successfully!")
    print(f"  Policy model parameters: {sum(p.numel() for p in policy_model.parameters()):,}")
    print(f"  (Reference model will be created by trainer)")
    print("=" * 80)
    
    return policy_model, processor


def preprocess_question(question: str) -> str:
    """Preprocess question text to match evaluation format."""
    # Replace "You are a robot beginning at" with "If you begin navigating at"
    question = question.replace("You are a robot beginning at", "If you begin navigating at")
    return question


def load_vsi_bench_questions_for_training(train_split=0.8, random_seed=42):
    """
    Load VSI-Bench questions filtered by arkitscenes + route_planning.
    Same filtering logic as render_point_cloud_qwen_angle.py.
    
    Args:
        train_split: Fraction for training (default 0.8 = 80%)
        random_seed: Random seed for reproducible splits (default 42)
    
    Returns:
        train_questions: List of question dicts for training (80%)
        val_questions: List of question dicts for validation (20%)
    """
    print("[INFO] 📥 Loading VSI-Bench dataset...")
    vsi = load_dataset("nyu-visionx/VSI-Bench", split="test")
    print(f"[INFO] Total VSI-Bench rows: {len(vsi)}")
    
    # Apply same filtering as render_point_cloud_qwen_angle.py
    filtered = vsi.filter(
        lambda x: x["dataset"] == "arkitscenes"
                  and x["question_type"] == "route_planning"
    )
    print(f"[INFO] ✅ Filtered to {len(filtered)} route_planning questions")
    
    # Convert to list of dicts with preprocessed questions
    questions = []
    for row in filtered:
        questions.append({
            "scene_id": row["scene_name"],  # This is the video_id (numeric)
            "question": preprocess_question(row["question"]),  # Preprocess question text
            "choices": row.get("options", ["A", "B", "C", "D"]),
            "ground_truth": row.get("ground_truth", "A"),
        })
    
    # Shuffle with fixed seed for reproducibility
    import random
    rng = random.Random(random_seed)
    rng.shuffle(questions)
    print(f"[INFO] Shuffled questions with seed={random_seed}")
    
    # Split into train/val
    split_idx = int(len(questions) * train_split)
    train_questions = questions[:split_idx]
    val_questions = questions[split_idx:]
    
    print(f"[INFO] Split: {len(train_questions)} train, {len(val_questions)} val")
    
    return train_questions, val_questions


def collect_episodes(args, policy_model, processor, update_idx=None):
    """Collect episodes using current policy from VSI-Bench filtered questions.
    
    Args:
        args: Training arguments
        policy_model: Current policy model
        processor: Model processor
        update_idx: Current update index for online RL (None for offline)
    """
    
    # Disable gradient checkpointing during inference
    if hasattr(policy_model, 'gradient_checkpointing_disable'):
        policy_model.gradient_checkpointing_disable()
    policy_model.eval()  # Set to eval mode
    
    # Create timestamped run directory if not exists
    if not hasattr(collect_episodes, '_run_dir'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        collect_episodes._run_dir = Path(args.episode_storage_dir) / timestamp
        collect_episodes._run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Episodes] Created run directory: {collect_episodes._run_dir}")
    
    # Create update-specific subdirectory
    if update_idx is not None:
        episode_output_dir = collect_episodes._run_dir / f"update_{update_idx + 1}"
    else:
        episode_output_dir = collect_episodes._run_dir / "offline_training"
    episode_output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    if update_idx is not None:
        print(f"COLLECTING EPISODES - Update {update_idx + 1}")
    else:
        print("COLLECTING EPISODES")
    print(f"Output dir: {episode_output_dir}")
    print("=" * 80)
    
    # Load VSI-Bench questions with same filtering as original pipeline
    # Cache this to avoid reloading every time
    if not hasattr(collect_episodes, '_train_questions'):
        collect_episodes._train_questions, collect_episodes._val_questions = load_vsi_bench_questions_for_training(args.train_split)
        print(f"[INFO] Loaded {len(collect_episodes._train_questions)} train, {len(collect_episodes._val_questions)} val questions")
    
    train_questions = collect_episodes._train_questions
    
    # Use training split for episode collection
    num_scenes_needed = min(len(train_questions), args.num_episodes // args.episodes_per_scene)
    scenes = train_questions[:num_scenes_needed]
    
    print(f"Using {len(scenes)} scenes from VSI-Bench (train split)")
    
    # Create simulator config
    simulator_config = SimulatorConfig(
        max_steps=2,
        track_action_tokens=True,
        do_sample=True,
        temperature=1.0,
        top_p=0.9,
        top_k=50,
        min_action_tokens=10,
        max_action_tokens=100
    )
    
    # Create episode simulator with current policy
    simulator = EpisodeSimulator(
        model=policy_model,
        processor=processor,
        config=simulator_config,
        device=args.device
    )
    
    # Create batch collector
    collector = EpisodeBatchCollector(
        simulator=simulator,
        output_dir=episode_output_dir
    )
    
    # Collect episodes
    all_episodes = []
    for scene_idx, scene_data in enumerate(scenes):
        scene_id = str(scene_data["scene_id"])
        question = scene_data["question"]
        choices = scene_data["choices"]
        ground_truth = str(scene_data["ground_truth"])
        
        print(f"\n--- Scene {scene_idx+1}/{len(scenes)}: {scene_id} ---")
        
        # Find mesh file using the same method as render_point_cloud_qwen_angle.py
        mesh_path = find_mesh_file(scene_id)
        if mesh_path is None:
            print(f"[✗] Mesh not found for scene {scene_id}, skipping")
            continue
        
        # Load mesh
        try:
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            if mesh.is_empty():
                print(f"[✗] Empty mesh for scene {scene_id}")
                continue
        except Exception as e:
            print(f"[✗] Failed to load mesh: {e}")
            continue
        
        # Compute initial camera pose using the same method as render_point_cloud_qwen_angle.py
        vertices = np.asarray(mesh.vertices)
        if len(vertices) == 0:
            print(f"[✗] No vertices in mesh for scene {scene_id}")
            continue
        
        # Get sky direction from metadata and convert to up vector
        sky_direction = get_sky_direction_for_scene(scene_id)
        up_vector = sky_direction_to_up_vector(sky_direction)
        
        # Place camera at center XY, middle of Z-range
        center_x = (vertices[:, 0].min() + vertices[:, 0].max()) / 2.0
        center_y = (vertices[:, 1].min() + vertices[:, 1].max()) / 2.0
        z_min = vertices[:, 2].min()
        cam_height_z = z_min + CAM_HEIGHT
        eye = np.array([center_x, center_y, cam_height_z], dtype=float)
        
        # Look forward along X-axis
        forward = np.array([1.0, 0.0, 0.0], dtype=float)
        initial_pose = look_at_camera_pose_center_from_forward(
            eye, forward=forward, up=up_vector
        )
        
        # Collect episodes for this scene
        for ep_idx in range(args.episodes_per_scene):
            # Create environment
            env = NavigationEnvironment(
                mesh=mesh,
                mesh_path=mesh_path,
                scene_id=scene_id,
                question=question,
                choices=choices,
                ground_truth=ground_truth,
                max_steps=10,
                output_dir=episode_output_dir / f"scene_{scene_id}_ep_{ep_idx}"
            )
            
            # Collect single episode
            try:
                episode = collector.collect_episode(
                    env=env,
                    initial_pose=initial_pose,
                    episode_id=f"train_scene_{scene_id}_ep_{ep_idx}",
                    verbose=True  # Enable turn-by-turn output
                )
                all_episodes.append(episode)
                print(f"  Episode {ep_idx+1}/{args.episodes_per_scene}: {'✓ Valid' if episode.is_valid else '✗ Invalid'}")
            except Exception as e:
                print(f"  Episode {ep_idx+1}/{args.episodes_per_scene}: ✗ Error - {e}")
                import traceback
                if args.device == "cuda":
                    print("  (Detailed traceback suppressed, run on CPU for debugging)")
                else:
                    traceback.print_exc()
    
    print(f"\nCollected {len(all_episodes)} episodes total")
    
    # Show statistics (with safety check)
    if len(all_episodes) == 0:
        print("[ERROR] No episodes collected! Check VSI-Bench data and mesh files.")
        raise RuntimeError("Failed to collect any episodes. Ensure VSI-Bench dataset is accessible and mesh files exist.")
    
    valid_episodes = [ep for ep in all_episodes if ep.is_valid]
    success_rate = sum(ep.final_reward for ep in valid_episodes) / len(valid_episodes) if valid_episodes else 0.0
    
    print(f"  Valid episodes: {len(valid_episodes)}/{len(all_episodes)}")
    print(f"  Success rate: {success_rate:.1%}")
    print(f"  Average turns: {sum(len(ep.turns) for ep in all_episodes) / len(all_episodes):.1f}")
    print("=" * 80)
    
    # Re-enable gradient checkpointing for training
    if hasattr(policy_model, 'gradient_checkpointing_enable'):
        policy_model.gradient_checkpointing_enable()
    policy_model.train()  # Set back to train mode
    
    return all_episodes


def validate(args, policy_model, processor, update_idx=None, max_val_episodes=None):
    """
    Run validation on val split with deterministic generation (do_sample=False).
    
    Args:
        args: Training arguments
        policy_model: Current policy model
        processor: Model processor
        update_idx: Current update index (for logging)
        max_val_episodes: Maximum validation episodes to run (None = all)
        
    Returns:
        Dict with validation metrics
    """
    # Disable gradient checkpointing during inference
    if hasattr(policy_model, 'gradient_checkpointing_disable'):
        policy_model.gradient_checkpointing_disable()
    policy_model.eval()
    
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)
    
    # Get val questions (cached from collect_episodes)
    if not hasattr(collect_episodes, '_val_questions'):
        _, collect_episodes._val_questions = load_vsi_bench_questions_for_training(args.train_split)
    
    val_questions = collect_episodes._val_questions
    
    # Limit validation episodes if specified
    if max_val_episodes is not None:
        val_questions = val_questions[:max_val_episodes]
    
    print(f"Running validation on {len(val_questions)} questions (deterministic generation)")
    
    # Create validation output directory
    val_output_dir = Path(args.episode_storage_dir) / "validation"
    if update_idx is not None:
        val_output_dir = val_output_dir / f"update_{update_idx + 1}"
    else:
        val_output_dir = val_output_dir / "final"
    val_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create simulator with deterministic generation
    simulator_config = SimulatorConfig(
        max_steps=2,
        track_action_tokens=True,
        do_sample=False,  # Deterministic for evaluation
        temperature=1.0,
        top_p=1.0,
        top_k=50,
        min_action_tokens=10,
        max_action_tokens=100
    )
    
    simulator = EpisodeSimulator(
        model=policy_model,
        processor=processor,
        config=simulator_config,
        device=args.device
    )
    
    collector = EpisodeBatchCollector(
        simulator=simulator,
        output_dir=val_output_dir
    )
    
    # Collect validation episodes
    correct = 0
    total = 0
    
    for val_idx, scene_data in enumerate(val_questions):
        scene_id = str(scene_data["scene_id"])
        question = scene_data["question"]
        choices = scene_data["choices"]
        ground_truth = str(scene_data["ground_truth"])
        
        # Find mesh
        mesh_path = find_mesh_file(scene_id)
        if mesh_path is None:
            continue
        
        try:
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            if mesh.is_empty():
                continue
        except Exception:
            continue
        
        # Compute initial pose
        vertices = np.asarray(mesh.vertices)
        if len(vertices) == 0:
            continue
        
        sky_direction = get_sky_direction_for_scene(scene_id)
        up_vector = sky_direction_to_up_vector(sky_direction)
        
        center_x = (vertices[:, 0].min() + vertices[:, 0].max()) / 2.0
        center_y = (vertices[:, 1].min() + vertices[:, 1].max()) / 2.0
        z_min = vertices[:, 2].min()
        cam_height_z = z_min + CAM_HEIGHT
        eye = np.array([center_x, center_y, cam_height_z], dtype=float)
        
        forward = np.array([1.0, 0.0, 0.0], dtype=float)
        initial_pose = look_at_camera_pose_center_from_forward(
            eye, forward=forward, up=up_vector
        )
        
        # Create environment
        env = NavigationEnvironment(
            mesh=mesh,
            mesh_path=mesh_path,
            scene_id=scene_id,
            question=question,
            choices=choices,
            ground_truth=ground_truth,
            max_steps=10,
            output_dir=val_output_dir / f"val_{val_idx}_scene_{scene_id}"
        )
        
        # Collect episode
        try:
            episode = collector.collect_episode(
                env=env,
                initial_pose=initial_pose,
                episode_id=f"val_scene_{scene_id}",
                verbose=False  # Less verbose for validation
            )
            
            # Check correctness
            if episode.final_answer is not None:
                total += 1
                if episode.final_answer == ground_truth:
                    correct += 1
                    print(f"  [{val_idx+1}/{len(val_questions)}] Scene {scene_id}: ✓ Correct ({episode.final_answer})")
                else:
                    print(f"  [{val_idx+1}/{len(val_questions)}] Scene {scene_id}: ✗ Wrong ({episode.final_answer} vs {ground_truth})")
            else:
                print(f"  [{val_idx+1}/{len(val_questions)}] Scene {scene_id}: No answer")
                
        except Exception as e:
            print(f"  [{val_idx+1}/{len(val_questions)}] Scene {scene_id}: Error - {e}")
    
    # Compute accuracy
    accuracy = correct / total if total > 0 else 0.0
    
    print(f"\nValidation Results:")
    print(f"  Accuracy: {correct}/{total} = {accuracy:.1%}")
    print("=" * 80)
    
    # Re-enable gradient checkpointing for training
    if hasattr(policy_model, 'gradient_checkpointing_enable'):
        policy_model.gradient_checkpointing_enable()
    policy_model.train()
    
    return {
        "val/accuracy": accuracy,
        "val/correct": correct,
        "val/total": total
    }


def create_dataloader(episodes, processor, args):
    """Create dataloader from episodes."""
    print("\n" + "=" * 80)
    print("CREATING DATALOADER")
    print("=" * 80)
    
    dataloader = EpisodeDataLoader(
        episodes=episodes,
        batch_size=args.batch_size,
        shuffle=True,
        filter_invalid=True,
        processor=processor,
        device=args.device
    )
    
    dataloader.log_statistics()
    
    return dataloader


def train(args, policy_model, dataloader):
    """Train policy model with RL (offline mode)."""
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    
    # Create trainer config
    config = TrainerConfig(
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        kl_coef=args.kl_coef,
        entropy_coef=args.entropy_coef,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        output_dir=args.output_dir,
        device=args.device,
        # Mixed precision training with bfloat16
        # Autocast enabled but no GradScaler (bfloat16 doesn't need it)
        use_amp=True,
        amp_dtype=torch.bfloat16,
        # Reference model config - DISABLED to save memory
        # With 40GB VRAM, 2 models + optimizer + activations is too much
        ref_model_strategy=None,           # Disable reference model to save ~4GB
        # If you need KL penalty, re-enable after fixing other memory issues:
        # ref_model_strategy="ema", ref_ema_tau=0.999
        # KL scheduling (disabled since no reference model)
        use_kl_scheduler=False,
        target_kl=0.01,
        kl_tolerance=0.005,
        kl_adaptation_rate=1.5
    )
    
    # Create trainer (it will create reference model internally)
    trainer = RLTrainer(
        model=policy_model,
        dataloader=dataloader,
        config=config
    )
    
    # Train
    trainer.train()
    
    return trainer


def train_online(args, policy_model, processor):
    """
    Train policy model with online RL.
    
    Collects new episodes with current policy after each model update.
    """
    print("\n" + "=" * 80)
    print("ONLINE RL TRAINING")
    print("=" * 80)
    
    # Create trainer config (single update per batch)
    config = TrainerConfig(
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        kl_coef=args.kl_coef,
        entropy_coef=args.entropy_coef,
        num_epochs=1,  # Single pass through each batch
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        output_dir=args.output_dir,
        device=args.device,
        use_amp=True,
        amp_dtype=torch.bfloat16,
        ref_model_strategy=None,
        use_kl_scheduler=False,
        target_kl=0.01,
        kl_tolerance=0.005,
        kl_adaptation_rate=1.5
    )
    
    # Create trainer
    trainer = RLTrainer(
        model=policy_model,
        dataloader=None,  # Will be updated each iteration
        config=config
    )
    
    print(f"Will perform {args.num_updates} policy updates")
    print(f"Each update: collect {args.num_episodes} episodes → train on {args.batch_size} batch")
    print("=" * 80)
    
    # Online training loop
    for update_idx in range(args.num_updates):
        print(f"\n{'='*80}")
        print(f"POLICY UPDATE {update_idx + 1}/{args.num_updates}")
        print(f"{'='*80}")
        
        # Step 1: Collect episodes with current policy
        print(f"[Update {update_idx + 1}] Collecting episodes with current policy...")
        episodes = collect_episodes(args, policy_model, processor, update_idx=update_idx)
        
        # Step 2: Create fresh dataloader
        dataloader = create_dataloader(episodes, processor, args)
        trainer.dataloader = dataloader
        
        # Step 3: Train on collected episodes (single epoch)
        print(f"[Update {update_idx + 1}] Training on collected episodes...")
        trainer.epoch = update_idx  # Track which update we're on
        epoch_metrics = trainer.train_epoch()
        
        # Step 4: Log update statistics
        epoch_stats = trainer._compute_epoch_stats(epoch_metrics)
        print(f"\n[Update {update_idx + 1}] completed:")
        print(f"  Loss: {epoch_stats.get('loss/total', 0):.4f}")
        print(f"  Mean reward: {epoch_stats.get('rewards/mean', 0):.3f}")
        print(f"  Mean advantage: {epoch_stats.get('advantages/mean', 0):.3f}")
        if 'entropy/mean' in epoch_stats:
            print(f"  Mean entropy: {epoch_stats['entropy/mean']:.4f}")
        
        # Step 5: Run validation periodically
        if (update_idx + 1) % 5 == 0 or (update_idx + 1) == args.num_updates:
            print(f"\n[Validation] Running validation at update {update_idx + 1}")
            val_metrics = validate(args, policy_model, processor, update_idx=update_idx, max_val_episodes=10)
            
            # Log validation metrics to wandb
            if trainer.logger:
                trainer.logger.log(val_metrics, step=trainer.global_step)
            
            print(f"  Validation accuracy: {val_metrics['val/accuracy']:.1%}")
        
        # Step 6: Save checkpoint periodically
        if (update_idx + 1) % 5 == 0:
            print(f"[Checkpoint] Saving at update {update_idx + 1}")
            trainer._save_training_checkpoint(is_epoch_end=True)
    
    # Final save
    print("\n" + "=" * 80)
    print("ONLINE TRAINING COMPLETED")
    print("=" * 80)
    trainer._save_training_checkpoint(is_final=True)
    
    return trainer


def main():
    args = parse_args()
    
    print("\n")
    print("=" * 80)
    print("VLM NAVIGATION - POLICY GRADIENT RL TRAINING")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_id}")
    print(f"  Dataset: {args.dataset_path}")
    print(f"  Mode: {'ONLINE RL' if args.online_rl else 'OFFLINE RL'}")
    if args.online_rl:
        print(f"  Policy updates: {args.num_updates}")
        print(f"  Episodes per update: {args.num_episodes}")
    else:
        print(f"  Episodes: {args.num_episodes}")
        print(f"  Epochs: {args.num_epochs}")
    print(f"  Micro batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  KL coefficient: {args.kl_coef}")
    print(f"  Entropy coefficient: {args.entropy_coef}")
    print(f"  Output: {args.output_dir}")
    print("=" * 80)
    
    # Load models
    policy_model, processor = load_models(args)
    
    if args.online_rl:
        # Online RL: integrate episode collection into training loop
        print("\n🔄 ONLINE RL MODE: Episodes will be re-collected after each policy update")
        train_online(args, policy_model, processor)
    else:
        # Offline RL: collect once, train multiple epochs
        print("\n📦 OFFLINE RL MODE: Training on fixed episode dataset")
        episodes = collect_episodes(args, policy_model, processor)
        dataloader = create_dataloader(episodes, processor, args)
        trainer = train(args, policy_model, dataloader)
    
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETED!")
    print("=" * 80)
    print(f"\nCheckpoints saved to: {args.output_dir}")
    print(f"Episodes saved to: {args.episode_storage_dir}")
    print("\nTo use the trained model:")
    print(f"  model = Qwen3VLForConditionalGeneration.from_pretrained('{args.output_dir}/final_model')")


if __name__ == "__main__":
    main()
