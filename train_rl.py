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

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_environment import NavigationEnvironment, EpisodeSimulator, EpisodeBatchCollector
from rl_trainer import TrainerConfig, RLTrainer, EpisodeDataLoader

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
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
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
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size (N >= 8 recommended for LOO)")
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
    
    print(f"Model loaded successfully!")
    print(f"  Policy model parameters: {sum(p.numel() for p in policy_model.parameters()):,}")
    print(f"  (Reference model will be created by trainer)")
    print("=" * 80)
    
    return policy_model, processor


def load_vsi_bench_questions_for_training(train_split=0.8):
    """
    Load VSI-Bench questions filtered by arkitscenes + route_planning.
    Same filtering logic as render_point_cloud_qwen_angle.py.
    
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
    
    # Convert to list of dicts
    questions = []
    for row in filtered:
        questions.append({
            "scene_id": row["scene_name"],  # This is the video_id (numeric)
            "question": row["question"],
            "choices": row.get("options", ["A", "B", "C", "D"]),
            "ground_truth": row.get("ground_truth", "A"),
        })
    
    # Split into train/val
    split_idx = int(len(questions) * train_split)
    train_questions = questions[:split_idx]
    val_questions = questions[split_idx:]
    
    print(f"[INFO] Split: {len(train_questions)} train, {len(val_questions)} val")
    
    return train_questions, val_questions


def collect_episodes(args, policy_model, processor):
    """Collect episodes using current policy from VSI-Bench filtered questions."""
    print("\n" + "=" * 80)
    print("COLLECTING EPISODES")
    print("=" * 80)
    
    # Load VSI-Bench questions with same filtering as original pipeline
    train_questions, val_questions = load_vsi_bench_questions_for_training(args.train_split)
    
    # Use training split for episode collection
    num_scenes_needed = min(len(train_questions), args.num_episodes // args.episodes_per_scene)
    scenes = train_questions[:num_scenes_needed]
    
    print(f"Using {len(scenes)} scenes from VSI-Bench (train split)")
    
    # Create episode simulator
    simulator = EpisodeSimulator(
        model=policy_model,
        processor=processor,
        device=args.device,
        max_steps=10,
        track_action_tokens=True,
        do_sample=True,
        temperature=1.0,
        top_p=0.9,
        top_k=50
    )
    
    # Create batch collector
    collector = EpisodeBatchCollector(
        simulator=simulator,
        output_dir=Path(args.episode_storage_dir)
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
                output_dir=Path(args.episode_storage_dir) / f"scene_{scene_id}_ep_{ep_idx}"
            )
            
            # Collect single episode
            try:
                episode = collector.collect_episode(
                    env=env,
                    initial_pose=initial_pose,
                    episode_id=f"train_scene_{scene_id}_ep_{ep_idx}",
                    verbose=False
                )
                all_episodes.append(episode)
                print(f"  Episode {ep_idx+1}/{args.episodes_per_scene}: {'✓ Valid' if episode.is_valid else '✗ Invalid'}")
            except Exception as e:
                print(f"  Episode {ep_idx+1}/{args.episodes_per_scene}: ✗ Error - {e}")
    
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
    
    return all_episodes
    
    return episodes


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
    """Train policy model with RL."""
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
        output_dir=args.output_dir,
        device=args.device,
        # Reference model config
        ref_model_strategy="ema",          # Use EMA (recommended)
        ref_ema_tau=0.999,                 # τ = 0.999 for stable reference
        # KL scheduling
        use_kl_scheduler=True,             # Enable adaptive β
        target_kl=0.01,                    # Target KL divergence
        kl_tolerance=0.005,                # ±0.005 tolerance
        kl_adaptation_rate=1.5             # Adapt β by 1.5x
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


def main():
    args = parse_args()
    
    print("\n")
    print("=" * 80)
    print("VLM NAVIGATION - POLICY GRADIENT RL TRAINING")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_id}")
    print(f"  Dataset: {args.dataset_path}")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  KL coefficient: {args.kl_coef}")
    print(f"  Entropy coefficient: {args.entropy_coef}")
    print(f"  Output: {args.output_dir}")
    print("=" * 80)
    
    # Load models
    policy_model, processor = load_models(args)
    
    # Collect episodes
    episodes = collect_episodes(args, policy_model, processor)
    
    # Create dataloader
    dataloader = create_dataloader(episodes, processor, args)
    
    # Train
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
