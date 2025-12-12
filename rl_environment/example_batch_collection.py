#!/usr/bin/env python3
"""
Example: Batch episode collection with JSONL storage.
Demonstrates Step 2 implementation with sampling and real-time token tracking.
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_environment import (
    NavigationEnvironment,
    EpisodeSimulator,
    EpisodeBatchCollector
)

from render_point_cloud_qwen_angle import (
    CACHE_DIR,
    MODEL_ID,
    get_sky_direction_for_scene,
    sky_direction_to_up_vector,
    find_mesh_file,
    select_best_initial_view,
    look_at_camera_pose_center_from_forward,
    render_mesh_from_pose,
    DEFAULT_FX_FY,
    INITIAL_VIEW_SELECTION_METRIC
)

import open3d as o3d
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image


def example_batch_collection():
    """
    Example: Collect multiple episodes with sampling and save to JSONL.
    """
    print("=" * 80)
    print("EXAMPLE: Batch Episode Collection with JSONL Storage")
    print("=" * 80)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[1] Loading model on {device}...")
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype="auto", device_map="auto", cache_dir=CACHE_DIR
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    model.to(device)
    print("[✓] Model loaded")
    
    # Create simulator with sampling enabled
    print(f"\n[2] Creating simulator with sampling...")
    simulator = EpisodeSimulator(
        model=model,
        processor=processor,
        device=device,
        max_steps=3,  # Short episodes for testing
        track_action_tokens=True,
        do_sample=True,          # Enable sampling
        temperature=0.8,         # Temperature for randomness
        top_p=0.9,              # Nucleus sampling
        top_k=50                # Top-k sampling
    )
    print("[✓] Simulator created with sampling parameters")
    print(f"    - do_sample: True")
    print(f"    - temperature: 0.8")
    print(f"    - top_p: 0.9")
    print(f"    - top_k: 50")
    
    # Create batch collector
    print(f"\n[3] Creating batch collector...")
    output_dir = Path("batch_collection_output")
    collector = EpisodeBatchCollector(
        simulator=simulator,
        output_dir=output_dir,
        save_format="both"  # Save both JSONL and full episode data
    )
    print(f"[✓] Batch collector created")
    print(f"    - Output: {output_dir}")
    print(f"    - JSONL: {collector.jsonl_path}")
    print(f"    - Tensors: {collector.tensors_dir}")
    
    # Test scenarios
    test_scenes = [
        {
            "scene_id": "42444953",
            "question": "If you begin navigating at the wall outlet, which direction should you move to reach the wooden chair?",
            "choices": ["Forward", "Left", "Right", "Backward"],
            "ground_truth": "A"
        },
        {
            "scene_id": "42444953",  # Same scene, different episode (sampling will vary)
            "question": "If you begin navigating at the wall outlet, which direction should you move to reach the wooden chair?",
            "choices": ["Forward", "Left", "Right", "Backward"],
            "ground_truth": "A"
        }
    ]
    
    # Collect episodes
    print(f"\n[4] Collecting {len(test_scenes)} episodes...")
    for i, scenario in enumerate(test_scenes):
        print(f"\n--- Episode {i+1}/{len(test_scenes)} ---")
        
        scene_id = scenario["scene_id"]
        question = scenario["question"]
        choices = scenario["choices"]
        ground_truth = scenario["ground_truth"]
        
        # Load mesh
        mesh_path = find_mesh_file(scene_id)
        if mesh_path is None:
            print(f"[✗] Mesh not found for scene {scene_id}, skipping")
            continue
        
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        
        # Compute initial pose (simplified version)
        vertices = np.asarray(mesh.vertices)
        center_x = (vertices[:, 0].min() + vertices[:, 0].max()) / 2.0
        center_y = (vertices[:, 1].min() + vertices[:, 1].max()) / 2.0
        z_min = vertices[:, 2].min()
        cam_height_z = z_min + 1.6
        eye = np.array([center_x, center_y, cam_height_z], dtype=float)
        
        # Use 0-degree view as initial
        forward = np.array([1.0, 0.0, 0.0], dtype=float)
        initial_pose = look_at_camera_pose_center_from_forward(eye, forward=forward, up=np.array([0.0, 0.0, -1.0]))
        
        # Create environment
        env = NavigationEnvironment(
            mesh=mesh,
            mesh_path=mesh_path,
            scene_id=scene_id,
            question=question,
            choices=choices,
            ground_truth=ground_truth,
            max_steps=3,
            output_dir=output_dir / f"env_episode_{i}"
        )
        
        # Collect episode
        episode = collector.collect_episode(
            env=env,
            initial_pose=initial_pose,
            episode_id=f"episode_{i:03d}",
            verbose=True
        )
        
        print(f"[✓] Episode {i+1} collected:")
        print(f"    - Answer: {episode.final_answer}")
        print(f"    - Correct: {episode.is_correct}")
        print(f"    - Reward: {episode.final_reward}")
        print(f"    - Turns: {len(episode.turns)}")
    
    # Show statistics
    print("\n" + "=" * 80)
    print("BATCH STATISTICS")
    print("=" * 80)
    stats = collector.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Demonstrate loading from JSONL
    print("\n" + "=" * 80)
    print("LOADING FROM JSONL")
    print("=" * 80)
    print(f"Reading episodes from: {collector.jsonl_path}\n")
    
    for i, episode_data in enumerate(collector.iter_episodes_jsonl()):
        print(f"Episode {i}:")
        print(f"  ID: {episode_data['episode_id']}")
        print(f"  Scene: {episode_data['scene_id']}")
        print(f"  Turns: {len(episode_data['turns'])}")
        print(f"  Final reward: {episode_data['final_reward']}")
        print(f"  Correct: {episode_data['is_correct']}")
        
        # Show token tracking info for each turn
        for turn_data in episode_data['turns']:
            print(f"    Turn {turn_data['turn_index']}:")
            print(f"      - Generated tokens: {turn_data['generated_ids_length']}")
            print(f"      - JSON tokens: [{turn_data['action_token_start_index']}, {turn_data['action_token_end_index']})")
            print(f"      - Action valid: {turn_data['action_valid']}")
    
    # Demonstrate loading tensors
    print("\n" + "=" * 80)
    print("LOADING TENSORS")
    print("=" * 80)
    
    # Load tensors for first episode, first turn
    episode_id = "episode_000"
    turn_index = 0
    
    tensor_data = collector.load_episode_tensors(episode_id, turn_index)
    print(f"\nLoaded tensors for {episode_id}, turn {turn_index}:")
    print(f"  generated_ids shape: {tensor_data['generated_ids'].shape}")
    print(f"  action_token_mask shape: {tensor_data['action_token_mask'].shape}")
    print(f"  action_token_mask sum: {tensor_data['action_token_mask'].sum().item()} (number of action tokens)")
    
    print("\n[✓] Batch collection example complete!")
    print(f"[✓] Data saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    example_batch_collection()
