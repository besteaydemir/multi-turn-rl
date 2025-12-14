#!/usr/bin/env python3
"""
Test script for RL environment and simulator.
Demonstrates how to run episodes and collect trajectory data.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_environment import (
    NavigationEnvironment,
    EpisodeSimulator,
    Episode
)

# Import from baseline pipeline
from render_point_cloud_qwen_angle import (
    CACHE_DIR,
    MODEL_ID,
    compute_initial_camera_pose,
    get_sky_direction_for_scene,
    sky_direction_to_up_vector,
    find_mesh_file,
    select_best_initial_view,
    look_at_camera_pose_center_from_forward,
    render_mesh_from_pose,
    IMAGE_WH,
    DEFAULT_FX_FY,
    INITIAL_VIEW_SELECTION_METRIC
)

import open3d as o3d
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image


def test_single_episode():
    """
    Test running a single episode with the environment and simulator.
    """
    print("=" * 80)
    print("TEST: Single Episode Simulation")
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
    
    # Test data
    scene_id = "42444953"  # Example scene
    question = "If you begin navigating at the wall outlet, which direction should you move to reach the wooden chair?"
    choices = ["Forward", "Left", "Right", "Backward"]
    ground_truth = "A"
    
    print(f"\n[2] Loading scene {scene_id}...")
    mesh_path = find_mesh_file(scene_id)
    if mesh_path is None:
        print(f"[✗] Mesh not found for scene {scene_id}")
        return
    
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    print(f"[✓] Mesh loaded: {len(np.asarray(mesh.vertices))} vertices")
    
    # Compute initial camera pose
    print(f"\n[3] Computing initial camera pose...")
    sky_dir = get_sky_direction_for_scene(scene_id)
    up_vector = sky_direction_to_up_vector(sky_dir)
    
    # Generate candidate views and select best
    vertices = np.asarray(mesh.vertices)
    center_x = (vertices[:, 0].min() + vertices[:, 0].max()) / 2.0
    center_y = (vertices[:, 1].min() + vertices[:, 1].max()) / 2.0
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
    cam_height = 1.6
    cam_height_z = z_min + cam_height
    eye = np.array([center_x, center_y, cam_height_z], dtype=float)
    
    # Generate 4 candidate views
    temp_dir = Path("test_episode_temp")
    temp_dir.mkdir(exist_ok=True)
    
    view_images = {}
    view_poses = {}
    for angle_deg in [0, 90, 180, 270]:
        angle_rad = np.deg2rad(angle_deg)
        forward = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0], dtype=float)
        pose = look_at_camera_pose_center_from_forward(eye, forward=forward, up=np.array([0.0, 0.0, -1.0]))
        view_poses[angle_deg] = pose
        
        img_path = temp_dir / f"candidate_{angle_deg}.png"
        render_mesh_from_pose(mesh, pose, img_path, fxfy=DEFAULT_FX_FY)
        
        img_pil = Image.open(img_path)
        img_array = np.array(img_pil).astype(float) / 255.0
        view_images[angle_deg] = img_array
    
    from render_point_cloud_qwen_angle import select_best_initial_view
    best_angle, best_score, all_scores = select_best_initial_view(view_images, metric=INITIAL_VIEW_SELECTION_METRIC)
    initial_pose = view_poses[best_angle]
    print(f"[✓] Initial pose computed (best angle: {best_angle}°)")
    
    # Create environment
    print(f"\n[4] Creating environment...")
    output_dir = Path("test_episode_output")
    env = NavigationEnvironment(
        mesh=mesh,
        mesh_path=mesh_path,
        scene_id=scene_id,
        question=question,
        choices=choices,
        ground_truth=ground_truth,
        max_steps=3,  # Short test
        output_dir=output_dir / "env_renders"
    )
    print("[✓] Environment created")
    
    # Create simulator config
    print(f"\n[5] Creating simulator config...")
    simulator_config = SimulatorConfig(
        max_steps=3,
        track_action_tokens=True,
        do_sample=True,
        temperature=1.0,
        top_p=0.9,
        top_k=50,
        min_action_tokens=10,
        max_action_tokens=100
    )
    
    # Create simulator
    print(f"\n[6] Creating episode simulator...")
    simulator = EpisodeSimulator(
        model=model,
        processor=processor,
        config=simulator_config,
        device=device
    )
    print("[✓] Simulator created")
    
    # Run episode
    print(f"\n[7] Running episode...")
    episode = simulator.run_episode(
        env=env,
        initial_pose=initial_pose,
        episode_id="test_episode_001",
        output_dir=output_dir,
        verbose=True
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("EPISODE RESULTS")
    print("=" * 80)
    print(f"Episode ID: {episode.episode_id}")
    print(f"Scene: {episode.scene_id}")
    print(f"Question: {episode.question}")
    print(f"Choices: {episode.choices}")
    print(f"Ground Truth: {episode.ground_truth}")
    print(f"Model Answer: {episode.final_answer}")
    print(f"Correct: {episode.is_correct}")
    print(f"Final Reward: {episode.final_reward}")
    print(f"Number of Turns: {len(episode.turns)}")
    print(f"Duration: {episode.metadata['duration_seconds']:.2f}s")
    
    print("\n--- Turn Summary ---")
    for i, turn in enumerate(episode.turns):
        print(f"Turn {i}:")
        print(f"  Generated tokens: {len(turn.generated_ids)}")
        print(f"  Action tokens: {turn.action_token_mask.sum().item() if turn.action_token_mask is not None else 'N/A'}")
        print(f"  Action valid: {turn.action_valid}")
        if turn.action:
            print(f"  Movement: forward={turn.action.forward_meters:.2f}m, left={turn.action.left_meters:.2f}m, "
                  f"rotation={turn.action.rotation_angle_degrees:.1f}°")
            print(f"  Done: {turn.action.done}, Answer: {turn.action.answer}")
    
    print(f"\n[✓] Episode data saved to: {output_dir}")
    print("\n" + "=" * 80)


def test_episode_data_loading():
    """
    Test loading saved episode data.
    """
    print("\n" + "=" * 80)
    print("TEST: Episode Data Loading")
    print("=" * 80)
    
    output_dir = Path("test_episode_output")
    if not output_dir.exists():
        print("[✗] No saved episode data found. Run test_single_episode() first.")
        return
    
    # Load metadata
    import json
    metadata_path = output_dir / "episode_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        print(f"\n[✓] Loaded episode metadata:")
        print(f"  Episode ID: {metadata['episode_id']}")
        print(f"  Scene: {metadata['scene_id']}")
        print(f"  Turns: {metadata['num_turns']}")
        print(f"  Final reward: {metadata['final_reward']}")
    
    # Load turn data
    for turn_idx in range(metadata['num_turns']):
        turn_dir = output_dir / f"turn_{turn_idx:02d}"
        if not turn_dir.exists():
            continue
        
        print(f"\nTurn {turn_idx}:")
        
        # Load tensors
        gen_ids_path = turn_dir / "generated_ids.pt"
        if gen_ids_path.exists():
            gen_ids = torch.load(gen_ids_path)
            print(f"  Generated IDs shape: {gen_ids.shape}")
        
        mask_path = turn_dir / "action_token_mask.pt"
        if mask_path.exists():
            mask = torch.load(mask_path)
            print(f"  Action token mask shape: {mask.shape}")
            print(f"  Action tokens: {mask.sum().item()}/{len(mask)}")
        
        # Load text
        gen_text_path = turn_dir / "generated_text.txt"
        if gen_text_path.exists():
            with open(gen_text_path) as f:
                gen_text = f.read()
            print(f"  Generated text length: {len(gen_text)} chars")
    
    print("\n[✓] Episode data loaded successfully")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RL environment and simulator")
    parser.add_argument("--test", choices=["episode", "load", "all"], default="all",
                        help="Which test to run")
    args = parser.parse_args()
    
    if args.test in ["episode", "all"]:
        test_single_episode()
    
    if args.test in ["load", "all"]:
        test_episode_data_loading()
