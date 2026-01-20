#!/usr/bin/env python3
"""
Extract trajectory data from experiment logs and generate visualization data.
This creates trajectory files that can be visualized on a local machine with the mesh.
"""

import json
import numpy as np
from pathlib import Path
import argparse


def extract_trajectory_from_question(question_dir):
    """
    Extract camera poses and metadata from a single question folder.
    
    Returns:
        dict with trajectory data or None if extraction fails
    """
    question_dir = Path(question_dir)
    
    # Find all camera pose files
    cam_pose_files = sorted(question_dir.glob("cam_pose_*.npy"))
    
    if not cam_pose_files:
        return None
    
    # Load all camera poses
    poses = []
    for cam_file in cam_pose_files:
        try:
            pose = np.load(str(cam_file))
            poses.append({
                "file": cam_file.name,
                "position": pose[:3, 3].tolist(),
                "rotation": pose[:3, :3].tolist(),
                "matrix": pose.tolist()
            })
        except Exception as e:
            print(f"[WARN] Failed to load {cam_file.name}: {e}")
            continue
    
    if not poses:
        return None
    
    # Load metadata if available
    metadata = {}
    results_file = question_dir / "results.json"
    if results_file.exists():
        try:
            with open(results_file) as f:
                metadata = json.load(f)
        except:
            pass
    
    initial_view_file = question_dir / "initial_view_selection.json"
    initial_view = {}
    if initial_view_file.exists():
        try:
            with open(initial_view_file) as f:
                initial_view = json.load(f)
        except:
            pass
    
    return {
        "poses": poses,
        "num_poses": len(poses),
        "metadata": metadata,
        "initial_view": initial_view
    }


def generate_visualization_script(trajectory_data_file, output_script):
    """
    Generate a Python script that can be run locally to visualize trajectories.
    """
    script_content = '''#!/usr/bin/env python3
"""
Trajectory Visualization Script
Load mesh + trajectory data to visualize camera path through the scene.

Usage:
    1. Update mesh_path to point to your local mesh file
    2. Update json_trajectory_file to point to your trajectory data
    3. Run: python visualize_trajectory.py
"""

import json
import numpy as np
import open3d as o3d
from pathlib import Path


def visualize_trajectory(mesh_path, trajectory_json_path):
    """
    Visualize mesh with camera trajectory markers.
    
    Args:
        mesh_path: Path to .ply mesh file
        trajectory_json_path: Path to extracted trajectory JSON
    """
    # Load mesh
    print(f"Loading mesh from: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    print(f"Mesh loaded: {len(np.asarray(mesh.vertices))} vertices, {len(np.asarray(mesh.triangles))} faces")
    
    # Load trajectory data
    print(f"Loading trajectory from: {trajectory_json_path}")
    with open(trajectory_json_path) as f:
        data = json.load(f)
    
    geometries = [mesh]  # Start with mesh
    
    # Add camera markers (red spheres for each pose)
    poses = data["poses"]
    print(f"Trajectory has {len(poses)} camera poses")
    
    for i, pose_data in enumerate(poses):
        pos = np.array(pose_data["position"])
        
        # Create sphere at camera position
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
        sphere.translate(pos)
        sphere.compute_vertex_normals()
        
        # Color: fade from red (start) to orange (end)
        color_intensity = 1.0 - (i / max(len(poses) - 1, 1)) * 0.5
        sphere.paint_uniform_color([color_intensity, 0.2, 0.0])
        
        geometries.append(sphere)
        
        # Add direction indicator (green arrow pointing forward)
        rotation = np.array(pose_data["rotation"])
        forward_dir = rotation[:, 2]  # Forward is z-axis
        arrow_length = 0.4
        arrow_end = pos + forward_dir * arrow_length
        
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=0.05,
            height=arrow_length
        )
        
        # Rotate and translate cylinder
        cylinder_center = (pos + arrow_end) / 2
        default_dir = np.array([0, 0, 1])
        rot_axis = np.cross(default_dir, forward_dir)
        if np.linalg.norm(rot_axis) > 1e-6:
            rot_axis = rot_axis / np.linalg.norm(rot_axis)
            angle = np.arccos(np.clip(np.dot(default_dir, forward_dir / np.linalg.norm(forward_dir)), -1, 1))
            R_align = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * angle)
            cylinder.rotate(R_align, center=[0, 0, 0])
        
        cylinder.translate(cylinder_center)
        cylinder.paint_uniform_color([0.0, 1.0, 0.0])  # Green
        geometries.append(cylinder)
        
        # Mark first pose with blue sphere on top
        if i == 0:
            start_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
            start_marker.translate(pos + np.array([0, 0, 0.3]))
            start_marker.paint_uniform_color([0.0, 0.0, 1.0])  # Blue
            geometries.append(start_marker)
        
        print(f"  Pose {i}: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    
    # Print metadata
    if data.get("metadata"):
        print(f"\\nMetadata: {data['metadata']}")
    
    print(f"\\nVisualization ready! Showing {len(poses)} camera poses.")
    print("Legend: Red=camera poses, Green=viewing direction, Blue=start position")
    
    # Visualize
    o3d.visualization.draw_geometries(geometries)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python visualize_trajectory.py <mesh_path> <trajectory_json>")
        print("")
        print("Example:")
        print("  python visualize_trajectory.py mesh.ply trajectory_q001.json")
        sys.exit(1)
    
    mesh_path = sys.argv[1]
    trajectory_json = sys.argv[2]
    
    if not Path(mesh_path).exists():
        print(f"Error: Mesh file not found: {mesh_path}")
        sys.exit(1)
    
    if not Path(trajectory_json).exists():
        print(f"Error: Trajectory file not found: {trajectory_json}")
        sys.exit(1)
    
    visualize_trajectory(mesh_path, trajectory_json)
'''
    
    with open(output_script, 'w') as f:
        f.write(script_content)
    
    print(f"Generated visualization script: {output_script}")


def main(exp_dir, output_dir=None):
    """
    Extract trajectories from all questions in experiment folder.
    
    Args:
        exp_dir: Path to experiment folder with q001, q002, etc.
        output_dir: Where to save trajectory JSON files (defaults to exp_dir/trajectories)
    """
    exp_dir = Path(exp_dir)
    
    if output_dir is None:
        output_dir = exp_dir / "trajectories"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}\n")
    
    # Find all question folders
    question_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("q")])
    print(f"Found {len(question_dirs)} question folders\n")
    
    trajectories_extracted = 0
    trajectories_failed = 0
    
    # Extract trajectory from each question
    for q_dir in question_dirs:
        q_name = q_dir.name
        print(f"Processing {q_name}...")
        
        traj_data = extract_trajectory_from_question(q_dir)
        
        if traj_data is None:
            print(f"  ❌ No trajectory data found")
            trajectories_failed += 1
            continue
        
        # Save trajectory JSON
        output_file = output_dir / f"trajectory_{q_name}.json"
        with open(output_file, 'w') as f:
            json.dump(traj_data, f, indent=2)
        
        print(f"  ✅ Saved: {output_file.name} ({traj_data['num_poses']} poses)")
        trajectories_extracted += 1
    
    print(f"\n{'='*60}")
    print(f"Summary: {trajectories_extracted} extracted, {trajectories_failed} failed")
    print(f"{'='*60}\n")
    
    # Generate visualization script
    viz_script = output_dir / "visualize_trajectory.py"
    generate_visualization_script(None, viz_script)
    
    # Create a README
    readme = output_dir / "README.md"
    with open(readme, 'w') as f:
        f.write(f"""# Trajectory Visualization Data

This folder contains extracted trajectory data from {trajectories_extracted} questions.

## Files

- `trajectory_qXXX.json` - Camera trajectory for each question (4x4 pose matrices, positions, rotations)
- `visualize_trajectory.py` - Standalone Python script to visualize trajectories with mesh

## Usage

### On Your Local Machine

1. Copy the trajectory JSON files to your computer:
   ```
   trajectory_q001.json
   trajectory_q002.json
   ...
   ```

2. Get the corresponding mesh file from the server and save locally

3. Run the visualization script:
   ```bash
   python visualize_trajectory.py /path/to/mesh.ply trajectory_q001.json
   ```

### Requirements

```bash
pip install open3d numpy
```

## Trajectory Format

Each `trajectory_qXXX.json` contains:
- `poses`: List of camera poses, each with:
  - `position`: [x, y, z] camera location in world coordinates
  - `rotation`: 3x3 rotation matrix
  - `matrix`: Full 4x4 camera-to-world transformation matrix
- `num_poses`: Number of camera positions
- `metadata`: Question metadata
- `initial_view`: Initial view selection data

## Visualization Legend

- **Red spheres**: Camera positions (fade from bright red to darker as you progress)
- **Green arrows**: Camera viewing direction (points toward where the camera is looking)
- **Blue sphere**: Starting camera position (marked with blue marker on top)
- **Mesh**: Gray background scene

You can rotate, zoom, and pan in the visualization window.
""")
    
    print(f"Generated README: {readme}")
    print(f"\n🎉 All trajectory files ready for download!")
    print(f"\nTo use on your computer:")
    print(f"1. Download all files from: {output_dir}")
    print(f"2. Get the mesh file: /dss/mcmlscratch/06/di38riq/arkit_vsi/raw/Validation/41125700/41125700_3dod_mesh.ply")
    print(f"3. Run: python visualize_trajectory.py /path/to/mesh.ply trajectory_q001.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract trajectories from experiment logs")
    parser.add_argument("exp_dir", help="Path to experiment folder")
    parser.add_argument("--output", "-o", default=None, help="Output directory for trajectory files")
    args = parser.parse_args()
    
    main(args.exp_dir, args.output)
