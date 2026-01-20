"""Visualization utilities for trajectory and scene rendering."""

import numpy as np
from PIL import Image
import open3d as o3d
from open3d.visualization import rendering
from pathlib import Path

from .camera import look_at_camera_pose_center_from_forward


def render_birds_eye_view_with_path(mesh: o3d.geometry.TriangleMesh, camera_positions: list, out_path: Path, marker_size=0.15):
    """
    Render multiple views with camera positions marked as red spheres with green direction indicators.
    
    Views include:
    - 4 BEV (bird's eye view from directly above, rotated)
    - 4 Dollhouse views (looking into the room from 4 sides, with obstructing wall removed)
    
    Args:
        mesh: Open3D TriangleMesh
        camera_positions: List of 4x4 camera pose matrices
        out_path: Base path for output images (will append view suffixes)
        marker_size: Size of camera position markers
    """
    # Get full mesh bounds first (before any filtering)
    vertices = np.asarray(mesh.vertices)
    x_min_full, x_max_full = vertices[:, 0].min(), vertices[:, 0].max()
    y_min_full, y_max_full = vertices[:, 1].min(), vertices[:, 1].max()
    z_min_full, z_max_full = vertices[:, 2].min(), vertices[:, 2].max()
    
    center_x = (x_min_full + x_max_full) / 2.0
    center_y = (y_min_full + y_max_full) / 2.0
    center_z = (z_min_full + z_max_full) / 2.0
    
    x_extent = x_max_full - x_min_full
    y_extent = y_max_full - y_min_full
    max_extent = max(x_extent, y_extent)
    
    # Camera positioning parameters
    dollhouse_height = z_max_full + max_extent * 0.3  # Higher up for better view
    dollhouse_distance = max_extent * 0.5  # Further back
    bev_height = z_max_full + max_extent * 0.4
    
    # Use high resolution
    width, height = 2560, 1920
    
    def create_filtered_mesh_for_view(view_type, view_direction=None):
        """Create a filtered mesh appropriate for the view type."""
        verts = np.asarray(mesh.vertices)
        
        if view_type == "bev":
            # For BEV: remove ceiling (top 15% by Z)
            z_threshold = np.percentile(verts[:, 2], 85)
            mask = verts[:, 2] < z_threshold
        elif view_type == "dollhouse":
            # For dollhouse: remove ceiling AND the wall facing the camera
            z_threshold = np.percentile(verts[:, 2], 85)
            z_mask = verts[:, 2] < z_threshold
            
            # Remove wall based on view direction
            wall_margin = 0.15  # Remove 15% of room from camera side
            if view_direction == "east":
                x_threshold = x_max_full - (x_extent * wall_margin)
                wall_mask = verts[:, 0] < x_threshold
            elif view_direction == "west":
                x_threshold = x_min_full + (x_extent * wall_margin)
                wall_mask = verts[:, 0] > x_threshold
            elif view_direction == "north":
                y_threshold = y_max_full - (y_extent * wall_margin)
                wall_mask = verts[:, 1] < y_threshold
            elif view_direction == "south":
                y_threshold = y_min_full + (y_extent * wall_margin)
                wall_mask = verts[:, 1] > y_threshold
            else:
                wall_mask = np.ones(len(verts), dtype=bool)
            
            mask = z_mask & wall_mask
        else:
            mask = np.ones(len(verts), dtype=bool)
        
        # Create filtered mesh
        filtered = o3d.geometry.TriangleMesh()
        vertex_indices = np.where(mask)[0]
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(vertex_indices)}
        
        filtered.vertices = o3d.utility.Vector3dVector(verts[mask])
        
        if mesh.has_vertex_colors():
            colors = np.asarray(mesh.vertex_colors)
            filtered.vertex_colors = o3d.utility.Vector3dVector(colors[mask])
        
        # Filter triangles
        triangles = np.asarray(mesh.triangles)
        valid_triangles = []
        for tri in triangles:
            if all(v in vertex_map for v in tri):
                valid_triangles.append([vertex_map[tri[0]], vertex_map[tri[1]], vertex_map[tri[2]]])
        
        if len(valid_triangles) > 0:
            filtered.triangles = o3d.utility.Vector3iVector(np.array(valid_triangles))
            filtered.compute_vertex_normals()
        
        return filtered
    
    # Create markers with direction indicators
    marker_geometries = []
    for i, pos in enumerate(camera_positions):
        # Red sphere for position
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=marker_size * 1.5)
        sphere.translate(pos[:3, 3])
        sphere.compute_vertex_normals()
        
        red_intensity = 1.0 - (i / max(len(camera_positions) - 1, 1)) * 0.5
        marker_geometries.append((f"sphere_{i}", sphere, [red_intensity, 0.0, 0.0, 1.0]))
        
        # Green direction indicator
        forward_direction = pos[:3, 2]
        arrow_length = marker_size * 3.0
        arrow_start = pos[:3, 3]
        arrow_end = arrow_start + forward_direction * arrow_length
        
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=marker_size*0.4, height=arrow_length)
        cylinder_direction = arrow_end - arrow_start
        cylinder_center = (arrow_start + arrow_end) / 2
        
        default_dir = np.array([0, 0, 1])
        rot_axis = np.cross(default_dir, cylinder_direction)
        if np.linalg.norm(rot_axis) > 1e-6:
            rot_axis = rot_axis / np.linalg.norm(rot_axis)
            angle = np.arccos(np.clip(np.dot(default_dir, cylinder_direction / np.linalg.norm(cylinder_direction)), -1, 1))
            R_align = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * angle)
            cylinder.rotate(R_align, center=[0, 0, 0])
        cylinder.translate(cylinder_center)
        cylinder.compute_vertex_normals()
        marker_geometries.append((f"direction_{i}", cylinder, [0.0, 1.0, 0.0, 1.0]))
        
        if i == 0:
            start_marker = o3d.geometry.TriangleMesh.create_sphere(radius=marker_size * 0.8)
            start_marker.translate(pos[:3, 3] + np.array([0, 0, marker_size * 2]))
            start_marker.compute_vertex_normals()
            marker_geometries.append((f"start_{i}", start_marker, [0.0, 0.0, 1.0, 1.0]))
    
    # Define views with proper orientation
    views = [
        # BEV views - top-down
        ("bev_0", "bev", None,
         np.array([center_x, center_y, bev_height]),
         np.array([0.0, 0.0, -1.0]),
         np.array([0.0, 1.0, 0.0])),
        ("bev_90", "bev", None,
         np.array([center_x, center_y, bev_height]),
         np.array([0.0, 0.0, -1.0]),
         np.array([-1.0, 0.0, 0.0])),
        ("bev_180", "bev", None,
         np.array([center_x, center_y, bev_height]),
         np.array([0.0, 0.0, -1.0]),
         np.array([0.0, -1.0, 0.0])),
        ("bev_270", "bev", None,
         np.array([center_x, center_y, bev_height]),
         np.array([0.0, 0.0, -1.0]),
         np.array([1.0, 0.0, 0.0])),
        
        # Dollhouse views - looking from sides at an angle
        ("dollhouse_east", "dollhouse", "east",
         np.array([x_max_full + dollhouse_distance, center_y, dollhouse_height]),
         None, None),
        ("dollhouse_west", "dollhouse", "west",
         np.array([x_min_full - dollhouse_distance, center_y, dollhouse_height]),
         None, None),
        ("dollhouse_north", "dollhouse", "north",
         np.array([center_x, y_max_full + dollhouse_distance, dollhouse_height]),
         None, None),
        ("dollhouse_south", "dollhouse", "south",
         np.array([center_x, y_min_full - dollhouse_distance, dollhouse_height]),
         None, None),
    ]
    
    for view_info in views:
        if len(view_info) == 6:
            view_name, view_type, wall_dir, eye, forward, up = view_info
        else:
            continue
            
        print(f"[INFO] 🎨 Rendering {view_name} view...")
        
        # Get appropriate filtered mesh
        filtered_mesh = create_filtered_mesh_for_view(view_type, wall_dir)
        
        # For dollhouse views, compute forward and up vectors properly
        if forward is None:
            target = np.array([center_x, center_y, center_z])
            forward = target - eye
            forward = forward / np.linalg.norm(forward)
            
            world_up = np.array([0.0, 0.0, 1.0])
            right = np.cross(forward, world_up)
            if np.linalg.norm(right) < 1e-6:
                right = np.array([1.0, 0.0, 0.0])
            right = right / np.linalg.norm(right)
            
            up = np.cross(right, forward)
            if up[2] < 0:
                up = -up
            up = up / np.linalg.norm(up)
        
        # Create renderer
        renderer = rendering.OffscreenRenderer(width, height)
        renderer.scene.view.set_post_processing(False)
        
        # Add mesh
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        mat.base_color = [1.0, 1.0, 1.0, 1.0]
        renderer.scene.add_geometry("mesh", filtered_mesh, mat, True)
        
        # Lighting
        renderer.scene.scene.set_sun_light([0.5, 0.5, -1.0], [1.0, 1.0, 1.0], 75000)
        renderer.scene.scene.enable_sun_light(True)
        
        # Add markers
        for geom_name, geom, color in marker_geometries:
            marker_mat = rendering.MaterialRecord()
            marker_mat.shader = "defaultUnlit"
            marker_mat.base_color = color
            renderer.scene.add_geometry(f"{geom_name}_{view_name}", geom, marker_mat, True)
        
        # Setup camera
        camera_pose = look_at_camera_pose_center_from_forward(eye, forward=forward, up=up)
        cx, cy = width / 2.0, height / 2.0
        
        if "bev" in view_name:
            focal_length = width / (max_extent * 0.5)
        else:
            focal_length = 800.0
            
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, focal_length, focal_length, cx, cy)
        extrinsic = np.linalg.inv(camera_pose)
        renderer.setup_camera(intrinsic, extrinsic)
        
        renderer.scene.set_background([0.95, 0.95, 0.95, 1.0])
        
        try:
            img = renderer.render_to_image()
            arr = np.asarray(img)
            view_path = out_path.parent / f"{out_path.stem}_{view_name}.png"
            Image.fromarray(arr).save(str(view_path))
            print(f"[INFO] 🗺️  Saved {view_name} view: {view_path}")
        except Exception as e:
            print(f"[WARN] Failed to render {view_name}: {e}")
        finally:
            del renderer
    
    print(f"[INFO] ✅ Completed rendering {len(views)} views with {len(camera_positions)} camera positions")
