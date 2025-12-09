import open3d as o3d
import numpy as np
import os
import glob

# Base folders
input_base = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/raw/Validation"
output_base = "views"

# Camera settings
camera_height = 1.6
width = 800
height = 600

# Find all subfolders with a .ply file
subfolders = [f.path for f in os.scandir(input_base) if f.is_dir()]

for folder in subfolders:
    ply_files = glob.glob(os.path.join(folder, "*.ply"))
    if not ply_files:
        print(f"No .ply file in {folder}, skipping.")
        continue

    mesh_file = ply_files[0]  # take first ply in folder
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()

    # Compute bounding box center
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    center[2] = bbox.min_bound[2] + camera_height  # set camera height

    # Prepare output folder
    mesh_name = os.path.basename(folder)
    output_folder = os.path.join(output_base, mesh_name)
    os.makedirs(output_folder, exist_ok=True)

    # Create renderer ONCE per mesh and reuse with proper reset
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    renderer.scene.add_geometry("mesh", mesh, mat)
    renderer.scene.set_lighting(o3d.visualization.rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, (0.5, 0.5, 0.5))

    # Four views: rotate around vertical axis
    for i, angle_deg in enumerate([0, 90, 180, 270]):
        angle_rad = np.deg2rad(angle_deg)
        cam_pos = center
        look_dir = np.array([np.cos(angle_rad), np.sin(angle_rad), 0])
        up_dir = np.array([0, 0, 1])
        look_at = cam_pos + look_dir
        
        # Set up proper camera with intrinsics and extrinsics
        cx = width / 2.0
        cy = height / 2.0
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, 400.0, 400.0, cx, cy)
        
        # Build camera-to-world matrix
        forward = look_at - cam_pos
        forward_norm = np.linalg.norm(forward)
        if forward_norm > 1e-8:
            forward = forward / forward_norm
        right = np.cross(up_dir, forward)
        right_norm = np.linalg.norm(right)
        if right_norm > 1e-8:
            right = right / right_norm
        true_up = np.cross(forward, right)
        
        cam_pose_world = np.eye(4, dtype=np.float64)
        cam_pose_world[:3, 0] = right
        cam_pose_world[:3, 1] = true_up
        cam_pose_world[:3, 2] = forward
        cam_pose_world[:3, 3] = cam_pos
        
        extrinsic_world_to_cam = np.linalg.inv(cam_pose_world)
        renderer.setup_camera(intrinsic, extrinsic_world_to_cam)
        
        renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
        
        # Render and save
        img = renderer.render_to_image()
        img_path = os.path.join(output_folder, f"view_{i+1}.png")
        o3d.io.write_image(img_path, img)

    print(f"Rendered 4 views for {mesh_file} -> {output_folder}")
    
    # Clean up renderer and mesh for this iteration
    del renderer
    del mesh
    del img
    import gc
    gc.collect()
