import open3d as o3d
import numpy as np
import os
import glob
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import open3d as o3d
mat = o3d.visualization.rendering.MaterialRecord()
print(sorted([a for a in dir(mat) if not a.startswith("_")]))


# Base folders
input_base = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/raw/Validation"
metadata_path = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/3dod/metadata.csv"
output_base = "views8"

# Camera settings
camera_height = 1.4
width = 800
height = 600

# Load metadata CSV
metadata = pd.read_csv(metadata_path)
video_to_direction = dict(zip(metadata['video_id'], metadata['sky_direction']))

# Map sky directions to rotation angles (in degrees)
direction_to_angles = {
    'Up': [0],           # Look up (North)
    'Down': [180],       # Look down (South)
    'Left': [270],       # Look left (West)
    'Right': [90],       # Look right (East)
    'NA': [0, 90, 180, 270]  # All directions if NA
}

# Find all subfolders with a .ply file
subfolders = [f.path for f in os.scandir(input_base) if f.is_dir()]

for folder in subfolders:
    ply_files = glob.glob(os.path.join(folder, "*.ply"))
    if not ply_files:
        print(f"No .ply file in {folder}, skipping.")
        continue

    mesh_file = ply_files[0]  # take first ply in folder
    mesh = o3d.io.read_triangle_mesh(mesh_file)

    # --------------------------- #
    # ✔️ Minimal Fix (important)
    # --------------------------- #
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()   # <-- Added
    # --------------------------- #

    # Compute bounding box center
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    center[2] = bbox.min_bound[2] + camera_height  # set camera height

    # Prepare output folder
    mesh_name = os.path.basename(folder)
    output_folder = os.path.join(output_base, mesh_name)
    os.makedirs(output_folder, exist_ok=True)

    # Get video_id from folder name and look up sky direction
    try:
        video_id = int(mesh_name)
        sky_direction = video_to_direction.get(video_id, 'NA')
    except ValueError:
        sky_direction = 'NA'
    
    # Always render 4 interior views (90 degree rotations around vertical axis)
    angles_to_render = [0, 90, 180, 270]
    
    print(f"Processing {mesh_name} (video_id: {video_id}, sky_direction: {sky_direction})")
    
    # Create renderer ONCE per mesh and reuse with proper reset
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.view.set_post_processing(False)
    mat = o3d.visualization.rendering.MaterialRecord()
    
    mat.shader = "defaultUnlit"      # <-- CRITICAL: forces nearest sampling in legacy renderer
    mat.base_color = [1, 1, 1, 1]


    renderer.scene.clear_geometry()
    renderer.scene.add_geometry("mesh", mesh, mat, True)
    # renderer.scene.set_use_fxaa(True)
    # renderer.scene.set_texture_quality(
    #     o3d.visualization.rendering.TextureQuality.High
    # )

    # renderer.scene.set_lighting(
    #     o3d.visualization.rendering.Open3DScene.LightingProfile.SOFT_SHADOWS,
    #     (0.5, 0.5, 0.5)
    # )

    # ---------------------------
    # Interior views (fixed normals)
    # ---------------------------
    for view_idx, angle_deg in enumerate(angles_to_render):
        angle_rad = np.deg2rad(angle_deg)
        cam_pos = center.copy()
        look_dir = np.array([np.cos(angle_rad), np.sin(angle_rad), 0])
        up_dir = np.array([0, 0, -1])  # Flip Y to fix upside-down issue
        look_at = cam_pos + look_dir
        
        # Set up proper camera with intrinsics and extrinsics
        cx = width / 2.0
        cy = height / 2.0
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, 300.0, 300.0, cx, cy)
        
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
        img_path = os.path.join(output_folder, f"interior_view_{view_idx+1}.png")
        o3d.io.write_image(img_path, img)

    # Exterior views (unchanged – no need to modify)
    # bbox_size = bbox.max_bound - bbox.min_bound
    # bbox_center = bbox.get_center()
    
    # axis_length = max(bbox_size) * 0.3
    # axis_frame = o3d.geometry.LineSet(
    #     points=o3d.utility.Vector3dVector([
    #         bbox_center,
    #         bbox_center + np.array([axis_length, 0, 0]),
    #         bbox_center + np.array([0, axis_length, 0]),
    #         bbox_center + np.array([0, 0, axis_length]),
    #     ]),
    #     lines=o3d.utility.Vector2iVector([
    #         [0, 1], [0, 2], [0, 3],
    #     ]),
    # )
    # axis_frame.colors = o3d.utility.Vector3dVector([
    #     [1, 0, 0], [1, 0, 0],
    #     [0, 1, 0], [0, 1, 0],
    #     [0, 0, 1], [0, 0, 1],
    # ])
    
    # exterior_positions = [
    #     (bbox_center[0] + bbox_size[0] * 0.8, bbox_center[1], bbox_center[2] + bbox_size[2] * 0.5, "Front"),
    #     (bbox_center[0] - bbox_size[0] * 0.8, bbox_center[1], bbox_center[2] + bbox_size[2] * 0.5, "Back"),
    #     (bbox_center[0], bbox_center[1] + bbox_size[1] * 0.8, bbox_center[2] + bbox_size[2] * 0.5, "Side"),
    # ]
    
    # for ext_idx, (cam_x, cam_y, cam_z, position_label) in enumerate(exterior_positions):
    #     cam_pos_ext = np.array([cam_x, cam_y, cam_z])
    #     look_at_ext = bbox_center
        
    #     renderer.scene.clear_geometry()
    #     mat = o3d.visualization.rendering.MaterialRecord()
    #     mat.shader = "defaultLit"
    #     renderer.scene.add_geometry("mesh", mesh, mat)
        
    #     axis_mat = o3d.visualization.rendering.MaterialRecord()
    #     axis_mat.shader = "unlitLine"
    #     axis_mat.line_width = 5.0
    #     renderer.scene.add_geometry("axis_frame", axis_frame, axis_mat)
        
    #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=bbox_size[0] * 0.1)
    #     sphere.translate(cam_pos_ext)
    #     sphere_mat = o3d.visualization.rendering.MaterialRecord()
    #     sphere_mat.shader = "defaultLit"
    #     sphere_mat.base_color = [1.0, 0.0, 0.0, 1.0]
    #     renderer.scene.add_geometry("camera_marker", sphere, sphere_mat)
        
    #     renderer.scene.set_lighting(
    #         o3d.visualization.rendering.Open3DScene.LightingProfile.SOFT_SHADOWS,
    #         (0.5, 0.5, 0.5)
    #     )
        
    #     cx = width / 2.0
    #     cy = height / 2.0
    #     intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, 300.0, 300.0, cx, cy)
        
    #     up_dir = np.array([0, 0, -1])
    #     forward = look_at_ext - cam_pos_ext
    #     forward /= np.linalg.norm(forward)
    #     right = np.cross(up_dir, forward)
    #     right /= np.linalg.norm(right)
    #     true_up = np.cross(forward, right)
        
    #     cam_pose_world = np.eye(4)
    #     cam_pose_world[:3, 0] = right
    #     cam_pose_world[:3, 1] = true_up
    #     cam_pose_world[:3, 2] = forward
    #     cam_pose_world[:3, 3] = cam_pos_ext
        
    #     extrinsic_world_to_cam = np.linalg.inv(cam_pose_world)
    #     renderer.setup_camera(intrinsic, extrinsic_world_to_cam)
    #     renderer.scene.set_background([1, 1, 1, 1])
        
    #     img = renderer.render_to_image()
    #     img_array = np.asarray(img)
        
    #     pil_img = Image.fromarray((img_array * 255).astype(np.uint8))
    #     draw = ImageDraw.Draw(pil_img)
        
    #     text = f"Camera: {position_label}"
    #     bbox_text = draw.textbbox((0, 0), text)
    #     text_width = bbox_text[2] - bbox_text[0]
    #     text_height = bbox_text[3] - bbox_text[1]
        
    #     margin = 10
    #     rect_coords = [
    #         (pil_img.width - text_width - margin * 2, pil_img.height - text_height - margin * 2),
    #         (pil_img.width - margin, pil_img.height - margin)
    #     ]
    #     draw.rectangle(rect_coords, fill=(0, 0, 0, 200), outline=(255, 0, 0, 255), width=3)
        
    #     draw.text(
    #         (pil_img.width - text_width - margin, pil_img.height - text_height - margin),
    #         text,
    #         fill=(255, 0, 0, 255)
    #     )
        
    #     ext_path = os.path.join(output_folder, f"exterior_view_{ext_idx+1}_{position_label.lower()}.png")
    #     pil_img.save(ext_path)

    print(f"Rendered {len(angles_to_render)} interior views and 3 exterior views for {mesh_file} -> {output_folder}")
    
    del renderer
    del mesh
    import gc
    gc.collect()
