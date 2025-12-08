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

    # Set up offscreen renderer
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    renderer.scene.add_geometry("mesh", mesh, mat)

    # Four views: rotate around vertical axis
    for i, angle_deg in enumerate([0, 90, 180, 270]):
        angle_rad = np.deg2rad(angle_deg)
        cam_pos = center
        look_dir = np.array([np.cos(angle_rad), np.sin(angle_rad), 0])
        up_dir = np.array([0, 0, 1])
        look_at = cam_pos + look_dir
        renderer.scene.camera.look_at(look_at, cam_pos, up_dir)

        # Render and save
        img = renderer.render_to_image()
        img_path = os.path.join(output_folder, f"view_{i+1}.png")
        o3d.io.write_image(img_path, img)

    #renderer.release()
    print(f"Rendered 4 views for {mesh_file} -> {output_folder}")
