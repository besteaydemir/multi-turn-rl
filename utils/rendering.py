"""Rendering utilities for 3D mesh visualization."""

import numpy as np
import cv2
from PIL import Image
import open3d as o3d
from open3d.visualization import rendering
from pathlib import Path


def compute_visibility_score(image_array, background_color=(1.0, 1.0, 1.0)):
    """
    Compute visibility score: fraction of non-background pixels.
    Higher score = more mesh geometry visible (less occlusion).
    
    Args:
        image_array: numpy array (H, W, 3) with values in [0, 1]
        background_color: tuple (R, G, B) for background
    
    Returns:
        score: float in [0, 1], higher is better
    """
    if image_array.size == 0:
        return 0.0
    
    # Convert to uint8 if needed for comparison
    img_uint8 = (image_array * 255).astype(np.uint8) if image_array.max() <= 1.0 else image_array.astype(np.uint8)
    bg_uint8 = tuple(int(c * 255) for c in background_color)
    
    # Count pixels that differ from background
    non_bg = np.any(img_uint8 != bg_uint8, axis=2)
    score = non_bg.sum() / (image_array.shape[0] * image_array.shape[1])
    return score


def compute_laplacian_variance_score(image_array):
    """
    Compute Laplacian variance: measures edge density and sharpness.
    Higher score = more structured/sharp features visible.
    
    Args:
        image_array: numpy array (H, W, 3) with values in [0, 1]
    
    Returns:
        score: float, higher is better (not normalized)
    """
    if image_array.size == 0:
        return 0.0
    
    # Convert to grayscale
    img_uint8 = (image_array * 255).astype(np.uint8) if image_array.max() <= 1.0 else image_array.astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    
    # Compute Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance


def select_best_initial_view(view_images, metric="visibility"):
    """
    Select the best view from 4 candidate views using the specified metric.
    
    Args:
        view_images: dict with keys 0, 90, 180, 270 -> numpy arrays
        metric: "visibility" or "laplacian"
    
    Returns:
        tuple: (best_angle, best_score, all_scores_dict)
    """
    scores = {}
    
    if metric == "visibility":
        for angle, img_array in view_images.items():
            scores[angle] = compute_visibility_score(img_array)
        print(f"[INFO] 🎯 Visibility scores: {scores}")
    elif metric == "laplacian":
        for angle, img_array in view_images.items():
            scores[angle] = compute_laplacian_variance_score(img_array)
        print(f"[INFO] 🎯 Laplacian variance scores: {scores}")
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    best_angle = max(scores, key=scores.get)
    best_score = scores[best_angle]
    print(f"[INFO] ✅ Selected view: {best_angle}° (score: {best_score:.4f})")
    
    return best_angle, best_score, scores


def render_mesh_from_pose(mesh: o3d.geometry.TriangleMesh, cam_pose_world: np.ndarray, out_path_img: Path, fxfy=300.0, image_wh=(1024, 768)):
    """
    Headless render of a mesh using OffscreenRenderer.
    
    Args:
        mesh: Open3D TriangleMesh object
        cam_pose_world: 4x4 camera-to-world matrix
        out_path_img: Path to save rendered image
        fxfy: focal length for camera intrinsics
        image_wh: (width, height) tuple for image resolution
    """
    width, height = image_wh
    renderer = rendering.OffscreenRenderer(width, height)
    renderer.scene.view.set_post_processing(False)

    # Create a material for the mesh with lit shader
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.base_color = [1, 1, 1, 1]
    renderer.scene.clear_geometry()
    renderer.scene.add_geometry("mesh", mesh, mat, True)

    cx = width / 2.0
    cy = height / 2.0
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, float(fxfy), float(fxfy), cx, cy)

    # Open3D expects world->camera extrinsic
    extrinsic_world_to_cam = np.linalg.inv(cam_pose_world)
    renderer.setup_camera(intrinsic, extrinsic_world_to_cam)

    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    img = renderer.render_to_image()
    arr = np.asarray(img)
    Image.fromarray(arr).save(str(out_path_img))
