#!/usr/bin/env python3
"""
Sequential pipeline with job splitting support:
 - Load VSI-Bench questions (arkitscenes + route_planning)
 - Support splitting questions across multiple jobs (--split N --num-splits M)
 - For each question, run Qwen3-VL multimodal reasoning loop sequentially
 - Collect answers and evaluate against ground truth
 - NO batching - all inference is sequential
"""

import argparse
import json
import re
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
import torch
from PIL import Image
from open3d.visualization import rendering
from datasets import load_dataset

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ----------------- Config -----------------
CACHE_DIR = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MESH_BASE_DIR = "/dss/mcmlscratch/06/di38riq/arkit_vsi/raw"
ARKIT_CSV_PATH = "/dss/dsshome1/06/di38riq/ARKitScenes/raw/raw_train_val_splits.csv"
METADATA_CSV_PATH = "/dss/mcmlscratch/06/di38riq/arkit_vsi/raw/metadata.csv"

NUM_STEPS = 8  # Max iterations per question
IMAGE_WH = (640, 480)
DEFAULT_FX_FY = 300.0
CAM_HEIGHT = 1.6
MAX_ATTEMPTS_PER_STEP = 1

# Initial view selection configuration
INITIAL_VIEW_SELECTION_METRIC = "visibility"  # "visibility" or "laplacian"

# ----------------- Utilities -----------------
def timestamp_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def compute_visibility_score(image_array, background_color=(1.0, 1.0, 1.0)):
    """Compute visibility score: fraction of non-background pixels."""
    if image_array.size == 0:
        return 0.0
    img_uint8 = (image_array * 255).astype(np.uint8) if image_array.max() <= 1.0 else image_array.astype(np.uint8)
    bg_uint8 = tuple(int(c * 255) for c in background_color)
    non_bg = np.any(img_uint8 != bg_uint8, axis=2)
    score = non_bg.sum() / (image_array.shape[0] * image_array.shape[1])
    return score

def compute_laplacian_variance_score(image_array):
    """Compute Laplacian variance: measures edge density and sharpness."""
    if image_array.size == 0:
        return 0.0
    img_uint8 = (image_array * 255).astype(np.uint8) if image_array.max() <= 1.0 else image_array.astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance

def select_best_initial_view(view_images, metric="visibility"):
    """Select the best view from 4 candidate views."""
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

def look_at_camera_pose_center_from_forward(eye, forward=np.array([1.0,0.0,0.0]), up=np.array([0,0,-1])):
    """Construct camera-to-world 4x4 matrix."""
    forward = np.asarray(forward, dtype=float)
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-8:
        forward = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        forward = forward / forward_norm
    up = np.asarray(up, dtype=float)
    right = np.cross(up, forward)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        up_tmp = np.array([0.0, 1.0, 0.0], dtype=float)
        right = np.cross(up_tmp, forward)
        right /= np.linalg.norm(right)
    else:
        right = right / right_norm
    true_up = np.cross(forward, right)
    R = np.column_stack((right, true_up, forward))
    T = np.eye(4, dtype=float)
    T[:3,:3] = R
    T[:3,3] = eye
    return T

def save_matrix(path: Path, mat: np.ndarray, text=True):
    np.save(str(path), mat)
    if text:
        with open(path.with_suffix(".txt"), "w") as f:
            f.write(str(mat))

def render_mesh_from_pose(mesh: o3d.geometry.TriangleMesh, cam_pose_world: np.ndarray, out_path_img: Path, fxfy=DEFAULT_FX_FY):
    """Headless render of a mesh using OffscreenRenderer."""
    width, height = IMAGE_WH
    renderer = rendering.OffscreenRenderer(width, height)
    renderer.scene.view.set_post_processing(False)

    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.base_color = [1, 1, 1, 1]
    renderer.scene.clear_geometry()
    renderer.scene.add_geometry("mesh", mesh, mat, True)

    cx = width / 2.0
    cy = height / 2.0
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, float(fxfy), float(fxfy), cx, cy)

    extrinsic_world_to_cam = np.linalg.inv(cam_pose_world)
    renderer.setup_camera(intrinsic, extrinsic_world_to_cam)

    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    img = renderer.render_to_image()
    arr = np.asarray(img)
    Image.fromarray(arr).save(str(out_path_img))

def find_mesh_file(scene_id, mesh_base_dir=MESH_BASE_DIR):
    """Find a mesh file for the given scene_id."""
    video_id = str(scene_id)
    for split in ["Validation", "Training"]:
        mesh_path = Path(mesh_base_dir) / split / video_id / f"{video_id}_3dod_mesh.ply"
        if mesh_path.exists():
            return mesh_path
    print(f"[WARN] Mesh file not found for scene {scene_id} in {mesh_base_dir}")
    return None

def build_instruction_text(R, t, question, bbox=None, options=None, is_final_step=False, movement_history=None, step_num=0, question_type="unknown", is_numerical=False):
    """Build instruction text for Qwen."""
    R_rounded = np.round(R, 2).tolist()
    t_rounded = np.round(t, 2).tolist()
    
    bbox_text = ""
    if bbox is not None:
        try:
            mins, maxs = bbox
            bbox_text = f"""**Scene Bounding Box (use this to reason about your position):**
- X-axis range: [{mins[0]:.2f}, {maxs[0]:.2f}] meters
- Y-axis range: [{mins[1]:.2f}, {maxs[1]:.2f}] meters  
- Z-axis range: [{mins[2]:.2f}, {maxs[2]:.2f}] meters
- Your current position: {t_rounded}
"""
        except Exception:
            pass

    options_text = ""
    if not is_numerical and options and isinstance(options, (list, dict)):
        options_text = "\n**Answer Options:**\n"
        if isinstance(options, list):
            for i, opt in enumerate(options):
                options_text += f"{chr(65+i)}. {opt}\n"
        elif isinstance(options, dict):
            for k, v in options.items():
                options_text += f"{k}. {v}\n"
        options_text += "\n"

    if is_numerical:
        answer_format = "a numerical value (e.g., 2.5 for meters, 3 for counting)"
        answer_example = "\"answer\": 2.5 (for numerical questions) or \"answer\": 3 (for counting)"
    else:
        answer_format = "A, B, C, or D"
        answer_example = "\"answer\": \"A\" (or B, C, D)"
    
    if is_final_step:
        movement_instruction = f"""
**THIS IS YOUR FINAL STEP:** You have explored the scene. You MUST provide your final answer.
- Prioritize outputting your best answer ({answer_format}).
- You may provide minimal camera movement or zero movement.
- The critical requirement is that you output your final answer in the JSON response.
- Set "done": true ONLY when you are providing a final answer.
"""
        important_note = f"IMPORTANT: This is your final opportunity to answer. You MUST output your best answer ({answer_format}) in the JSON and set done=true."
    else:
        movement_instruction = f"""
**Movement Guidelines:**

1. **Do NOT stay in the same location.** Always provide at least one non-zero movement.

2. **AVOID REPETITIVE MOVEMENTS!** Look at your action history - do NOT keep rotating back and forth.
   - Explore NEW areas you haven't seen yet.
   - Think strategically: What haven't you seen? What information is missing?

3. **Use the bounding box information above to reason about your position and plan movements.**

4. **Answering too early is a mistake.** ONLY output an answer when you are CERTAIN.
   - Only when completely confident: set "done": true AND provide your answer.

5. **When you are confident:** If you have seen enough viewpoints and are absolutely certain, set "done": true AND provide your answer ({answer_format}).
"""
        important_note = "IMPORTANT: Keep exploring meaningfully and DO NOT output an answer unless you are ready to stop (done=true). Answer and done=true should come together."

    if "You are a robot beginning at" in question:
        question = question.replace("You are a robot beginning at", "If you begin navigating at")

    if question_type == "route_planning":
        starting_place_note = "\n**Note:** Your first task is to identify the starting place mentioned in the question and understand the route.\n"
    else:
        starting_place_note = "\n**Note:** Your first task is to locate and identify the items/objects mentioned in the question.\n"

    movement_history_text = ""
    if movement_history:
        movement_history_text = "\n---\nYour previous movements and actions (IMPORTANT: Do not repeat these movements exactly):\n"
        for i, movement in enumerate(movement_history, 1):
            movement_history_text += f"{i}. Rotation: {movement['rotation']:.1f}°, Forward: {movement['forward']:.2f}m, Left: {movement['forward']:.2f}m, Left: {movement['left']:.2f}m, Z-delta: {movement['z_delta']:.2f}m\n"
        movement_history_text += "\nAvoid repeating these exact movements. Explore new areas.\n"

    # Question type header
    if is_numerical:
        question_type_header = """
**QUESTION TYPE: NUMERICAL ANSWER**
You must provide a numerical value as your answer (e.g., a distance in meters, a count of objects, a size measurement).
Do NOT select from multiple choice options. Provide only a number.
"""
    else:
        question_type_header = """
**QUESTION TYPE: MULTIPLE CHOICE**
You must select ONE option from the choices provided below (A, B, C, or D).
Your answer must be the letter corresponding to your chosen option.
"""

    instr = f"""You are exploring a 3D room scene to answer a navigation question. You control the camera by specifying movements.

# Current camera position: {t_rounded} (x, y, z in meters)
# This is exploration step {step_num}

{bbox_text}
---
{question_type_header}
Question: {question}
{options_text}
{starting_place_note}
{important_note}

---
**DETAILED REASONING INSTRUCTIONS:**

1. **Analyze the current view thoroughly:**
   - What objects and areas are visible?
   - What areas are hidden or blocked?
   - What would you need to see to better answer the question?

2. **Plan your camera movement:**
   - Decide how much to move forward/backward
   - Decide how much to move left/right
   - Think about what rotation might help

3. **Specify the movement:**
   - `rotation_angle_degrees`: positive = turn left, negative = turn right (e.g., 15, -30, 90)
   - `forward_meters`: positive = move forward, negative = move backward. Range: -0.5 to +0.5
   - `left_meters`: positive = strafe left, negative = strafe right. Range: -0.5 to +0.5
   - `z_delta_meters`: move up (+) or down (-). Range: -0.3 to +0.3

{movement_instruction}

---
**OUTPUT FORMAT:**

First, write your detailed reasoning (3-4 sentences explaining what you see and why you're moving).

Then end with a JSON object:
{{
  "rotation_angle_degrees": <number>,
  "forward_meters": <number between -0.5 and 0.5>,
  "left_meters": <number between -0.5 and 0.5>,
  "z_delta_meters": <number between -0.3 and 0.3>,
  "answer": {answer_example} ONLY if done=true, otherwise null,
  "done": true or false
}}

CRITICAL ANSWER FORMAT:
- If done=false: answer MUST be null (still exploring)
- If done=true and NUMERICAL question: answer MUST be a NUMBER (e.g., 2.5, 3, 10.8)
- If done=true and MULTIPLE CHOICE question: answer MUST be a LETTER (A, B, C, or D)
- Answer format: {answer_format}

{movement_history_text}
"""
    return instr

_JSON_OBJ_RE = re.compile(r"(\{[\s\S]*?\})", re.DOTALL)

def extract_first_json(text):
    """Extract JSON object from text."""
    m = _JSON_OBJ_RE.search(text)
    if m:
        s = m.group(1)
        try:
            return json.loads(s)
        except Exception:
            pass
    return None

def parse_rotation_angle(angle_degrees, R_current):
    """Apply rotation angle around z-axis."""
    try:
        angle_rad = float(angle_degrees) * np.pi / 180.0
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
        R_new = Rz @ np.array(R_current, dtype=float)
        return R_new
    except Exception as e:
        print(f"[WARN] Failed to apply rotation: {e}")
        return np.array(R_current, dtype=float)

def apply_movement_in_camera_frame(R_current, t_current, forward_m, left_m, z_delta_m):
    """Apply movement relative to camera's current frame."""
    try:
        R = np.array(R_current, dtype=float)
        t = np.array(t_current, dtype=float).reshape(3,)
        
        right_axis = R[:, 0]
        up_axis = R[:, 1]
        forward_axis = R[:, 2]
        
        movement = forward_m * forward_axis + left_m * right_axis
        movement[2] += z_delta_m
        
        t_new = t + movement
        return t_new
    except Exception as e:
        print(f"[WARN] Failed to apply movement: {e}")
        return np.array(t_current, dtype=float)

def parse_qwen_output_and_get_movement(output_text):
    """Parse JSON from output_text and extract movement commands."""
    obj = extract_first_json(output_text)
    if obj is None:
        return None, None, None, None, None, None, False

    reasoning = obj.get("reasoning") if isinstance(obj, dict) else None
    answer = obj.get("answer") if isinstance(obj, dict) else None
    done = obj.get("done", False) if isinstance(obj, dict) else False

    rotation_angle = None
    forward_m = None
    left_m = None
    z_delta_m = None
    
    if isinstance(obj, dict):
        if "rotation_angle_degrees" in obj:
            rotation_angle = float(obj["rotation_angle_degrees"])
        if "forward_meters" in obj:
            forward_m = float(obj["forward_meters"])
        if "left_meters" in obj:
            left_m = float(obj["left_meters"])
        if "z_delta_meters" in obj:
            z_delta_m = float(obj["z_delta_meters"])

    return rotation_angle, forward_m, left_m, z_delta_m, reasoning, obj, done

def calculate_mra(predicted, ground_truth):
    """
    Calculate Mean Relative Accuracy (MRA) for numerical predictions.
    
    MRA = (1/10) * Σ(θ∈C) 1[|ŷ - y|/y < 1 - θ]
    where C = {0.5, 0.55, 0.60, ..., 0.95}
    
    Args:
        predicted: Model's predicted value (float)
        ground_truth: Ground truth value (float)
    
    Returns:
        MRA score between 0 and 1
    """
    if ground_truth == 0:
        # Avoid division by zero
        return 1.0 if predicted == 0 else 0.0
    
    thresholds = [0.5 + 0.05 * i for i in range(10)]  # [0.5, 0.55, ..., 0.95]
    relative_error = abs(predicted - ground_truth) / abs(ground_truth)
    
    score = 0.0
    for theta in thresholds:
        if relative_error < (1 - theta):
            score += 1.0
    
    return score / 10.0

def render_birds_eye_view_with_path(mesh: o3d.geometry.TriangleMesh, camera_positions: list, out_path: Path, marker_size=0.15):
    """Render multiple views with camera positions marked as red spheres with green direction indicators."""
    # Filter out ceiling - remove top 15% of vertices by z-coordinate
    vertices = np.asarray(mesh.vertices)
    z_threshold = np.percentile(vertices[:, 2], 85)  # Keep bottom 85%
    vertex_mask = vertices[:, 2] < z_threshold
    
    # Create filtered mesh
    filtered_mesh = o3d.geometry.TriangleMesh()
    vertex_indices = np.where(vertex_mask)[0]
    vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(vertex_indices)}
    
    filtered_mesh.vertices = o3d.utility.Vector3dVector(vertices[vertex_mask])
    
    # Preserve vertex colors if they exist
    if mesh.has_vertex_colors():
        vertex_colors = np.asarray(mesh.vertex_colors)
        filtered_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors[vertex_mask])
    
    # Filter triangles that reference removed vertices
    triangles = np.asarray(mesh.triangles)
    valid_triangles = []
    for tri in triangles:
        if all(v in vertex_map for v in tri):
            valid_triangles.append([vertex_map[tri[0]], vertex_map[tri[1]], vertex_map[tri[2]]])
    
    if len(valid_triangles) > 0:
        filtered_mesh.triangles = o3d.utility.Vector3iVector(np.array(valid_triangles))
        filtered_mesh.compute_vertex_normals()
    
    # Get scene bounds for camera positioning
    filtered_vertices = np.asarray(filtered_mesh.vertices)
    center_x = (filtered_vertices[:, 0].min() + filtered_vertices[:, 0].max()) / 2.0
    center_y = (filtered_vertices[:, 1].min() + filtered_vertices[:, 1].max()) / 2.0
    z_max = filtered_vertices[:, 2].max()
    
    # Calculate scene extent for better zoom
    x_extent = filtered_vertices[:, 0].max() - filtered_vertices[:, 0].min()
    y_extent = filtered_vertices[:, 1].max() - filtered_vertices[:, 1].min()
    max_extent = max(x_extent, y_extent)
    
    # Position camera closer based on scene size for better framing
    bev_height = z_max + max_extent * 0.6  # Camera height based on scene extent
    
    # Define camera views with closer zoom - 4 BEV rotations + side + isometric
    views = [
        ("bev_0", np.array([center_x, center_y, bev_height]), np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])),
        ("bev_90", np.array([center_x, center_y, bev_height]), np.array([0.0, 0.0, -1.0]), np.array([-1.0, 0.0, 0.0])),
        ("bev_180", np.array([center_x, center_y, bev_height]), np.array([0.0, 0.0, -1.0]), np.array([0.0, -1.0, 0.0])),
        ("bev_270", np.array([center_x, center_y, bev_height]), np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0])),
        ("side", np.array([center_x + 3.5, center_y, z_max * 0.5]), np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        ("isometric", np.array([center_x + 2.5, center_y + 2.5, z_max + 1.5]), np.array([-0.5, -0.5, -0.7]), np.array([0.0, 0.0, 1.0])),
    ]
    
    # Use even higher resolution
    width, height = 2048, 1536
    
    # Create markers with direction indicators
    marker_geometries = []
    for i, pos in enumerate(camera_positions):
        # Red sphere for position
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=marker_size)
        sphere.translate(pos[:3, 3])
        sphere.compute_vertex_normals()
        marker_geometries.append((f"sphere_{i}", sphere, [1.0, 0.0, 0.0, 1.0]))
        
        # Green arrow/cylinder for direction (forward axis)
        forward_direction = pos[:3, 2]  # Camera's forward is Z-axis
        arrow_length = marker_size * 2.5
        arrow_start = pos[:3, 3]
        arrow_end = arrow_start + forward_direction * arrow_length
        
        # Create cylinder as direction indicator
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=marker_size*0.3, height=arrow_length)
        # Rotate and translate cylinder to point in camera direction
        cylinder_direction = arrow_end - arrow_start
        cylinder_center = (arrow_start + arrow_end) / 2
        
        # Align cylinder with direction
        default_dir = np.array([0, 0, 1])
        rot_axis = np.cross(default_dir, cylinder_direction)
        if np.linalg.norm(rot_axis) > 1e-6:
            rot_axis = rot_axis / np.linalg.norm(rot_axis)
            angle = np.arccos(np.dot(default_dir, cylinder_direction / np.linalg.norm(cylinder_direction)))
            R_align = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * angle)
            cylinder.rotate(R_align, center=[0, 0, 0])
        cylinder.translate(cylinder_center)
        cylinder.compute_vertex_normals()
        marker_geometries.append((f"direction_{i}", cylinder, [0.0, 1.0, 0.0, 1.0]))
    
    for view_name, eye, forward, up in views:
        print(f"[INFO] 🎨 Rendering {view_name} view...")
        
        # Create fresh renderer for each view
        renderer = rendering.OffscreenRenderer(width, height)
        renderer.scene.view.set_post_processing(False)
        
        # Add filtered mesh with original colors
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        renderer.scene.add_geometry("mesh", filtered_mesh, mat, True)
        
        # Add markers with direction indicators
        for geom_name, geom, color in marker_geometries:
            marker_mat = rendering.MaterialRecord()
            marker_mat.shader = "defaultUnlit"
            marker_mat.base_color = color
            renderer.scene.add_geometry(f"{geom_name}_{view_name}", geom, marker_mat, True)
        
        # Setup camera with closer zoom
        camera_pose = look_at_camera_pose_center_from_forward(eye, forward=forward / np.linalg.norm(forward), up=up)
        cx = width / 2.0
        cy = height / 2.0
        
        # Adjust FOV based on view type for better framing
        if "bev" in view_name:
            # For BEV, use higher focal length (more zoom) based on scene extent
            focal_length = width / (max_extent * 1.2)  # Fit scene with small margin
            intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, focal_length, focal_length, cx, cy)
        else:
            fov = 280.0
            intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fov, fov, cx, cy)
        
        extrinsic_world_to_cam = np.linalg.inv(camera_pose)
        renderer.setup_camera(intrinsic, extrinsic_world_to_cam)
        
        renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
        
        try:
            img = renderer.render_to_image()
            arr = np.asarray(img)
            
            # Save with view name
            view_path = out_path.parent / f"{out_path.stem}_{view_name}.png"
            Image.fromarray(arr).save(str(view_path))
            print(f"[INFO] 🗺️  Saved {view_name} view: {view_path}")
        except Exception as e:
            print(f"[WARN] Failed to render {view_name}: {e}")
        finally:
            # Clean up renderer
            del renderer
    
    print(f"[INFO] ✅ Completed rendering all views with {len(camera_positions)} red markers")

# --------------- Qwen model init ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[INFO] Loading Qwen3 model on device: {device}")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype="auto", device_map="auto", cache_dir=CACHE_DIR
)
processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
processor.tokenizer.padding_side = 'left'
model.to(device)
print("[INFO] Qwen3 model loaded.")

def load_vsi_bench_questions():
    """Load VSI-Bench questions filtered by arkitscenes with numerical and MCA question types."""
    print("[INFO] 📥 Loading VSI-Bench dataset...")
    vsi = load_dataset("nyu-visionx/VSI-Bench", split="test")
    print(f"[INFO] Total VSI-Bench rows: {len(vsi)}")
    
    # Question types: numerical (NA) and multiple choice (MCA)
    numerical_types = ["object_size_estimation", "room_size_estimation", "object_counting", "object_abs_distance"]
    mca_types = ["object_rel_distance"]
    all_types = numerical_types + mca_types
    
    filtered = vsi.filter(
        lambda x: x["dataset"] == "arkitscenes"
                  and x["question_type"] in all_types
    )
    print(f"[INFO] ✅ Filtered to {len(filtered)} questions")
    print(f"[INFO]    - Numerical (NA): {len([x for x in filtered if x['question_type'] in numerical_types])} questions")
    print(f"[INFO]    - Multiple Choice (MCA): {len([x for x in filtered if x['question_type'] in mca_types])} questions")
    
    questions = []
    numerical_types = ["object_size_estimation", "room_size_estimation", "object_counting", "object_abs_distance"]
    for row in filtered:
        q_type = row.get("question_type", "unknown")
        is_numerical = q_type in numerical_types
        questions.append({
            "scene_name": row["scene_name"],
            "question": row["question"],
            "choices": row.get("options", []),
            "answer_id": row.get("ground_truth", -1),
            "question_type": q_type,
            "is_numerical": is_numerical,
        })
    
    return questions

def run_single_question(mesh_path, question, choices, question_id, experiment_base_dir, scene_id, num_steps=NUM_STEPS, question_type="unknown", is_numerical=False):
    """Run reasoning pipeline for a single question - fully sequential."""
    start_time = time.time()
    
    base_out = Path(experiment_base_dir) / f"q{question_id:03d}"
    base_out.mkdir(parents=True, exist_ok=True)
    print(f"\n[Q{question_id:03d}] 📁 Outputs -> {base_out.resolve()}")

    # Load mesh
    print(f"[Q{question_id:03d}] 📂 Loading mesh: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise RuntimeError("Loaded mesh is empty")

    vertices = np.asarray(mesh.vertices)
    bbox_mins = vertices.min(axis=0).tolist()
    bbox_maxs = vertices.max(axis=0).tolist()

    # Generate 4 candidate views and select best
    center_x = (vertices[:, 0].min() + vertices[:, 0].max()) / 2.0
    center_y = (vertices[:, 1].min() + vertices[:, 1].max()) / 2.0
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
    cam_height_z = z_min + CAM_HEIGHT
    eye = np.array([center_x, center_y, cam_height_z], dtype=float)

    view_images = {}
    view_poses = {}
    for angle_deg in [0, 90, 180, 270]:
        angle_rad = np.deg2rad(angle_deg)
        forward = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0], dtype=float)
        pose = look_at_camera_pose_center_from_forward(eye, forward=forward, up=np.array([0.0, 0.0, -1.0]))
        view_poses[angle_deg] = pose

        img_path = base_out / f"render_candidate_{angle_deg}.png"
        render_mesh_from_pose(mesh, pose, img_path, fxfy=DEFAULT_FX_FY)

        img_pil = Image.open(img_path)
        img_array = np.array(img_pil).astype(float) / 255.0
        view_images[angle_deg] = img_array

    best_angle, best_score, all_scores = select_best_initial_view(view_images, metric=INITIAL_VIEW_SELECTION_METRIC)

    scores_record = {
        "metric": INITIAL_VIEW_SELECTION_METRIC,
        "all_scores": {str(k): float(v) for k, v in all_scores.items()},
        "best_angle": best_angle,
        "best_score": float(best_score)
    }
    with open(base_out / "initial_view_selection.json", "w") as f:
        json.dump(scores_record, f, indent=2)

    cam_pose = view_poses[best_angle]
    save_matrix(base_out / "cam_pose_00.npy", cam_pose)
    img0 = base_out / "render_00.png"
    Image.fromarray((view_images[best_angle] * 255).astype(np.uint8)).save(str(img0))

    image_history = [str(img0)]
    cam_history = [cam_pose.copy()]
    R_current = cam_pose[:3,:3]
    t_current = cam_pose[:3,3]
    position_history = []
    final_answer = None

    # Sequential reasoning loop
    print(f"[Q{question_id:03d}] 🤖 Starting reasoning loop (max {num_steps} steps)...")
    for step in range(0, num_steps+1):
        print(f"\n[Q{question_id:03d}] [Step {step:02d}] " + "─" * 40)
        is_final_step = (step == num_steps)

        instruction_text = build_instruction_text(
            R_current, t_current, question,
            bbox=(bbox_mins, bbox_maxs),
            options=choices,
            is_final_step=is_final_step,
            movement_history=position_history,
            step_num=step,
            question_type=question_type,
            is_numerical=is_numerical
        )

        history_context = "## Image History (numbered for reference):\n"
        for hist_step, hist_t in enumerate(cam_history):
            history_context += f"Image {hist_step}: Position {np.round(hist_t[:3,3], 2).tolist()}\n"
        history_context += "\nAbove are all the images you have seen so far.\n\n"

        full_prompt = history_context + instruction_text

        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": full_prompt}]
        }]

        for img_path in image_history:
            messages[0]["content"].insert(len(messages[0]["content"]) - 1, 
                                         {"type": "image", "image": img_path})

        step_folder = base_out / f"step_{step:02d}"
        step_folder.mkdir(parents=True, exist_ok=True)

        # Save input prompt for debugging
        with open(step_folder / "qwen_input_prompt.txt", "w", encoding="utf-8") as f:
            f.write(full_prompt)
        with open(step_folder / "qwen_input_messages.json", "w", encoding="utf-8") as f:
            # Save messages structure (without actual image data, just paths)
            messages_debug = []
            for msg in messages:
                msg_copy = {"role": msg["role"], "content": []}
                for item in msg["content"]:
                    if item["type"] == "text":
                        msg_copy["content"].append({"type": "text", "text": item["text"]})
                    elif item["type"] == "image":
                        msg_copy["content"].append({"type": "image", "path": item["image"]})
                messages_debug.append(msg_copy)
            json.dump(messages_debug, f, indent=2)

        # Apply chat template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Sequential inference
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)

        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)]
        output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        output_text = output_texts[0]

        with open(step_folder / "qwen_raw_output.txt", "w", encoding="utf-8") as f:
            f.write(output_text)

        # Parse output
        rotation_angle, forward_m, left_m, z_delta_m, reasoning_text, raw_obj, done_flag = parse_qwen_output_and_get_movement(output_text)

        # Check for answer
        if raw_obj and isinstance(raw_obj, dict):
            answer_value = raw_obj.get("answer")
            has_answer = answer_value is not None and str(answer_value).strip().upper() in "ABCDEFGHIJ"
            
            if has_answer and done_flag:
                final_answer = str(answer_value).strip().upper()
                print(f"[Q{question_id:03d}] ✅ Final answer: {final_answer}")

        # If we got an answer and done flag, or if this is the final step, don't render another image
        if (done_flag and final_answer) or is_final_step:
            if done_flag and final_answer:
                print(f"[Q{question_id:03d}] 🏁 Model signaled done=true, breaking early at step {step}")
            else:
                print(f"[Q{question_id:03d}] 🏁 Reached final step {step}, ending exploration")
            break

        # Apply movement
        if rotation_angle is not None and forward_m is not None and left_m is not None and z_delta_m is not None:
            R_new = parse_rotation_angle(rotation_angle, R_current)
            t_new = apply_movement_in_camera_frame(R_current, t_current, forward_m, left_m, z_delta_m)
            next_pose = np.eye(4, dtype=float)
            next_pose[:3, :3] = R_new
            next_pose[:3, 3] = t_new
            
            position_history.append({
                "rotation": rotation_angle,
                "forward": forward_m,
                "left": left_m,
                "z_delta": z_delta_m
            })
        else:
            # Default movement if parsing fails
            last = cam_history[-1]
            angle = 10.0 * np.pi / 180.0
            Rz = np.array([[np.cos(angle), -np.sin(angle), 0.0], 
                          [np.sin(angle), np.cos(angle), 0.0], 
                          [0.0, 0.0, 1.0]])
            next_pose = last.copy()
            next_pose[:3, :3] = Rz @ next_pose[:3, :3]

        # Render next image
        img_next = base_out / f"render_{step+1:02d}.png"
        render_mesh_from_pose(mesh, next_pose, img_next, fxfy=DEFAULT_FX_FY)

        image_history.append(str(img_next))
        cam_history.append(next_pose)
        R_current = next_pose[:3, :3]
        t_current = next_pose[:3, 3]

    elapsed_time = time.time() - start_time
    print(f"\n[Q{question_id:03d}] ⏱️  Completed in {elapsed_time:.2f}s")
    print(f"[Q{question_id:03d}] Final answer: {final_answer}")

    # Render bird's eye view with path
    birds_eye_path = base_out / "birds_eye_view_path.png"
    try:
        render_birds_eye_view_with_path(mesh, cam_history, birds_eye_path)
    except Exception as e:
        print(f"[WARN] Failed to render bird's eye view: {e}")

    return final_answer, elapsed_time, len(image_history) - 1

def main_sequential_split(mesh_base_dir=MESH_BASE_DIR, num_steps_per_question=NUM_STEPS, 
                          split=1, num_splits=1, continue_from=None, test_mode=False, max_questions=None):
    """
    Main loop - fully sequential with job splitting support.
    
    Args:
        split: Which split to run (1-indexed)
        num_splits: Total number of splits
        test_mode: If True, use "test" folder and limit to 1 question
        max_questions: Maximum number of questions to process
    """
    print("\n" + "=" * 80)
    print("🚀 VSI-BENCH ROUTE PLANNING EVALUATION (SEQUENTIAL WITH SPLITS)")
    if test_mode:
        print("   🧪 TEST MODE - Running 1 question to 'test' folder")
    else:
        print(f"   Running split {split}/{num_splits}")
    print("=" * 80 + "\n")

    questions = load_vsi_bench_questions()
    total_questions = len(questions)
    
    # Calculate split ranges
    questions_per_split = total_questions // num_splits
    remainder = total_questions % num_splits
    
    # Distribute remainder across first splits
    split_sizes = [questions_per_split + (1 if i < remainder else 0) for i in range(num_splits)]
    
    # Calculate start and end indices for this split
    start_idx = sum(split_sizes[:split-1])
    end_idx = start_idx + split_sizes[split-1]
    
    print(f"[INFO] Total questions: {total_questions}")
    print(f"[INFO] Split {split}/{num_splits}: questions {start_idx+1} to {end_idx} (count: {end_idx - start_idx})")
    
    # Filter to this split's questions
    split_questions = questions[start_idx:end_idx]
    
    # Apply max_questions limit if specified
    if max_questions is not None:
        split_questions = split_questions[:max_questions]
        print(f"[INFO] Limited to {max_questions} question(s)\n")
    
    # Determine experiment directory
    if test_mode:
        exp_base_dir = Path("test")
        exp_base_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] 🧪 Test mode: {exp_base_dir.resolve()}\n")
    elif continue_from:
        exp_base_dir = Path(continue_from)
        print(f"[INFO] Resuming from: {exp_base_dir.resolve()}\n")
    else:
        exp_timestamp = timestamp_str()
        exp_base_dir = Path("experiment_logs") / f"{exp_timestamp}_sequential_split{split}of{num_splits}"
        exp_base_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] 📁 Experiment logs: {exp_base_dir.resolve()}\n")

    results_file = exp_base_dir / "results.json"
    csv_file = exp_base_dir / "results.csv"

    # Load existing results
    if continue_from and results_file.exists():
        with open(results_file, "r") as f:
            results = json.load(f)
        print(f"[INFO] Loaded {len(results)} existing results\n")
    else:
        results = []

    if continue_from and csv_file.exists():
        csv_df = pd.read_csv(csv_file)
        csv_rows = csv_df.to_dict('records')
        print(f"[INFO] Loaded {len(csv_rows)} existing CSV rows\n")
    else:
        csv_rows = []

    # Track completed questions
    completed_questions = set()
    for subfolder in exp_base_dir.iterdir():
        if subfolder.is_dir() and subfolder.name.startswith("q"):
            completed_questions.add(subfolder.name)

    # Process questions in this split
    processed_count = start_idx  # Track actual question number for folder naming
    
    for local_idx, q_data in enumerate(split_questions, 1):
        processed_count += 1
        scene_id = q_data["scene_name"]
        mesh_file = find_mesh_file(scene_id, mesh_base_dir)
        
        if mesh_file is None:
            print(f"[WARN] Question {processed_count} (local {local_idx}/{len(split_questions)}): No mesh for scene {scene_id}. Skipping.\n")
            continue
        
        question_folder = f"q{processed_count:03d}"
        
        if question_folder in completed_questions:
            print(f"[INFO] Question {processed_count} already completed. Skipping.\n")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing Question {processed_count} (split {split}, local {local_idx}/{len(split_questions)})")
        print(f"Scene: {scene_id}")
        print(f"{'='*80}")
        
        try:
            model_answer, elapsed_time, actual_steps = run_single_question(
                mesh_file,
                q_data["question"],
                q_data["choices"],
                processed_count,
                str(exp_base_dir),
                scene_id,
                num_steps=num_steps_per_question,
                question_type=q_data["question_type"],
                is_numerical=q_data["is_numerical"]
            )
        except Exception as e:
            print(f"[ERROR] Failed to process Q{processed_count}: {e}")
            continue
        
        # Handle None answer
        if model_answer is None:
            print(f"[WARN] Q{processed_count} never provided valid answer. Recording as 'NO_ANSWER'.")
            model_answer = "NO_ANSWER"
        
        # Save results
        run_timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
        ground_truth = q_data["answer_id"]
        
        # Evaluate based on question type
        is_numerical = q_data["is_numerical"]
        if is_numerical and model_answer != "NO_ANSWER":
            # Parse numerical answer
            try:
                predicted_value = float(model_answer)
                gt_value = float(ground_truth)
                mra_score = calculate_mra(predicted_value, gt_value)
                is_correct = (mra_score > 0.5)  # Consider correct if MRA > 0.5
                
                print(f"\n[Q{processed_count:03d}] Model Answer: {predicted_value}")
                print(f"[Q{processed_count:03d}] Ground Truth: {gt_value}")
                print(f"[Q{processed_count:03d}] MRA Score: {mra_score:.4f}")
                print(f"[Q{processed_count:03d}] Result: {'✅ CORRECT (MRA > 0.5)' if is_correct else '❌ INCORRECT (MRA ≤ 0.5)'}\n")
            except (ValueError, TypeError) as e:
                print(f"[ERROR] Failed to parse numerical answer: {model_answer}. Error: {e}")
                mra_score = 0.0
                is_correct = False
        else:
            # Multiple choice answer
            mra_score = None
            is_correct = (model_answer == ground_truth)
            
            print(f"\n[Q{processed_count:03d}] Model Answer: {model_answer}")
            print(f"[Q{processed_count:03d}] Ground Truth: {ground_truth}")
            print(f"[Q{processed_count:03d}] Result: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}\n")
        
        results.append({
            "scene_id": scene_id,
            "question": q_data["question"],
            "status": "COMPLETED",
            "model_answer": model_answer,
            "ground_truth": ground_truth,
            "correct": is_correct,
            "is_numerical": is_numerical,
            "mra_score": mra_score if is_numerical else None
        })
        
        csv_rows.append({
            "question_id": f"q{processed_count:03d}",
            "scene_id": scene_id,
            "question_type": q_data["question_type"],
            "is_numerical": is_numerical,
            "gt_answer": ground_truth,
            "model_answer": model_answer,
            "mra_score": mra_score if is_numerical else None,
            "time_seconds": elapsed_time,
            "num_steps": actual_steps,
            "timestamp": run_timestamp,
            "question": q_data["question"]
        })

        # Save after each question
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        csv_df = pd.DataFrame(csv_rows)
        csv_df.to_csv(csv_file, index=False)
        print(f"[INFO] 💾 Saved progress (Q{processed_count:03d} completed)")

        # Print running stats
        correct_so_far = sum(1 for r in results if r["correct"])
        total_so_far = len([r for r in results if r["status"] == "COMPLETED"])
        running_accuracy = (100 * correct_so_far / total_so_far) if total_so_far > 0 else 0
        print(f"[RUNNING] Split {split} accuracy: {correct_so_far}/{total_so_far} = {running_accuracy:.1f}%")
        print(f"[RUNNING] Split {split} progress: {local_idx}/{len(split_questions)}")
        print("=" * 80)

    # Final summary
    print("\n" + "=" * 80)
    print(f"📊 SUMMARY (Split {split}/{num_splits})")
    print("=" * 80)
    correct_count = sum(1 for r in results if r["correct"])
    total_count = len([r for r in results if r["status"] == "COMPLETED"])
    print(f"Accuracy: {correct_count}/{total_count} = {100*correct_count/total_count:.1f}%\n")
    
    print(f"[INFO] 🏁 Results saved to: {results_file}")
    print(f"[INFO] 🏁 CSV saved to: {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VSI-Bench reasoning loop (sequential, with job splitting)")
    parser.add_argument("--steps", type=int, default=NUM_STEPS, help="Number of reasoning steps per question")
    parser.add_argument("--split", type=int, default=1, help="Which split to run (1-indexed)")
    parser.add_argument("--num-splits", type=int, default=1, help="Total number of splits")
    parser.add_argument("--continue", dest="continue_from", default=None, 
                       help="Resume from experiment folder and skip completed questions")
    parser.add_argument("--test", action="store_true", help="Test mode: run 1 question to 'test' folder")
    parser.add_argument("--max-questions", type=int, default=None, help="Maximum number of questions to process")
    args = parser.parse_args()

    if not args.test and (args.split < 1 or args.split > args.num_splits):
        print(f"[ERROR] Invalid split: {args.split}. Must be between 1 and {args.num_splits}")
        exit(1)

    if args.test:
        print(f"[INFO] Running in test mode (5 questions)")
        main_sequential_split(
            num_steps_per_question=args.steps,
            split=1,
            num_splits=1,
            continue_from=None,
            test_mode=True,
            max_questions=5
        )
    else:
        print(f"[INFO] Running split {args.split} of {args.num_splits}")
        main_sequential_split(
            num_steps_per_question=args.steps,
            split=args.split,
            num_splits=args.num_splits,
            continue_from=args.continue_from,
            test_mode=False,
            max_questions=args.max_questions
        )
