#!/usr/bin/env python3
"""
Full pipeline:
 - Load VSI-Bench questions (arkitscenes + route_planning)
 - For each question, find matching PLY file
 - Qwen3-VL multimodal reasoning loop (images + movement commands)
 - Collect answers and evaluate against ground truth
"""

import argparse
import json
import re
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
import torch
import open3d as o3d
from open3d.visualization import rendering
from datasets import load_dataset
import time

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info  # assumed available in your environment

# ----------------- Config -----------------
CACHE_DIR = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MESH_BASE_DIR = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/raw"
ARKIT_CSV_PATH = "/dss/dsshome1/06/di38riq/ARKitScenes/raw/raw_train_val_splits.csv"
METADATA_CSV_PATH = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/raw/metadata.csv"

import pandas as pd
import cv2

# NUM_STEPS = 5
NUM_STEPS = 10  # Max iterations, but model can terminate early with "done": true
IMAGE_WH = (1024, 768)
DEFAULT_FX_FY = 300.0   # wider FOV
CAM_HEIGHT = 1.6        # meters above floor (heuristic)
MAX_ATTEMPTS_PER_STEP = 3
COND_THRESHOLD = 1e12   # threshold for condition number marking near-singular

# Initial view selection configuration
INITIAL_VIEW_SELECTION_METRIC = "visibility"  # "visibility" or "laplacian"
# visibility: picks view with most mesh geometry visible (least occlusion)
# laplacian: picks view with highest edge density/sharpness

# Sky direction mapping (loaded from metadata CSV)
_METADATA_CACHE = None

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

def get_metadata_df():
    """Load metadata CSV with caching."""
    global _METADATA_CACHE
    if _METADATA_CACHE is None:
        _METADATA_CACHE = pd.read_csv(METADATA_CSV_PATH)
    return _METADATA_CACHE

def sky_direction_to_up_vector(sky_direction):
    """
    Convert sky direction string to camera up vector.
    - "Up": camera sees sky above (normal orientation) -> up = [0, 0, 1]
    - "Down": camera points down (device upside down) -> up = [0, 0, -1]
    - "Left": device tilted left -> up = [-1, 0, 0]
    - "Right": device tilted right -> up = [1, 0, 0]
    - "NA" or None: default to up = [0, 0, 1]
    """
    direction_map = {
        "Up": np.array([0.0, 0.0, 1.0], dtype=float),
        "Down": np.array([0.0, 0.0, -1.0], dtype=float),
        "Left": np.array([-1.0, 0.0, 0.0], dtype=float),
        "Right": np.array([1.0, 0.0, 0.0], dtype=float),
    }
    
    sky_dir_str = str(sky_direction).strip() if sky_direction is not None else "NA"
    up_vector = direction_map.get(sky_dir_str, np.array([0.0, 0.0, 1.0], dtype=float))
    return up_vector

def get_sky_direction_for_scene(scene_id):
    """
    Look up the sky direction for a given scene_id from metadata CSV.
    Returns the sky_direction string ("Up", "Down", "Left", "Right", or "NA").
    """
    try:
        df = get_metadata_df()
        row = df[df['video_id'] == int(scene_id)]
        if len(row) > 0:
            sky_dir = row.iloc[0]['sky_direction']
            return sky_dir
    except Exception as e:
        print(f"[WARN] Error looking up sky direction for scene {scene_id}: {e}")
    return "NA"

# ----------------- Utilities -----------------
def timestamp_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def look_at_camera_pose_center_from_forward(eye, forward=np.array([1.0,0.0,0.0]), up=np.array([0,0,-1])):
    """
    Construct camera-to-world 4x4 matrix at `eye` oriented along `forward` with up `up`.
    Note: Default up=[0,0,-1] gives correct orientation for Open3D rendering.
    """
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
        # pick orthogonal up if degenerate
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

def compute_initial_camera_pose(mesh, cam_height=None, up_vector=None):
    """
    Simple and robust initial camera placement:
     - Place camera at center of mesh in X,Y
     - Use half the Z-range as camera height (middle of the scene)
     - Look directly forward (along positive X-axis)
     - Respect the device's sky direction (up_vector orientation)
    
    Args:
        mesh: Open3D TriangleMesh
        cam_height: optional fixed camera height, else use middle of z-range
        up_vector: camera up direction (e.g., [0,0,1] for normal, [0,0,-1] for upside down)
    """
    if up_vector is None:
        up_vector = np.array([0.0, 0.0, 1.0], dtype=float)
    
    vertices = np.asarray(mesh.vertices)
    if vertices.size == 0:
        raise ValueError("Empty mesh")

    # Get bounding box limits
    x_min, y_min, z_min = vertices.min(axis=0)
    x_max, y_max, z_max = vertices.max(axis=0)
    
    # Place camera at center XY
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    
    # Use half the Z-range as camera height
    if cam_height is None:
        cam_height = (z_min + z_max) / 2.0
    else:
        cam_height = z_min + cam_height
    
    eye = np.array([center_x, center_y, cam_height], dtype=float)
    
    # Look directly forward (along positive X-axis)
    forward = np.array([1.0, 0.0, 0.0], dtype=float)
    
    pose = look_at_camera_pose_center_from_forward(eye, forward=forward, up=up_vector)
    
    print(f"[INFO] 📐 Z-range: [{z_min:.2f}, {z_max:.2f}], cam_height: {cam_height:.2f}")
    print(f"[INFO] 📐 Camera at center: ({center_x:.2f}, {center_y:.2f}, {cam_height:.2f}), looking forward")
    print(f"[INFO] 📐 Up vector: {np.round(up_vector, 2)}")
    
    return pose

def save_matrix(path: Path, mat: np.ndarray, text=True):
    np.save(str(path), mat)
    if text:
        with open(path.with_suffix(".txt"), "w") as f:
            f.write(np.array2string(mat, precision=2, separator=', '))

def render_mesh_from_pose(mesh: o3d.geometry.TriangleMesh, cam_pose_world: np.ndarray, out_path_img: Path, fxfy=DEFAULT_FX_FY):
    """
    Headless render of a mesh using OffscreenRenderer.
    """
    width, height = IMAGE_WH
    renderer = rendering.OffscreenRenderer(width, height)
    renderer.scene.view.set_post_processing(False)

    # Create a material for the mesh with lit shader
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"      # <-- CRITICAL: forces nearest sampling in legacy renderer
    mat.base_color = [1, 1, 1, 1]
    renderer.scene.clear_geometry()
    renderer.scene.add_geometry("mesh", mesh, mat, True)

    # Setup lighting for better visualization
    #renderer.scene.set_lighting(rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, (0.5, 0.5, 0.5))

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

# --------------- Qwen model init ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[INFO] Loading Qwen3 model on device: {device} (this may take a while)")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype="auto", device_map="auto", cache_dir=CACHE_DIR
)
processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
model.to(device)
print("[INFO] Qwen3 model loaded.")

# --------------- Qwen interaction and parsing --------------
def build_instruction_text(R, t, question, bbox=None, options=None, is_final_step=False, movement_history=None, step_num=0):
    R_rounded = np.round(R, 2).tolist()
    t_rounded = np.round(t, 2).tolist()
    instr = build_instruction_natural(R_rounded, t_rounded, question, bbox=bbox, options=options, is_final_step=is_final_step, step_num=step_num)

    # Add movement history to the instruction with clear numbering
    movement_history_text = ""
    if movement_history:
        movement_history_text = "\n---\nYour previous movements and actions (IMPORTANT: Do not repeat these movements exactly - vary your exploration strategy):\n"
        for i, movement in enumerate(movement_history, 1):
            movement_history_text += f"  Step {i}: Moved forward={movement['forward']:.2f}m, left={movement['left']:.2f}m, up={movement['z_delta']:.2f}m, rotated={movement['rotation']:.1f}°\n"
        movement_history_text += "\nAvoid repeating these exact movements. Explore new areas of the scene.\n"

    return instr + movement_history_text

def find_mesh_file(scene_id, mesh_base_dir=MESH_BASE_DIR):
    """
    Find a mesh file for the given scene_id (video_id).
    Looks in {mesh_base_dir}/[Validation|Training]/{video_id}/{video_id}_3dod_mesh.ply
    Returns the mesh file path, or None if not found.
    """
    video_id = str(scene_id)
    
    # Try both Validation and Training splits
    for split in ["Validation", "Training"]:
        mesh_path = Path(mesh_base_dir) / split / video_id / f"{video_id}_3dod_mesh.ply"
        if mesh_path.exists():
            print(f"[INFO] ✅ Found mesh (scene {scene_id}, split {split}): {mesh_path}")
            return mesh_path
    
    print(f"[WARN] Mesh file not found for scene {scene_id} in {mesh_base_dir}")
    return None


def build_instruction_natural(R_rounded, t_rounded, question, bbox=None, options=None, is_final_step=False, step_num=0):
    bbox_text = ""
    if bbox is not None:
        try:
            mins, maxs = bbox
            # Limit to 2 significant figures
            mins_2sf = [float(f"{x:.2g}") for x in mins]
            maxs_2sf = [float(f"{x:.2g}") for x in maxs]
            bbox_text = (
                "Scene bounding box limits (meters):\n"
                f"  x: [{mins_2sf[0]}, {maxs_2sf[0]}]\n"
                f"  y: [{mins_2sf[1]}, {maxs_2sf[1]}]\n"
                f"  z: [{mins_2sf[2]}, {maxs_2sf[2]}]\n\n"
            )
        except Exception:
            bbox_text = ""

    # Format answer options if provided
    options_text = ""
    if options and isinstance(options, (list, dict)):
        options_text = "\n**Answer Options:**\n"
        if isinstance(options, list):
            for i, opt in enumerate(options):
                options_text += f"  {chr(65+i)}) {opt}\n"
        elif isinstance(options, dict):
            for key, val in options.items():
                options_text += f"  {key}) {val}\n"
        options_text += "\n"

    # Conditional instruction text based on whether this is the final step
    if is_final_step:
        movement_instruction = """
4. **THIS IS YOUR FINAL STEP:** You have now explored the scene thoroughly. You MUST provide your final answer choice (A, B, C, or D).
   - Prioritize outputting your best answer.
   - You may provide minimal camera movement (e.g., small adjustments) or zero movement.
   - The critical requirement is that you output your final answer in the JSON response.
   - Set "done": true ONLY when you are providing a final answer.
"""
        important_note = "IMPORTANT: This is your final opportunity to answer. You MUST output your best answer choice (A/B/C/D) in the JSON and set done=true."
    else:
        movement_instruction = """
4. **Do NOT stay in the same location.** Always provide at least one non-zero movement.

5. **Answering too early is a mistake.** ONLY output an answer (A, B, C, or D) when you have thoroughly explored the scene and are CERTAIN of your answer. 
   - Most of the time, you should NOT provide an answer yet - just explore and set "done": false.
   - Only when you are completely confident should you set both "answer": "X" AND "done": true together.
   - Never output an answer while still exploring (done=false). Either explore (no answer) OR finalize (answer + done=true).

6. **When you are confident:** If you have seen enough viewpoints and are absolutely certain of your answer, set "done": true AND provide your answer.
"""
        important_note = "IMPORTANT: Keep exploring and DO NOT output an answer unless you are ready to stop (done=true). Answer and done=true should come together, not separately."

    # Replace the first sentence of the question with a coherent starting point
    question = question.replace("You are a robot beginning at", "If you begin navigating at")

    # Add a note for Qwen to find the starting place first
    starting_place_note = "\n**Note:** Your first task is to identify the starting place mentioned in the question.\n"

    instr = f"""
            You are given rendered views of a 3D room scene. You control the camera by specifying how to move it. 
            Your goal is to find the answer to the question below by exploring the scene with camera movements.

            {bbox_text}

            # Current camera position: {t_rounded} (x, y, z in meters)
            # This is exploration step {step_num}

            Question: {question}
            {options_text}

            {starting_place_note}
            {important_note}

            ---
            DETAILED REASONING INSTRUCTIONS:

            1. **Analyze the current view thoroughly:**
            - What objects and areas are visible?
            - What areas are hidden or blocked from the current viewpoint?
            - What would you need to see to better answer the question?

            2. **Plan your camera movement:**
            - Decide how much to move forward/backward (toward/away from what you're looking at)
            - Decide how much to move left/right (perpendicular to where you're looking)
            - Think about what rotation (turning left/right) might help
            - Write out your reasoning in plain English before giving the JSON.

            3. **Specify the movement:**
            - `rotation_angle_degrees`: positive = turn left, negative = turn right (e.g., 15, -30, 90)
            - `forward_meters`: positive = move forward (in the direction you're facing), negative = move backward. Range: -0.5 to +0.5 meters
            - `left_meters`: positive = strafe left, negative = strafe right. Range: -0.5 to +0.5 meters
            - `z_delta_meters`: move up (+) or down (-) to change height. Range: -0.3 to +0.3 meters

            {movement_instruction}

            ---
            FORMAT:

            First, write your detailed reasoning (3-4 sentences explaining what you see and why you're moving the camera, or why you're ready to finalize your answer).

            Then end with a JSON object:
            {{
            "rotation_angle_degrees": <number, e.g., 15 or -30>,
            "forward_meters": <number between -0.5 and 0.5>,
            "left_meters": <number between -0.5 and 0.5>,
            "z_delta_meters": <number between -0.3 and 0.3>,
            "answer": "your final answer choice (A, B, C, or D) ONLY if done=true, otherwise null",
            "done": true or false
            }}

            CRITICAL: "answer" and "done" should be consistent:
            - If done=false: answer MUST be null (you are still exploring)
            - If done=true: answer MUST be A, B, C, or D (you have decided to finalize)
            """
    return instr



_JSON_OBJ_RE = re.compile(r"(\{[\s\S]*?\})", re.DOTALL)
_JSON_ARRAY_RE = re.compile(r"(\[[\s\S]*?\])", re.DOTALL)

def extract_first_json(text):
    """
    Try to extract a JSON object or array from arbitrary model text.
    Returns parsed Python object or None.
    """
    # try object first
    m = _JSON_OBJ_RE.search(text)
    if m:
        s = m.group(1)
        try:
            return json.loads(s)
        except Exception:
            pass
    # try array (e.g. [ { ... } ])
    m = _JSON_ARRAY_RE.search(text)
    if m:
        s = m.group(1)
        try:
            parsed = json.loads(s)
            # if array, return first element if it's an object
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed[0]
            return parsed
        except Exception:
            pass
    return None

def validate_rotation_matrix(R):
    """
    Check if R is 3x3, numeric, orthonormal-ish, and det close to +1.
    If R is close but not quite valid, attempt to fix it via SVD projection.
    Return (valid:bool, reason:str, R_corrected:ndarray or None)
    """
    try:
        R = np.array(R, dtype=float)
        if R.shape != (3,3):
            return False, f"shape {R.shape} != (3,3)", None
        
        # Check orthonormality
        RtR = R.T @ R
        err = np.linalg.norm(RtR - np.eye(3))
        det = np.linalg.det(R)
        
        # If orthonormal and det ~1, already valid
        if err < 1e-2 and (0.9 < det < 1.1):
            return True, "", None
        
        # If very close, try to fix via SVD
        if err < 0.05:  # relaxed threshold for attempting fix
            U, S, Vt = np.linalg.svd(R)
            R_fixed = U @ Vt
            # Ensure det = +1 (not -1)
            if np.linalg.det(R_fixed) < 0:
                Vt[-1, :] *= -1
                R_fixed = U @ Vt
            # Verify the corrected matrix is valid
            RtR_fixed = R_fixed.T @ R_fixed
            err_fixed = np.linalg.norm(RtR_fixed - np.eye(3))
            det_fixed = np.linalg.det(R_fixed)
            if err_fixed < 1e-2 and (0.9 < det_fixed < 1.1):
                return True, f"Fixed via SVD (was err={err:.4f}, det={det:.4f})", R_fixed
            else:
                return False, f"SVD correction failed (err_fixed={err_fixed:.4f}, det_fixed={det_fixed:.4f})", None
        
        # Not fixable
        if err > 1e-2:
            return False, f"R^T R error {err:.4f}", None
        if not (0.9 < det < 1.1):
            return False, f"determinant {det:.4f} not ~1", None
        return False, "unknown", None
    except Exception as e:
        return False, f"exception {e}", None

def validate_translation_vector(t):
    try:
        t = np.array(t, dtype=float).reshape(3,)
        if not np.isfinite(t).all():
            return False, "non-finite"
        # no further checks, but could add bounds if desired
        return True, ""
    except Exception as e:
        return False, f"exception {e}"

def parse_rotation_angle(angle_degrees, R_current):
    """
    Apply a rotation angle (in degrees, around z-axis) to the current rotation matrix.
    Simulates a human standing in the room: if R_current is their orientation and they turn left/right,
    the rotation is applied in world frame: R_new = Rz(angle) @ R_current.
    Positive angle = counterclockwise/left, negative = clockwise/right.
    Returns the updated rotation matrix.
    """
    try:
        angle_rad = float(angle_degrees) * np.pi / 180.0
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
        # World-frame rotation: camera turns in world space (like a person turning their head)
        R_new = Rz @ np.array(R_current, dtype=float)
        return R_new
    except Exception as e:
        print(f"[WARN] Failed to apply rotation angle: {e}")
        return np.array(R_current, dtype=float)  # Return unchanged if parsing fails

def apply_movement_in_camera_frame(R_current, t_current, forward_m, left_m, z_delta_m):
    """
    Apply movement relative to the camera's current frame.
    - forward_m: positive moves in the direction the camera is facing (Z-axis of camera frame)
    - left_m: positive moves left (X-axis of camera frame, perpendicular to forward)
    - z_delta_m: positive moves up in world space (Z-axis of world frame)
    
    Returns new t_3x1 position.
    """
    try:
        R = np.array(R_current, dtype=float)
        t = np.array(t_current, dtype=float).reshape(3,)
        
        # Camera frame axes (columns of R are the camera's local axes in world coordinates)
        # R = [right_axis | up_axis | forward_axis]
        right_axis = R[:, 0]    # camera's right direction in world coords
        up_axis = R[:, 1]       # camera's up direction in world coords
        forward_axis = R[:, 2]  # camera's forward direction in world coords
        
        # Movement in world frame
        movement = forward_m * forward_axis + left_m * right_axis
        # Add vertical movement (in world Z)
        movement[2] += z_delta_m
        
        t_new = t + movement
        return t_new
    except Exception as e:
        print(f"[WARN] Failed to apply movement: {e}")
        return np.array(t_current, dtype=float)

def parse_qwen_output_and_get_movement(output_text):
    """
    Parse JSON from output_text and extract movement commands: rotation_angle_degrees, forward_meters, left_meters, z_delta_meters, done flag.
    Returns (rotation_angle_degrees, forward_m, left_m, z_delta_m, reasoning, raw_json_obj, done)
    """
    obj = extract_first_json(output_text)
    if obj is None:
        return None, None, None, None, None, None, False

    # Reasoning string if present
    reasoning = obj.get("reasoning") if isinstance(obj, dict) else None
    answer = obj.get("answer") if isinstance(obj, dict) else None
    done = obj.get("done", False) if isinstance(obj, dict) else False

    # Try to extract movement parameters
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

def load_vsi_bench_questions():
    """
    Load VSI-Bench questions filtered by:
     - dataset == "arkitscenes"
     - question_type == "route_planning"
    Returns list of dicts with: scene_name, question, choices, answer_id
    """
    print("[INFO] 📥 Loading VSI-Bench dataset...")
    vsi = load_dataset("nyu-visionx/VSI-Bench", split="test")
    print(f"[INFO] Total VSI-Bench rows: {len(vsi)}")
    
    filtered = vsi.filter(
        lambda x: x["dataset"] == "arkitscenes"
                  and x["question_type"] == "route_planning"
    )
    print(f"[INFO] ✅ Filtered to {len(filtered)} route_planning questions")
    
    questions = []
    for row in filtered:
        questions.append({
            "scene_name": row["scene_name"],
            "question": row["question"],
            "choices": row.get("options", []),
            "answer_id": row.get("ground_truth", -1),
        })
    
    return questions

# Main entry point for running all questions
# Main entry point for running all questions
def main_vsi_bench_loop(mesh_base_dir=MESH_BASE_DIR, num_steps_per_question=NUM_STEPS, continue_from=None):
    """
    Main loop: iterate through all VSI-Bench questions, find PLY files, and run reasoning.
    If `continue_from` is provided, resumes from the specified folder and skips completed questions.
    """
    print("\n" + "=" * 80)
    print("🚀 VSI-BENCH ROUTE PLANNING EVALUATION")
    print("=" * 80 + "\n")

    questions = load_vsi_bench_questions()
    print(f"[INFO] Loaded {len(questions)} questions\n")

    results = []

    # Determine experiment directory
    if continue_from:
        exp_base_dir = Path(continue_from)
        print(f"[INFO] Resuming from existing folder: {exp_base_dir.resolve()}\n")
    else:
        exp_timestamp = timestamp_str()
        exp_base_dir = Path("experiment_logs") / exp_timestamp
        exp_base_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] 📁 Experiment logs: {exp_base_dir.resolve()}\n")

    # Track completed questions
    completed_questions = set()
    for subfolder in exp_base_dir.iterdir():
        if subfolder.is_dir() and subfolder.name.startswith("q"):
            completed_questions.add(subfolder.name)

    for q_idx, q_data in enumerate(questions, 1):
        scene_id = q_data["scene_name"]
        question_text = q_data["question"]
        choices = q_data["choices"]
        ground_truth_id = q_data["answer_id"]

        question_folder = f"q{q_idx:03d}"
        if question_folder in completed_questions:
            print(f"[INFO] Skipping already completed question: {question_folder}")
            continue

        print("\n" + "─" * 80)
        print(f"[Q{q_idx:03d}] Scene: {scene_id}")
        print(f"[Q{q_idx:03d}] Question: {question_text}")
        print(f"[Q{q_idx:03d}] Options:")
        for i, choice in enumerate(choices):
            print(f"[Q{q_idx:03d}]   {chr(65+i)}) {choice}")
        if ground_truth_id >= 0 and ground_truth_id < len(choices):
            print(f"[Q{q_idx:03d}] Ground Truth: {chr(65+ground_truth_id)}) {choices[ground_truth_id]}")
        print("─" * 80)

        # Find mesh file
        mesh_file = find_mesh_file(scene_id, mesh_base_dir)
        if mesh_file is None:
            print(f"[ERROR] Could not find mesh file for scene {scene_id}. Skipping.\n")
            results.append({
                "scene_id": scene_id,
                "question": question_text,
                "status": "SKIPPED - No mesh found",
                "model_answer": None,
                "ground_truth": chr(65+ground_truth_id) if ground_truth_id >= 0 else "Unknown",
                "correct": False
            })
            continue

        # Run reasoning pipeline with shared experiment base directory
        print(f"[INFO] Starting reasoning pipeline for question {q_idx}...")
        model_answer = run_pipeline(
            mesh_file,
            question=question_text,
            choices=choices,
            num_steps=num_steps_per_question,
            question_id=q_idx,
            experiment_base_dir=str(exp_base_dir),
            scene_id=scene_id
        )

        # Check correctness
        ground_truth_letter = chr(65+ground_truth_id) if ground_truth_id >= 0 else "Unknown"
        is_correct = (model_answer == ground_truth_letter) if ground_truth_id >= 0 else False

        print(f"\n[Q{q_idx:03d}] Model Answer: {model_answer}")
        print(f"[Q{q_idx:03d}] Ground Truth: {ground_truth_letter}")
        print(f"[Q{q_idx:03d}] Result: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}\n")

        results.append({
            "scene_id": scene_id,
            "question": question_text,
            "status": "COMPLETED",
            "model_answer": model_answer,
            "ground_truth": ground_truth_letter,
            "correct": is_correct
        })

        # Print running accuracy
        correct_so_far = sum(1 for r in results if r["correct"])
        total_so_far = len([r for r in results if r["status"] == "COMPLETED"])
        running_accuracy = (100 * correct_so_far / total_so_far) if total_so_far > 0 else 0
        print(f"[RUNNING] Accuracy so far: {correct_so_far}/{total_so_far} = {running_accuracy:.1f}%")
        print(f"[RUNNING] Progress: {total_so_far}/{len(questions)} questions completed.")
        print("=" * 80)

    # Print summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    correct_count = sum(1 for r in results if r["correct"])
    total_count = len([r for r in results if r["status"] == "COMPLETED"])
    print(f"Accuracy: {correct_count}/{total_count} = {100*correct_count/total_count:.1f}%\n")

    for r in results:
        status_icon = "✅" if r["correct"] else ("⏭️ " if "SKIPPED" in r["status"] else "❌")
        print(f"{status_icon} {r['scene_id']:6s} | {r['model_answer']:1s} vs {r['ground_truth']:1s} | {r['question'][:60]}")

    # Save results to JSON in the experiment directory
    results_file = exp_base_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Results saved to: {results_file}")

def run_pipeline(mesh_path: Path, question="", choices=None, cache_dir=CACHE_DIR, num_steps=NUM_STEPS, question_id=0, experiment_base_dir="experiment_logs", scene_id=None):
    """
    Run the reasoning pipeline for a single question.
    Returns the model's final answer (A, B, C, D, etc.)
    """
    start_time = time.time()  # Start timing

    if choices is None:
        choices = []

    # Extract scene_id from mesh_path if not provided
    if scene_id is None:
        # mesh_path format: .../Validation|Training/{video_id}/{video_id}_3dod_mesh.ply
        scene_id = mesh_path.parent.name

    # Create nested directory: experiment_logs/YYYYMMDD_HHMMSS/q00X
    base_out = Path(experiment_base_dir) / f"q{question_id:03d}"
    base_out.mkdir(parents=True, exist_ok=True)
    print(f"\n[INFO] 📁 Outputs -> {base_out.resolve()}")

    # load mesh
    print(f"[INFO] 📂 Loading mesh: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise RuntimeError("Loaded mesh is empty")
    print(f"[INFO] ✅ Mesh loaded, {len(np.asarray(mesh.vertices))} vertices, {len(np.asarray(mesh.triangles))} triangles")

    # compute mesh axis-aligned bounding box once and reuse for all prompts
    vertices = np.asarray(mesh.vertices)
    bbox_mins = vertices.min(axis=0).tolist()
    bbox_maxs = vertices.max(axis=0).tolist()

    print(f"[INFO] 📐 Bounding box: x [{bbox_mins[0]:.2f}, {bbox_maxs[0]:.2f}], y [{bbox_mins[1]:.2f}, {bbox_maxs[1]:.2f}], z [{bbox_mins[2]:.2f}, {bbox_maxs[2]:.2f}]")

    # Define the camera position (eye) before generating candidate views
    vertices = np.asarray(mesh.vertices)
    center_x = (vertices[:, 0].min() + vertices[:, 0].max()) / 2.0
    center_y = (vertices[:, 1].min() + vertices[:, 1].max()) / 2.0
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
    cam_height_z = z_min + CAM_HEIGHT
    eye = np.array([center_x, center_y, cam_height_z], dtype=float)

    # Generate 4 candidate views (0°, 90°, 180°, 270°) with correct up_vector
    view_images = {}
    view_poses = {}
    view_angles = [0, 90, 180, 270]

    for angle_deg in view_angles:
        angle_rad = np.deg2rad(angle_deg)
        forward = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0], dtype=float)
        pose = look_at_camera_pose_center_from_forward(eye, forward=forward, up=np.array([0.0, 0.0, -1.0]))
        view_poses[angle_deg] = pose

        # Render and save temporary image
        img_path = base_out / f"render_candidate_{angle_deg}.png"
        render_mesh_from_pose(mesh, pose, img_path, fxfy=DEFAULT_FX_FY)

        # Load image for scoring
        img_pil = Image.open(img_path)
        img_array = np.array(img_pil).astype(float) / 255.0
        view_images[angle_deg] = img_array

    # Select best view using configured metric
    best_angle, best_score, all_scores = select_best_initial_view(view_images, metric=INITIAL_VIEW_SELECTION_METRIC)

    # Save all candidate scores to file
    scores_record = {
        "metric": INITIAL_VIEW_SELECTION_METRIC,
        "all_scores": {str(k): float(v) for k, v in all_scores.items()},
        "best_angle": best_angle,
        "best_score": float(best_score)
    }
    with open(base_out / "initial_view_selection.json", "w") as f:
        json.dump(scores_record, f, indent=2)

    # Use the best pose as the initial camera pose
    cam_pose = view_poses[best_angle]
    save_matrix(base_out / "cam_pose_00.npy", cam_pose)
    img0 = base_out / "render_00.png"
    Image.fromarray((view_images[best_angle] * 255).astype(np.uint8)).save(str(img0))
    print(f"[INFO] ✅ Initial render saved (best view: {best_angle}°)")

    image_history = [str(img0)]
    cam_history = [cam_pose.copy()]

    # initial R/t to send to Qwen
    R_current = cam_pose[:3,:3]
    t_current = cam_pose[:3,3]

    # Track position history for context
    position_history = []  # Store as list of dictionaries

    # Track the final answer
    final_answer = None

    # send initial step and then iterate
    print(f"[INFO] 🤖 Starting reasoning loop (max {num_steps} steps)...")
    for step in range(0, num_steps+1):
        print(f"\n[Step {step:02d}] " + "─" * 40)
        # Check if this is the final step
        is_final_step = (step == num_steps)

        # build single instruction + messages: send ALL images so far with their positions
        instruction_text = build_instruction_text(
            R_current, t_current, question, bbox=(bbox_mins, bbox_maxs), options=choices, is_final_step=is_final_step, movement_history=position_history, step_num=step
        )

        # Build history context showing where each image was taken with clear numbering
        history_context = "## Image History (numbered for reference):\n"
        for hist_step, hist_t in enumerate(cam_history):
            history_context += f"  Image {hist_step}: position [x={hist_t[0,3]:.2f}m, y={hist_t[1,3]:.2f}m, z={hist_t[2,3]:.2f}m]\n"
        history_context += "\nAbove are all the images you have seen so far in this exploration.\n\n"

        # Build messages with ALL images accumulated so far
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": history_context + instruction_text}
                ]
            }
        ]

        # Add all images to the message content
        for img_path in image_history:
            messages[0]["content"].insert(len(messages[0]["content"]) - 1, {"type": "image", "image": img_path})

        # Save the messages passed
        step_folder = base_out / f"step_{step:02d}"
        step_folder.mkdir(parents=True, exist_ok=True)
        with open(step_folder / "qwen_input_messages.json", "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2)
        # Save instruction text separately for easy inspection
        with open(step_folder / "qwen_input_instruction.txt", "w", encoding="utf-8") as f:
            f.write(history_context + instruction_text)

        # Prepare inputs for Qwen
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        attempt = 0
        parsed_ok = False
        last_valid_pose = cam_history[-1].copy()
        final_parsed_matrix = None
        last_output_text = None

        while attempt < MAX_ATTEMPTS_PER_STEP and not parsed_ok:
            attempt_folder = step_folder / f"attempt_{attempt}"
            attempt_folder.mkdir(parents=True, exist_ok=True)
            try:
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)]
                output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                # Join if batch-like
                output_text = output_texts[0] if isinstance(output_texts, (list, tuple)) else str(output_texts)
            except Exception as e:
                output_text = f"Generation error: {e}"

            # Save raw output
            with open(attempt_folder / "qwen_raw_output.txt", "w", encoding="utf-8") as f:
                f.write(output_text)

            # Print to terminal for inspection
            print(f"\n[QWEN OUTPUT Step {step}, Attempt {attempt}]:")
            print("=" * 80)
            print(output_text)
            print("=" * 80 + "\n")

            # Try to parse JSON and extract movement commands
            rotation_angle, forward_m, left_m, z_delta_m, reasoning_text, raw_obj, done_flag = parse_qwen_output_and_get_movement(output_text)

            # Validate consistency: answer should only be present if done=true
            if raw_obj and isinstance(raw_obj, dict):
                answer_value = raw_obj.get("answer")
                has_answer = answer_value is not None and str(answer_value).strip().upper() in "ABCDEFGHIJ"
                
                # Enforce consistency: done=true with answer, or done=false without answer
                if has_answer and not done_flag:
                    print(f"[WARN] Model provided answer '{answer_value}' but done=false. Rejecting answer - must finalize with done=true to provide answer.")
                    has_answer = False
                elif not has_answer and done_flag:
                    print(f"[WARN] Model set done=true but provided no valid answer. Ignoring done flag - must provide answer with done=true.")
                    done_flag = False
                
                # Only capture answer if done=true
                if has_answer and done_flag:
                    model_answer = str(answer_value).strip().upper()
                    final_answer = model_answer
                    print(f"[DEBUG] Captured final answer: {final_answer}")
            else:
                has_answer = False

            # Debug logging
            print(f"[DEBUG Step {step}, Attempt {attempt}] Extracted: rotation={rotation_angle}, forward={forward_m}, left={left_m}, z_delta={z_delta_m}, done={done_flag}")
            if done_flag:
                print(f"[DEBUG] Model signaled done=true")

            # Validate and process
            if rotation_angle is not None and forward_m is not None and left_m is not None and z_delta_m is not None:
                # Apply rotation angle to current R
                R_new = parse_rotation_angle(rotation_angle, R_current)
                # Apply movement in camera frame to get new t
                t_new = apply_movement_in_camera_frame(R_current, t_current, forward_m, left_m, z_delta_m)

                print(f"[DEBUG] Movement applied: forward={forward_m}m, left={left_m}m, z_delta={z_delta_m}m")
                print(f"[DEBUG] Applied rotation of {rotation_angle} degrees to R")

                # Build homogeneous matrix
                M = np.eye(4, dtype=float)
                M[:3, :3] = R_new
                M[:3, 3] = t_new
                # Check condition number
                try:
                    cond = np.linalg.cond(M)
                except Exception:
                    cond = float('inf')
                if cond < COND_THRESHOLD:
                    parsed_ok = True
                    final_parsed_matrix = M
                    with open(attempt_folder / "qwen_valid_marker.txt", "w") as f:
                        f.write(f"VALID pose with rotation_angle={rotation_angle}°, cond={cond:.4e}\n")
                else:
                    # Record invalid due to conditioning
                    with open(attempt_folder / "qwen_invalid_marker.txt", "w") as f:
                        f.write(f"INVALID due to condition number {cond:.4e}\n")
            else:
                with open(attempt_folder / "qwen_invalid_marker.txt", "w") as f:
                    f.write("No movement parameters parsed from Qwen output.\n")

            # Increment attempt counter
            attempt += 1

        # After attempts, decide next camera pose
        if final_parsed_matrix is not None:
            # Use Qwen's suggestion
            next_pose = final_parsed_matrix
            with open(step_folder / "qwen_chosen_matrix.npy", "wb") as f:
                np.save(f, next_pose)
        else:
            # Fallback: small perturbation from last valid pose
            print(f"[WARN] Step {step}: Qwen did not provide a valid pose after {MAX_ATTEMPTS_PER_STEP} attempts. Using fallback perturbation.")
            last = last_valid_pose
            angle = 10.0 * np.pi / 180.0  # 10 degrees
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            Rz = np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]])
            next_pose = last.copy()
            next_pose[:3, :3] = Rz @ next_pose[:3, :3]
            # Small forward translate
            forward = next_pose[:3, 2]  # Camera forward
            next_pose[:3, 3] = next_pose[:3, 3] + 0.2 * forward  # 0.2 meters forward

            # Save fallback
            with open(step_folder / "fallback_used.txt", "w") as f:
                f.write("Fallback perturbation used because Qwen suggestions invalid or none.\n")
            save_matrix(step_folder / f"fallback_pose.npy", next_pose)

        # Save chosen pose and render next image (unless this was the last step)
        save_matrix(step_folder / f"cam_pose_chosen_step_{step:02d}.npy", next_pose)
        img_next = base_out / f"render_{step:02d}.png"
        render_mesh_from_pose(mesh, next_pose, img_next, fxfy=DEFAULT_FX_FY)

        # Append to histories
        image_history.append(str(img_next))
        cam_history.append(next_pose)
        R_current = next_pose[:3, :3]
        t_current = next_pose[:3, 3]

        # Also save the raw output last seen for easy top-level inspection
        with open(step_folder / "qwen_last_raw_text.txt", "w", encoding="utf-8") as f:
            f.write(last_output_text if last_output_text is not None else "")

        print(f"[INFO] Completed step {step}, saved render and pose in {step_folder}")

        # If model signaled done=true, break early
        if done_flag:
            print(f"[INFO] Model signaled completion at step {step}. Terminating pipeline.")
            break

        # Update position history for next iteration (only if movement was valid)
        if rotation_angle is not None and forward_m is not None and left_m is not None and z_delta_m is not None:
            position_history.append({
                "rotation": rotation_angle,
                "forward": forward_m,
                "left": left_m,
                "z_delta": z_delta_m
            })

    print(f"\n[DONE] Pipeline finished. See folder: {base_out.resolve()}")
    print(f"[DONE] Final answer captured: {final_answer}")

    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time
    print(f"[INFO] Time taken to answer question {question_id}: {elapsed_time:.2f} seconds")

    return final_answer

# Main entry for batch evaluation
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VSI-Bench reasoning loop or single PLY file.")
    parser.add_argument("--ply", default=None, help="Path to single PLY file (optional, for single run)")
    parser.add_argument("--question", default="", help="Question text (for single run)")
    parser.add_argument("--batch", action="store_true", help="Run full VSI-Bench batch evaluation")
    parser.add_argument("--steps", type=int, default=NUM_STEPS, help="Number of reasoning steps per question")
    parser.add_argument("--continue", dest="continue_from", default=None, help="Resume from the most recent experiment folder and skip completed questions")
    args = parser.parse_args()

    if args.batch:
        print("[INFO] Running VSI-Bench batch evaluation...")
        if args.continue_from == "recent":
            # Find the most recent experiment folder
            exp_logs = Path("experiment_logs")
            if exp_logs.exists():
                recent_folder = max(exp_logs.iterdir(), key=lambda p: p.stat().st_mtime, default=None)
                if recent_folder:
                    print(f"[INFO] Found recent folder: {recent_folder.resolve()}")
                    args.continue_from = str(recent_folder)
                else:
                    args.continue_from = None
        main_vsi_bench_loop(num_steps_per_question=args.steps, continue_from=args.continue_from)
    elif args.ply:
        print(f"[INFO] Running single mesh: {args.ply}")
        answer = run_pipeline(Path(args.ply), question=args.question, num_steps=args.steps)
        print(f"\n[RESULT] Model Answer: {answer}")
    else:
        parser.print_help()
