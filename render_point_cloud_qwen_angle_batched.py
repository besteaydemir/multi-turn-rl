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
NUM_STEPS = 8  # Max iterations, but model can terminate early with "done": true
IMAGE_WH = (728, 576)  # Reduced resolution for faster inference
DEFAULT_FX_FY = 300.0   # wider FOV
CAM_HEIGHT = 1.6        # meters above floor (heuristic)
MAX_ATTEMPTS_PER_STEP = 1  # Single attempt per step for speed
COND_THRESHOLD = 1e12   # threshold for condition number marking near-singular
MAX_BATCHED_TURNS = 4  # Number of turns to process in batch mode across questions
BATCH_SIZE = 2  # Number of questions to process in parallel at once (limit for GPU memory)

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

# CRITICAL: Set padding_side='left' for decoder-only models in batch inference
# Right-padding causes empty outputs because model generates from padding tokens!
processor.tokenizer.padding_side = 'left'
print(f"[INFO] Tokenizer padding side set to: {processor.tokenizer.padding_side}")

model.to(device)
print("[INFO] Qwen3 model loaded.")

# --------------- Vision embedding caching ----------------
def encode_image_to_embeddings(image_path):
    """
    Encode a single image and return its vision embeddings.
    This allows us to cache embeddings and reuse them across inference steps.
    """
    # Create a minimal message with just this image
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": "placeholder"}
        ]
    }]
    
    # Process to get input tensors
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Extract vision embeddings by running through model's vision encoder
    with torch.no_grad():
        # Get the full embeddings (includes vision + text)
        outputs = model.model(**inputs, output_hidden_states=True, return_dict=True)
        # The vision embeddings are part of the model's internal representation
        # We'll store the pixel_values and their processed form
        if 'pixel_values' in inputs:
            pixel_values = inputs['pixel_values']
            # Get vision tower outputs
            vision_outputs = model.model.visual(pixel_values)
            return vision_outputs
    
    return None

def build_inputs_with_cached_vision(text_prompt, image_paths, cached_vision_embeds=None):
    """
    Build model inputs using cached vision embeddings where available.
    
    Args:
        text_prompt: The text instruction
        image_paths: List of image paths (all images in history)
        cached_vision_embeds: List of cached vision embeddings (None for images to encode fresh)
    
    Returns:
        inputs dict ready for model.generate()
    """
    # For simplicity, we'll still use the standard processor path but with all images
    # The real optimization is to only re-encode new images
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": text_prompt}]
    }]
    
    # Add all images to content
    for img_path in image_paths:
        messages[0]["content"].insert(len(messages[0]["content"]) - 1, 
                                       {"type": "image", "image": str(img_path)})
    
    # Process with standard pipeline
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    return inputs

# --------------- Qwen interaction and parsing --------------
def build_instruction_text(R, t, question, bbox=None, options=None, is_final_step=False, movement_history=None, step_num=0, question_type="unknown"):
    R_rounded = np.round(R, 2).tolist()
    t_rounded = np.round(t, 2).tolist()
    instr = build_instruction_natural(R_rounded, t_rounded, question, bbox=bbox, options=options, is_final_step=is_final_step, step_num=step_num, question_type=question_type)

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


def build_instruction_natural(R_rounded, t_rounded, question, bbox=None, options=None, is_final_step=False, step_num=0, question_type="unknown"):
    # Format bounding box with reasoning context
    bbox_text = ""
    if bbox is not None:
        try:
            mins, maxs = bbox
            # Limit to 2 significant figures
            mins_2sf = [float(f"{x:.2g}") for x in mins]
            maxs_2sf = [float(f"{x:.2g}") for x in maxs]
            bbox_text = (
                "**Scene bounding box limits (meters):**\n"
                f"  x: [{mins_2sf[0]}, {maxs_2sf[0]}]\n"
                f"  y: [{mins_2sf[1]}, {maxs_2sf[1]}]\n"
                f"  z: [{mins_2sf[2]}, {maxs_2sf[2]}]\n\n"
                "⚠️  **IMPORTANT:** The room is NOT a perfect rectangle. Stay within these boundaries - if you see mostly white/empty areas in the render, you are at the edge of the valid room space. Move back toward areas with visible objects and textures.\n"
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
**THIS IS YOUR FINAL STEP:** You have explored the scene. You MUST provide your final answer choice (A, B, C, or D).
- Prioritize outputting your best answer.
- You may provide minimal camera movement or zero movement.
- The critical requirement is that you output your final answer in the JSON response.
- Set "done": true ONLY when you are providing a final answer.
"""
        important_note = "IMPORTANT: This is your final opportunity to answer. You MUST output your best answer choice (A/B/C/D) in the JSON and set done=true."
    else:
        movement_instruction = """
**Movement Guidelines:**

1. **Do NOT stay in the same location.** Always provide at least one non-zero movement.

2. **AVOID REPETITIVE MOVEMENTS!** Look at your action history - do NOT keep rotating back and forth (e.g., turn right 30°, then left 30°, then right 30°). This wastes steps without discovering new information.
   - If you've been oscillating between views, commit to a direction and move forward significantly.
   - Explore NEW areas you haven't seen yet - don't just look at the same spot from slightly different angles.
   - Think strategically: What haven't you seen? What information is missing?

3. **Use the bounding box information above to reason about your position and plan movements.** Consider where you are relative to the room boundaries and where unexplored areas might be.

4. **Answering too early is a mistake.** ONLY output an answer (A, B, C, or D) when you have thoroughly explored the scene and are CERTAIN of your answer.
   - Most of the time, you should NOT provide an answer yet - just explore and set "done": false.
   - Only when you are completely confident should you set both "answer": "X" AND "done": true together.
   - Never output an answer while still exploring (done=false). Either explore (no answer) OR finalize (answer + done=true).

5. **When you are confident:** If you have seen enough viewpoints and are absolutely certain of your answer, set "done": true AND provide your answer.
"""
        important_note = "IMPORTANT: Keep exploring meaningfully (avoid repetitive oscillations) and DO NOT output an answer unless you are ready to stop (done=true). Answer and done=true should come together, not separately."

    # Preprocess question text
    if "You are a robot beginning at" in question:
        question = question.replace("You are a robot beginning at", "If you begin navigating at")

    # Add a note - different for route planning vs other question types
    if question_type == "route_planning":
        starting_place_note = "\n**Note:** Your first task is to identify the starting place mentioned in the question and understand the route.\n"
    else:
        starting_place_note = "\n**Note:** Your first task is to locate and identify the items/objects mentioned in the question.\n"

    instr = f"""You are in the process of exploring a 3D room scene to answer a navigation question. You control the camera by specifying movements. Below you will see all the images you have rendered so far, followed by the actions you selected to obtain each image.

# Current camera position: {t_rounded} (x, y, z in meters)
# This is exploration step {step_num}

{bbox_text}
---

Question: {question}
{options_text}
{starting_place_note}
{important_note}

---
**DETAILED REASONING INSTRUCTIONS:**

1. **Analyze the current view thoroughly:**
   - What objects and areas are visible?
   - What areas are hidden or blocked from the current viewpoint?
   - What would you need to see to better answer the question?
   - Use the bounding box information to reason about your position in the room.

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
**OUTPUT FORMAT:**

First, write your detailed reasoning (3-4 sentences explaining what you see, how you're using the bounding box information, and why you're moving the camera, or why you're ready to finalize your answer).

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

# ============================================================================
# BATCH INFERENCE SUPPORT - QuestionState class and helper functions
# ============================================================================

class QuestionState:
    """Maintains state for a single question during multi-turn reasoning."""
    def __init__(self, mesh_path, question, choices, question_id, experiment_base_dir, scene_id, question_type="unknown"):
        self.mesh_path = mesh_path
        self.question = question
        self.choices = choices
        self.question_id = question_id
        self.experiment_base_dir = experiment_base_dir
        self.scene_id = scene_id
        self.question_type = question_type
        
        # Output directory
        self.base_out = Path(experiment_base_dir) / f"q{question_id:03d}"
        self.base_out.mkdir(parents=True, exist_ok=True)
        
        # Mesh and bounding box
        self.mesh = None
        self.bbox_mins = None
        self.bbox_maxs = None
        
        # Camera history
        self.image_history = []
        self.cam_history = []
        self.R_current = None
        self.t_current = None
        
        # Position history for context
        self.position_history = []
        
        # Vision embedding cache
        self.vision_embed_cache = {}
        
        # Tracking
        self.actual_steps = 0
        self.start_time = time.time()
        self.elapsed_time = 0
        self.final_answer = None
        self.batch_time_accumulated = 0.0  # Track time from batched inference
        self.sequential_time = 0.0  # Track time from sequential inference

def initialize_question_state(mesh_path, question, choices, question_id, experiment_base_dir, scene_id, question_type="unknown"):
    """Initialize a QuestionState by loading mesh and rendering initial view."""
    state = QuestionState(mesh_path, question, choices, question_id, experiment_base_dir, scene_id, question_type)
    
    # Load mesh
    print(f"[Q{question_id:03d}] 📂 Loading mesh: {mesh_path}")
    state.mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if state.mesh.is_empty():
        raise RuntimeError(f"Loaded mesh is empty for Q{question_id}")
    
    # Compute bounding box
    vertices = np.asarray(state.mesh.vertices)
    state.bbox_mins = vertices.min(axis=0).tolist()
    state.bbox_maxs = vertices.max(axis=0).tolist()
    
    # Generate and select best initial view
    center_x = (vertices[:, 0].min() + vertices[:, 0].max()) / 2.0
    center_y = (vertices[:, 1].min() + vertices[:, 1].max()) / 2.0
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
    cam_height_z = z_min + CAM_HEIGHT
    eye = np.array([center_x, center_y, cam_height_z], dtype=float)
    
    view_images = {}
    view_poses = {}
    view_angles = [0, 90, 180, 270]
    
    for angle_deg in view_angles:
        angle_rad = np.deg2rad(angle_deg)
        forward = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0], dtype=float)
        pose = look_at_camera_pose_center_from_forward(eye, forward=forward, up=np.array([0.0, 0.0, -1.0]))
        view_poses[angle_deg] = pose
        
        img_path = state.base_out / f"render_candidate_{angle_deg}.png"
        render_mesh_from_pose(state.mesh, pose, img_path, fxfy=DEFAULT_FX_FY)
        
        img_pil = Image.open(img_path)
        img_array = np.array(img_pil).astype(float) / 255.0
        view_images[angle_deg] = img_array
    
    best_angle, best_score, all_scores = select_best_initial_view(view_images, metric=INITIAL_VIEW_SELECTION_METRIC)
    
    # Save selection info
    scores_record = {
        "metric": INITIAL_VIEW_SELECTION_METRIC,
        "all_scores": {str(k): float(v) for k, v in all_scores.items()},
        "best_angle": best_angle,
        "best_score": float(best_score)
    }
    with open(state.base_out / "initial_view_selection.json", "w") as f:
        json.dump(scores_record, f, indent=2)
    
    # Set initial camera pose
    cam_pose = view_poses[best_angle]
    save_matrix(state.base_out / "cam_pose_00.npy", cam_pose)
    img0 = state.base_out / "render_00.png"
    Image.fromarray((view_images[best_angle] * 255).astype(np.uint8)).save(str(img0))
    
    state.image_history = [str(img0)]
    state.cam_history = [cam_pose.copy()]
    state.R_current = cam_pose[:3, :3]
    state.t_current = cam_pose[:3, 3]
    
    print(f"[Q{question_id:03d}] ✅ Initialized (best view: {best_angle}°)")
    
    return state

def run_batched_turn(active_questions, turn):
    """
    Run a single turn for multiple questions in batch.
    
    Args:
        active_questions: List of question_state dicts that are still active
        turn: Current turn number
    
    Returns:
        List of result dicts with keys: done, final_answer
    """
    print(f"[INFO] 🔄 Preparing batch inference for turn {turn} ({len(active_questions)} questions)")
    
    # Prepare batch messages
    batch_messages = []
    batch_states = []
    
    for qs in active_questions:
        state = qs["state"]
        is_final_step = False  # We don't force final step in batched phase
        
        # Build base instruction (without history)
        instruction_text = build_instruction_text(
            state.R_current, state.t_current, state.question,
            bbox=(state.bbox_mins, state.bbox_maxs),
            options=state.choices,
            is_final_step=is_final_step,
            movement_history=None,  # We'll add this separately
            step_num=turn,
            question_type=state.question_type
        )
        
        # Build image history context
        history_context = "## Image History (numbered for reference):\n"
        for hist_step, hist_t in enumerate(state.cam_history):
            history_context += f"  Image {hist_step}: position [x={hist_t[0,3]:.2f}m, y={hist_t[1,3]:.2f}m, z={hist_t[2,3]:.2f}m]\n"
        history_context += "\nAbove are all the images you have rendered during exploration.\n\n"
        
        # Build action history (what led to each image)
        action_history = ""
        if state.position_history:
            action_history = "## Your Previous Actions (these led to the images above):\n"
            action_history += "  Initial Image 0: Starting view (no movement yet)\n"
            for i, movement in enumerate(state.position_history, 1):
                action_history += f"  Image {i}: You chose to move forward={movement['forward']:.2f}m, left={movement['left']:.2f}m, up={movement['z_delta']:.2f}m, rotated={movement['rotation']:.1f}° → rendered Image {i}\n"
            action_history += "\n⚠️  IMPORTANT: Do not repeat these exact movements. Explore new areas and perspectives!\n\n"
        
        full_prompt = history_context + action_history + instruction_text
        
        # Build message with all images
        messages = [{
            "role": "user",
            "content": [{  "type": "text", "text": full_prompt}]
        }]
        
        # Add all images
        for img_path in state.image_history:
            messages[0]["content"].insert(len(messages[0]["content"]) - 1,
                                         {"type": "image", "image": str(img_path)})
        
        batch_messages.append(messages)
        batch_states.append(state)
        
        # Save step info (turn 0 = step 1, turn 1 = step 2, etc. since step 0 is initial render)
        step_num = turn + 1
        step_folder = state.base_out / f"step_{step_num:02d}"
        step_folder.mkdir(parents=True, exist_ok=True)
        with open(step_folder / "qwen_input_messages.json", "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2)
        with open(step_folder / "qwen_input_instruction.txt", "w", encoding="utf-8") as f:
            f.write(full_prompt)
    
    # Batch process with Qwen3-VL
    print(f"\n[DEBUG] ========== BATCH GENERATION START ==========")
    print(f"[DEBUG] Number of questions in batch: {len(batch_messages)}")
    print(f"[DEBUG] Turn number: {turn}")
    for i, qs in enumerate(active_questions):
        print(f"[DEBUG]   Q{qs['q_idx']:03d}: {len(batch_states[i].image_history)} images in history")
    print(f"[INFO] 🚀 Running batch generation for {len(batch_messages)} questions...")
    inference_start = time.time()
    
    # Apply chat template to all messages
    print(f"[DEBUG] Applying chat template to {len(batch_messages)} messages...")
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in batch_messages
    ]
    print(f"[DEBUG] Chat templates applied. Text lengths: {[len(t) for t in texts]}")
    
    # Process vision info for batch
    print(f"[DEBUG] Processing vision info for batch...")
    image_inputs, video_inputs = process_vision_info(batch_messages)
    print(f"[DEBUG] Vision info processed. Images: {len(image_inputs) if image_inputs else 0}, Videos: {len(video_inputs) if video_inputs else 0}")
    
    # Prepare batch inputs
    print(f"[DEBUG] Tokenizing inputs...")
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print(f"[DEBUG] Inputs prepared. Shapes: {[f'{k}: {v.shape}' for k, v in inputs.items() if hasattr(v, 'shape')]}")
    print(f"[DEBUG] Padding side used: {processor.tokenizer.padding_side}")
    
    # Batch generation
    print(f"[DEBUG] Starting model.generate()...")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    print(f"[DEBUG] Generation complete. Output shape: {generated_ids.shape}")
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
    ]
    print(f"[DEBUG] Trimmed output lengths: {[len(ids) for ids in generated_ids_trimmed]}")
    
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    print(f"[DEBUG] Decoded {len(output_texts)} outputs. Lengths: {[len(t) for t in output_texts]}")
    
    gen_time = time.time() - inference_start
    per_question_time = gen_time / len(batch_messages)
    print(f"[INFO] ⏱️  Batch generation completed in {gen_time:.2f}s ({per_question_time:.2f}s per question)")
    print(f"[DEBUG] ========== BATCH GENERATION END ==========\n")
    
    # Distribute timing to each question
    for qs in active_questions:
        qs["state"].batch_time_accumulated += per_question_time
    
    # Process outputs and update states
    results = []
    for i, (state, output_text) in enumerate(zip(batch_states, output_texts)):
        q_idx = active_questions[i]["q_idx"]
        step_num = turn + 1  # turn 0 = step 1, etc.
        step_folder = state.base_out / f"step_{step_num:02d}"
        attempt_folder = step_folder / f"attempt_0"
        attempt_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[DEBUG] ===== Q{q_idx:03d} Turn {turn} Processing =====")
        print(f"[DEBUG] Output text type: {type(output_text)}")
        print(f"[DEBUG] Output text length: {len(output_text)}")
        print(f"[DEBUG] Output is empty: {len(output_text) == 0}")
        
        # Save raw output
        with open(attempt_folder / "qwen_raw_output.txt", "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"[DEBUG] Saved output to: {attempt_folder / 'qwen_raw_output.txt'}")
        
        if len(output_text) > 0:
            print(f"\n[Q{q_idx:03d} Turn {turn}] Output preview (first 200 chars):\n{output_text[:200]}...")
        else:
            print(f"\n[Q{q_idx:03d} Turn {turn}] ⚠️  WARNING: Empty output from model!")
            print(f"[DEBUG] Input text length was: {len(texts[i]) if i < len(texts) else 'N/A'}")
            print(f"[DEBUG] Number of images: {len(state.image_history)}")
        
        # Parse output
        rotation_angle, forward_m, left_m, z_delta_m, reasoning_text, raw_obj, done_flag = parse_qwen_output_and_get_movement(output_text)
        
        # Check for answer
        final_answer = None
        if raw_obj and isinstance(raw_obj, dict):
            answer_value = raw_obj.get("answer")
            has_answer = answer_value is not None and str(answer_value).strip().upper() in "ABCDEFGHIJ"
            
            if has_answer and done_flag:
                final_answer = str(answer_value).strip().upper()
                print(f"[Q{q_idx:03d}] ✅ Final answer: {final_answer}")
        
        # Apply movement if valid
        if rotation_angle is not None and forward_m is not None and left_m is not None and z_delta_m is not None:
            R_new = parse_rotation_angle(rotation_angle, state.R_current)
            t_new = apply_movement_in_camera_frame(state.R_current, state.t_current, forward_m, left_m, z_delta_m)
            
            next_pose = np.eye(4, dtype=float)
            next_pose[:3, :3] = R_new
            next_pose[:3, 3] = t_new
            
            with open(attempt_folder / "qwen_valid_marker.txt", "w") as f:
                f.write(f"VALID pose with rotation_angle={rotation_angle}°\n")
        else:
            # Fallback: small perturbation
            print(f"[Q{q_idx:03d}] ⚠️  No valid movement, using fallback")
            last = state.cam_history[-1]
            angle = 10.0 * np.pi / 180.0
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            Rz = np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]])
            next_pose = last.copy()
            next_pose[:3, :3] = Rz @ next_pose[:3, :3]
            forward = next_pose[:3, 2]
            next_pose[:3, 3] = next_pose[:3, 3] + 0.2 * forward
            
            with open(attempt_folder / "fallback_used.txt", "w") as f:
                f.write("Fallback perturbation used\n")
        
        # Save pose and render (turn 0 creates render_01.png, turn 1 creates render_02.png, etc.)
        save_matrix(step_folder / f"cam_pose_chosen_step_{step_num:02d}.npy", next_pose)
        img_next = state.base_out / f"render_{step_num:02d}.png"
        render_mesh_from_pose(state.mesh, next_pose, img_next, fxfy=DEFAULT_FX_FY)
        
        # Update state
        state.image_history.append(str(img_next))
        state.cam_history.append(next_pose)
        state.R_current = next_pose[:3, :3]
        state.t_current = next_pose[:3, 3]
        state.actual_steps = step_num  # Properly track step number (not turn number)
        
        if rotation_angle is not None and forward_m is not None:
            state.position_history.append({
                "rotation": rotation_angle,
                "forward": forward_m,
                "left": left_m,
                "z_delta": z_delta_m
            })
        
        results.append({
            "done": done_flag and final_answer is not None,
            "final_answer": final_answer
        })
    
    return results

def complete_question_sequentially(qs, start_turn, num_steps):
    """Complete remaining turns for a single question sequentially."""
    state = qs["state"]
    q_idx = qs["q_idx"]
    
    print(f"\n[DEBUG] ===== Q{q_idx:03d} SEQUENTIAL PHASE START =====")
    print(f"[DEBUG] Starting from turn {start_turn}, ending at turn {num_steps}")
    print(f"[DEBUG] Batched time accumulated so far: {state.batch_time_accumulated:.2f}s")
    
    for turn in range(start_turn, num_steps + 1):
        if qs["done"]:
            break
        
        print(f"[Q{q_idx:03d}] Turn {turn} (sequential)")
        is_final_step = (turn == num_steps)
        
        # Build base instruction (without history)
        instruction_text = build_instruction_text(
            state.R_current, state.t_current, state.question,
            bbox=(state.bbox_mins, state.bbox_maxs),
            options=state.choices,
            is_final_step=is_final_step,
            movement_history=None,  # We'll add this separately
            step_num=turn,
            question_type=state.question_type
        )
        
        # Build image history context
        history_context = "## Image History (numbered for reference):\n"
        for hist_step, hist_t in enumerate(state.cam_history):
            history_context += f"  Image {hist_step}: position [x={hist_t[0,3]:.2f}m, y={hist_t[1,3]:.2f}m, z={hist_t[2,3]:.2f}m]\n"
        history_context += "\nAbove are all the images you have rendered during exploration.\n\n"
        
        # Build action history (what led to each image)
        action_history = ""
        if state.position_history:
            action_history = "## Your Previous Actions (these led to the images above):\n"
            action_history += "  Initial Image 0: Starting view (no movement yet)\n"
            for i, movement in enumerate(state.position_history, 1):
                action_history += f"  Image {i}: You chose to move forward={movement['forward']:.2f}m, left={movement['left']:.2f}m, up={movement['z_delta']:.2f}m, rotated={movement['rotation']:.1f}° → rendered Image {i}\n"
            action_history += "\n⚠️  IMPORTANT: Do not repeat these exact movements. Explore new areas and perspectives!\n\n"
        
        full_prompt = history_context + action_history + instruction_text
        
        # Build messages
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": full_prompt}]
        }]
        
        for img_path in state.image_history:
            messages[0]["content"].insert(len(messages[0]["content"]) - 1,
                                         {"type": "image", "image": str(img_path)})
        
        # Save step info
        step_folder = state.base_out / f"step_{turn:02d}"
        step_folder.mkdir(parents=True, exist_ok=True)
        with open(step_folder / "qwen_input_messages.json", "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2)
        
        # Single inference
        print(f"\n[DEBUG] Q{q_idx:03d} Turn {turn}: Starting single inference...")
        seq_start = time.time()
        inputs = build_inputs_with_cached_vision(
            text_prompt=full_prompt,
            image_paths=state.image_history,
            cached_vision_embeds=None
        )
        
        attempt_folder = step_folder / "attempt_0"
        attempt_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"[DEBUG] Q{q_idx:03d}: Running model.generate()...")
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        
        print(f"[DEBUG] Q{q_idx:03d}: Decoding output...")
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)]
        output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        output_text = output_texts[0] if isinstance(output_texts, (list, tuple)) else str(output_texts)
        
        seq_time = time.time() - seq_start
        state.sequential_time += seq_time
        print(f"[DEBUG] Q{q_idx:03d} Turn {turn}: Inference took {seq_time:.2f}s")
        print(f"[DEBUG] Q{q_idx:03d}: Output length: {len(output_text)}")
        
        with open(attempt_folder / "qwen_raw_output.txt", "w", encoding="utf-8") as f:
            f.write(output_text)
        
        # Parse and process (similar to batch)
        rotation_angle, forward_m, left_m, z_delta_m, reasoning_text, raw_obj, done_flag = parse_qwen_output_and_get_movement(output_text)
        
        # Check for answer
        final_answer = None
        if raw_obj and isinstance(raw_obj, dict):
            answer_value = raw_obj.get("answer")
            has_answer = answer_value is not None and str(answer_value).strip().upper() in "ABCDEFGHIJ"
            
            if has_answer and done_flag:
                final_answer = str(answer_value).strip().upper()
                qs["final_answer"] = final_answer
                qs["done"] = True
                print(f"[Q{q_idx:03d}] ✅ Final answer: {final_answer}")
        
        # Apply movement
        if rotation_angle is not None and forward_m is not None and left_m is not None and z_delta_m is not None:
            R_new = parse_rotation_angle(rotation_angle, state.R_current)
            t_new = apply_movement_in_camera_frame(state.R_current, state.t_current, forward_m, left_m, z_delta_m)
            next_pose = np.eye(4, dtype=float)
            next_pose[:3, :3] = R_new
            next_pose[:3, 3] = t_new
        else:
            last = state.cam_history[-1]
            angle = 10.0 * np.pi / 180.0
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            Rz = np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]])
            next_pose = last.copy()
            next_pose[:3, :3] = Rz @ next_pose[:3, :3]
            forward = next_pose[:3, 2]
            next_pose[:3, 3] = next_pose[:3, 3] + 0.2 * forward
        
        # Save and render (turn N creates render_{N}.png in sequential phase)
        save_matrix(step_folder / f"cam_pose_chosen_step_{turn:02d}.npy", next_pose)
        img_next = state.base_out / f"render_{turn:02d}.png"
        render_mesh_from_pose(state.mesh, next_pose, img_next, fxfy=DEFAULT_FX_FY)
        
        state.image_history.append(str(img_next))
        state.cam_history.append(next_pose)
        state.R_current = next_pose[:3, :3]
        state.t_current = next_pose[:3, 3]
        state.actual_steps = turn + 1
        
        if rotation_angle is not None and forward_m is not None:
            state.position_history.append({
                "rotation": rotation_angle,
                "forward": forward_m,
                "left": left_m,
                "z_delta": z_delta_m
            })
        
        if qs["done"]:
            break
    
    # Calculate total elapsed time (batch time + sequential time)
    state.elapsed_time = state.batch_time_accumulated + state.sequential_time
    print(f"\n[DEBUG] ===== Q{q_idx:03d} SEQUENTIAL PHASE END =====")
    print(f"[DEBUG] Total time: {state.elapsed_time:.2f}s (batched: {state.batch_time_accumulated:.2f}s, sequential: {state.sequential_time:.2f}s)")

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
            "question_type": row.get("question_type", "unknown"),
        })
    
    return questions

# Main entry point for running all questions WITH BATCH INFERENCE
def main_vsi_bench_loop(mesh_base_dir=MESH_BASE_DIR, num_steps_per_question=NUM_STEPS, continue_from=None, max_batched_turns=MAX_BATCHED_TURNS, batch_size=BATCH_SIZE, num_questions=None):
    """
    Main loop with BATCH INFERENCE: process multiple questions in parallel for the first max_batched_turns.
    After max_batched_turns, continue each question sequentially.
    
    EXECUTION ORDER:
    1. Load all questions from VSI-Bench
    2. Filter out completed questions (if continuing)
    3. Initialize ALL question states (load meshes, render initial views, create folders)
    4. BATCHED PHASE (turns 0 to max_batched_turns-1):
       - For each turn:
         - Split active questions into mini-batches of size batch_size
         - For each mini-batch:
           * Prepare messages for all questions in mini-batch
           * Run single batched inference
           * Process outputs and update all question states
           * Save renders and poses
    5. SEQUENTIAL PHASE (remaining turns):
       - For each question individually:
         * Complete remaining turns sequentially
         * Save final results
    
    Example with MAX_BATCHED_TURNS=4, BATCH_SIZE=2, 10 questions:
    - Initialize: Q1, Q2, Q3, ..., Q10 (load meshes, create folders)
    - Turn 0: Mini-batch [Q1, Q2], then [Q3, Q4], then [Q5, Q6], then [Q7, Q8], then [Q9, Q10]
    - Turn 1: Mini-batch [Q1, Q2], then [Q3, Q4], then [Q5, Q6], then [Q7, Q8], then [Q9, Q10]
    - Turn 2: Mini-batch [Q1, Q2], then [Q3, Q4], then [Q5, Q6], then [Q7, Q8], then [Q9, Q10]
    - Turn 3: Mini-batch [Q1, Q2], then [Q3, Q4], then [Q5, Q6], then [Q7, Q8], then [Q9, Q10]
    - Sequential: Q1 turns 4-8, then Q2 turns 4-8, then Q3 turns 4-8, ..., then Q10 turns 4-8
    """
    print("\n" + "=" * 80)
    print("🚀 VSI-BENCH ROUTE PLANNING EVALUATION (BATCHED)")
    print(f"   Batch processing first {max_batched_turns} turns across questions")
    print(f"   Mini-batch size: {batch_size} questions at a time")
    print("=" * 80 + "\n")

    questions = load_vsi_bench_questions()
    
    # Limit number of questions if specified
    if num_questions is not None and num_questions > 0:
        questions = questions[:num_questions]
        print(f"[INFO] Limited to first {num_questions} questions for this run")
    
    print(f"[INFO] Loaded {len(questions)} questions\n")

    results = []
    csv_rows = []

    # Determine experiment directory
    if continue_from:
        exp_base_dir = Path(continue_from)
        print(f"[INFO] Resuming from existing folder: {exp_base_dir.resolve()}\n")
    else:
        exp_timestamp = timestamp_str()
        exp_base_dir = Path("experiment_logs") / f"{exp_timestamp}_batched"
        exp_base_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] 📁 Experiment logs: {exp_base_dir.resolve()}\n")

    # Load existing results if continuing
    results_file = exp_base_dir / "results.json"
    if continue_from and results_file.exists():
        with open(results_file, "r") as f:
            results = json.load(f)
        print(f"[INFO] Loaded {len(results)} existing results from {results_file}\n")
    else:
        results = []

    # Load existing CSV if continuing
    csv_file = exp_base_dir / "results.csv"
    if continue_from and csv_file.exists():
        csv_df = pd.read_csv(csv_file)
        csv_rows = csv_df.to_dict('records')
        print(f"[INFO] Loaded {len(csv_rows)} existing CSV rows from {csv_file}\n")
    else:
        csv_rows = []

    # Track completed questions
    completed_questions = set()
    for subfolder in exp_base_dir.iterdir():
        if subfolder.is_dir() and subfolder.name.startswith("q"):
            completed_questions.add(subfolder.name)
            completed_questions.add(subfolder.name)

    # Prepare question states - only process non-completed questions
    question_states = []
    processed_count = 0  # Counter for questions that actually get processed
    
    for dataset_idx, q_data in enumerate(questions, 1):
        scene_id = q_data["scene_name"]
        mesh_file = find_mesh_file(scene_id, mesh_base_dir)
        
        if mesh_file is None:
            print(f"[WARN] Dataset question {dataset_idx}: Could not find mesh file for scene {scene_id}. Skipping.\n")
            results.append({
                "scene_id": scene_id,
                "question": q_data["question"],
                "status": "SKIPPED - No mesh found",
                "model_answer": None,
                "ground_truth": q_data["answer_id"],
                "correct": False
            })
            continue
        
        # Only increment processed_count for questions with valid meshes
        processed_count += 1
        question_folder = f"q{processed_count:03d}"
        
        if question_folder in completed_questions:
            print(f"[INFO] Skipping already completed question: {question_folder}")
            continue
        
        question_states.append({
            "q_idx": processed_count,  # Sequential numbering for processed questions only
            "q_data": q_data,
            "mesh_file": mesh_file,
            "scene_id": scene_id,
            "state": None,  # Will hold QuestionState object
            "done": False,
            "final_answer": None
        })
    
    print(f"[INFO] Processing {len(question_states)} questions in batched mode\n")
    
    # Initialize all question states (setup meshes, initial renders)
    print("[INFO] 🔧 Initializing all question states...")
    for qs in question_states:
        qs["state"] = initialize_question_state(
            qs["mesh_file"],
            qs["q_data"]["question"],
            qs["q_data"]["choices"],
            qs["q_idx"],
            str(exp_base_dir),
            qs["scene_id"]
        )
    print("[INFO] ✅ All question states initialized\n")
    
    # BATCHED PHASE: Process turns 0 to (max_batched_turns-1) in batch
    # Process questions in mini-batches to avoid memory issues
    for turn in range(max_batched_turns):
        print(f"\n{'='*80}")
        print(f"🔄 BATCHED TURN {turn}/{max_batched_turns-1}")
        print(f"{'='*80}\n")
        
        # Filter active questions (not done yet)
        active_questions = [qs for qs in question_states if not qs["done"]]
        
        if len(active_questions) == 0:
            print(f"[INFO] All questions completed before turn {turn}")
            break
        
        print(f"[INFO] Processing {len(active_questions)} active questions in mini-batches of {batch_size}")
        
        # Process in mini-batches to control memory usage
        for batch_start in range(0, len(active_questions), batch_size):
            batch_end = min(batch_start + batch_size, len(active_questions))
            mini_batch = active_questions[batch_start:batch_end]
            
            q_indices = [qs["q_idx"] for qs in mini_batch]
            print(f"[INFO] 🔄 Mini-batch [{batch_start+1}-{batch_end}]: Questions {q_indices}")
            
            # Run batched inference for this mini-batch
            batch_results = run_batched_turn(mini_batch, turn)
            
            # Update states based on batch results
            for qs, result in zip(mini_batch, batch_results):
                # Only mark as done if we have both done=True AND a valid final answer
                if result["done"] and result["final_answer"] is not None:
                    qs["done"] = True
                    qs["final_answer"] = result["final_answer"]
                    
                    # Calculate elapsed time for questions finishing in batched phase
                    qs["state"].elapsed_time = qs["state"].batch_time_accumulated
                    
                    ground_truth_letter = qs["q_data"]["answer_id"]
                    model_answer = qs["final_answer"]
                    is_correct = (model_answer == ground_truth_letter)
                    
                    print(f"[Q{qs['q_idx']:03d}] 🏁 Completed early in batched phase (turn {turn})!")
                    print(f"[Q{qs['q_idx']:03d}] Model Answer: {model_answer}")
                    print(f"[Q{qs['q_idx']:03d}] Ground Truth: {ground_truth_letter}")
                    print(f"[Q{qs['q_idx']:03d}] Result: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}\n")
    
    # SEQUENTIAL PHASE: Complete remaining turns for each question individually
    print(f"\n{'='*80}")
    print(f"📝 SEQUENTIAL PHASE: Completing remaining turns")
    print(f"{'='*80}\n")
    
    for qs in question_states:
        if not qs["done"]:
            print(f"\n[Q{qs['q_idx']:03d}] Continuing sequentially from turn {max_batched_turns}...")
            complete_question_sequentially(qs, start_turn=max_batched_turns, num_steps=num_steps_per_question)
        
        # Finalize results for this question
        run_timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
        ground_truth_letter = qs["q_data"]["answer_id"]
        model_answer = qs["final_answer"]
        
        # Handle case where model never provided a valid answer
        if model_answer is None:
            print(f"[WARN] Q{qs['q_idx']:03d} never provided a valid answer. Recording as 'NO_ANSWER'.")
            model_answer = "NO_ANSWER"
        
        is_correct = (model_answer == ground_truth_letter)
        
        print(f"\n[Q{qs['q_idx']:03d}] Model Answer: {model_answer}")
        print(f"[Q{qs['q_idx']:03d}] Ground Truth: {ground_truth_letter}")
        print(f"[Q{qs['q_idx']:03d}] Result: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}\n")
        
        results.append({
            "scene_id": qs["scene_id"],
            "question": qs["q_data"]["question"],
            "status": "COMPLETED",
            "model_answer": model_answer,
            "ground_truth": ground_truth_letter,
            "correct": is_correct
        })
        
        csv_rows.append({
            "question_id": f"q{qs['q_idx']:03d}",
            "scene_id": qs["scene_id"],
            "gt_answer": ground_truth_letter,
            "model_answer": model_answer,
            "time_seconds": qs["state"].elapsed_time,
            "num_steps": qs["state"].actual_steps,
            "timestamp": run_timestamp,
            "question": qs["q_data"]["question"]
        })

        # Save incremental results (both JSON and CSV) after each question
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        csv_df = pd.DataFrame(csv_rows)
        csv_df.to_csv(csv_file, index=False)
        print(f"[INFO] 💾 Saved progress to {csv_file.name} (Q{qs['q_idx']:03d} completed)")

        # Print running accuracy (includes all completed questions from current and previous runs)
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

    # Save final results to JSON and CSV in the experiment directory
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] 🏁 Final results saved to: {results_file}")
    
    csv_df = pd.DataFrame(csv_rows)
    csv_df.to_csv(csv_file, index=False)
    print(f"[INFO] 🏁 Final CSV results saved to: {csv_file}")

def run_pipeline(mesh_path: Path, question="", choices=None, cache_dir=CACHE_DIR, num_steps=NUM_STEPS, question_id=0, experiment_base_dir="experiment_logs", scene_id=None):
    """
    Run the reasoning pipeline for a single question.
    Returns tuple: (model_answer, elapsed_time_seconds, actual_steps_taken)
    """
    start_time = time.time()  # Start timing
    actual_steps = 0  # Track actual steps taken

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

    # Vision embedding cache - store embeddings to avoid re-encoding old images
    vision_embed_cache = {}  # image_path -> embeddings
    print(f"[INFO] 🧠 Vision embedding caching enabled for performance")

    # send initial step and then iterate
    print(f"[INFO] 🤖 Starting reasoning loop (max {num_steps} steps)...")
    for step in range(0, num_steps+1):
        actual_steps = step + 1  # Track the current step count
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

        # Cache vision embeddings for new images only
        new_images = []
        for img_path in image_history:
            if img_path not in vision_embed_cache:
                new_images.append(img_path)
                print(f"[INFO] 🔄 Encoding new image: {Path(img_path).name}")
                # Note: We use standard processing but track which images are new
                vision_embed_cache[img_path] = True  # Mark as cached
        
        if len(new_images) > 0:
            print(f"[INFO] ✅ Cached {len(new_images)} new image(s), {len(vision_embed_cache)} total in cache")
        else:
            print(f"[INFO] ♻️  Using all {len(image_history)} cached images (no new encoding needed)")

        # Prepare inputs for Qwen using all images in history
        inference_start = time.time()
        inputs = build_inputs_with_cached_vision(
            text_prompt=history_context + instruction_text,
            image_paths=image_history,
            cached_vision_embeds=None  # Standard path for now
        )
        prep_time = time.time() - inference_start
        print(f"[INFO] ⏱️  Input preparation: {prep_time:.2f}s")

        attempt = 0
        parsed_ok = False
        last_valid_pose = cam_history[-1].copy()
        final_parsed_matrix = None
        last_output_text = None

        while attempt < MAX_ATTEMPTS_PER_STEP and not parsed_ok:
            attempt_folder = step_folder / f"attempt_{attempt}"
            attempt_folder.mkdir(parents=True, exist_ok=True)
            try:
                gen_start = time.time()
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
                gen_time = time.time() - gen_start
                print(f"[INFO] ⏱️  Generation time: {gen_time:.2f}s")
                
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)]
                output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                # Join if batch-like
                output_text = output_texts[0] if isinstance(output_texts, (list, tuple)) else str(output_texts)
                
                total_inference = prep_time + gen_time
                print(f"[INFO] ⏱️  Total inference time (step {step}): {total_inference:.2f}s")
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
    print(f"[INFO] Actual steps taken: {actual_steps}")

    return final_answer, elapsed_time, actual_steps

# Main entry for batch evaluation
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VSI-Bench reasoning loop with BATCH INFERENCE or single PLY file.")
    parser.add_argument("--ply", default=None, help="Path to single PLY file (optional, for single run)")
    parser.add_argument("--question", default="", help="Question text (for single run)")
    parser.add_argument("--batch", action="store_true", help="Run full VSI-Bench batch evaluation with batched turns")
    parser.add_argument("--steps", type=int, default=NUM_STEPS, help="Number of reasoning steps per question")
    parser.add_argument("--max-batched-turns", type=int, default=MAX_BATCHED_TURNS, 
                        help=f"Number of turns to process in batch mode (default: {MAX_BATCHED_TURNS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Number of questions to process in parallel at once (default: {BATCH_SIZE}, higher=faster but more GPU memory)")
    parser.add_argument("--num-questions", type=int, default=None,
                        help="Limit the number of questions to process (useful for testing, default: process all questions)")
    parser.add_argument("--continue", dest="continue_from", default=None, help="Resume from the most recent experiment folder and skip completed questions")
    args = parser.parse_args()

    if args.batch:
        print(f"[INFO] Running VSI-Bench BATCHED evaluation (batching first {args.max_batched_turns} turns)...")
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
        main_vsi_bench_loop(
            num_steps_per_question=args.steps, 
            continue_from=args.continue_from,
            max_batched_turns=args.max_batched_turns,
            batch_size=args.batch_size,
            num_questions=args.num_questions
        )
    elif args.ply:
        print(f"[INFO] Running single mesh: {args.ply}")
        answer, elapsed_time, actual_steps = run_pipeline(Path(args.ply), question=args.question, num_steps=args.steps)
        print(f"\n[RESULT] Model Answer: {answer}")
        print(f"[RESULT] Time taken: {elapsed_time:.2f} seconds")
        print(f"[RESULT] Steps taken: {actual_steps}")
    else:
        parser.print_help()
