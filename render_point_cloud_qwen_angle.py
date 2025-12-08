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

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info  # assumed available in your environment

# ----------------- Config -----------------
CACHE_DIR = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MESH_BASE_DIR = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/raw"
ARKIT_CSV_PATH = "/dss/dsshome1/06/di38riq/ARKitScenes/raw/raw_train_val_splits.csv"
METADATA_CSV_PATH = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/raw/metadata.csv"

import pandas as pd


# NUM_STEPS = 5
NUM_STEPS = 5  # Max iterations, but model can terminate early with "done": true
IMAGE_WH = (1024, 768)
DEFAULT_FX_FY = 400.0   # wider FOV
CAM_HEIGHT = 1.2        # meters above floor (heuristic)
MAX_ATTEMPTS_PER_STEP = 3
COND_THRESHOLD = 1e12   # threshold for condition number marking near-singular

# Sky direction mapping (loaded from metadata CSV)
_METADATA_CACHE = None
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
    renderer.scene.clear_geometry()

    # Create a material for the mesh with lit shader
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    renderer.scene.add_geometry("mesh", mesh, mat)

    # Setup lighting for better visualization
    renderer.scene.set_lighting(rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, (0.5, 0.5, 0.5))

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
def build_instruction_text(R, t, question, bbox=None, options=None, is_final_step=False):
    R_rounded = np.round(R, 2).tolist()
    t_rounded = np.round(t, 2).tolist()
    instr = build_instruction_natural(R_rounded, t_rounded, question, bbox=bbox, options=options, is_final_step=is_final_step)
    return instr

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


def build_instruction_natural(R_rounded, t_rounded, question, bbox=None, options=None, is_final_step=False):
    bbox_text = ""
    if bbox is not None:
        try:
            mins, maxs = bbox
            # Limit to 2 significant figures
            mins_2sf = [float(f"{x:.2g}") for x in mins]
            maxs_2sf = [float(f"{x:.2g}") for x in maxs]
            bbox_text = (
                "Scene bounding box (meters):\n"
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
"""
        important_note = "IMPORTANT: This is your final opportunity to answer. You MUST output your best answer choice (A/B/C/D) in the JSON."
    else:
        movement_instruction = """
4. **Do NOT stay in the same location.** Always provide at least one non-zero movement.

5. **When you know the answer:** If you have seen enough viewpoints and are confident in your answer, set "done": true to terminate early.
"""
        important_note = "IMPORTANT: You MUST always output a camera movement command. This is not a one-time question. Your movements will be applied to re-render the scene from a new viewpoint and shown to you again."

    instr = f"""
You are given a rendered view of a 3D room scene. You control the camera by specifying how to move it.

{bbox_text}

Current camera position: {t_rounded} (x, y, z in meters)
Current camera orientation: R_3x3 = {R_rounded}

Question: {question}
{options_text}

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

First, write your detailed reasoning (2-3 sentences explaining what you see and why you're moving the camera, or why you're ready to answer).

Then end with a JSON object:
{{
  "rotation_angle_degrees": <number, e.g., 15 or -30>,
  "forward_meters": <number between -0.5 and 0.5>,
  "left_meters": <number between -0.5 and 0.5>,
  "z_delta_meters": <number between -0.3 and 0.3>,
  "answer": "your answer choice (e.g., A, B, C, or D)",
  "done": false
}}

Set "done": true when you are confident in your answer and wish to stop exploring.
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
            "choices": row.get("choices", []),
            "answer_id": row.get("answer_id", -1),
        })
    
    return questions

# Main entry point for running all questions
def main_vsi_bench_loop(mesh_base_dir=MESH_BASE_DIR, num_steps_per_question=NUM_STEPS):
    """
    Main loop: iterate through all VSI-Bench questions, find PLY files, and run reasoning.
    """
    print("\n" + "=" * 80)
    print("🚀 VSI-BENCH ROUTE PLANNING EVALUATION")
    print("=" * 80 + "\n")
    
    questions = load_vsi_bench_questions()
    print(f"[INFO] Loaded {len(questions)} questions\n")
    
    results = []
    
    # Create a single timestamp for all questions in this batch
    exp_timestamp = timestamp_str()
    exp_base_dir = Path("experiment_logs") / exp_timestamp
    exp_base_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 📁 Experiment logs: {exp_base_dir.resolve()}\n")
    
    for q_idx, q_data in enumerate(questions, 1):
        scene_id = q_data["scene_name"]
        question_text = q_data["question"]
        choices = q_data["choices"]
        ground_truth_id = q_data["answer_id"]
        
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
            experiment_base_dir=str(exp_base_dir.parent)
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

def run_pipeline(mesh_path: Path, question="", choices=None, cache_dir=CACHE_DIR, num_steps=NUM_STEPS, question_id=0, experiment_base_dir="experiment_logs"):
    """
    Run the reasoning pipeline for a single question.
    Returns the model's final answer (A, B, C, D, etc.)
    """
    if choices is None:
        choices = []
    
    run_ts = timestamp_str()
    # Create nested directory: experiment_logs/YYYYMMDD_HHMMSS/q00X
    exp_base = Path(experiment_base_dir) / run_ts
    base_out = exp_base / f"q{question_id:03d}"
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

    # Look up sky direction from metadata CSV
    print(f"[INFO] 🌍 Looking up sky direction for scene {scene_id}...")
    sky_dir = get_sky_direction_for_scene(scene_id)
    up_vector = sky_direction_to_up_vector(sky_dir)
    print(f"[INFO] 🌍 Sky direction: {sky_dir} -> up_vector: {np.round(up_vector, 2)}")
    
    # initial camera pose (robust) with sky direction awareness
    print(f"[INFO] 🎥 Computing initial camera pose...")
    cam_pose = compute_initial_camera_pose(mesh, cam_height=CAM_HEIGHT, up_vector=up_vector)
    save_matrix(base_out / "cam_pose_00.npy", cam_pose)
    img0 = base_out / "render_00.png"
    render_mesh_from_pose(mesh, cam_pose, img0, fxfy=DEFAULT_FX_FY)
    print(f"[INFO] ✅ Initial render saved")

    image_history = [str(img0)]
    cam_history = [cam_pose.copy()]

    # initial R/t to send to Qwen
    R_current = cam_pose[:3,:3]
    t_current = cam_pose[:3,3]

    # Track position history for context
    position_history = [(0, t_current.copy())]  # (step, t_vector)
    
    # Track the final answer
    final_answer = None

    # send initial step and then iterate
    print(f"[INFO] 🤖 Starting reasoning loop (max {num_steps} steps)...")
    for step in range(0, num_steps+1):
        print(f"\n[Step {step:02d}] " + "─" * 40)
        # Check if this is the final step
        is_final_step = (step == num_steps)
        # build single instruction + messages: send ALL images so far with their positions
        instruction_text = build_instruction_text(R_current, t_current, question, bbox=(bbox_mins, bbox_maxs), options=choices, is_final_step=is_final_step)

        # Build history context showing where each image was taken
        history_context = "## Image History (where each view was taken):\n"
        for hist_step, hist_t in position_history:
            history_context += f"  Image {hist_step}: position [tx={hist_t[0]:.2f}, ty={hist_t[1]:.2f}, tz={hist_t[2]:.2f}]\n"
        history_context += "\n"

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

        # save the messages passed
        step_folder = base_out / f"step_{step:02d}"
        step_folder.mkdir(parents=True, exist_ok=True)
        with open(step_folder / "qwen_input_messages.json", "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2)
        # Save instruction text separately for easy inspection
        with open(step_folder / "qwen_input_instruction.txt", "w", encoding="utf-8") as f:
            f.write(history_context + instruction_text)

        # prepare inputs for Qwen
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = {k:v.to(device) for k,v in inputs.items()}

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
                output_text = output_texts[0] if isinstance(output_texts, (list,tuple)) else str(output_texts)
            except Exception as e:
                output_text = f"Generation error: {e}"

            # save raw output
            with open(attempt_folder / "qwen_raw_output.txt", "w", encoding="utf-8") as f:
                f.write(output_text)
            
            # print to terminal for inspection
            print(f"\n[QWEN OUTPUT Step {step}, Attempt {attempt}]:")
            print("=" * 80)
            print(output_text)
            print("=" * 80 + "\n")

            # try to parse JSON and extract movement commands
            rotation_angle, forward_m, left_m, z_delta_m, reasoning_text, raw_obj, done_flag = parse_qwen_output_and_get_movement(output_text)
            
            # Capture the answer if present
            if raw_obj and isinstance(raw_obj, dict) and "answer" in raw_obj:
                model_answer = str(raw_obj["answer"]).strip().upper()
                if model_answer and len(model_answer) == 1 and model_answer in "ABCDEFGHIJ":
                    final_answer = model_answer
                    print(f"[DEBUG] Captured answer: {final_answer}")
            
            parsed_record = {
                "rotation_angle_degrees": rotation_angle,
                "forward_meters": forward_m,
                "left_meters": left_m,
                "z_delta_meters": z_delta_m,
                "reasoning": reasoning_text,
                "answer_field": raw_obj.get("answer") if isinstance(raw_obj, dict) else None,
                "done": done_flag,
                "raw_obj": raw_obj
            }
            with open(attempt_folder / "qwen_parsed_attempt.json", "w", encoding="utf-8") as f:
                json.dump(parsed_record, f, indent=2, default=lambda x: x if x is None else x)

            last_output_text = output_text

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
                
                # build homogeneous matrix
                M = np.eye(4, dtype=float)
                M[:3,:3] = R_new
                M[:3,3] = t_new
                # check condition number
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
                    # record invalid due to conditioning
                    with open(attempt_folder / "qwen_invalid_marker.txt", "w") as f:
                        f.write(f"INVALID due to condition number {cond:.4e}\n")
            else:
                with open(attempt_folder / "qwen_invalid_marker.txt", "w") as f:
                    f.write("No movement parameters parsed from Qwen output.\n")

            # increment attempt counter
            attempt += 1

        # After attempts, decide next camera pose
        if final_parsed_matrix is not None:
            # Use Qwen's suggestion
            next_pose = final_parsed_matrix
            with open(step_folder / "qwen_chosen_matrix.npy", "wb") as f:
                np.save(f, next_pose)
        else:
            # fallback: small perturbation from last valid pose
            print(f"[WARN] Step {step}: Qwen did not provide a valid pose after {MAX_ATTEMPTS_PER_STEP} attempts. Using fallback perturbation.")
            last = last_valid_pose
            angle = 10.0 * np.pi / 180.0  # 10 degrees
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            Rz = np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]])
            next_pose = last.copy()
            next_pose[:3,:3] = Rz @ next_pose[:3,:3]
            # small forward translate
            forward = next_pose[:3,2]  # camera forward
            next_pose[:3,3] = next_pose[:3,3] + 0.2 * forward  # 0.2 meters forward

            # save fallback
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
        R_current = next_pose[:3,:3]
        t_current = next_pose[:3,3]

        # also save the raw output last seen for easy top-level inspection
        with open(step_folder / "qwen_last_raw_text.txt", "w", encoding="utf-8") as f:
            f.write(last_output_text if last_output_text is not None else "")

        print(f"[INFO] Completed step {step}, saved render and pose in {step_folder}")

        # If model signaled done=true, break early
        if done_flag:
            print(f"[INFO] Model signaled completion at step {step}. Terminating pipeline.")
            break

        # Update position history for next iteration
        position_history.append((step + 1, t_current.copy()))

    print(f"\n[DONE] Pipeline finished. See folder: {base_out.resolve()}")
    print(f"[DONE] Final answer captured: {final_answer}")
    
    return final_answer

# Main entry for batch evaluation
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VSI-Bench reasoning loop or single PLY file.")
    parser.add_argument("--ply", default=None, help="Path to single PLY file (optional, for single run)")
    parser.add_argument("--question", default="", help="Question text (for single run)")
    parser.add_argument("--batch", action="store_true", help="Run full VSI-Bench batch evaluation")
    parser.add_argument("--steps", type=int, default=NUM_STEPS, help="Number of reasoning steps per question")
    args = parser.parse_args()
    
    if args.batch:
        print("[INFO] Running VSI-Bench batch evaluation...")
        main_vsi_bench_loop(num_steps_per_question=args.steps)
    elif args.ply:
        print(f"[INFO] Running single mesh: {args.ply}")
        answer = run_pipeline(Path(args.ply), question=args.question, num_steps=args.steps)
        print(f"\n[RESULT] Model Answer: {answer}")
    else:
        parser.print_help()
