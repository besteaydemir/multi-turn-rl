#!/usr/bin/env python3
"""
Full pipeline:
 - headless Open3D rendering of a room PLY
 - Qwen3-VL multimodal reasoning loop (images + R/t)
 - parse Qwen's R_3x3 and t_3x1 and use them for the next camera pose
 - robust validation/retry/fallback and full logging of attempts
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

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info  # assumed available in your environment

# ----------------- Config -----------------
CACHE_DIR = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir"
MODEL_ID = "Qwen/Qwen3-VL-14B-Instruct"

NUM_STEPS = 5
IMAGE_WH = (1024, 768)
DEFAULT_FX_FY = 400.0   # wider FOV
CAM_HEIGHT = 1.5        # meters above floor (heuristic)
MAX_ATTEMPTS_PER_STEP = 3
COND_THRESHOLD = 1e12   # threshold for condition number marking near-singular

# ----------------- Utilities -----------------
def timestamp_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def look_at_camera_pose_center_from_forward(eye, forward=np.array([1.0,0.0,0.0]), up=np.array([0,0,1])):
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

def compute_initial_camera_pose(pcd, cam_height=CAM_HEIGHT):
    """
    Robust placement:
     - use median X,Y of points (to avoid long tails)
     - set camera z at floor + cam_height
     - look toward centroid of points in front of the camera's X direction (heuristic)
    """
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        raise ValueError("Empty point cloud")

    floor_z = float(np.min(pts[:,2]))
    cam_z = floor_z + cam_height

    median_xy = np.median(pts[:,:2], axis=0)
    eye = np.array([median_xy[0], median_xy[1], cam_z], dtype=float)

    # choose forward direction: toward centroid of points that are slightly forward in X
    # we consider a local front region relative to median_x (robust to odd shapes)
    median_x = median_xy[0]
    front_region = pts[pts[:,0] > median_x - 0.1]  # slightly include near-center points
    if front_region.shape[0] == 0:
        target_xy = median_xy
    else:
        target_xy = np.mean(front_region[:,:2], axis=0)

    target = np.array([target_xy[0], target_xy[1], cam_z], dtype=float)
    forward = target - eye
    # fallback if degenerate
    if np.linalg.norm(forward) < 1e-6:
        forward = np.array([1.0, 0.0, 0.0], dtype=float)
    return look_at_camera_pose_center_from_forward(eye, forward=forward, up=np.array([0,0,1], dtype=float))

def save_matrix(path: Path, mat: np.ndarray, text=True):
    np.save(str(path), mat)
    if text:
        with open(path.with_suffix(".txt"), "w") as f:
            f.write(np.array2string(mat, precision=2, separator=', '))

def render_pc_from_pose(pcd: o3d.geometry.PointCloud, cam_pose_world: np.ndarray, out_path_img: Path, fxfy=DEFAULT_FX_FY):
    """
    Headless render using OffscreenRenderer. No renderer.release() call.
    """
    width, height = IMAGE_WH
    renderer = rendering.OffscreenRenderer(width, height)
    renderer.scene.clear_geometry()

    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    renderer.scene.add_geometry("pcd", pcd, mat)

    cx = width / 2.0
    cy = height / 2.0
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, float(fxfy), float(fxfy), cx, cy)

    # Open3D expects world->camera extrinsic
    extrinsic_world_to_cam = np.linalg.inv(cam_pose_world)
    renderer.setup_camera(intrinsic, extrinsic_world_to_cam)

    renderer.scene.set_background([1.0,1.0,1.0,1.0])
    img = renderer.render_to_image()
    arr = np.asarray(img)
    Image.fromarray(arr).save(str(out_path_img))
    # no explicit release

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
def build_instruction_text(R, t, question, bbox=None):
    R_rounded = np.round(R, 2).tolist()
    t_rounded = np.round(t, 2).tolist()
    instr = build_instruction_natural(R_rounded, t_rounded, question, bbox=bbox)

    return instr


def build_instruction_natural(R_rounded, t_rounded, question, bbox=None):
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

    instr = f"""
You are given a rendered view of a 3D room scene along with the camera rotation R and translation t that produced it.

{bbox_text}

Current camera pose (already correct, don't check validity):
  R_3x3: {R_rounded}
  t_3x1: {t_rounded}

Question: {question}

Think about how this camera pose relates to the visible scene.  
Reason about how you would move or rotate the camera—based on R and t—to obtain a better view that helps answer the question, given the current R and t.  
If the current view is already sufficient, explain why and provide the answer directly. 

The new R should be an orthonormal rotation matrix (3x3) and t should be within the bounds of the scene.
Do not return the same R and t that I am inputting. Give a new R and t that will give a different view of the scene.
After your reasoning, end with a JSON object containing only:
{{
  "R_3x3": [[...],[...],[...]],
  "t_3x1": [...]
}}
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

def parse_qwen_output_and_get_pose(output_text):
    """
    Attempt to parse JSON from output_text and return (R, t, reasoning, raw_json_obj) or None on failure.
    Accepts several forms:
     - {"reasoning": "...", "R_3x3": [[...]], "t_3x1":[...], "answer":"..."}
     - {"matrix_2d": [16-element list], "answer": "..."}
     - nested arrays etc.
    """
    obj = extract_first_json(output_text)
    if obj is None:
        return None, None, None, None

    # Reasoning string if present
    reasoning = obj.get("reasoning") if isinstance(obj, dict) else None

    # Try R_3x3 and t_3x1
    R = None
    t = None
    if isinstance(obj, dict):
        if "R_3x3" in obj:
            R = np.array(obj["R_3x3"], dtype=float)
        if "t_3x1" in obj:
            t = np.array(obj["t_3x1"], dtype=float)
        # older/formats: matrix_2d 16-element row-major
        if R is None and "matrix_2d" in obj:
            arr = np.array(obj["matrix_2d"], dtype=float)
            if arr.size == 16:
                M = arr.reshape(4,4)
                R = M[:3,:3]
                t = M[:3,3]
    # If dictionary didn't have those, maybe obj itself is array or flattened
    if R is None and isinstance(obj, list) and len(obj) >= 1:
        maybe = obj[0]
        if isinstance(maybe, dict):
            if "R_3x3" in maybe:
                R = np.array(maybe["R_3x3"], dtype=float)
            if "t_3x1" in maybe:
                t = np.array(maybe["t_3x1"], dtype=float)

    # Last fallback: search for any 16-number sequence in text (heuristic)
    if R is None or t is None:
        nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", output_text)
        if len(nums) >= 16:
            # take first 16
            vals = np.array([float(x) for x in nums[:16]], dtype=float)
            M = vals.reshape(4,4)
            R = M[:3,:3]
            t = M[:3,3]

    return R, t, reasoning, obj

# ----------------- Main driver -----------------
def run_pipeline(ply_path: Path, cache_dir=CACHE_DIR, num_steps=NUM_STEPS):
    run_ts = timestamp_str()
    base_out = Path(f"reasoning_chain_{run_ts}")
    base_out.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] outputs -> {base_out.resolve()}")

    # load point cloud
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if pcd.is_empty():
        raise RuntimeError("Loaded point cloud is empty or empty PLY")

    # compute point cloud axis-aligned bounding box once and reuse for all prompts
    pts_all = np.asarray(pcd.points)
    bbox_mins = pts_all.min(axis=0).tolist()
    bbox_maxs = pts_all.max(axis=0).tolist()

    # initial camera pose (robust)
    cam_pose = compute_initial_camera_pose(pcd, cam_height=CAM_HEIGHT)
    save_matrix(base_out / "cam_pose_00.npy", cam_pose)
    img0 = base_out / "render_00.png"
    render_pc_from_pose(pcd, cam_pose, img0, fxfy=DEFAULT_FX_FY)

    image_history = [str(img0)]
    cam_history = [cam_pose.copy()]

    # initial R/t to send to Qwen
    R_current = cam_pose[:3,:3]
    t_current = cam_pose[:3,3]

    # send initial step and then iterate
    for step in range(0, num_steps+1):
        # build single instruction + messages: send the latest image and the rounded R/t
        #question = "Initial camera view" if step == 0 else ("Please answer now." if step == num_steps else f"Step {step}: prepare info.")
        question = "What is accross the fireplace?"  # fixed question for all steps
        instruction_text = build_instruction_text(R_current, t_current, question, bbox=(bbox_mins, bbox_maxs))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_history[-1]},
                    {"type": "text", "text": instruction_text}
                ]
            }
        ]

        # save the messages passed
        step_folder = base_out / f"step_{step:02d}"
        step_folder.mkdir(parents=True, exist_ok=True)
        with open(step_folder / "qwen_input_messages.json", "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2)
        # Save instruction text separately for easy inspection
        with open(step_folder / "qwen_input_instruction.txt", "w", encoding="utf-8") as f:
            f.write(instruction_text)

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
                    generated_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)]
                output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                # Join if batch-like
                output_text = output_texts[0] if isinstance(output_texts, (list,tuple)) else str(output_texts)
            except Exception as e:
                output_text = f"Generation error: {e}"

            # save raw output
            with open(attempt_folder / "qwen_raw_output.txt", "w", encoding="utf-8") as f:
                f.write(output_text)

            # try to parse JSON and extract R/t
            R_parsed, t_parsed, reasoning_text, raw_obj = parse_qwen_output_and_get_pose(output_text)
            parsed_record = {
                "R_parsed": (None if R_parsed is None else R_parsed.tolist()),
                "t_parsed": (None if t_parsed is None else t_parsed.tolist()),
                "reasoning": reasoning_text,
                "raw_obj": raw_obj
            }
            with open(attempt_folder / "qwen_parsed_attempt.json", "w", encoding="utf-8") as f:
                json.dump(parsed_record, f, indent=2, default=lambda x: x if x is None else x)

            last_output_text = output_text

            # Debug logging
            print(f"[DEBUG Step {step}, Attempt {attempt}] Extracted R_parsed: {R_parsed is not None}, t_parsed: {t_parsed is not None}")
            if R_parsed is not None:
                print(f"[DEBUG] R_parsed shape: {R_parsed.shape}, dtype: {R_parsed.dtype}")
                print(f"[DEBUG] R_parsed (values):\n{R_parsed}")
            if t_parsed is not None:
                print(f"[DEBUG] t_parsed shape: {t_parsed.shape}, dtype: {t_parsed.dtype}")
                print(f"[DEBUG] t_parsed (values): {t_parsed}")
                print(f"[DEBUG] t_parsed (formatted): tx={t_parsed[0]:.4f}, ty={t_parsed[1]:.4f}, tz={t_parsed[2]:.4f}")

            # Validate parsed matrices if present
            if R_parsed is not None and t_parsed is not None:
                validR, reasonR, R_corrected = validate_rotation_matrix(R_parsed)
                validt, reasont = validate_translation_vector(t_parsed)
                print(f"[DEBUG] Validation: R_valid={validR} ({reasonR}), t_valid={validt} ({reasont})")
                if validR and validt:
                    # Use corrected R if available
                    if R_corrected is not None:
                        R_to_use = R_corrected
                        print(f"[DEBUG] Using corrected R matrix")
                    else:
                        R_to_use = R_parsed
                    # build homogeneous matrix
                    M = np.eye(4, dtype=float)
                    M[:3,:3] = R_to_use
                    M[:3,3] = np.array(t_parsed, dtype=float)
                    # check condition number
                    try:
                        cond = np.linalg.cond(M)
                    except Exception:
                        cond = float('inf')
                    if cond < COND_THRESHOLD:
                        parsed_ok = True
                        final_parsed_matrix = M
                        with open(attempt_folder / "qwen_valid_marker.txt", "w") as f:
                            f.write(f"VALID parsed matrix with cond={cond:.4e}\n")
                    else:
                        # record invalid due to conditioning
                        with open(attempt_folder / "qwen_invalid_marker.txt", "w") as f:
                            f.write(f"INVALID due to condition number {cond:.4e}\n")
                else:
                    # record invalid reasons
                    with open(attempt_folder / "qwen_invalid_marker.txt", "w") as f:
                        f.write(f"R valid? {validR} ({reasonR}); t valid? {validt} ({reasont})\n")
            else:
                with open(attempt_folder / "qwen_invalid_marker.txt", "w") as f:
                    f.write("No R/t parsed from Qwen output.\n")

            # increment attempt counter
            attempt += 1

            # if not valid and attempts remain, continue (Qwen will be run again with same inputs)
            # optionally you might modify the prompt or add a 'please be concise' hint on retry

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
        render_pc_from_pose(pcd, next_pose, img_next, fxfy=DEFAULT_FX_FY)

        # Append to histories
        image_history.append(str(img_next))
        cam_history.append(next_pose)
        R_current = next_pose[:3,:3]
        t_current = next_pose[:3,3]

        # also save the raw output last seen for easy top-level inspection
        with open(step_folder / "qwen_last_raw_text.txt", "w", encoding="utf-8") as f:
            f.write(last_output_text if last_output_text is not None else "")

        print(f"[INFO] Completed step {step}, saved render and pose in {step_folder}")

    print("[DONE] Pipeline finished. See folder:", base_out.resolve())

# ----------------- CLI -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a PLY and do multimodal reasoning with Qwen3-VL.")
    parser.add_argument("--ply", required=True, help="Path to input room PLY file")
    parser.add_argument("--steps", type=int, default=NUM_STEPS, help="Number of reasoning steps (default: 5)")
    args = parser.parse_args()
    run_pipeline(Path(args.ply), num_steps=args.steps)
