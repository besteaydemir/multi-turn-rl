#!/usr/bin/env python3
"""
Sequential pipeline with job splitting support:
 - Load VSI-Bench questions (arkitscenes + multiple choice types)
 - Support splitting questions across multiple jobs (--split N --num-splits M)
 - For each question, run Qwen3-VL multimodal reasoning loop sequentially
 - Collect answers and evaluate against ground truth
 - NO batching - all inference is sequential
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from PIL import Image

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Import utilities from the utils package
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    timestamp_str,
    look_at_camera_pose_center_from_forward,
    save_matrix,
    parse_rotation_angle,
    apply_movement_in_camera_frame,
    extract_first_json,
    parse_qwen_output_and_get_movement,
    select_best_initial_view,
    render_mesh_from_pose,
    find_mesh_file,
    load_mesh_cached,
    get_mesh_bounds,
    calculate_mra,
    render_birds_eye_view_with_path,
    MCA_QUESTION_TYPES,
)
from utils.data import load_vsi_bench_questions as _load_vsi_bench_questions

# ----------------- Config -----------------
CACHE_DIR = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MESH_BASE_DIR = "/dss/mcmlscratch/06/di38riq/arkit_vsi/raw"

NUM_STEPS = 8  # 8 reasoning steps = 9 total images (1 initial + 8 new renders)
IMAGE_WH = (640, 480)
DEFAULT_FX_FY = 300.0
CAM_HEIGHT = 1.6

# Initial view selection configuration
INITIAL_VIEW_SELECTION_METRIC = "qwen"  # "visibility", "laplacian", or "qwen"

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


# ----------------- Prompt Builder (kept in main file as requested) -----------------
def build_instruction_text(R, t, question, bbox=None, options=None, is_final_step=False, 
                           movement_history=None, step_num=0, question_type="unknown", is_numerical=False):
    """Build instruction text for Qwen - optimized for multiple choice spatial reasoning."""
    R_rounded = np.round(R, 2).tolist()
    t_rounded = np.round(t, 2).tolist()
    
    bbox_text = ""
    room_size_hint = ""
    if bbox is not None:
        try:
            mins, maxs = bbox
            room_width = maxs[0] - mins[0]
            room_depth = maxs[1] - mins[1]
            bbox_text = f"""**Your Position & Room Bounds:**
- Room dimensions: {room_width:.1f}m (X) x {room_depth:.1f}m (Y)
- X range: [{mins[0]:.2f}, {maxs[0]:.2f}] meters
- Y range: [{mins[1]:.2f}, {maxs[1]:.2f}] meters  
- **Your current position:** X={t_rounded[0]:.2f}m, Y={t_rounded[1]:.2f}m, Z={t_rounded[2]:.2f}m
"""
            room_size_hint = f"Stay within the room bounds. "
        except Exception:
            pass

    # Format options as they appear in VSI-Bench
    options_text = ""
    if options and isinstance(options, (list, dict)):
        options_text = "\n**Answer Options:**\n"
        if isinstance(options, list):
            for opt in options:
                options_text += f"- {opt}\n"
        elif isinstance(options, dict):
            for k, v in options.items():
                options_text += f"{k}. {v}\n"
        options_text += "\n"

    # Question-type specific guidance
    task_description, answer_hint, direction_note = _get_question_type_guidance(question_type)
    
    if is_final_step:
        movement_instruction = f"""
**⚠️ FINAL STEP {step_num}/8 - ANSWER REQUIRED**

You have explored the scene and seen all images from your journey. Based on ALL images you've collected:
- {answer_hint}
- Set "done": true in your JSON output
- Set "answer" to your choice (A, B, C, or D)
- Do NOT include any movement commands when done=true
"""
        important_note = f"⚠️ FINAL STEP: Provide your answer now! NO MOVEMENT needed!"
    else:
        movement_instruction = f"""
**STEP {step_num}/8 - EXPLORATION PHASE ({8 - step_num} steps remaining)**

⚙️ **Movement Controls** (rotation FIRST, then translation in new direction):
- **rotation_angle_degrees**: -90 to +90 (negative=right, positive=left)
- **forward_meters**: -1.0 to +1.0 (walk forward/back in your facing direction)
- **left_meters**: -0.5 to +0.5 (strafe left/right perpendicular to facing)

📐 **Movement Examples:**
- **Rotate only (just look around):** {{"rotation_angle_degrees": 45, "forward_meters": 0.0, "left_meters": 0.0, ...}}
- **Turn left and approach:** {{"rotation_angle_degrees": 60, "forward_meters": 0.8, "left_meters": 0.0, ...}}
- **Turn around and move back:** {{"rotation_angle_degrees": 180, "forward_meters": 0.6, "left_meters": 0.0, ...}}

⚠️ **Exploration Strategy:**
- Focus on the MOST RECENT image and connect it to previous images
- You can rotate in place to look around without moving
- Combine rotation + translation to efficiently explore new areas
- {room_size_hint}
- Explore until you have sufficient information to answer

✅ **Decision Point:** When you have enough information to confidently choose the answer:
- Set "done": true
- Provide "answer": "A", "B", "C", or "D"
- Do NOT include movement commands
"""
        important_note = f"Step {step_num}/8 - Focus on the current image and how it connects to your journey!"

    movement_history_text = _format_movement_history(movement_history)

    instr = f"""You are an embodied agent exploring a 3D room to answer a spatial reasoning question.

---
{task_description}

**Question:** {question}
{options_text}{direction_note}
{bbox_text}
---
## Your Exploration Journey:

Below are ALL the images you have captured as you moved through the room.
**IMPORTANT: Focus on the MOST RECENT image (bottom of list) first, then connect it to your journey.**
- Image 0 = your initial starting position
- Image 1+ = views after each of your movements

---
**OUTPUT FORMAT:**

Think step-by-step BEFORE providing JSON:

1. **Current observation:** What do I see in the most recent image?
2. **Objects found so far:** Which objects from the question have I identified?
3. **Missing information:** What objects/perspectives do I still need to see?
4. **Current opinion on answer:** Based on what I've seen, what's my best guess?
5. **Next action:** Should I explore more or answer now?

**DECISION RULE:**
- If you have STRONG confidence in your answer → Set "done": true, provide "answer", NO movement needed
- If you need more exploration → Provide movement commands, set "done": false

Then output JSON:
```json
{{
  "rotation_angle_degrees": <-90 to 90, or 0 if done=true>,
  "forward_meters": <-1.0 to 1.0, or 0 if done=true>,
  "left_meters": <-0.5 to 0.5, or 0 if done=true>,
  "z_delta_meters": <-0.3 to 0.3, or 0 if done=true>,
  "answer": <"A"/"B"/"C"/"D" or null>,
  "done": <true/false>
}}
```

---
{important_note}

{movement_instruction}

{movement_history_text}
"""
    return instr


def _get_question_type_guidance(question_type):
    """Get task-specific guidance based on question type."""
    if question_type == "route_planning":
        task_description = """**TASK: Route Planning**
You need to understand a navigation route between objects in the room.
- First, locate the STARTING object mentioned
- Identify the DESTINATION object 
- Trace the route and determine which turns are needed"""
        answer_hint = "Choose the correct sequence of turns"
        direction_note = ""
        
    elif question_type == "object_rel_distance":
        task_description = """**TASK: Relative Distance Comparison**
You need to determine which object is CLOSEST to a reference object.
- Locate the reference object mentioned in the question
- Find ALL candidate objects listed in the options
- Compare their distances to the reference object"""
        answer_hint = "Choose the object that is closest (A, B, C, or D)"
        direction_note = ""
        
    elif question_type in ["object_rel_direction_easy", "object_rel_direction_medium", "object_rel_direction_hard"]:
        task_description = """**TASK: Relative Direction**
You need to determine the direction of one object relative to another FROM A SPECIFIC VIEWPOINT.
- First, locate WHERE you would be standing
- Then, determine which direction you would be FACING
- From that perspective, determine where the TARGET object is"""
        if question_type == "object_rel_direction_easy":
            answer_hint = "Choose: left or right"
        elif question_type == "object_rel_direction_medium":
            answer_hint = "Choose: left, right, or back"
        else:
            answer_hint = "Choose: front-left, front-right, back-left, or back-right"
        direction_note = """
**Direction Reference:**
- Imagine standing at the viewpoint and facing in the specified direction
- Front-left: 45° to your left, Front-right: 45° to your right
- Back-left: 45° behind you to the left, Back-right: 45° behind you to the right
"""
    else:
        task_description = "**TASK: Spatial Reasoning**\nAnalyze the scene to answer the question."
        answer_hint = "Choose the correct option (A, B, C, or D)"
        direction_note = ""
    
    return task_description, answer_hint, direction_note


def _format_movement_history(movement_history):
    """Format movement history for the prompt."""
    if not movement_history or len(movement_history) == 0:
        return ""
    
    text = "\n---\n**How you got here - Movement & Image correspondence:**\n"
    text += "Each movement you made resulted in a new image.\n\n"
    
    for i, movement in enumerate(movement_history, 1):
        rot_desc = f"turned {abs(movement['rotation']):.0f}° {'left' if movement['rotation'] > 0 else 'right' if movement['rotation'] < 0 else 'stayed facing same direction'}"
        fwd_desc = f"moved {abs(movement['forward']):.1f}m {'forward' if movement['forward'] > 0 else 'backward' if movement['forward'] < 0 else 'stayed in place'}"
        left_desc = f"strafed {abs(movement['left']):.1f}m {'left' if movement['left'] > 0 else 'right' if movement['left'] < 0 else ''}" if abs(movement['left']) > 0.01 else ""
        
        parts = [rot_desc, fwd_desc]
        if left_desc:
            parts.append(left_desc)
        
        pos_info = f"Position: {movement.get('position', 'unknown')}"
        text += f"Step {i}: {', '.join(parts)} → Image {i} ({pos_info})\n"
    
    text += "\n💡 Try exploring areas you haven't visited yet.\n"
    return text


# ----------------- Qwen-based View Selection -----------------
def select_initial_view_with_qwen(view_image_paths, question, choices, question_type):
    """Use Qwen to select the best initial view based on the question."""
    print("[INFO] 🤖 Using Qwen to select best initial view...")
    
    prompt_intro = f"""You are looking at a 3D room scene from 4 different angles.
Your task is to select which view is the BEST starting point to answer the following question:

**Question:** {question}

**Options:** {choices}

**Question Type:** {question_type}

Below are 4 images, each labeled with its viewing angle. Look at all 4 and select the best one.
"""
    
    prompt_end = """
Select the view that:
1. Shows the most relevant objects mentioned in the question
2. Has the clearest visibility of the scene
3. Would be the best starting point for exploration

Respond with ONLY a JSON object:
{"selected_angle": <0, 90, 180, or 270>, "reasoning": "<brief explanation>"}
"""
    
    # Build message with LABELED images
    content = [{"type": "text", "text": prompt_intro}]
    
    for angle in [0, 90, 180, 270]:
        img_path = view_image_paths.get(angle)
        if img_path:
            content.append({"type": "text", "text": f"\n**View at {angle}° angle:**"})
            content.append({"type": "image", "image": str(img_path)})
    
    content.append({"type": "text", "text": prompt_end})
    
    messages = [{"role": "user", "content": content}]
    
    # Apply chat template and run inference
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
    
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    
    print(f"[INFO] 🤖 Qwen view selection response: {output_text[:200]}...")
    
    # Parse response
    try:
        json_obj = extract_first_json(output_text)
        if json_obj and "selected_angle" in json_obj:
            selected = int(json_obj["selected_angle"])
            if selected in [0, 90, 180, 270]:
                reasoning = json_obj.get("reasoning", "No reasoning provided")
                print(f"[INFO] ✅ Qwen selected view: {selected}° - {reasoning}")
                return selected, reasoning
    except Exception as e:
        print(f"[WARN] Failed to parse Qwen view selection: {e}")
    
    # Fallback to 0 if parsing fails
    print("[WARN] Falling back to 0° view")
    return 0, "Fallback to default"


# ----------------- Data Loading -----------------
def load_vsi_bench_questions():
    """Load VSI-Bench questions with MULTIPLE CHOICE question types only."""
    return _load_vsi_bench_questions(question_types=MCA_QUESTION_TYPES, dataset="arkitscenes")


# ----------------- Main Question Runner -----------------
def run_single_question(mesh_path, question, choices, question_id, experiment_base_dir, scene_id, 
                        num_steps=NUM_STEPS, question_type="unknown", is_numerical=False, max_questions=None):
    """Run reasoning pipeline for a single question - fully sequential."""
    start_time = time.time()
    
    base_out = Path(experiment_base_dir) / f"q{question_id:03d}"
    base_out.mkdir(parents=True, exist_ok=True)
    print(f"\n[Q{question_id:03d}] 📁 Outputs -> {base_out.resolve()}")

    # Load mesh (with caching)
    mesh = load_mesh_cached(mesh_path)
    
    # Get bounding box with outlier filtering
    bbox_mins, bbox_maxs = get_mesh_bounds(mesh, percentile_filter=True)
    
    print(f"[Q{question_id:03d}] 📏 Robust bbox (2-98%): X[{bbox_mins[0]:.2f}, {bbox_maxs[0]:.2f}], Y[{bbox_mins[1]:.2f}, {bbox_maxs[1]:.2f}]")

    # Generate 4 candidate views and select best
    center_x = (bbox_mins[0] + bbox_maxs[0]) / 2.0
    center_y = (bbox_mins[1] + bbox_maxs[1]) / 2.0
    cam_height_z = bbox_mins[2] + CAM_HEIGHT
    eye = np.array([center_x, center_y, cam_height_z], dtype=float)

    view_images = {}
    view_poses = {}
    for angle_deg in [0, 90, 180, 270]:
        angle_rad = np.deg2rad(angle_deg)
        forward = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0], dtype=float)
        pose = look_at_camera_pose_center_from_forward(eye, forward=forward, up=np.array([0.0, 0.0, -1.0]))
        view_poses[angle_deg] = pose

        img_path = base_out / f"render_candidate_{angle_deg}.png"
        render_mesh_from_pose(mesh, pose, img_path, fxfy=DEFAULT_FX_FY, image_wh=IMAGE_WH)

        img_pil = Image.open(img_path)
        img_array = np.array(img_pil).astype(float) / 255.0
        view_images[angle_deg] = img_array

    # Select best initial view
    view_image_paths = {angle: base_out / f"render_candidate_{angle}.png" for angle in [0, 90, 180, 270]}
    
    if INITIAL_VIEW_SELECTION_METRIC == "qwen":
        best_angle, reasoning = select_initial_view_with_qwen(view_image_paths, question, choices, question_type)
        scores_record = {"metric": "qwen", "selected_angle": best_angle, "reasoning": reasoning}
    else:
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

        history_context = _build_history_context(cam_history)
        full_prompt = history_context + instruction_text

        # Build message with LABELED images
        content = []
        for img_idx, img_path in enumerate(image_history):
            label = f"\n**Image {img_idx} (Initial view):**" if img_idx == 0 else f"\n**Image {img_idx} (After movement {img_idx}):**"
            content.append({"type": "text", "text": label})
            content.append({"type": "image", "image": img_path})
        
        content.append({"type": "text", "text": f"\n\n{full_prompt}"})
        messages = [{"role": "user", "content": content}]

        step_folder = base_out / f"step_{step:02d}"
        step_folder.mkdir(parents=True, exist_ok=True)

        # Save input prompt for debugging
        with open(step_folder / "qwen_input_prompt.txt", "w", encoding="utf-8") as f:
            f.write(full_prompt)

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

        # Sequential inference with FP16
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
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

        # If we got an answer and done flag, or if this is the final step, stop
        if (done_flag and final_answer) or is_final_step:
            if done_flag and final_answer:
                print(f"[Q{question_id:03d}] 🏁 Model signaled done=true, breaking early at step {step}")
            else:
                print(f"[Q{question_id:03d}] 🏁 Reached final step {step}, ending exploration")
            break

        # Apply movement
        if rotation_angle is not None and forward_m is not None and left_m is not None and z_delta_m is not None:
            R_new = parse_rotation_angle(rotation_angle, R_current)
            t_new = apply_movement_in_camera_frame(R_new, t_current, forward_m, left_m, z_delta_m, 
                                                   bbox_mins=bbox_mins, bbox_maxs=bbox_maxs)
            next_pose = np.eye(4, dtype=float)
            next_pose[:3, :3] = R_new
            next_pose[:3, 3] = t_new
            
            position_history.append({
                "rotation": rotation_angle,
                "forward": forward_m,
                "left": left_m,
                "z_delta": z_delta_m,
                "position": f"X={t_new[0]:.2f}m, Y={t_new[1]:.2f}m, Z={t_new[2]:.2f}m"
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
        render_mesh_from_pose(mesh, next_pose, img_next, fxfy=DEFAULT_FX_FY, image_wh=IMAGE_WH)

        image_history.append(str(img_next))
        cam_history.append(next_pose)
        R_current = next_pose[:3, :3]
        t_current = next_pose[:3, 3]

    elapsed_time = time.time() - start_time
    print(f"\n[Q{question_id:03d}] ⏱️  Completed in {elapsed_time:.2f}s")
    print(f"[Q{question_id:03d}] Final answer: {final_answer}")

    # Render bird's eye view with path (for small tests or every 10th question)
    enable_viz = (max_questions and max_questions <= 100) or (question_id % 10 == 0) or (question_id <= 5)
    if enable_viz:
        birds_eye_path = base_out / "birds_eye_view_path.png"
        try:
            print(f"[Q{question_id:03d}] 🗺️  Rendering trajectory visualization...")
            render_birds_eye_view_with_path(mesh, cam_history, birds_eye_path)
        except Exception as e:
            print(f"[WARN] Failed to render bird's eye view: {e}")

    return final_answer, elapsed_time, len(image_history) - 1


def _build_history_context(cam_history):
    """Build history context text for the prompt."""
    history_context = "## Your Exploration History:\n"
    history_context += "Below are images you captured as you moved through the room.\n"
    history_context += "- Image 0 = your initial starting position\n"
    if len(cam_history) > 1:
        history_context += f"- Images 1-{len(cam_history)-1} = views after each of your {len(cam_history)-1} movement(s)\n"
    history_context += "\n"
    for hist_step, hist_t in enumerate(cam_history):
        pos_str = f"[X={hist_t[0,3]:.2f}m, Y={hist_t[1,3]:.2f}m, Z={hist_t[2,3]:.2f}m]"
        if hist_step == 0:
            history_context += f"Image {hist_step} (Initial view): {pos_str}\n"
        else:
            history_context += f"Image {hist_step} (After movement {hist_step}): {pos_str}\n"
    history_context += "\n"
    return history_context


# ----------------- Main Entry Point -----------------
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
    
    split_sizes = [questions_per_split + (1 if i < remainder else 0) for i in range(num_splits)]
    
    start_idx = sum(split_sizes[:split-1])
    end_idx = start_idx + split_sizes[split-1]
    
    print(f"[INFO] Total questions: {total_questions}")
    print(f"[INFO] Split {split}/{num_splits}: questions {start_idx+1} to {end_idx} (count: {end_idx - start_idx})")
    
    split_questions = questions[start_idx:end_idx]
    
    if max_questions is not None:
        split_questions = split_questions[:max_questions]
        print(f"[INFO] Limited to {max_questions} question(s)\n")
    
    # Determine experiment directory
    if test_mode:
        exp_base_dir = Path("test")
        exp_base_dir.mkdir(parents=True, exist_ok=True)
    elif continue_from:
        exp_base_dir = Path(continue_from)
    else:
        exp_timestamp = timestamp_str()
        exp_base_dir = Path("experiment_logs") / f"{exp_timestamp}_sequential_split{split}of{num_splits}"
        exp_base_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 📁 Experiment logs: {exp_base_dir.resolve()}\n")

    results_file = exp_base_dir / "results.json"
    csv_file = exp_base_dir / "results.csv"

    # Load existing results
    results = []
    csv_rows = []
    if continue_from:
        if results_file.exists():
            with open(results_file, "r") as f:
                results = json.load(f)
        if csv_file.exists():
            csv_df = pd.read_csv(csv_file)
            csv_rows = csv_df.to_dict('records')

    # Track completed questions
    completed_questions = set()
    for subfolder in exp_base_dir.iterdir():
        if subfolder.is_dir() and subfolder.name.startswith("q"):
            completed_questions.add(subfolder.name)

    # Process questions
    processed_count = start_idx
    
    for local_idx, q_data in enumerate(split_questions, 1):
        processed_count += 1
        scene_id = q_data["scene_name"]
        mesh_file = find_mesh_file(scene_id, mesh_base_dir)
        
        if mesh_file is None:
            print(f"[WARN] Question {processed_count}: No mesh for scene {scene_id}. Skipping.\n")
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
                is_numerical=q_data["is_numerical"],
                max_questions=max_questions
            )
        except Exception as e:
            print(f"[ERROR] Failed to process Q{processed_count}: {e}")
            continue
        
        if model_answer is None:
            model_answer = "NO_ANSWER"
        
        # Evaluate
        ground_truth = q_data["answer_id"]
        is_numerical = q_data["is_numerical"]
        
        if is_numerical and model_answer != "NO_ANSWER":
            try:
                predicted_value = float(model_answer)
                gt_value = float(ground_truth)
                mra_score = calculate_mra(predicted_value, gt_value)
                is_correct = (mra_score > 0.5)
            except (ValueError, TypeError):
                mra_score = 0.0
                is_correct = False
        else:
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
            "mra_score": mra_score,
            "question_type": q_data["question_type"]
        })
        
        csv_rows.append({
            "question_id": f"q{processed_count:03d}",
            "scene_id": scene_id,
            "question_type": q_data["question_type"],
            "is_numerical": is_numerical,
            "gt_answer": ground_truth,
            "model_answer": model_answer,
            "mra_score": mra_score,
            "time_seconds": elapsed_time,
            "num_steps": actual_steps,
            "timestamp": datetime.now().strftime("%y%m%d-%H%M%S"),
            "question": q_data["question"]
        })

        # Save after each question
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        csv_df = pd.DataFrame(csv_rows)
        csv_df.to_csv(csv_file, index=False)

        # Print running stats
        correct_so_far = sum(1 for r in results if r["correct"])
        total_so_far = len([r for r in results if r["status"] == "COMPLETED"])
        running_accuracy = (100 * correct_so_far / total_so_far) if total_so_far > 0 else 0
        print(f"[RUNNING] Split {split} accuracy: {correct_so_far}/{total_so_far} = {running_accuracy:.1f}%")
        print("=" * 80)

    # Final summary
    print("\n" + "=" * 80)
    print(f"📊 SUMMARY (Split {split}/{num_splits})")
    print("=" * 80)
    correct_count = sum(1 for r in results if r["correct"])
    total_count = len([r for r in results if r["status"] == "COMPLETED"])
    if total_count > 0:
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
