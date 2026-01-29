#!/usr/bin/env python3
"""
Video Baseline Evaluation for VSI-Bench.

This script evaluates Qwen3-VL-8B on VSI-Bench MCQ questions using actual video frames
from ARKitScenes instead of rendered mesh images. It samples N equally-spaced frames
from the video and inputs them as a sequence to the model.

Usage:
    python video_baseline.py --num-frames 8 --split 1 --num-splits 4
    python video_baseline.py --num-frames 16 --split 1 --num-splits 4
"""

# CRITICAL: Set multiprocessing start method to 'spawn' BEFORE importing torch/CUDA
import multiprocessing
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image

# Import utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    timestamp_str,
    MCA_QUESTION_TYPES,
    create_inference_backend,
    calculate_mra,
)
from utils.data import load_vsi_bench_questions as _load_vsi_bench_questions

# ----------------- Config -----------------
import os
CACHE_DIR = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir"
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")
VIDEO_BASE_DIR = "/dss/mcmlscratch/06/di38riq/arkit_vsi/raw"
SCANNET_VIDEO_BASE_DIR = "/dss/mcmlscratch/06/di38riq/scans/scans"
METADATA_CSV = "/dss/dsshome1/06/di38riq/ARKitScenes/metadata.csv"

IMAGE_WH = (640, 480)  # Same as sequential.py

# Global inference backend
inference_backend = None

# Load metadata with sky_direction
metadata_df = pd.read_csv(METADATA_CSV)
sky_direction_map = dict(zip(metadata_df['video_id'].astype(str), metadata_df['sky_direction']))


def get_video_frames_dir(scene_id: str, base_dir: str = VIDEO_BASE_DIR) -> Optional[Path]:
    """Find the vga_wide directory for a given scene."""
    base = Path(base_dir)
    
    # Check both Training and Validation
    for split in ["Training", "Validation"]:
        video_dir = base / split / str(scene_id) / "vga_wide"
        if video_dir.exists():
            return video_dir
    
    return None


def sample_frames(video_dir: Path, num_frames: int, scene_id: str) -> List[Path]:
    """Sample num_frames equally-spaced frames from the video directory."""
    # Get all frames sorted by timestamp
    frame_files = sorted(video_dir.glob(f"{scene_id}_*.png"))
    
    if len(frame_files) == 0:
        return []
    
    if len(frame_files) <= num_frames:
        return frame_files
    
    # Sample equally spaced frames
    indices = np.linspace(0, len(frame_files) - 1, num_frames, dtype=int)
    sampled_frames = [frame_files[i] for i in indices]
    
    return sampled_frames


def get_scannet_frames_dir(scene_id: str, base_dir: str = SCANNET_VIDEO_BASE_DIR) -> Optional[Path]:
    """Find the frames/color directory for a given ScanNet scene."""
    frames_dir = Path(base_dir) / scene_id / "frames" / "color"
    if frames_dir.exists():
        return frames_dir
    return None


def sample_scannet_frames(frames_dir: Path, num_frames: int) -> List[Path]:
    """Sample num_frames equally-spaced frames from ScanNet frames directory."""
    # Get all .jpg frames sorted numerically
    frame_files = sorted(frames_dir.glob("*.jpg"), key=lambda x: int(x.stem))
    
    if len(frame_files) == 0:
        return []
    
    if len(frame_files) <= num_frames:
        return frame_files
    
    # Sample equally spaced frames
    indices = np.linspace(0, len(frame_files) - 1, num_frames, dtype=int)
    sampled_frames = [frame_files[i] for i in indices]
    
    return sampled_frames


def rotate_image_by_sky_direction(img: Image.Image, sky_direction: str) -> Image.Image:
    """Rotate image so sky is up based on sky_direction metadata.
    
    Args:
        img: PIL Image
        sky_direction: One of 'Up', 'Down', 'Left', 'Right'
    
    Returns:
        Rotated image with sky pointing up
    """
    if sky_direction == 'Up':
        return img
    elif sky_direction == 'Down':
        return img.rotate(180)
    elif sky_direction == 'Left':
        # Sky on left → rotate clockwise (-90) to bring left edge to top
        return img.rotate(-90, expand=True)
    elif sky_direction == 'Right':
        # Sky on right → rotate counter-clockwise (90) to bring right edge to top
        return img.rotate(90, expand=True)
    else:
        print(f"[WARN] Unknown sky_direction: {sky_direction}, returning original")
        return img


def resize_image(image_path: Path, video_id: str, target_size: Tuple[int, int] = IMAGE_WH, dataset: str = "arkitscenes") -> Image.Image:
    """Load, rotate based on sky_direction (ARKitScenes only), and resize image.
    
    For ARKitScenes Left/Right rotations, the image becomes portrait so we use swapped dimensions.
    For ScanNet, no rotation is applied.
    """
    img = Image.open(image_path)
    
    if dataset == "arkitscenes":
        # Get sky direction from metadata
        sky_direction = sky_direction_map.get(video_id, 'Up')
        
        # Rotate to make sky point up
        img = rotate_image_by_sky_direction(img, sky_direction)
        
        # For Left/Right, image is now portrait - use swapped dimensions
        if sky_direction in ('Left', 'Right'):
            final_size = (target_size[1], target_size[0])  # Swap to (480, 640)
        else:
            final_size = target_size  # Keep (640, 480)
    else:
        # ScanNet: no rotation needed, use standard size
        final_size = target_size
    
    # Resize to target size
    if img.size != final_size:
        img = img.resize(final_size, Image.LANCZOS)
    
    return img


def build_video_prompt(question: str, choices: list, question_type: str, num_frames: int) -> str:
    """Build a simple prompt for video-based MCQ answering."""
    
    # Format choices
    choices_text = "\n".join([f"  {choice}" for choice in choices])
    
    # Question type specific guidance
    if question_type == "route_planning":
        task_hint = "Analyze the navigation route shown in the video to answer."
    elif question_type == "object_rel_distance":
        task_hint = "Compare the distances between objects visible in the video."
    elif "object_rel_direction" in question_type:
        task_hint = "Determine the relative direction of objects from the specified viewpoint."
    else:
        task_hint = "Analyze the spatial relationships shown in the video."

    prompt = f"""You are viewing {num_frames} frames from a video walkthrough of an indoor scene.

{task_hint}

**Question:** {question}

**Answer Options:**
{choices_text}

**Instructions:**
1. Observe the objects and spatial layout across all frames
2. Determine which answer is correct
3. Respond with a JSON object containing brief reasoning and your answer

**Response format:**
{{"reasoning": "<brief explanation>", "answer": "<A, B, C, or D>"}}

Be concise. Avoid repeating yourself."""
    return prompt


def extract_answer(output_text: str) -> Optional[str]:
    """Extract the answer from model output."""
    import re
    
    # Try to find JSON
    try:
        # Look for JSON pattern
        json_match = re.search(r'\{[^{}]*"answer"\s*:\s*"([A-D])?"[^{}]*\}', output_text, re.DOTALL)
        if json_match:
            answer = json_match.group(1)
            if answer and answer.upper() in "ABCD":
                return answer.upper()
        
        # Try parsing as JSON
        import json
        json_start = output_text.find('{')
        json_end = output_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = output_text[json_start:json_end]
            obj = json.loads(json_str)
            if "answer" in obj and obj["answer"]:
                return str(obj["answer"]).strip().upper()
    except:
        pass
    
    # Fallback: look for standalone A, B, C, D
    for letter in ["A", "B", "C", "D"]:
        if f'"{letter}"' in output_text or f"'{letter}'" in output_text:
            return letter
        # Check for "Answer: X" pattern
        if re.search(rf'\banswer\s*[:=]\s*{letter}\b', output_text, re.IGNORECASE):
            return letter
    
    return None


def run_single_question(
    scene_id: str,
    question: str,
    choices: list,
    question_id: int,
    experiment_dir: Path,
    num_frames: int,
    question_type: str,
    dataset: str = "arkitscenes",
) -> Tuple[Optional[str], float, int]:
    """Run video baseline for a single question."""
    global inference_backend
    
    start_time = time.time()
    
    # Create output directory
    out_dir = experiment_dir / f"q{question_id:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Find video frames based on dataset
    if dataset == "scannet":
        video_dir = get_scannet_frames_dir(scene_id)
        if video_dir is None:
            print(f"[Q{question_id:03d}] ❌ No frames found for ScanNet scene {scene_id}")
            return None, time.time() - start_time, 0
        frame_paths = sample_scannet_frames(video_dir, num_frames)
    else:  # arkitscenes
        video_dir = get_video_frames_dir(scene_id)
        if video_dir is None:
            print(f"[Q{question_id:03d}] ❌ No video found for scene {scene_id}")
            return None, time.time() - start_time, 0
        frame_paths = sample_frames(video_dir, num_frames, scene_id)
    
    if len(frame_paths) == 0:
        print(f"[Q{question_id:03d}] ❌ No frames found in {video_dir}")
        return None, time.time() - start_time, 0
    
    actual_frames = len(frame_paths)
    print(f"[Q{question_id:03d}] 📹 Sampled {actual_frames} frames from {video_dir.name}")
    
    # Build prompt
    prompt = build_video_prompt(question, choices, question_type, actual_frames)
    
    # Save prompt for debugging
    with open(out_dir / "prompt.txt", "w") as f:
        f.write(prompt)
    
    # Build message with video frames (using video mode for >16 frames support)
    # Save resized frames
    resized_paths = []
    for i, frame_path in enumerate(frame_paths):
        # Resize and save frame (with proper rotation based on sky_direction for ARKitScenes)
        img = resize_image(frame_path, scene_id, dataset=dataset)
        resized_path = out_dir / f"frame_{i:02d}.png"
        img.save(resized_path)
        resized_paths.append(str(resized_path))
    
    # Use multi-image approach: each frame is a separate image entry
    # This works with vLLM's image mode (limit_mm_per_prompt={"image": N})
    # Reference: vLLM docs - treating video frames as multiple images
    content = [{"type": "text", "text": prompt}]
    
    # Add each frame as a separate image entry
    for frame_path in resized_paths:
        content.append({
            "type": "image",
            "image": frame_path,
        })
    
    messages = [{"role": "user", "content": content}]
    
    # Generate response
    output_text = inference_backend.generate(messages, max_new_tokens=1024)
    
    # Save output
    with open(out_dir / "output.txt", "w") as f:
        f.write(output_text)
    
    # Extract answer
    answer = extract_answer(output_text)
    
    elapsed_time = time.time() - start_time
    print(f"[Q{question_id:03d}] ✅ Answer: {answer} ({elapsed_time:.1f}s)")
    
    # Save result
    result = {
        "question_id": question_id,
        "scene_id": scene_id,
        "question": question,
        "choices": choices,
        "question_type": question_type,
        "dataset": dataset,
        "num_frames": actual_frames,
        "model_answer": answer,
        "elapsed_time": elapsed_time,
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    return answer, elapsed_time, actual_frames


def load_vsi_bench_questions(dataset="arkitscenes"):
    """Load VSI-Bench MCQ questions."""
    if dataset == "combined":
        # Load both datasets
        arkit = _load_vsi_bench_questions(question_types=MCA_QUESTION_TYPES, dataset="arkitscenes")
        scannet = _load_vsi_bench_questions(question_types=MCA_QUESTION_TYPES, dataset="scannet")
        return arkit + scannet
    return _load_vsi_bench_questions(question_types=MCA_QUESTION_TYPES, dataset=dataset)


def main(num_frames: int, split: int, num_splits: int, max_questions: Optional[int] = None, dataset="arkitscenes"):
    """Main evaluation loop."""
    global inference_backend
    
    print("\n" + "=" * 80)
    print(f"🎬 VIDEO BASELINE EVALUATION - {dataset.upper()} ({num_frames} frames)")
    print(f"   Split {split}/{num_splits}")
    print("=" * 80 + "\n")
    
    # Load questions
    questions = load_vsi_bench_questions(dataset=dataset)
    total_questions = len(questions)
    
    # Calculate split range
    questions_per_split = total_questions // num_splits
    remainder = total_questions % num_splits
    split_sizes = [questions_per_split + (1 if i < remainder else 0) for i in range(num_splits)]
    
    start_idx = sum(split_sizes[:split-1])
    end_idx = start_idx + split_sizes[split-1]
    
    print(f"[INFO] Total questions: {total_questions}")
    print(f"[INFO] Split {split}/{num_splits}: questions {start_idx+1} to {end_idx}")
    
    split_questions = questions[start_idx:end_idx]
    
    if max_questions:
        split_questions = split_questions[:max_questions]
        print(f"[INFO] Limited to {max_questions} questions\n")
    
    # Create experiment directory
    EXPERIMENT_BASE = Path("/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs")
    exp_timestamp = timestamp_str()
    model_name = MODEL_ID.split("/")[-1].replace("-Instruct", "")  # e.g., "Qwen3-VL-8B"
    model_size = "4B" if "4B" in model_name else "8B"
    exp_dir = EXPERIMENT_BASE / "Video" / model_size / f"{exp_timestamp}_video_{model_name}_{dataset}_{num_frames}frames_split{split}of{num_splits}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] 📁 Output: {exp_dir.resolve()}\n")
    
    # Initialize backend - using vllm with increased image limit for multi-frame support
    print("[INFO] Initializing vLLM backend (multi-image mode for video frames)...")
    inference_backend = create_inference_backend(
        backend="vllm",
        model_id=MODEL_ID,
        cache_dir=CACHE_DIR,
        max_images=48,  # Support up to 48 frames (extra headroom)
    )
    print("[INFO] Backend ready.\n")
    
    # Track results
    results = []
    csv_rows = []
    
    # Process questions
    for local_idx, q_data in enumerate(split_questions, 1):
        global_idx = start_idx + local_idx
        scene_id = q_data["scene_name"]
        # Use the question's source dataset for combined mode
        q_dataset = q_data.get("dataset", dataset) if dataset == "combined" else dataset
        
        print(f"\n{'='*60}")
        print(f"Question {global_idx} (local {local_idx}/{len(split_questions)})")
        print(f"Scene: {scene_id} | Type: {q_data['question_type']} | Dataset: {q_dataset}")
        print(f"{'='*60}")
        
        try:
            model_answer, elapsed_time, actual_frames = run_single_question(
                scene_id=scene_id,
                question=q_data["question"],
                choices=q_data["choices"],
                question_id=global_idx,
                experiment_dir=exp_dir,
                num_frames=num_frames,
                question_type=q_data["question_type"],
                dataset=q_dataset,
            )
        except Exception as e:
            print(f"[ERROR] Failed: {e}")
            model_answer = None
            elapsed_time = 0
            actual_frames = 0
        
        if model_answer is None:
            model_answer = "NO_ANSWER"
        
        # Evaluate
        gt_answer = q_data["answer_id"]
        is_correct = (model_answer == gt_answer)
        
        print(f"[Q{global_idx:03d}] Model: {model_answer} | GT: {gt_answer} | {'✅' if is_correct else '❌'}")
        
        results.append({
            "scene_id": scene_id,
            "question": q_data["question"],
            "model_answer": model_answer,
            "ground_truth": gt_answer,
            "correct": is_correct,
            "question_type": q_data["question_type"],
            "num_frames": actual_frames,
        })
        
        csv_rows.append({
            "question_id": f"q{global_idx:03d}",
            "scene_id": scene_id,
            "question_type": q_data["question_type"],
            "is_numerical": q_data.get("is_numerical", False),
            "gt_answer": gt_answer,
            "model_answer": model_answer,
            "mra_score": None,  # Not used for MCQ
            "time_seconds": elapsed_time,
            "num_steps": actual_frames,  # Using num_steps to match sequential.py format
            "timestamp": datetime.now().strftime("%y%m%d-%H%M%S"),
            "question": q_data["question"],
        })
        
        # Save after each question
        with open(exp_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        pd.DataFrame(csv_rows).to_csv(exp_dir / "results.csv", index=False)
        
        # Running stats
        correct_count = sum(1 for r in results if r["correct"])
        total_count = len(results)
        print(f"[RUNNING] Accuracy: {correct_count}/{total_count} = {100*correct_count/total_count:.1f}%")
    
    # Final summary
    print("\n" + "=" * 80)
    print(f"📊 FINAL RESULTS ({num_frames} frames, split {split}/{num_splits})")
    print("=" * 80)
    
    correct_count = sum(1 for r in results if r["correct"])
    total_count = len(results)
    no_answer_count = sum(1 for r in results if r["model_answer"] == "NO_ANSWER")
    
    print(f"Total: {total_count}")
    print(f"Correct: {correct_count} ({100*correct_count/total_count:.1f}%)")
    print(f"No Answer: {no_answer_count}")
    
    # By question type
    print("\nBy Question Type:")
    df = pd.DataFrame(results)
    for qtype in df["question_type"].unique():
        qtype_df = df[df["question_type"] == qtype]
        acc = qtype_df["correct"].mean() * 100
        print(f"  {qtype}: {acc:.1f}% ({qtype_df['correct'].sum()}/{len(qtype_df)})")
    
    print(f"\n[INFO] Results saved to: {exp_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video baseline evaluation for VSI-Bench")
    parser.add_argument("--dataset", type=str, default="arkitscenes", choices=["arkitscenes", "scannet", "combined"],
                       help="Dataset to evaluate on (default: arkitscenes, use 'combined' for both)")
    parser.add_argument("--num-frames", type=int, default=8, choices=[4, 8, 16, 32],
                       help="Number of frames to sample (4, 8, 16, or 32)")
    parser.add_argument("--split", type=int, default=1, help="Which split to run (1-indexed)")
    parser.add_argument("--num-splits", type=int, default=1, help="Total number of splits")
    parser.add_argument("--max-questions", type=int, default=None, help="Max questions to process")
    args = parser.parse_args()
    
    if args.split < 1 or args.split > args.num_splits:
        print(f"[ERROR] Invalid split: {args.split}")
        exit(1)
    
    main(
        num_frames=args.num_frames,
        split=args.split,
        num_splits=args.num_splits,
        max_questions=args.max_questions,
        dataset=args.dataset,
    )
