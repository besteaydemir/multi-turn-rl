#!/usr/bin/env python3
"""
Video-R1 Baseline Evaluation for VSI-Bench.

Evaluates Video-R1/Video-R1-7B on VSI-Bench questions using actual video frames.
Video-R1 uses a reasoning-focused prompt with <think></think> and <answer></answer> tags,
and natively processes video input via vLLM's video modality.

Usage:
    python video_baseline_r1.py --num-frames 16 --split 1 --num-splits 4
    python video_baseline_r1.py --num-frames 16 --dataset combined --question-types all --split 1 --num-splits 2
"""

# CRITICAL: Set multiprocessing start method to 'spawn' BEFORE importing torch/CUDA
import multiprocessing
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

import argparse
import json
import re
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
CACHE_DIR = "/dss/mcmlscratch/06/di38riq/hf_cache"
MODEL_ID = os.environ.get("MODEL_ID", "Video-R1/Video-R1-7B")
VIDEO_BASE_DIR = "/dss/mcmlscratch/06/di38riq/arkit_vsi/raw"
SCANNET_VIDEO_BASE_DIR = "/dss/mcmlscratch/06/di38riq/scans/scans"
SCANNETPP_VIDEO_BASE_DIR = "/dss/mcmlscratch/06/di38riq/data"
METADATA_CSV = "/dss/dsshome1/06/di38riq/ARKitScenes/metadata.csv"

IMAGE_WH = (640, 480)

# Global inference backend
inference_backend = None

# Load metadata with sky_direction
metadata_df = pd.read_csv(METADATA_CSV)
sky_direction_map = dict(zip(metadata_df['video_id'].astype(str), metadata_df['sky_direction']))

# ============================================================================
# Video-R1 Prompt Templates
# ============================================================================

QUESTION_TEMPLATE = (
    "{Question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
)

TYPE_TEMPLATE = {
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
    "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "free-form": " Please provide your text answer within the <answer> </answer> tags.",
}

# ============================================================================
# Frame Extraction (reused from video_baseline.py)
# ============================================================================

def get_video_frames_dir(scene_id: str, base_dir: str = VIDEO_BASE_DIR) -> Optional[Path]:
    """Find the vga_wide directory for a given ARKitScenes scene."""
    base = Path(base_dir)
    for split in ["Training", "Validation"]:
        video_dir = base / split / str(scene_id) / "vga_wide"
        if video_dir.exists():
            return video_dir
    return None


def get_scannet_frames_dir(scene_id: str, base_dir: str = SCANNET_VIDEO_BASE_DIR) -> Optional[Path]:
    """Find the frames/color directory for a given ScanNet scene."""
    frames_dir = Path(base_dir) / scene_id / "frames" / "color"
    if frames_dir.exists():
        return frames_dir
    return None


def get_scannetpp_frames_dir(scene_id: str, base_dir: str = SCANNETPP_VIDEO_BASE_DIR) -> Optional[Path]:
    """Find the resized_undistorted_images directory for a given ScanNet++ scene."""
    frames_dir = Path(base_dir) / scene_id / "dslr" / "resized_undistorted_images"
    if frames_dir.exists():
        return frames_dir
    return None


def sample_frames_generic(frames_dir: Path, num_frames: int, pattern: str = "*.png", sort_key=None) -> List[Path]:
    """Sample num_frames equally-spaced frames from a directory."""
    frame_files = sorted(frames_dir.glob(pattern), key=sort_key)
    if len(frame_files) == 0:
        return []
    if len(frame_files) <= num_frames:
        return frame_files
    indices = np.linspace(0, len(frame_files) - 1, num_frames, dtype=int)
    return [frame_files[i] for i in indices]


def sample_arkit_frames(video_dir: Path, num_frames: int, scene_id: str) -> List[Path]:
    """Sample frames from ARKitScenes video directory."""
    return sample_frames_generic(video_dir, num_frames, pattern=f"{scene_id}_*.png")


def sample_scannet_frames(frames_dir: Path, num_frames: int) -> List[Path]:
    """Sample frames from ScanNet."""
    return sample_frames_generic(frames_dir, num_frames, pattern="*.jpg", sort_key=lambda x: int(x.stem))


def sample_scannetpp_frames(frames_dir: Path, num_frames: int) -> List[Path]:
    """Sample frames from ScanNet++."""
    frames = sample_frames_generic(frames_dir, num_frames, pattern="*.JPG")
    if len(frames) == 0:
        frames = sample_frames_generic(frames_dir, num_frames, pattern="*.jpg")
    return frames


def rotate_image_by_sky_direction(img: Image.Image, sky_direction: str) -> Image.Image:
    """Rotate image so sky is up based on sky_direction metadata."""
    if sky_direction == 'Up':
        return img
    elif sky_direction == 'Down':
        return img.rotate(180)
    elif sky_direction == 'Left':
        return img.rotate(-90, expand=True)
    elif sky_direction == 'Right':
        return img.rotate(90, expand=True)
    else:
        print(f"[WARN] Unknown sky_direction: {sky_direction}, returning original")
        return img


def resize_image(image_path: Path, video_id: str, target_size: Tuple[int, int] = IMAGE_WH, dataset: str = "arkitscenes") -> Image.Image:
    """Load, rotate (ARKitScenes only), and resize image."""
    img = Image.open(image_path)
    
    if dataset == "arkitscenes":
        sky_direction = sky_direction_map.get(video_id, 'Up')
        img = rotate_image_by_sky_direction(img, sky_direction)
        if sky_direction in ('Left', 'Right'):
            final_size = (target_size[1], target_size[0])
        else:
            final_size = target_size
    else:
        final_size = target_size
    
    if img.size != final_size:
        img = img.resize(final_size, Image.LANCZOS)
    
    return img


# ============================================================================
# Video-R1 Prompt Building
# ============================================================================

def build_video_r1_prompt(question: str, choices: list, question_type: str, is_numerical: bool = False) -> str:
    """Build prompt using Video-R1 template format."""
    # Construct the full question with options if MCQ
    if choices and len(choices) > 0:
        choices_text = "\n".join([f"  {c}" for c in choices])
        full_question = f"{question}\n\nOptions:\n{choices_text}"
        ptype = "multiple choice"
    elif is_numerical:
        full_question = question
        ptype = "numerical"
    else:
        full_question = question
        ptype = "free-form"
    
    prompt = QUESTION_TEMPLATE.format(Question=full_question) + TYPE_TEMPLATE.get(ptype, TYPE_TEMPLATE["free-form"])
    return prompt


# ============================================================================
# Answer Extraction for Video-R1
# ============================================================================

def extract_answer_r1(output_text: str, is_numerical: bool = False) -> Optional[str]:
    """Extract answer from Video-R1 output (uses <answer></answer> tags)."""
    # Try to find <answer>...</answer> tags
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', output_text, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer = answer_match.group(1).strip()
        if is_numerical:
            # Extract numerical value
            num_match = re.search(r'[\d\.\-]+', answer)
            if num_match:
                return num_match.group(0)
        else:
            # Extract letter for MCQ
            letter_match = re.search(r'^([A-D])\b', answer)
            if letter_match:
                return letter_match.group(1).upper()
            # Try single letter
            if len(answer) == 1 and answer.upper() in "ABCD":
                return answer.upper()
            # If the answer contains a letter in context
            for letter in "ABCD":
                if answer.upper().startswith(letter):
                    return letter
        return answer
    
    # Fallback: try JSON extraction
    try:
        json_match = re.search(r'\{[^{}]*"answer"\s*:\s*"?([^"}\s]+)"?\s*[^{}]*\}', output_text, re.DOTALL)
        if json_match:
            answer = json_match.group(1).strip()
            if not is_numerical and answer.upper() in "ABCD":
                return answer.upper()
            return answer
    except:
        pass
    
    # Fallback: look for standalone letter
    if not is_numerical:
        for letter in "ABCD":
            if re.search(rf'\banswer\s*[:=]\s*{letter}\b', output_text, re.IGNORECASE):
                return letter
    
    return None


# ============================================================================
# Main Question Processing
# ============================================================================

def run_single_question(
    scene_id: str,
    question: str,
    choices: list,
    question_id: int,
    experiment_dir: Path,
    num_frames: int,
    question_type: str,
    dataset: str = "arkitscenes",
    is_numerical: bool = False,
) -> Tuple[Optional[str], float, int]:
    """Run Video-R1 inference for a single question."""
    global inference_backend
    
    start_time = time.time()
    
    out_dir = experiment_dir / f"q{question_id:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Find and sample video frames
    if dataset == "scannetpp":
        video_dir = get_scannetpp_frames_dir(scene_id)
        if video_dir is None:
            print(f"[Q{question_id:03d}] ❌ No frames found for ScanNet++ scene {scene_id}")
            return None, time.time() - start_time, 0
        frame_paths = sample_scannetpp_frames(video_dir, num_frames)
    elif dataset == "scannet":
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
        frame_paths = sample_arkit_frames(video_dir, num_frames, scene_id)
    
    if len(frame_paths) == 0:
        print(f"[Q{question_id:03d}] ❌ No frames found in {video_dir}")
        return None, time.time() - start_time, 0
    
    actual_frames = len(frame_paths)
    print(f"[Q{question_id:03d}] 📹 Sampled {actual_frames} frames from {video_dir.name}")
    
    # Resize and save frames
    resized_paths = []
    for i, frame_path in enumerate(frame_paths):
        img = resize_image(frame_path, scene_id, dataset=dataset)
        resized_path = out_dir / f"frame_{i:02d}.png"
        img.save(resized_path)
        resized_paths.append(str(resized_path))
    
    # Build Video-R1 prompt
    prompt_text = build_video_r1_prompt(question, choices, question_type, is_numerical=is_numerical)
    
    # Save prompt for debugging
    with open(out_dir / "prompt.txt", "w") as f:
        f.write(prompt_text)
    
    # Build messages in Video-R1 format (video modality)
    # Video-R1 expects frames as a "video" type entry with list of frame paths
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": resized_paths,
                    "max_pixels": 200704,  # 200704 = ~448x448 per frame
                    "nframes": actual_frames,
                },
                {
                    "type": "text",
                    "text": prompt_text,
                },
            ],
        }
    ]
    
    # Generate response (max_tokens higher for reasoning traces)
    output_text = inference_backend.generate(messages, max_new_tokens=2048)
    
    # Save output
    with open(out_dir / "output.txt", "w") as f:
        f.write(output_text)
    
    # Extract answer
    answer = extract_answer_r1(output_text, is_numerical=is_numerical)
    
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
        "model_output": output_text,
        "elapsed_time": elapsed_time,
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    return answer, elapsed_time, actual_frames


def load_vsi_bench_questions(dataset="arkitscenes", include_numerical=False, include_temporal=True):
    """Load VSI-Bench questions."""
    question_types = MCA_QUESTION_TYPES.copy()
    if include_numerical:
        from utils import NUMERICAL_QUESTION_TYPES
        question_types.extend(NUMERICAL_QUESTION_TYPES)
    if include_temporal:
        question_types.append("obj_appearance_order")
    return _load_vsi_bench_questions(question_types=question_types, dataset=dataset)


def main(num_frames: int, split: int, num_splits: int, max_questions: Optional[int] = None,
         dataset="arkitscenes", test_mode=False, question_types=None, include_temporal=False):
    """Main evaluation loop."""
    global inference_backend
    
    model_name = MODEL_ID.split("/")[-1]
    
    print("\n" + "=" * 80)
    print(f"🎬 VIDEO-R1 EVALUATION - {dataset.upper()} ({num_frames} frames)")
    print(f"   Model: {MODEL_ID}")
    if test_mode:
        print(f"   🧪 TEST MODE - Running {max_questions or 3} questions to 'test' folder")
    else:
        print(f"   Split {split}/{num_splits}")
    print("=" * 80 + "\n")
    
    # Load questions
    if question_types is not None:
        questions = _load_vsi_bench_questions(question_types=question_types, dataset=dataset)
    else:
        questions = load_vsi_bench_questions(dataset=dataset, include_numerical=False, include_temporal=include_temporal)
    total_questions = len(questions)
    
    # Apply question filter if specified
    filter_file = os.environ.get("QUESTION_FILTER_FILE")
    if filter_file and os.path.exists(filter_file):
        print(f"[INFO] Applying question filter from: {filter_file}")
        with open(filter_file, 'r') as f:
            allowed_scenes = set(json.load(f))
        questions = [q for q in questions if q["scene_name"] in allowed_scenes]
        print(f"[INFO] Filtered to {len(questions)} questions (from {total_questions})")
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
    if test_mode:
        exp_dir = Path("test") / f"video_r1_{dataset}"
        exp_dir.mkdir(parents=True, exist_ok=True)
    else:
        EXPERIMENT_BASE = Path("/dss/mcmlscratch/06/di38riq/experiment_logs")
        date_folder = datetime.now().strftime("%Y-%m-%d")
        exp_timestamp = timestamp_str()
        frames_folder = f"{num_frames}_frames"
        exp_dir = EXPERIMENT_BASE / "Video" / model_name / frames_folder / date_folder / f"{exp_timestamp}_video_{model_name}_{dataset}_{num_frames}frames_split{split}of{num_splits}"
        exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] 📁 Output: {exp_dir.resolve()}\n")
    
    # Initialize Video-R1 backend
    print(f"[INFO] Initializing Video-R1 backend via vLLM...")
    inference_backend = create_inference_backend(
        backend="video_r1",
        model_id=MODEL_ID,
        cache_dir=CACHE_DIR,
        nframes=num_frames,
    )
    print("[INFO] Backend ready.\n")
    
    # Track results
    results = []
    csv_rows = []
    
    for local_idx, q_data in enumerate(split_questions, 1):
        global_idx = start_idx + local_idx
        scene_id = q_data["scene_name"]
        q_dataset = q_data.get("dataset", dataset)
        
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
                is_numerical=q_data.get("is_numerical", False),
            )
        except Exception as e:
            import traceback
            print(f"[ERROR] Failed: {e}")
            traceback.print_exc()
            model_answer = None
            elapsed_time = 0
            actual_frames = 0
        
        if model_answer is None:
            model_answer = "NO_ANSWER"
        
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
            "mra_score": None,
            "time_seconds": elapsed_time,
            "num_steps": actual_frames,
            "timestamp": datetime.now().strftime("%y%m%d-%H%M%S"),
            "question": q_data["question"],
        })
        
        # Save after each question
        with open(exp_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        pd.DataFrame(csv_rows).to_csv(exp_dir / "results.csv", index=False)
        
        correct_count = sum(1 for r in results if r["correct"])
        total_count = len(results)
        print(f"[RUNNING] Accuracy: {correct_count}/{total_count} = {100*correct_count/total_count:.1f}%")
    
    # Final summary
    print("\n" + "=" * 80)
    print(f"📊 FINAL RESULTS - Video-R1 ({num_frames} frames, split {split}/{num_splits})")
    print("=" * 80)
    
    correct_count = sum(1 for r in results if r["correct"])
    total_count = len(results)
    no_answer_count = sum(1 for r in results if r["model_answer"] == "NO_ANSWER")
    
    print(f"Total: {total_count}")
    print(f"Correct: {correct_count} ({100*correct_count/total_count:.1f}%)")
    print(f"No Answer: {no_answer_count}")
    
    print("\nBy Question Type:")
    df = pd.DataFrame(results)
    for qtype in df["question_type"].unique():
        qtype_df = df[df["question_type"] == qtype]
        acc = qtype_df["correct"].mean() * 100
        print(f"  {qtype}: {acc:.1f}% ({qtype_df['correct'].sum()}/{len(qtype_df)})")
    
    print(f"\n[INFO] Results saved to: {exp_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video-R1 evaluation for VSI-Bench")
    parser.add_argument("--dataset", type=str, default="arkitscenes",
                       choices=["arkitscenes", "scannet", "scannetpp", "all", "combined"],
                       help="Dataset to evaluate on. Use 'all' for all datasets combined.")
    parser.add_argument("--num-frames", type=int, default=16, choices=[4, 8, 16, 32],
                       help="Number of frames to sample (default: 16)")
    parser.add_argument("--split", type=int, default=1, help="Which split to run (1-indexed)")
    parser.add_argument("--num-splits", type=int, default=1, help="Total number of splits")
    parser.add_argument("--max-questions", type=int, default=None, help="Max questions to process")
    parser.add_argument("--test", action="store_true", help="Test mode: limited questions to 'test' folder")
    parser.add_argument("--question-types", type=str, default="mcq",
                       choices=["mcq", "numerical", "temporal", "all"],
                       help="Question types: 'mcq' (default), 'numerical', 'temporal', or 'all'")
    args = parser.parse_args()
    
    # Map question-types argument
    include_temporal = False
    if args.question_types == "mcq":
        question_types = MCA_QUESTION_TYPES
    elif args.question_types == "numerical":
        from utils import NUMERICAL_QUESTION_TYPES
        question_types = NUMERICAL_QUESTION_TYPES
    elif args.question_types == "temporal":
        question_types = ["obj_appearance_order"]
        include_temporal = True
    else:  # all
        from utils import ALL_SEQUENTIAL_QUESTION_TYPES
        question_types = ALL_SEQUENTIAL_QUESTION_TYPES + ["obj_appearance_order"]
        include_temporal = True
    
    if not args.test and (args.split < 1 or args.split > args.num_splits):
        print(f"[ERROR] Invalid split: {args.split}")
        exit(1)
    
    if args.test and args.max_questions is None:
        args.max_questions = 3
    
    main(
        num_frames=args.num_frames,
        split=args.split,
        num_splits=args.num_splits,
        max_questions=args.max_questions,
        dataset=args.dataset,
        test_mode=args.test,
        question_types=question_types,
        include_temporal=include_temporal,
    )
