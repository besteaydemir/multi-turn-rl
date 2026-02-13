#!/usr/bin/env python3
"""
Resume sequential evaluation from where a previous run left off.
This script identifies remaining questions and splits them across new jobs.

Usage:
  python sequential_resume.py --model 4B --frames 16 --split 1 --num-splits 4
  
This will:
  1. Find all existing results for 4B/16_frames
  2. Identify which questions haven't been completed
  3. Split remaining questions across 4 jobs
  4. Run the specified split
"""

# CRITICAL: Set multiprocessing start method to 'spawn' BEFORE importing torch/CUDA
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    MCA_QUESTION_TYPES,
    NUMERICAL_QUESTION_TYPES,
    ALL_SEQUENTIAL_QUESTION_TYPES,
)
from utils.data import load_vsi_bench_questions as _load_vsi_bench_questions


# ----------------- Config -----------------
CACHE_DIR = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir"
# Old location (read completed questions from here)
EXPERIMENT_BASE_OLD = Path("/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs")
# New location on scratch (write new results here — old location has disk quota issues)
EXPERIMENT_BASE_SCRATCH = Path("/dss/mcmlscratch/06/di38riq/experiment_logs")
# Default output location
EXPERIMENT_BASE = EXPERIMENT_BASE_SCRATCH


def load_vsi_bench_questions(dataset="arkitscenes", question_types=None):
    """Load VSI-Bench questions for the specified dataset(s)."""
    if dataset in ["all", "combined"]:
        all_questions = []
        for ds in ["arkitscenes", "scannet", "scannetpp"]:
            ds_questions = _load_vsi_bench_questions(dataset=ds, question_types=question_types)
            for q in ds_questions:
                q["dataset"] = ds
            all_questions.extend(ds_questions)
        return all_questions
    else:
        questions = _load_vsi_bench_questions(dataset=dataset, question_types=question_types)
        for q in questions:
            q["dataset"] = dataset
        return questions


def get_completed_questions(model_size: str, num_frames: int) -> set:
    """
    Get set of completed questions from existing results.
    Returns set of (scene_name, question_text) tuples.
    
    Structure: Sequential/{model}/{frames}/YYYY-MM-DD/{run_folder}/results.csv
    We only check the top-level results.csv in each run folder, not the per-question ones.
    
    Also checks Sequential/Sequential (bug from earlier runs) for backwards compatibility.
    """
    completed = set()
    
    frames_folder = f"{num_frames}_frames"
    
    # Check ALL locations where results might exist:
    # 1. Old location (original experiment logs)
    # 2. Old location buggy path (from earlier runs)
    # 3. New scratch location (current runs)
    base_dirs = [
        EXPERIMENT_BASE_OLD / "Sequential" / model_size / frames_folder,
        EXPERIMENT_BASE_OLD / "Sequential" / "Sequential" / model_size / frames_folder,  # Bug path
        EXPERIMENT_BASE_SCRATCH / "Sequential" / model_size / frames_folder,
    ]
    
    for base_dir in base_dirs:
        if not base_dir.exists():
            continue
        
        print(f"[INFO] Scanning {base_dir} for completed questions...")
        
        # Only look at date folders (2026-XX-XX)
        for date_folder in base_dir.iterdir():
            if not date_folder.is_dir() or not date_folder.name.startswith("20"):
                continue
            
            # Look at run folders inside date folder
            for run_folder in date_folder.iterdir():
                if not run_folder.is_dir():
                    continue
                
                # Check for results.csv in this run folder (not recursively)
                csv_file = run_folder / "results.csv"
                if csv_file.exists():
                    try:
                        df = pd.read_csv(csv_file)
                        for _, row in df.iterrows():
                            scene_id = row.get("scene_id", "")
                            question = row.get("question", "")
                            if scene_id and question:
                                completed.add((scene_id, question))
                        print(f"  ✓ {run_folder.name}: {len(df)} questions")
                    except Exception as e:
                        print(f"  ✗ {run_folder.name}: Error - {e}")
    
    return completed


def main():
    parser = argparse.ArgumentParser(description="Resume VSI-Bench sequential evaluation")
    parser.add_argument("--model", type=str, required=True, choices=["4B", "8B"],
                       help="Model size: 4B or 8B")
    parser.add_argument("--frames", type=int, required=True, choices=[4, 8, 16, 32],
                       help="Number of frames (4, 8, 16, or 32)")
    parser.add_argument("--split", type=int, required=True, help="Which split to run (1-indexed)")
    parser.add_argument("--num-splits", type=int, required=True, help="Total number of splits for remaining questions")
    parser.add_argument("--dataset", type=str, default="combined",
                       choices=["arkitscenes", "scannet", "scannetpp", "all", "combined"],
                       help="Dataset to evaluate on")
    parser.add_argument("--question-types", type=str, default="all",
                       choices=["mcq", "numerical", "all"],
                       help="Question types to process")
    parser.add_argument("--dry-run", action="store_true",
                       help="Just show what would be processed without running")
    args = parser.parse_args()
    
    # Map question-types argument to actual types
    if args.question_types == "mcq":
        question_types = MCA_QUESTION_TYPES
    elif args.question_types == "numerical":
        question_types = NUMERICAL_QUESTION_TYPES
    else:
        question_types = ALL_SEQUENTIAL_QUESTION_TYPES
    
    print("\n" + "=" * 80)
    print(f"🔄 RESUME SEQUENTIAL EVALUATION")
    print(f"   Model: {args.model}, Frames: {args.frames}")
    print(f"   Split: {args.split}/{args.num_splits}")
    print("=" * 80 + "\n")
    
    # Load all questions
    all_questions = load_vsi_bench_questions(dataset=args.dataset, question_types=question_types)
    print(f"[INFO] Total questions in dataset: {len(all_questions)}")
    
    # Get completed questions
    completed = get_completed_questions(args.model, args.frames)
    print(f"[INFO] Already completed: {len(completed)}")
    
    # Filter to remaining questions
    remaining_questions = []
    for q in all_questions:
        key = (q["scene_name"], q["question"])
        if key not in completed:
            remaining_questions.append(q)
    
    print(f"[INFO] Remaining questions: {len(remaining_questions)}")
    
    if len(remaining_questions) == 0:
        print("\n✅ All questions already completed!")
        return
    
    # Split remaining questions
    total_remaining = len(remaining_questions)
    questions_per_split = total_remaining // args.num_splits
    remainder = total_remaining % args.num_splits
    
    split_sizes = [questions_per_split + (1 if i < remainder else 0) for i in range(args.num_splits)]
    
    start_idx = sum(split_sizes[:args.split-1])
    end_idx = start_idx + split_sizes[args.split-1]
    
    split_questions = remaining_questions[start_idx:end_idx]
    
    print(f"\n[INFO] Split {args.split}/{args.num_splits}:")
    print(f"       Questions {start_idx+1} to {end_idx} of remaining (count: {len(split_questions)})")
    
    if args.dry_run:
        print("\n[DRY RUN] Would process these questions:")
        for i, q in enumerate(split_questions[:10], 1):
            print(f"  {i}. {q['scene_name']}: {q['question'][:60]}...")
        if len(split_questions) > 10:
            print(f"  ... and {len(split_questions) - 10} more")
        return
    
    # Create a temp file with the question IDs to process
    import tempfile
    question_ids = [[q["scene_name"], q["question"]] for q in split_questions]
    
    # Write to temp file
    temp_dir = Path("/tmp")
    question_ids_file = temp_dir / f"resume_questions_{args.model}_{args.frames}f_split{args.split}.json"
    with open(question_ids_file, "w") as f:
        json.dump(question_ids, f)
    
    print(f"[INFO] Question IDs file: {question_ids_file}")
    
    # Set environment variable and run sequential.py
    os.environ["QUESTION_IDS_FILE"] = str(question_ids_file)
    
    # Calculate steps from frames
    steps = args.frames - 1
    
    # Import and run sequential
    from evaluation.sequential import main_sequential_split, initialize_backend, MODEL_ID
    
    # Set model ID based on model size
    model_id = f"Qwen/Qwen3-VL-{args.model}-Instruct"
    os.environ["MODEL_ID"] = model_id
    
    # Re-import to get updated MODEL_ID
    import importlib
    import evaluation.sequential as seq_module
    importlib.reload(seq_module)
    
    # Initialize backend
    seq_module.initialize_backend("vllm")
    
    # Create output directory for this resume run - with unique split identifier
    date_folder = datetime.now().strftime("%Y-%m-%d")
    exp_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"Qwen3-VL-{args.model}"
    frames_folder = f"{args.frames}_frames"
    
    # Use --continue with a unique folder path to save directly there
    # Write to SCRATCH to avoid disk quota issues on old location
    output_dir = EXPERIMENT_BASE_SCRATCH / "Sequential" / args.model / frames_folder / date_folder / f"{exp_timestamp}_resume_{model_name}_{args.dataset}_split{args.split}of{args.num_splits}_{steps}steps"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Output directory: {output_dir}")
    
    # Run with --continue pointing to our unique folder
    # This makes sequential.py use our folder directly instead of creating a new one
    seq_module.main_sequential_split(
        num_steps_per_question=steps,
        split=1,
        num_splits=1,
        continue_from=str(output_dir),  # Use continue_from to save to our unique folder
        test_mode=False,
        max_questions=None,
        dataset=args.dataset,
        question_types=question_types,
        output_base=None  # Not used when continue_from is set
    )


if __name__ == "__main__":
    main()
