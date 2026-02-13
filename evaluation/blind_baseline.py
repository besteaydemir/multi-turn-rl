#!/usr/bin/env python3
"""
Blind Baseline Evaluation for VSI-Bench.

This script evaluates the model on VSI-Bench questions WITHOUT passing any images.
It serves as a "blind" baseline to measure how well the model can answer spatial
reasoning questions from text alone (i.e., language-prior / guessing performance).

Usage:
    python blind_baseline.py --model 4B
    python blind_baseline.py --model 8B
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

# Import utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    timestamp_str,
    MCA_QUESTION_TYPES,
    NUMERICAL_QUESTION_TYPES,
    ALL_SEQUENTIAL_QUESTION_TYPES,
    create_inference_backend,
    calculate_mra,
)
from utils.data import load_vsi_bench_questions as _load_vsi_bench_questions

# ----------------- Config -----------------
import os
CACHE_DIR = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir"
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")

# Global inference backend
inference_backend = None


def build_blind_prompt(question: str, choices: list, question_type: str, is_numerical: bool = False) -> str:
    """Build a prompt for blind (no-image) evaluation."""

    # Question type specific guidance (same as video baseline, but referencing "a scene")
    if question_type == "route_planning":
        task_hint = "Consider the navigation route described to answer."
    elif question_type == "object_rel_distance":
        task_hint = "Compare the distances between the objects mentioned."
    elif "object_rel_direction" in question_type:
        task_hint = "Determine the relative direction of objects from the specified viewpoint."
    elif question_type == "object_counting":
        task_hint = "Estimate how many instances of the target object are present."
    elif question_type == "object_abs_distance":
        task_hint = "Estimate the distance between the two objects in meters."
    elif question_type == "object_size_estimation":
        task_hint = "Estimate the size of the specified object dimension in centimeters."
    elif question_type == "room_size_estimation":
        task_hint = "Estimate the total floor area of the room in square meters."
    elif question_type == "obj_appearance_order":
        task_hint = "Determine the temporal order in which objects might appear as you move through the scene."
    else:
        task_hint = "Analyze the spatial relationships described."

    prompt = f"""You are answering a question about an indoor scene.

{task_hint}

**Question:** {question}
"""

    # Add options or numerical instruction
    if choices and len(choices) > 0:
        choices_text = "\n".join([f"  {choice}" for choice in choices])
        prompt += f"""
**Answer Options:**
{choices_text}
"""
        answer_format = '"answer": "<A, B, C, or D>"'
    else:
        # Numerical answer
        if question_type == "object_counting":
            prompt += "\n**Provide your answer as an integer (e.g., 3).**\n"
            answer_format = '"answer": <integer>'
        elif question_type in ["object_abs_distance", "room_size_estimation"]:
            prompt += "\n**Provide your answer as a number with one decimal place (e.g., 1.5).**\n"
            answer_format = '"answer": <number>'
        elif question_type == "object_size_estimation":
            prompt += "\n**Provide your answer in centimeters as an integer (e.g., 75).**\n"
            answer_format = '"answer": <integer>'
        else:
            answer_format = '"answer": <your answer>'

    prompt += f"""
**Instructions:**
1. Think about the question and use your best judgment
2. Determine the answer
3. Respond with a JSON object containing brief reasoning and your answer

**Response format:**
{{"reasoning": "<brief explanation>", {answer_format}}}

Be concise. Avoid repeating yourself."""
    return prompt


def extract_answer(output_text: str, is_numerical: bool = False) -> Optional[str]:
    """Extract the answer from model output."""
    import re

    # Try to find JSON
    try:
        # Look for JSON pattern
        if is_numerical:
            # Extract numerical answer
            json_match = re.search(r'\{[^{}]*"answer"\s*:\s*([\d\.\-]+)[^{}]*\}', output_text, re.DOTALL)
            if json_match:
                return json_match.group(1)
        else:
            # Extract letter answer
            json_match = re.search(r'\{[^{}]*"answer"\s*:\s*"([A-D])?"[^{}]*\}', output_text, re.DOTALL)
            if json_match:
                answer = json_match.group(1)
                if answer and answer.upper() in "ABCD":
                    return answer.upper()

        # Try parsing as JSON
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
    question_type: str,
    dataset: str = "arkitscenes",
    is_numerical: bool = False,
) -> Tuple[Optional[str], float]:
    """Run blind evaluation for a single question (no images)."""
    global inference_backend

    start_time = time.time()

    # Create output directory
    out_dir = experiment_dir / f"q{question_id:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build prompt (no images)
    prompt = build_blind_prompt(question, choices, question_type, is_numerical=is_numerical)

    # Save prompt for debugging
    with open(out_dir / "prompt.txt", "w") as f:
        f.write(prompt)

    # Build message with TEXT ONLY — no images
    messages = [{"role": "user", "content": prompt}]

    # Generate response
    output_text = inference_backend.generate(messages, max_new_tokens=1024)

    # Save output
    with open(out_dir / "output.txt", "w") as f:
        f.write(output_text)

    # Extract answer
    answer = extract_answer(output_text, is_numerical=is_numerical)

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
        "num_frames": 0,  # Blind — no frames
        "model_answer": answer,
        "elapsed_time": elapsed_time,
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    return answer, elapsed_time


def main(model_size: str, max_questions: Optional[int] = None, test_mode: bool = False, backend: str = "vllm"):
    """Main evaluation loop."""
    global inference_backend

    print("\n" + "=" * 80)
    print(f"🚫 BLIND BASELINE EVALUATION — {model_size} (NO IMAGES)")
    if test_mode:
        print(f"   🧪 TEST MODE — Running {max_questions or 3} questions to 'test' folder")
    print("=" * 80 + "\n")

    # Load ALL question types (same as video baseline with --question-types all)
    question_types = ALL_SEQUENTIAL_QUESTION_TYPES + ["obj_appearance_order"]
    questions = _load_vsi_bench_questions(question_types=question_types, dataset="combined")
    total_questions = len(questions)

    print(f"[INFO] Total questions: {total_questions}")

    if max_questions:
        questions = questions[:max_questions]
        print(f"[INFO] Limited to {max_questions} questions\n")

    # Create experiment directory
    if test_mode:
        exp_dir = Path("test") / f"blind_{model_size}"
        exp_dir.mkdir(parents=True, exist_ok=True)
    else:
        EXPERIMENT_BASE = Path("/dss/mcmlscratch/06/di38riq/experiment_logs")
        date_folder = datetime.now().strftime("%Y-%m-%d")
        exp_timestamp = timestamp_str()
        model_name = MODEL_ID.split("/")[-1].replace("-Instruct", "")
        exp_dir = EXPERIMENT_BASE / "Blind" / model_size / date_folder / f"{exp_timestamp}_blind_{model_name}_combined"
        exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 📁 Output: {exp_dir.resolve()}\n")

    # Initialize backend
    print(f"[INFO] Initializing {backend.upper()} backend (text-only mode)...")
    backend_kwargs = {
        "backend": backend,
        "model_id": MODEL_ID,
        "cache_dir": CACHE_DIR,
    }
    # Even though blind, vLLM still needs max_images set (it's part of model config)
    if backend.lower() in ["vllm", "vllm_video"]:
        backend_kwargs["max_images"] = 1  # Minimal — won't use any
    inference_backend = create_inference_backend(**backend_kwargs)
    print("[INFO] Backend ready.\n")

    # Track results
    results = []
    csv_rows = []

    # Process questions
    for idx, q_data in enumerate(questions, 1):
        scene_id = q_data["scene_name"]
        q_dataset = q_data.get("dataset", "unknown")

        print(f"\n{'='*60}")
        print(f"Question {idx}/{len(questions)}")
        print(f"Scene: {scene_id} | Type: {q_data['question_type']} | Dataset: {q_dataset}")
        print(f"{'='*60}")

        try:
            model_answer, elapsed_time = run_single_question(
                scene_id=scene_id,
                question=q_data["question"],
                choices=q_data["choices"],
                question_id=idx,
                experiment_dir=exp_dir,
                question_type=q_data["question_type"],
                dataset=q_dataset,
                is_numerical=q_data.get("is_numerical", False),
            )
        except Exception as e:
            print(f"[ERROR] Failed: {e}")
            import traceback
            traceback.print_exc()
            model_answer = None
            elapsed_time = 0

        if model_answer is None:
            model_answer = "NO_ANSWER"

        # Evaluate
        gt_answer = q_data["answer_id"]
        is_correct = (model_answer == gt_answer)

        print(f"[Q{idx:03d}] Model: {model_answer} | GT: {gt_answer} | {'✅' if is_correct else '❌'}")

        results.append({
            "scene_id": scene_id,
            "question": q_data["question"],
            "model_answer": model_answer,
            "ground_truth": gt_answer,
            "correct": is_correct,
            "question_type": q_data["question_type"],
            "num_frames": 0,
        })

        csv_rows.append({
            "question_id": f"q{idx:03d}",
            "scene_id": scene_id,
            "question_type": q_data["question_type"],
            "is_numerical": q_data.get("is_numerical", False),
            "gt_answer": gt_answer,
            "model_answer": model_answer,
            "mra_score": None,
            "time_seconds": elapsed_time,
            "num_steps": 0,
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
    print(f"📊 FINAL RESULTS — Blind {model_size}")
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
    parser = argparse.ArgumentParser(description="Blind baseline evaluation for VSI-Bench (no images)")
    parser.add_argument("--backend", type=str, default="vllm", choices=["hf", "vllm"],
                       help="Inference backend: 'vllm' (default) or 'hf'")
    parser.add_argument("--model", type=str, default="8B", choices=["4B", "8B"],
                       help="Model size (4B or 8B)")
    parser.add_argument("--max-questions", type=int, default=None, help="Max questions to process")
    parser.add_argument("--test", action="store_true", help="Test mode: run limited questions to 'test' folder")
    args = parser.parse_args()

    if args.test and args.max_questions is None:
        args.max_questions = 3

    main(
        model_size=args.model,
        max_questions=args.max_questions,
        test_mode=args.test,
        backend=args.backend,
    )
