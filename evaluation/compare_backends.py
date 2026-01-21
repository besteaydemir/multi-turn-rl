#!/usr/bin/env python3
"""
Speedup comparison between HuggingFace and vLLM backends.

This script runs a small test set with both backends and compares:
- Total inference time
- Per-question time
- Per-step time
- Speedup factor
- GPU memory usage

Usage:
  # Compare both backends
  python compare_backends.py --num-questions 5
  
  # Run single backend with wandb tracking
  python compare_backends.py --backend hf --num-questions 5 --use-wandb
  python compare_backends.py --backend vllm --num-questions 5 --use-wandb
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import os

import numpy as np
import pandas as pd
import torch
import pynvml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    timestamp_str,
    create_inference_backend,
    find_mesh_file,
    load_mesh_cached,
    get_mesh_bounds,
    render_mesh_from_pose,
    look_at_camera_pose_center_from_forward,
    MCA_QUESTION_TYPES,
    parse_rotation_angle,
    apply_movement_in_camera_frame,
    parse_qwen_output_and_get_movement,
    save_matrix,
)
from utils.data import load_vsi_bench_questions

# Import the instruction builder from sequential.py to ensure consistency
from sequential import build_instruction_text, _get_question_type_guidance, _build_history_context

# Config
CACHE_DIR = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MESH_BASE_DIR = "/dss/mcmlscratch/06/di38riq/arkit_vsi/raw"
IMAGE_WH = (640, 480)
DEFAULT_FX_FY = 300.0
CAM_HEIGHT = 1.6


class GPUMemoryTracker:
    """Track GPU memory usage using pynvml."""
    
    def __init__(self, device_id=0):
        self.device_id = device_id
        self.torch_device = f"cuda:{device_id}"
        # Initialize PyTorch CUDA context first
        torch.cuda.set_device(self.torch_device)
        torch.cuda.init()
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        
    def get_memory_info(self):
        """Get current GPU memory usage in MB."""
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return {
            "used_mb": info.used / 1024**2,
            "free_mb": info.free / 1024**2,
            "total_mb": info.total / 1024**2,
            "utilization": (info.used / info.total) * 100,
        }
    
    def get_peak_memory(self):
        """Get peak GPU memory allocated by PyTorch."""
        return torch.cuda.max_memory_allocated(self.torch_device) / 1024**2
    
    def reset_peak(self):
        """Reset peak memory counter."""
        torch.cuda.reset_peak_memory_stats(self.torch_device)
    
    def cleanup(self):
        """Cleanup NVML."""
        try:
            pynvml.nvmlShutdown()
        except:
            pass


def compute_text_similarity(text1: str, text2: str) -> float:
    """
    Compute similarity between two texts using character-level overlap.
    Returns a score between 0 and 1.
    """
    # Simple character-level Jaccard similarity
    set1 = set(text1.lower())
    set2 = set(text2.lower())
    
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0


def compare_outputs(hf_outputs: list, vllm_outputs: list):
    """
    Compare outputs between HF and vLLM backends.
    
    Args:
        hf_outputs: List of output dicts from HF backend
        vllm_outputs: List of output dicts from vLLM backend
    
    Returns:
        dict with comparison statistics
    """
    print("\n" + "="*80)
    print("OUTPUT COMPARISON")
    print("="*80)
    
    comparisons = []
    
    for hf_q, vllm_q in zip(hf_outputs, vllm_outputs):
        q_idx = hf_q["question_idx"]
        question = hf_q["question"]
        
        print(f"\n[Q{q_idx}] {question[:60]}...")
        
        for step_idx, (hf_out, vllm_out) in enumerate(zip(hf_q["outputs"], vllm_q["outputs"]), 1):
            # Compute similarity
            similarity = compute_text_similarity(hf_out, vllm_out)
            
            # Check if outputs are identical
            identical = (hf_out.strip() == vllm_out.strip())
            
            comparison = {
                "question_idx": q_idx,
                "step": step_idx,
                "identical": identical,
                "similarity": similarity,
                "hf_output": hf_out,
                "vllm_output": vllm_out,
            }
            comparisons.append(comparison)
            
            status = "✅ IDENTICAL" if identical else f"≈ {similarity:.1%} similar"
            print(f"  Step {step_idx}: {status}")
            
            if not identical and similarity < 0.9:
                print(f"    HF:   {hf_out[:100]}...")
                print(f"    vLLM: {vllm_out[:100]}...")
    
    # Compute statistics
    total_comparisons = len(comparisons)
    num_identical = sum(1 for c in comparisons if c["identical"])
    avg_similarity = np.mean([c["similarity"] for c in comparisons]) if comparisons else 0
    
    print("\n" + "-"*80)
    print(f"Total comparisons: {total_comparisons}")
    print(f"Identical outputs: {num_identical}/{total_comparisons} ({num_identical/total_comparisons*100:.1f}%)")
    print(f"Average similarity: {avg_similarity:.1%}")
    print("="*80)
    
    return {
        "comparisons": comparisons,
        "total_comparisons": total_comparisons,
        "num_identical": num_identical,
        "percent_identical": num_identical / total_comparisons * 100 if total_comparisons > 0 else 0,
        "avg_similarity": avg_similarity,
    }


def run_inference_test(backend_type: str, num_questions: int = 5, num_steps: int = 3, use_wandb: bool = False, wandb_run=None):
    """
    Run inference test with specified backend.
    
    Args:
        backend_type: "hf" or "vllm"
        num_questions: Number of questions to test
        num_steps: Number of steps per question
        use_wandb: Whether to log to wandb
        wandb_run: Wandb run object (if already initialized)
    
    Returns:
        dict with timing statistics, memory usage, and generated outputs
    """
    print(f"\n{'='*60}")
    print(f"Testing {backend_type.upper()} Backend")
    print(f"{'='*60}")
    
    # Initialize memory tracker
    gpu_tracker = GPUMemoryTracker(device_id=0)
    
    # Get initial memory state
    initial_mem = gpu_tracker.get_memory_info()
    print(f"[{backend_type}] Initial GPU memory: {initial_mem['used_mb']:.1f}MB / {initial_mem['total_mb']:.1f}MB")
    
    # Reset peak memory counter
    gpu_tracker.reset_peak()
    
    # Initialize backend
    start_init = time.time()
    backend = create_inference_backend(
        backend=backend_type,
        model_id=MODEL_ID,
        cache_dir=CACHE_DIR,
    )
    init_time = time.time() - start_init
    
    # Get memory after model loading
    post_load_mem = gpu_tracker.get_memory_info()
    model_memory_mb = post_load_mem['used_mb'] - initial_mem['used_mb']
    
    print(f"[{backend_type}] Initialization time: {init_time:.2f}s")
    print(f"[{backend_type}] Model memory: {model_memory_mb:.1f}MB")
    print(f"[{backend_type}] Post-load GPU memory: {post_load_mem['used_mb']:.1f}MB / {post_load_mem['total_mb']:.1f}MB ({post_load_mem['utilization']:.1f}%)")
    
    if use_wandb and wandb_run:
        wandb_run.log({
            f"{backend_type}/init_time_s": init_time,
            f"{backend_type}/model_memory_mb": model_memory_mb,
            f"{backend_type}/post_load_memory_mb": post_load_mem['used_mb'],
            f"{backend_type}/memory_utilization": post_load_mem['utilization'],
        })
    
    # Load questions
    questions = load_vsi_bench_questions(question_types=MCA_QUESTION_TYPES, dataset="arkitscenes")
    questions = questions[:num_questions]
    
    # Create experiment output directory
    exp_timestamp = timestamp_str()
    exp_dir = Path(f"compare_logs/{backend_type}_{exp_timestamp}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{backend_type}] Experiment dir: {exp_dir}")
    
    # Timing stats and outputs
    question_times = []
    step_times = []
    outputs_list = []  # Store all outputs for comparison
    accuracy_results = []  # Track answers and correctness
    total_start = time.time()
    
    for q_idx, q_data in enumerate(questions, 1):
        scene_id = q_data["scene_name"]
        mesh_file = find_mesh_file(scene_id, MESH_BASE_DIR)
        
        if mesh_file is None:
            print(f"[{backend_type}] Q{q_idx}: No mesh for {scene_id}, skipping")
            continue
        
        print(f"\n[{backend_type}] Q{q_idx}/{num_questions}: {q_data['question'][:50]}...")
        q_start = time.time()
        
        # Create question folder
        q_dir = exp_dir / f"q{q_idx:03d}"
        q_dir.mkdir(parents=True, exist_ok=True)
        
        # Load mesh and get bounds
        mesh = load_mesh_cached(mesh_file)
        bbox_mins, bbox_maxs = get_mesh_bounds(mesh, percentile_filter=True)
        
        # Create initial camera pose
        center_x = (bbox_mins[0] + bbox_maxs[0]) / 2.0
        center_y = (bbox_mins[1] + bbox_maxs[1]) / 2.0
        cam_height_z = bbox_mins[2] + CAM_HEIGHT
        eye = np.array([center_x, center_y, cam_height_z], dtype=float)
        forward = np.array([1.0, 0.0, 0.0], dtype=float)
        current_pose = look_at_camera_pose_center_from_forward(eye, forward=forward, up=np.array([0.0, 0.0, -1.0]))
        
        # Initialize exploration state
        R_current = current_pose[:3, :3]
        t_current = current_pose[:3, 3]
        cam_history = [current_pose.copy()]  # Track all poses for trajectory
        
        # Render initial image to question folder
        img_path_0 = q_dir / "render_00.png"
        render_mesh_from_pose(mesh, current_pose, img_path_0, fxfy=DEFAULT_FX_FY, image_wh=IMAGE_WH)
        
        # Image history grows with each step
        image_history = [str(img_path_0)]
        position_history = []  # Track movement history for prompt
        question_outputs = []  # Store outputs for this question
        final_answer = None
        question_type = q_data.get('question_type', 'unknown')
        is_numerical = q_data.get('is_numerical', False)
        
        # Run inference steps with growing image history (0 to num_steps inclusive, like sequential.py)
        for step in range(num_steps + 1):
            step_start = time.time()
            is_final_step = (step == num_steps)
            
            # Get memory before inference
            pre_inference_mem = gpu_tracker.get_memory_info()
            
            # Build full instruction text using the same function as sequential.py
            instruction_text = build_instruction_text(
                R_current, t_current, q_data['question'],
                bbox=(bbox_mins, bbox_maxs),
                options=q_data['choices'],
                is_final_step=is_final_step,
                movement_history=position_history,
                step_num=step,
                question_type=question_type,
                is_numerical=is_numerical
            )
            
            # Build history context
            history_context = _build_history_context(cam_history)
            full_prompt = history_context + instruction_text
            
            # Build message with ALL images from history (growing context)
            content = []
            
            # Add all images with labels (same as sequential.py)
            for img_idx, img_path in enumerate(image_history):
                label = f"\n**Image {img_idx} (Initial view):**" if img_idx == 0 else f"\n**Image {img_idx} (After movement {img_idx}):**"
                content.append({"type": "text", "text": label})
                content.append({"type": "image", "image": img_path})
            
            content.append({"type": "text", "text": f"\n\n{full_prompt}"})
            
            messages = [{"role": "user", "content": content}]
            
            # Create step folder and save inputs (same as sequential.py)
            step_folder = q_dir / f"step_{step:02d}"
            step_folder.mkdir(parents=True, exist_ok=True)
            
            # Save input prompt for debugging
            with open(step_folder / "qwen_input_prompt.txt", "w", encoding="utf-8") as f:
                f.write(full_prompt)
            
            # Run inference (use 1024 tokens like sequential.py)
            output = backend.generate(messages, max_new_tokens=1024)
            question_outputs.append(output)
            
            # Save raw output
            with open(step_folder / "qwen_raw_output.txt", "w", encoding="utf-8") as f:
                f.write(output)
            
            # Get memory after inference
            post_inference_mem = gpu_tracker.get_memory_info()
            
            step_time = time.time() - step_start
            step_times.append(step_time)
            print(f"    Step {step}/{num_steps}: {step_time:.2f}s | Memory: {post_inference_mem['used_mb']:.1f}MB | Images: {len(image_history)}")
            
            if use_wandb and wandb_run:
                wandb_run.log({
                    f"{backend_type}/step_time_s": step_time,
                    f"{backend_type}/step_memory_mb": post_inference_mem['used_mb'],
                    f"{backend_type}/step": q_idx * num_steps + step,
                    f"{backend_type}/num_images": len(image_history),
                })
            
            # Parse movement from output using the same robust parser as sequential.py
            rotation_angle, forward_m, left_m, z_delta_m, reasoning_text, raw_obj, done_flag = parse_qwen_output_and_get_movement(output)
            
            # Check for answer (same logic as sequential.py)
            if raw_obj and isinstance(raw_obj, dict):
                answer_value = raw_obj.get("answer")
                has_answer = answer_value is not None and str(answer_value).strip().upper() in "ABCDEFGHIJ"
                
                if has_answer and done_flag:
                    final_answer = str(answer_value).strip().upper()
                    print(f"    ✅ Final answer: {final_answer}")
            
            # If we got an answer and done flag, or if this is the final step, stop
            if (done_flag and final_answer) or is_final_step:
                break
            
            # Apply movement and render new image if not done (same logic as sequential.py)
            if rotation_angle is not None and forward_m is not None and left_m is not None and z_delta_m is not None:
                # Apply rotation first using the same function as sequential.py
                R_new = parse_rotation_angle(rotation_angle, R_current)
                # Then apply movement using the same function as sequential.py
                t_new = apply_movement_in_camera_frame(R_new, t_current, forward_m, left_m, z_delta_m,
                                                       bbox_mins=bbox_mins, bbox_maxs=bbox_maxs)
                
                # Update pose
                new_pose = np.eye(4, dtype=float)
                new_pose[:3, :3] = R_new
                new_pose[:3, 3] = t_new
                
                # Track movement history for prompt (same as sequential.py)
                position_history.append({
                    "rotation": rotation_angle,
                    "forward": forward_m,
                    "left": left_m,
                    "z_delta": z_delta_m,
                    "position": f"X={t_new[0]:.2f}m, Y={t_new[1]:.2f}m, Z={t_new[2]:.2f}m"
                })
                
                R_current = R_new
                t_current = t_new
                cam_history.append(new_pose.copy())
                
                # Render new image to question folder
                new_img_path = q_dir / f"render_{step+1:02d}.png"
                render_mesh_from_pose(mesh, new_pose, new_img_path, fxfy=DEFAULT_FX_FY, image_wh=IMAGE_WH)
                image_history.append(str(new_img_path))
            else:
                # No valid movement parsed - skip rendering new image this step
                print(f"    ⚠️  No valid movement parsed at step {step}, skipping render")
        
        # Save camera pose files (same as sequential.py)
        for pose_idx, pose in enumerate(cam_history):
            pose_file = q_dir / f"cam_pose_{pose_idx:02d}.npy"
            save_matrix(pose_file, pose)
        
        # Save trajectory data for this question
        trajectory_data = {
            "poses": [
                {
                    "file": f"cam_pose_{i:02d}.npy",
                    "position": pose[:3, 3].tolist(),
                    "rotation": pose[:3, :3].tolist(),
                    "matrix": pose.tolist()
                }
                for i, pose in enumerate(cam_history)
            ],
            "num_poses": len(cam_history),
            "question_id": q_idx,
            "scene_id": scene_id,
            "question": q_data['question'],
            "choices": q_data['choices'],
            "ground_truth": q_data.get('answer_id', None),
            "model_answer": final_answer,
            "question_type": q_data.get('question_type', 'unknown'),
        }
        
        with open(q_dir / "trajectory.json", "w") as f:
            json.dump(trajectory_data, f, indent=2)
        
        # Evaluate answer
        ground_truth = q_data.get('answer_id', None)
        is_correct = (final_answer == ground_truth) if final_answer and ground_truth else False
        
        accuracy_results.append({
            "question_idx": q_idx,
            "scene_id": scene_id,
            "question": q_data['question'],
            "ground_truth": ground_truth,
            "model_answer": final_answer,
            "is_correct": is_correct,
            "question_type": q_data.get('question_type', 'unknown'),
            "num_steps": len(image_history) - 1,
        })
        
        # Save per-question results
        with open(q_dir / "results.json", "w") as f:
            json.dump(accuracy_results[-1], f, indent=2)
        
        outputs_list.append({
            "question_idx": q_idx,
            "question": q_data['question'],
            "outputs": question_outputs,
            "num_images": len(image_history),
            "model_answer": final_answer,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
        })
        
        q_time = time.time() - q_start
        question_times.append(q_time)
        print(f"  Question total: {q_time:.2f}s | Total images rendered: {len(image_history)}")
    
    total_time = time.time() - total_start
    
    # Get peak memory usage
    peak_memory_mb = gpu_tracker.get_peak_memory()
    final_mem = gpu_tracker.get_memory_info()
    
    # Cleanup
    backend.cleanup()
    torch.cuda.empty_cache()
    
    # Get memory after cleanup
    post_cleanup_mem = gpu_tracker.get_memory_info()
    
    print(f"\n[{backend_type}] Memory Summary:")
    print(f"  Peak allocated: {peak_memory_mb:.1f}MB")
    print(f"  Final memory: {final_mem['used_mb']:.1f}MB")
    print(f"  After cleanup: {post_cleanup_mem['used_mb']:.1f}MB")
    
    # Cleanup tracker
    gpu_tracker.cleanup()
    
    # Compute stats
    # Calculate accuracy statistics
    correct_count = sum(1 for r in accuracy_results if r['is_correct'])
    total_count = len(accuracy_results)
    accuracy_percent = 100 * correct_count / total_count if total_count > 0 else 0.0
    
    # Group accuracy by question type
    accuracy_by_type = {}
    for r in accuracy_results:
        q_type = r.get('question_type', 'unknown')
        if q_type not in accuracy_by_type:
            accuracy_by_type[q_type] = {'correct': 0, 'total': 0}
        accuracy_by_type[q_type]['total'] += 1
        if r['is_correct']:
            accuracy_by_type[q_type]['correct'] += 1
    
    for q_type in accuracy_by_type:
        stats_type = accuracy_by_type[q_type]
        stats_type['accuracy'] = 100 * stats_type['correct'] / stats_type['total'] if stats_type['total'] > 0 else 0.0
    
    stats = {
        "backend": backend_type,
        "num_questions": len(question_times),
        "num_steps": num_steps,
        "init_time_s": init_time,
        "total_time_s": total_time,
        "avg_question_time_s": np.mean(question_times) if question_times else 0,
        "avg_step_time_s": np.mean(step_times) if step_times else 0,
        "total_inference_time_s": total_time - init_time,
        "question_times": question_times,
        "step_times": step_times,
        "outputs": outputs_list,
        # Memory stats
        "memory": {
            "model_memory_mb": model_memory_mb,
            "peak_memory_mb": peak_memory_mb,
            "post_load_memory_mb": post_load_mem['used_mb'],
            "final_memory_mb": final_mem['used_mb'],
            "post_cleanup_memory_mb": post_cleanup_mem['used_mb'],
        },
        # Accuracy stats
        "accuracy": {
            "correct": correct_count,
            "total": total_count,
            "accuracy_percent": accuracy_percent,
            "by_question_type": accuracy_by_type,
        },
        "accuracy_results": accuracy_results,
        "experiment_dir": str(exp_dir),
    }
    
    # Save overall experiment summary
    with open(exp_dir / "experiment_summary.json", "w") as f:
        # Create a serializable copy without numpy arrays
        summary = {k: v for k, v in stats.items() if k not in ['question_times', 'step_times']}
        summary['question_times'] = [float(t) for t in question_times]
        summary['step_times'] = [float(t) for t in step_times]
        json.dump(summary, f, indent=2)
    
    if use_wandb and wandb_run:
        wandb_run.log({
            f"{backend_type}/total_time_s": total_time,
            f"{backend_type}/avg_question_time_s": stats['avg_question_time_s'],
            f"{backend_type}/avg_step_time_s": stats['avg_step_time_s'],
            f"{backend_type}/peak_memory_mb": peak_memory_mb,
            f"{backend_type}/accuracy_percent": accuracy_percent,
            f"{backend_type}/correct_count": correct_count,
        })

    print(f"\n[{backend_type}] Summary:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Avg per question: {stats['avg_question_time_s']:.2f}s")
    print(f"  Avg per step: {stats['avg_step_time_s']:.2f}s")
    print(f"  Accuracy: {correct_count}/{total_count} ({accuracy_percent:.1f}%)")
    print(f"  Experiment dir: {exp_dir}")
    
    return stats


def compare_backends(num_questions: int = 5, num_steps: int = 3):
    """
    Compare HuggingFace and vLLM backends.
    
    IMPORTANT NOTES:
    ---------------
    Pitfall #5: This test runs SEQUENTIAL inference (one step at a time).
                vLLM optimizes for THROUGHPUT, not single-request latency!
                
    Expected results:
    - Sequential mode: vLLM may be only 1-2x faster (or even slower)
    - True vLLM advantage comes from batching multiple requests
    
    Pitfall #6: Outputs may differ slightly due to:
    - Different CUDA kernels (different numerical precision)
    - Different KV cache management
    - Minor floating-point rounding differences
    
    With temperature=0, outputs should be SEMANTICALLY identical even if
    character-for-character matching isn't perfect.
    """
    
    print("\n" + "="*80)
    print("BACKEND COMPARISON: HuggingFace vs vLLM")
    print("="*80)
    print(f"Testing with {num_questions} questions, {num_steps} steps each")
    print(f"Model: {MODEL_ID}")
    print("\n⚠️  NOTE: This runs SEQUENTIAL inference (not batched).")
    print("   vLLM's real advantage comes from batching multiple requests!")
    print("="*80)
    
    # Run tests
    hf_stats = run_inference_test("hf", num_questions, num_steps)
    
    # Clear GPU memory before loading vLLM
    torch.cuda.empty_cache()
    
    vllm_stats = run_inference_test("vllm", num_questions, num_steps)
    
    # Compare outputs
    output_comparison = compare_outputs(hf_stats["outputs"], vllm_stats["outputs"])
    
    # Compute speedup
    speedup_total = hf_stats["total_time_s"] / vllm_stats["total_time_s"] if vllm_stats["total_time_s"] > 0 else 0
    speedup_inference = hf_stats["total_inference_time_s"] / vllm_stats["total_inference_time_s"] if vllm_stats["total_inference_time_s"] > 0 else 0
    speedup_step = hf_stats["avg_step_time_s"] / vllm_stats["avg_step_time_s"] if vllm_stats["avg_step_time_s"] > 0 else 0
    
    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(f"\n{'Metric':<30} {'HuggingFace':>15} {'vLLM':>15} {'Speedup':>10}")
    print("-"*70)
    print(f"{'Initialization time (s)':<30} {hf_stats['init_time_s']:>15.2f} {vllm_stats['init_time_s']:>15.2f} {'N/A':>10}")
    print(f"{'Total time (s)':<30} {hf_stats['total_time_s']:>15.2f} {vllm_stats['total_time_s']:>15.2f} {speedup_total:>10.2f}x")
    print(f"{'Inference time (s)':<30} {hf_stats['total_inference_time_s']:>15.2f} {vllm_stats['total_inference_time_s']:>15.2f} {speedup_inference:>10.2f}x")
    print(f"{'Avg per question (s)':<30} {hf_stats['avg_question_time_s']:>15.2f} {vllm_stats['avg_question_time_s']:>15.2f} {hf_stats['avg_question_time_s']/vllm_stats['avg_question_time_s'] if vllm_stats['avg_question_time_s'] > 0 else 0:>10.2f}x")
    print(f"{'Avg per step (s)':<30} {hf_stats['avg_step_time_s']:>15.2f} {vllm_stats['avg_step_time_s']:>15.2f} {speedup_step:>10.2f}x")
    print("="*80)
    
    print(f"\n🚀 vLLM is {speedup_inference:.1f}x faster than HuggingFace for inference!")
    
    # Print output consistency summary
    print(f"\n📊 Output Consistency:")
    print(f"   Identical outputs: {output_comparison['num_identical']}/{output_comparison['total_comparisons']} ({output_comparison['percent_identical']:.1f}%)")
    print(f"   Average similarity: {output_comparison['avg_similarity']:.1%}")
    
    if output_comparison['percent_identical'] > 95:
        print("   ✅ Outputs are highly consistent between backends!")
    elif output_comparison['avg_similarity'] > 0.90:
        print("   ⚠️  Outputs are similar but not identical. This is expected due to numerical differences.")
    else:
        print("   ⚠️  WARNING: Significant differences in outputs detected!")
    
    if speedup_inference < 1.5:
        print("\n⚠️  Speedup lower than expected - this is NORMAL for sequential inference!")
        print("   Pitfall #5: vLLM optimizes THROUGHPUT, not single-request latency")
        print("   ")
        print("   Why you're not seeing big speedup:")
        print("   - This test runs one request at a time (sequential)")
        print("   - vLLM allocates large KV cache upfront (overhead)")
        print("   - First-token latency may be worse than HF")
        print("   ")
        print("   To see real vLLM advantages:")
        print("   - Batch multiple questions together")
        print("   - Use API serving with concurrent requests")
        print("   - Run full evaluation with many questions")
        print("   ")
        print("   Expected speedup:")
        print("   - Sequential (no batching): 0.8-2x")
        print("   - Small batches (4-8): 2-4x")
        print("   - Large batches (16+): 4-8x")
        print("   - API serving: 5-10x throughput")
    elif speedup_inference > 3:
        print("\n🎉 Excellent speedup! vLLM is working well.")
        print("   Even in sequential mode, you're seeing good improvements.")
    
    # Save results
    results = {
        "timestamp": timestamp_str(),
        "num_questions": num_questions,
        "num_steps": num_steps,
        "model_id": MODEL_ID,
        "hf_stats": {k: v for k, v in hf_stats.items() if k not in ["question_times", "step_times", "outputs"]},
        "vllm_stats": {k: v for k, v in vllm_stats.items() if k not in ["question_times", "step_times", "outputs"]},
        "speedup": {
            "total": speedup_total,
            "inference": speedup_inference,
            "per_step": speedup_step,
        },
        "output_comparison": {
            "total_comparisons": output_comparison["total_comparisons"],
            "num_identical": output_comparison["num_identical"],
            "percent_identical": output_comparison["percent_identical"],
            "avg_similarity": output_comparison["avg_similarity"],
        }
    }
    
    output_file = Path("comparison_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Results saved to {output_file}")
    
    # Save detailed outputs for inspection
    detailed_file = Path("comparison_outputs_detailed.json")
    detailed_results = {
        "timestamp": timestamp_str(),
        "hf_outputs": hf_stats["outputs"],
        "vllm_outputs": vllm_stats["outputs"],
        "comparisons": output_comparison["comparisons"],
    }
    with open(detailed_file, "w") as f:
        json.dump(detailed_results, f, indent=2)
    print(f"[INFO] Detailed outputs saved to {detailed_file}")
    
    return results


def run_single_backend(backend_type: str, num_questions: int = 5, num_steps: int = 3, use_wandb: bool = False):
    """
    Run a single backend test with wandb tracking.
    
    Args:
        backend_type: "hf" or "vllm"
        num_questions: Number of questions to test
        num_steps: Number of steps per question
        use_wandb: Whether to use wandb for tracking
    """
    wandb_run = None
    
    if use_wandb:
        import wandb
        wandb_run = wandb.init(
            project="vllm-hf-comparison",
            name=f"{backend_type}_{timestamp_str()}",
            config={
                "backend": backend_type,
                "model_id": MODEL_ID,
                "num_questions": num_questions,
                "num_steps": num_steps,
            },
            tags=[backend_type, "memory-profiling"],
        )
    
    print("\n" + "="*80)
    print(f"SINGLE BACKEND TEST: {backend_type.upper()}")
    print("="*80)
    print(f"Testing with {num_questions} questions, {num_steps} steps each")
    print(f"Model: {MODEL_ID}")
    if use_wandb:
        print(f"Wandb run: {wandb_run.name}")
    print("="*80)
    
    # Run test
    stats = run_inference_test(backend_type, num_questions, num_steps, use_wandb, wandb_run)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\n{'Metric':<30} {'Value':>15}")
    print("-"*50)
    print(f"{'Total time (s)':<30} {stats['total_time_s']:>15.2f}")
    print(f"{'Avg per question (s)':<30} {stats['avg_question_time_s']:>15.2f}")
    print(f"{'Avg per step (s)':<30} {stats['avg_step_time_s']:>15.2f}")
    print(f"{'Model memory (MB)':<30} {stats['memory']['model_memory_mb']:>15.1f}")
    print(f"{'Peak memory (MB)':<30} {stats['memory']['peak_memory_mb']:>15.1f}")
    print(f"{'Post-load memory (MB)':<30} {stats['memory']['post_load_memory_mb']:>15.1f}")
    print(f"{'Post-cleanup memory (MB)':<30} {stats['memory']['post_cleanup_memory_mb']:>15.1f}")
    print("="*80)
    
    # Save results
    output_file = Path(f"{backend_type}_results_{timestamp_str()}.json")
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": timestamp_str(),
            "backend": backend_type,
            "config": {
                "model_id": MODEL_ID,
                "num_questions": num_questions,
                "num_steps": num_steps,
            },
            "stats": {k: v for k, v in stats.items() if k != "outputs"},
        }, f, indent=2)
    print(f"\n[INFO] Results saved to {output_file}")
    
    if use_wandb:
        wandb_run.finish()
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare HuggingFace vs vLLM backends")
    parser.add_argument("--backend", type=str, choices=["hf", "vllm", "both"], default="both",
                       help="Which backend to run: hf, vllm, or both")
    parser.add_argument("--num-questions", type=int, default=5, help="Number of questions to test")
    parser.add_argument("--num-steps", type=int, default=3, help="Number of steps per question")
    parser.add_argument("--use-wandb", action="store_true", help="Enable wandb tracking")
    args = parser.parse_args()
    
    if args.backend == "both":
        compare_backends(args.num_questions, args.num_steps)
    else:
        run_single_backend(args.backend, args.num_questions, args.num_steps, args.use_wandb)

