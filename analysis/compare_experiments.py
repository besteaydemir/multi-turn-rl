#!/usr/bin/env python3
"""
Compare VSI-Bench experiment results across different models, methods, and configurations.
Creates comprehensive comparison tables and statistics.

Usage:
    python analysis/compare_experiments.py
    python analysis/compare_experiments.py --filter combined  # Only combined dataset experiments
    python analysis/compare_experiments.py --export results.csv
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# Base directory for experiments
EXPERIMENT_BASE = Path("/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs")


def load_experiment_results(exp_dir: Path) -> Tuple[List[Dict], Dict]:
    """Load all question results from an experiment directory."""
    results = []
    metadata = {
        "total_questions": 0,
        "completed": 0,
        "failed": 0,
    }
    
    # Try loading results.json first (contains all results with ground truth)
    results_file = exp_dir / "results.json"
    if results_file.exists():
        try:
            with open(results_file) as f:
                results = json.load(f)
                metadata["completed"] = len(results)
                metadata["total_questions"] = len(results)
                return results, metadata
        except Exception as e:
            print(f"Warning: Failed to load {results_file}: {e}")
    
    # Fallback: load from individual q* folders
    q_folders = sorted([f for f in exp_dir.iterdir() if f.is_dir() and f.name.startswith("q")])
    metadata["total_questions"] = len(q_folders)
    
    for q_folder in q_folders:
        result_file = q_folder / "result.json"
        if result_file.exists():
            try:
                with open(result_file) as f:
                    result = json.load(f)
                    results.append(result)
                    metadata["completed"] += 1
            except Exception as e:
                metadata["failed"] += 1
        else:
            metadata["failed"] += 1
    
    return results, metadata


def parse_experiment_name(exp_name: str) -> Dict:
    """Parse experiment directory name to extract configuration."""
    parts = exp_name.split("_")
    
    config = {
        "timestamp": parts[0],
        "method": None,
        "model": None,
        "dataset": None,
        "frames_steps": None,
        "split": None,
    }
    
    # Method (sequential or video)
    if "sequential" in exp_name:
        config["method"] = "sequential"
    elif "video" in exp_name:
        config["method"] = "video"
    
    # Model
    for part in parts:
        if "Qwen3-VL" in part:
            config["model"] = part
            break
    
    # Dataset
    for part in parts:
        if part in ["arkitscenes", "scannet", "combined"]:
            config["dataset"] = part
            break
    
    # Frames/Steps
    for part in parts:
        if "frames" in part or "steps" in part:
            config["frames_steps"] = part
            break
    
    # Split info
    for part in parts:
        if "split" in part:
            config["split"] = part
            break
    
    return config


def aggregate_splits(experiments: List[Tuple[Path, Dict, List[Dict], Dict]]) -> Dict:
    """Aggregate results from multiple splits of the same experiment."""
    all_results = []
    total_completed = 0
    total_failed = 0
    elapsed_times = []
    
    for exp_dir, config, results, metadata in experiments:
        all_results.extend(results)
        total_completed += metadata["completed"]
        total_failed += metadata["failed"]
        
        # Collect elapsed times for std/stderr calculation
        for result in results:
            if "elapsed_time" in result:
                elapsed_times.append(result["elapsed_time"])
    
    if not all_results:
        return None
    
    # Calculate statistics
    df = pd.DataFrame(all_results)
    
    # Determine correctness
    correct = []
    for _, row in df.iterrows():
        if row.get("model_answer") == "NO_ANSWER":
            correct.append(False)
        else:
            gt = row.get("ground_truth")
            pred = row.get("model_answer")
            correct.append(pred == gt)
    
    df["correct"] = correct
    
    # Calculate per-split statistics for error bars
    split_accuracies = []
    if len(experiments) > 1:  # If we have multiple splits
        for exp_dir, config, results, metadata in experiments:
            if results:
                split_df = pd.DataFrame(results)
                split_correct = []
                for _, row in split_df.iterrows():
                    if row.get("model_answer") == "NO_ANSWER":
                        split_correct.append(False)
                    else:
                        gt = row.get("ground_truth")
                        pred = row.get("model_answer")
                        split_correct.append(pred == gt)
                split_df["correct"] = split_correct
                split_accuracies.append(split_df["correct"].mean() * 100)
    
    stats = {
        "method": config["method"],
        "model": config["model"],
        "dataset": config["dataset"],
        "frames_steps": config["frames_steps"],
        "total_questions": len(df),
        "completed": total_completed,
        "failed": total_failed,
        "accuracy": df["correct"].mean() * 100,
        "accuracy_std": np.std(split_accuracies) if len(split_accuracies) > 1 else 0,
        "accuracy_sem": np.std(split_accuracies) / np.sqrt(len(split_accuracies)) if len(split_accuracies) > 1 else 0,
        "num_splits": len(experiments),
        "success_rate": (df["model_answer"] != "NO_ANSWER").mean() * 100,
        "avg_time": np.mean(elapsed_times) if elapsed_times else 0,
        "time_std": np.std(elapsed_times) if len(elapsed_times) > 1 else 0,
    }
    
    # Breakdown by dataset (if combined)
    if "dataset" in df.columns and config["dataset"] == "combined":
        for ds in ["arkitscenes", "scannet"]:
            ds_df = df[df["dataset"] == ds]
            if len(ds_df) > 0:
                stats[f"acc_{ds}"] = ds_df["correct"].mean() * 100
                stats[f"n_{ds}"] = len(ds_df)
    
    # Breakdown by question type
    if "question_type" in df.columns:
        by_type = {}
        for qtype in df["question_type"].unique():
            type_df = df[df["question_type"] == qtype]
            by_type[qtype] = {
                "count": len(type_df),
                "accuracy": type_df["correct"].mean() * 100,
            }
        stats["by_question_type"] = by_type
    
    return stats


def find_and_group_experiments(base_dir: Path, filter_dataset: str = None) -> Dict:
    """Find all experiments and group by configuration (merging splits)."""
    grouped = defaultdict(list)
    
    for method_dir in ["Video", "Sequential"]:
        method_path = base_dir / method_dir
        if not method_path.exists():
            continue
        
        for model_size_dir in ["4B", "8B"]:
            model_path = method_path / model_size_dir
            if not model_path.exists():
                continue
            
            for exp_dir in model_path.iterdir():
                if not exp_dir.is_dir():
                    continue
                
                config = parse_experiment_name(exp_dir.name)
                
                # Filter by dataset if specified
                if filter_dataset and config["dataset"] != filter_dataset:
                    continue
                
                # Load results
                results, metadata = load_experiment_results(exp_dir)
                
                if not results:
                    continue
                
                # Group by method + model + dataset + frames_steps (ignore split)
                key = (config["method"], config["model"], config["dataset"], config["frames_steps"])
                grouped[key].append((exp_dir, config, results, metadata))
    
    return grouped


def create_comparison_table(aggregated_stats: List[Dict]) -> pd.DataFrame:
    """Create a comparison table from aggregated statistics."""
    rows = []
    
    for stats in aggregated_stats:
        row = {
            "Method": stats["method"].title(),
            "Model": stats["model"],
            "Dataset": stats["dataset"],
            "Frames/Steps": stats["frames_steps"],
            "Questions": stats["total_questions"],
            "Accuracy (%)": f"{stats['accuracy']:.2f}",
            "Success Rate (%)": f"{stats['success_rate']:.2f}",
            "Avg Time (s)": f"{stats['avg_time']:.2f}",
        }
        
        # Add dataset breakdowns if available
        if stats["dataset"] == "combined":
            if "acc_arkitscenes" in stats:
                row["ARKit Acc (%)"] = f"{stats['acc_arkitscenes']:.2f}"
                row["ARKit N"] = stats["n_arkitscenes"]
            if "acc_scannet" in stats:
                row["ScanNet Acc (%)"] = f"{stats['acc_scannet']:.2f}"
                row["ScanNet N"] = stats["n_scannet"]
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by method, model, frames/steps
    df = df.sort_values(["Method", "Model", "Frames/Steps"])
    
    return df


def create_question_type_table(aggregated_stats: List[Dict]) -> pd.DataFrame:
    """Create a detailed table breaking down accuracy by question type."""
    rows = []
    
    for stats in aggregated_stats:
        if "by_question_type" not in stats:
            continue
        
        base_info = {
            "Method": stats["method"].title(),
            "Model": stats["model"],
            "Dataset": stats["dataset"],
            "Frames/Steps": stats["frames_steps"],
        }
        
        for qtype, qtype_stats in stats["by_question_type"].items():
            row = base_info.copy()
            row["Question Type"] = qtype
            row["Count"] = qtype_stats["count"]
            row["Accuracy (%)"] = f"{qtype_stats['accuracy']:.2f}"
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort
    df = df.sort_values(["Method", "Model", "Frames/Steps", "Question Type"])
    
    return df


def create_best_configs_table(aggregated_stats: List[Dict]) -> pd.DataFrame:
    """Create a table showing best configurations for each model/method combination."""
    best_configs = {}
    
    for stats in aggregated_stats:
        key = (stats["method"], stats["model"], stats["dataset"])
        
        if key not in best_configs or stats["accuracy"] > best_configs[key]["accuracy"]:
            best_configs[key] = stats
    
    rows = []
    for (method, model, dataset), stats in best_configs.items():
        rows.append({
            "Method": method.title(),
            "Model": model,
            "Dataset": dataset,
            "Best Config": stats["frames_steps"],
            "Accuracy (%)": f"{stats['accuracy']:.2f}",
            "Questions": stats["total_questions"],
            "Avg Time (s)": f"{stats['avg_time']:.2f}",
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values(["Dataset", "Method", "Model"])
    
    return df


def print_summary(aggregated_stats: List[Dict]):
    """Print a summary of all experiments."""
    print("\n" + "=" * 80)
    print("VSI-BENCH EXPERIMENT COMPARISON")
    print("=" * 80 + "\n")
    
    print(f"Total experiment configurations: {len(aggregated_stats)}\n")
    
    # Overall comparison table
    print("=" * 80)
    print("OVERALL COMPARISON")
    print("=" * 80)
    comparison_df = create_comparison_table(aggregated_stats)
    print(comparison_df.to_string(index=False))
    print()
    
    # Best configurations
    print("=" * 80)
    print("BEST CONFIGURATIONS")
    print("=" * 80)
    best_df = create_best_configs_table(aggregated_stats)
    print(best_df.to_string(index=False))
    print()
    
    # Method comparison
    print("=" * 80)
    print("METHOD COMPARISON (Combined Dataset)")
    print("=" * 80)
    combined_stats = [s for s in aggregated_stats if s["dataset"] == "combined"]
    if combined_stats:
        method_comparison = []
        for method in ["sequential", "video"]:
            method_stats = [s for s in combined_stats if s["method"] == method]
            if method_stats:
                avg_acc = np.mean([s["accuracy"] for s in method_stats])
                avg_time = np.mean([s["avg_time"] for s in method_stats])
                method_comparison.append({
                    "Method": method.title(),
                    "Avg Accuracy (%)": f"{avg_acc:.2f}",
                    "Avg Time (s)": f"{avg_time:.2f}",
                    "Configs": len(method_stats),
                })
        method_df = pd.DataFrame(method_comparison)
        print(method_df.to_string(index=False))
    else:
        print("No combined dataset experiments found.")
    print()
    
    # Model size comparison
    print("=" * 80)
    print("MODEL SIZE COMPARISON (Combined Dataset)")
    print("=" * 80)
    if combined_stats:
        model_comparison = []
        for model in ["Qwen3-VL-4B", "Qwen3-VL-8B"]:
            model_stats = [s for s in combined_stats if model in s["model"]]
            if model_stats:
                avg_acc = np.mean([s["accuracy"] for s in model_stats])
                avg_time = np.mean([s["avg_time"] for s in model_stats])
                model_comparison.append({
                    "Model": model,
                    "Avg Accuracy (%)": f"{avg_acc:.2f}",
                    "Avg Time (s)": f"{avg_time:.2f}",
                    "Configs": len(model_stats),
                })
        model_df = pd.DataFrame(model_comparison)
        print(model_df.to_string(index=False))
    else:
        print("No combined dataset experiments found.")
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare VSI-Bench experiment results")
    parser.add_argument("--filter", type=str, choices=["arkitscenes", "scannet", "combined"],
                       help="Filter experiments by dataset")
    parser.add_argument("--export", type=str, help="Export comparison table to CSV")
    parser.add_argument("--export-types", type=str, help="Export question type breakdown to CSV")
    args = parser.parse_args()
    
    print("Loading experiments...")
    grouped_experiments = find_and_group_experiments(EXPERIMENT_BASE, args.filter)
    
    print(f"Found {len(grouped_experiments)} experiment configurations")
    
    # Aggregate splits
    print("Aggregating splits...")
    aggregated_stats = []
    for key, experiments in grouped_experiments.items():
        stats = aggregate_splits(experiments)
        if stats:
            aggregated_stats.append(stats)
    
    if not aggregated_stats:
        print("No experiment results found!")
        return
    
    # Print summary
    print_summary(aggregated_stats)
    
    # Question type breakdown
    print("=" * 80)
    print("ACCURACY BY QUESTION TYPE (Top 10 rows)")
    print("=" * 80)
    qtype_df = create_question_type_table(aggregated_stats)
    print(qtype_df.head(10).to_string(index=False))
    print(f"\n... ({len(qtype_df)} total rows)")
    print()
    
    # Export if requested
    if args.export:
        comparison_df = create_comparison_table(aggregated_stats)
        comparison_df.to_csv(args.export, index=False)
        print(f"Exported comparison table to: {args.export}")
    
    if args.export_types:
        qtype_df = create_question_type_table(aggregated_stats)
        qtype_df.to_csv(args.export_types, index=False)
        print(f"Exported question type breakdown to: {args.export_types}")


if __name__ == "__main__":
    main()
