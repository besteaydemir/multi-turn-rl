#!/usr/bin/env python3
"""
Analyze Embodied-R1/Embodied-R1-7B-Stage1 evaluation results on VSI-Bench.
Aggregates split results, computes accuracy/MRA, and generates plots
matching the style of the existing analysis directory.

Usage:
    python analysis/analyze_embodied_r1.py
"""

import os
import re
import json
import math
import csv
import warnings
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────

VIDEO_R1_BASE = Path("/dss/mcmlscratch/06/di38riq/experiment_logs/Video/Embodied-R1-7B-Stage1")
EXISTING_ANALYSIS = Path("/dss/dsshome1/06/di38riq/rl_multi_turn/analysis")
OUTPUT_DIR = EXISTING_ANALYSIS / "embodied_r1"
FRAME_CONFIGS = [4, 8, 16, 32]

# Question type categories
MCA_QUESTION_TYPES = [
    "route_planning",
    "object_rel_distance",
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
]

NUMERICAL_QUESTION_TYPES = [
    "object_counting",
    "object_abs_distance",
    "object_size_estimation",
    "room_size_estimation",
]

TEMPORAL_TYPES = ["obj_appearance_order"]

# Scene-to-dataset mapping (based on scene_id prefix patterns)
# ARKitScenes: 8-digit numeric IDs (e.g., 41069025)
# ScanNet:     scene*
# ScanNetpp:   various hex IDs

# Map dataset from question_id ranges (from the original VSI-Bench dataset)
# We'll infer from the data itself.

# ─── Styling ──────────────────────────────────────────────────────────────────

def setup_style():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "serif"]
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.5,
        "axes.edgecolor": "0.3",
        "axes.axisbelow": True,
    })
    sns.set_theme(style="whitegrid", context="paper", palette="muted")

COLORS = {
    "frames": ["#229e4a", "#443acf", "#bd0a0a", "#659df2"],
    "method": ["#443acf", "#bd0a0a", "#ff8c00"],  # Sequential, Video-Qwen, Embodied-R1
    "model": ["#229e4a", "#659df2", "#e15759"],    # 4B, 8B, R1-7B
    "qtype": sns.color_palette("Set2", 10),
}

# ─── MRA Computation ─────────────────────────────────────────────────────────

def calculate_mra(predicted, ground_truth):
    """Multi-Resolution Accuracy with 10 thresholds from 0.5 to 0.95."""
    try:
        predicted = float(predicted)
        ground_truth = float(ground_truth)
    except (ValueError, TypeError):
        return 0.0

    if ground_truth == 0:
        return 1.0 if predicted == 0 else 0.0

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    score = 0
    for theta in thresholds:
        relative_error = abs(predicted - ground_truth) / abs(ground_truth)
        if relative_error < (1 - theta):
            score += 1
    return score / len(thresholds)


def evaluate_answer(model_answer, gt_answer, is_numerical):
    """Evaluate a single answer. Returns (is_correct, mra_score)."""
    if model_answer is None or str(model_answer).strip() in ("", "NO_ANSWER", "None"):
        return False, 0.0

    model_answer = str(model_answer).strip()
    gt_answer = str(gt_answer).strip()

    if is_numerical:
        try:
            pred_val = float(model_answer)
            gt_val = float(gt_answer)
        except (ValueError, TypeError):
            return False, 0.0
        mra = calculate_mra(pred_val, gt_val)
        return mra > 0.5, mra
    else:
        # MCA: case-insensitive exact match
        is_correct = model_answer.upper() == gt_answer.upper()
        return is_correct, 1.0 if is_correct else 0.0

# ─── Data Loading ─────────────────────────────────────────────────────────────

def find_latest_split_dirs(base_dir: Path, n_frames: int, n_splits: int = 4):
    """Find the latest result directories for each split of a given frame config."""
    frame_dir = base_dir / f"{n_frames}_frames"
    if not frame_dir.exists():
        print(f"  [WARN] {frame_dir} does not exist")
        return {}

    # Find all result directories
    all_dirs = {}
    for date_dir in sorted(frame_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        for result_dir in sorted(date_dir.iterdir()):
            if not result_dir.is_dir():
                continue
            # Parse split from directory name
            match = re.search(r"split(\d+)of(\d+)", result_dir.name)
            if match:
                split_num = int(match.group(1))
                results_csv = result_dir / "results.csv"
                if results_csv.exists():
                    line_count = sum(1 for _ in open(results_csv))
                    # Only consider files with > 500 lines (valid runs)
                    if line_count > 500:
                        all_dirs[split_num] = result_dir

    return all_dirs


def load_split_results(result_dir: Path) -> pd.DataFrame:
    """Load results CSV from a split directory."""
    csv_path = result_dir / "results.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    return df


def aggregate_frame_config(base_dir: Path, n_frames: int) -> pd.DataFrame:
    """Aggregate all splits for a given frame configuration."""
    split_dirs = find_latest_split_dirs(base_dir, n_frames)

    if not split_dirs:
        print(f"  [WARN] No valid splits found for {n_frames} frames")
        return pd.DataFrame()

    dfs = []
    for split_num in sorted(split_dirs.keys()):
        df = load_split_results(split_dirs[split_num])
        if len(df) > 0:
            print(f"    Split {split_num}: {len(df)} rows from {split_dirs[split_num].name}")
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    # Remove duplicates by question_id (keep last = latest)
    combined = combined.drop_duplicates(subset="question_id", keep="last")
    return combined


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute is_correct and mra_score for each row."""
    results = []
    for _, row in df.iterrows():
        is_num = row.get("is_numerical", False)
        if isinstance(is_num, str):
            is_num = is_num.lower() == "true"
        gt = str(row["gt_answer"]).strip()
        pred = str(row["model_answer"]).strip()
        is_correct, mra = evaluate_answer(pred, gt, is_num)
        row_dict = row.to_dict()
        row_dict["is_correct"] = is_correct
        row_dict["mra_score_computed"] = mra
        results.append(row_dict)
    return pd.DataFrame(results)

# ─── Infer dataset from scene_id ──────────────────────────────────────────────

def infer_dataset(scene_id):
    """Infer which dataset a scene belongs to based on scene_id format."""
    s = str(scene_id)
    if s.isdigit() and len(s) == 8:
        return "arkitscenes"
    elif s.startswith("scene"):
        return "scannet"
    else:
        return "scannetpp"

# ─── Analysis Functions ────────────────────────────────────────────────────────

def compute_summary_stats(df: pd.DataFrame, n_frames: int):
    """Compute summary statistics for a frame configuration."""
    total = len(df)
    no_answer = df["model_answer"].isin(["NO_ANSWER", "None", ""]).sum() + df["model_answer"].isna().sum()
    answered = total - no_answer
    correct = df["is_correct"].sum()

    accuracy = 100 * correct / total if total > 0 else 0
    success_rate = 100 * answered / total if total > 0 else 0

    # MRA for numerical questions only
    numerical_df = df[df["is_numerical"].astype(str).str.lower() == "true"]
    avg_mra = numerical_df["mra_score_computed"].mean() * 100 if len(numerical_df) > 0 else 0

    # Per-question-type accuracy
    qtype_stats = {}
    for qt in df["question_type"].unique():
        qt_df = df[df["question_type"] == qt]
        qt_correct = qt_df["is_correct"].sum()
        qt_total = len(qt_df)
        qt_no_answer = qt_df["model_answer"].isin(["NO_ANSWER", "None", ""]).sum() + qt_df["model_answer"].isna().sum()
        qtype_stats[qt] = {
            "total": qt_total,
            "correct": qt_correct,
            "accuracy": 100 * qt_correct / qt_total if qt_total > 0 else 0,
            "no_answer": qt_no_answer,
            "no_answer_rate": 100 * qt_no_answer / qt_total if qt_total > 0 else 0,
        }
        # MRA for numerical types
        if qt in NUMERICAL_QUESTION_TYPES and qt in df["question_type"].values:
            qt_num_df = qt_df[qt_df["is_numerical"].astype(str).str.lower() == "true"]
            qtype_stats[qt]["avg_mra"] = qt_num_df["mra_score_computed"].mean() * 100 if len(qt_num_df) > 0 else 0

    # Per-dataset accuracy
    df["dataset_inferred"] = df["scene_id"].apply(infer_dataset)
    dataset_stats = {}
    for ds in df["dataset_inferred"].unique():
        ds_df = df[df["dataset_inferred"] == ds]
        ds_correct = ds_df["is_correct"].sum()
        ds_total = len(ds_df)
        dataset_stats[ds] = {
            "total": ds_total,
            "correct": ds_correct,
            "accuracy": 100 * ds_correct / ds_total if ds_total > 0 else 0,
        }

    return {
        "n_frames": n_frames,
        "total": total,
        "answered": answered,
        "no_answer": no_answer,
        "correct": correct,
        "accuracy": accuracy,
        "success_rate": success_rate,
        "avg_mra": avg_mra,
        "qtype_stats": qtype_stats,
        "dataset_stats": dataset_stats,
        "avg_time": df["time_seconds"].mean() if "time_seconds" in df.columns else 0,
    }


# ─── Plot Functions ───────────────────────────────────────────────────────────

def plot_accuracy_vs_frames(all_stats, output_dir):
    """Plot overall accuracy and MRA vs number of frames."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    frames = [s["n_frames"] for s in all_stats]
    accuracies = [s["accuracy"] for s in all_stats]
    mras = [s["avg_mra"] for s in all_stats]
    success_rates = [s["success_rate"] for s in all_stats]

    # Accuracy
    ax1.plot(frames, accuracies, "o-", color="#e15759", linewidth=2.5, markersize=10,
             label="Embodied-R1-7B-Stage1", zorder=3)
    for x, y in zip(frames, accuracies):
        ax1.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Number of Frames")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Embodied-R1-7B-Stage1: Accuracy vs Frames")
    ax1.set_xticks(frames)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    # MRA
    ax2.plot(frames, mras, "s-", color="#e15759", linewidth=2.5, markersize=10,
             label="Embodied-R1-7B-Stage1 (MRA)", zorder=3)
    for x, y in zip(frames, mras):
        ax2.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Number of Frames")
    ax2.set_ylabel("MRA (%)")
    ax2.set_title("Embodied-R1-7B-Stage1: MRA vs Frames (Numerical Questions)")
    ax2.set_xticks(frames)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_vs_frames.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "accuracy_vs_frames.pdf", dpi=600, bbox_inches="tight")
    plt.close()


def plot_category_accuracy(all_stats, output_dir):
    """Grouped bar chart: accuracy by question type for each frame config."""
    # Collect all question types
    all_qtypes = set()
    for s in all_stats:
        all_qtypes.update(s["qtype_stats"].keys())
    all_qtypes = sorted(all_qtypes)

    # Rename for display
    display_names = {
        "route_planning": "Route\nPlanning",
        "object_rel_distance": "Rel.\nDistance",
        "object_rel_direction_easy": "Rel. Dir.\n(Easy)",
        "object_rel_direction_medium": "Rel. Dir.\n(Med)",
        "object_rel_direction_hard": "Rel. Dir.\n(Hard)",
        "object_counting": "Object\nCounting",
        "object_abs_distance": "Abs.\nDistance",
        "object_size_estimation": "Size\nEstimation",
        "room_size_estimation": "Room\nSize",
        "obj_appearance_order": "Appearance\nOrder",
    }

    fig, ax = plt.subplots(figsize=(16, 6))

    n_groups = len(all_qtypes)
    n_bars = len(all_stats)
    width = 0.8 / n_bars
    x = np.arange(n_groups)

    for i, stats in enumerate(all_stats):
        accs = []
        for qt in all_qtypes:
            if qt in stats["qtype_stats"]:
                accs.append(stats["qtype_stats"][qt]["accuracy"])
            else:
                accs.append(0)
        offset = (i - n_bars / 2 + 0.5) * width
        bars = ax.bar(x + offset, accs, width, label=f'{stats["n_frames"]} frames',
                      color=COLORS["frames"][i], alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Question Type")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Embodied-R1-7B-Stage1: Accuracy by Question Type and Frame Count")
    ax.set_xticks(x)
    ax.set_xticklabels([display_names.get(qt, qt) for qt in all_qtypes])
    ax.legend(title="Frames")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "category_accuracy.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "category_accuracy.pdf", dpi=600, bbox_inches="tight")
    plt.close()


def plot_category_mra(all_stats, output_dir):
    """MRA by question type (numerical only) for each frame config."""
    num_qtypes = NUMERICAL_QUESTION_TYPES
    display_names = {
        "object_counting": "Object\nCounting",
        "object_abs_distance": "Abs.\nDistance",
        "object_size_estimation": "Size\nEstimation",
        "room_size_estimation": "Room\nSize",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    n_groups = len(num_qtypes)
    n_bars = len(all_stats)
    width = 0.8 / n_bars
    x = np.arange(n_groups)

    for i, stats in enumerate(all_stats):
        mras = []
        for qt in num_qtypes:
            if qt in stats["qtype_stats"] and "avg_mra" in stats["qtype_stats"][qt]:
                mras.append(stats["qtype_stats"][qt]["avg_mra"])
            else:
                mras.append(0)
        offset = (i - n_bars / 2 + 0.5) * width
        bars = ax.bar(x + offset, mras, width, label=f'{stats["n_frames"]} frames',
                      color=COLORS["frames"][i], alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Question Type")
    ax.set_ylabel("MRA (%)")
    ax.set_title("Embodied-R1-7B-Stage1: MRA by Numerical Question Type")
    ax.set_xticks(x)
    ax.set_xticklabels([display_names.get(qt, qt) for qt in num_qtypes])
    ax.legend(title="Frames")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "category_mra.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "category_mra.pdf", dpi=600, bbox_inches="tight")
    plt.close()


def plot_dataset_breakdown(all_stats, output_dir):
    """Accuracy by dataset (arkitscenes, scannet, scannetpp) for each frame config."""
    datasets = ["arkitscenes", "scannet", "scannetpp"]
    ds_colors = {"arkitscenes": "#2E8B57", "scannet": "#4682B4", "scannetpp": "#9467bd"}

    fig, ax = plt.subplots(figsize=(10, 6))

    n_groups = len(all_stats)
    n_bars = len(datasets)
    width = 0.8 / n_bars
    x = np.arange(n_groups)

    for i, ds in enumerate(datasets):
        accs = []
        for stats in all_stats:
            if ds in stats["dataset_stats"]:
                accs.append(stats["dataset_stats"][ds]["accuracy"])
            else:
                accs.append(0)
        offset = (i - n_bars / 2 + 0.5) * width
        ax.bar(x + offset, accs, width, label=ds.capitalize(),
               color=ds_colors[ds], alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Frame Count")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Embodied-R1-7B-Stage1: Accuracy by Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s["n_frames"]} frames' for s in all_stats])
    ax.legend(title="Dataset")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "dataset_breakdown.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "dataset_breakdown.pdf", dpi=600, bbox_inches="tight")
    plt.close()


def plot_no_answer_rate(all_stats, output_dir):
    """No-answer rate by question type for each frame config."""
    all_qtypes = set()
    for s in all_stats:
        all_qtypes.update(s["qtype_stats"].keys())
    all_qtypes = sorted(all_qtypes)

    display_names = {
        "route_planning": "Route Plan.",
        "object_rel_distance": "Rel. Dist.",
        "object_rel_direction_easy": "Dir. (Easy)",
        "object_rel_direction_medium": "Dir. (Med)",
        "object_rel_direction_hard": "Dir. (Hard)",
        "object_counting": "Counting",
        "object_abs_distance": "Abs. Dist.",
        "object_size_estimation": "Size Est.",
        "room_size_estimation": "Room Size",
        "obj_appearance_order": "Appear. Order",
    }

    fig, ax = plt.subplots(figsize=(14, 5))

    n_groups = len(all_qtypes)
    n_bars = len(all_stats)
    width = 0.8 / n_bars
    x = np.arange(n_groups)

    for i, stats in enumerate(all_stats):
        rates = []
        for qt in all_qtypes:
            if qt in stats["qtype_stats"]:
                rates.append(stats["qtype_stats"][qt]["no_answer_rate"])
            else:
                rates.append(0)
        offset = (i - n_bars / 2 + 0.5) * width
        ax.bar(x + offset, rates, width, label=f'{stats["n_frames"]} frames',
               color=COLORS["frames"][i], alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Question Type")
    ax.set_ylabel("No-Answer Rate (%)")
    ax.set_title("Embodied-R1-7B-Stage1: No-Answer Rate by Question Type")
    ax.set_xticks(x)
    ax.set_xticklabels([display_names.get(qt, qt) for qt in all_qtypes], rotation=30, ha="right")
    ax.legend(title="Frames")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "no_answer_rate.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "no_answer_rate.pdf", dpi=600, bbox_inches="tight")
    plt.close()


def plot_overall_combined(all_stats, output_dir):
    """2x2 summary figure: accuracy, MRA, success rate, avg time."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    frames = [s["n_frames"] for s in all_stats]

    # (0,0) Accuracy
    ax = axes[0, 0]
    accs = [s["accuracy"] for s in all_stats]
    bars = ax.bar(range(len(frames)), accs, color=COLORS["frames"], alpha=0.85,
                  edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(frames)))
    ax.set_xticklabels([f"{f}f" for f in frames])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Overall Accuracy")
    ax.grid(axis="y", alpha=0.3)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)

    # (0,1) MRA
    ax = axes[0, 1]
    mras = [s["avg_mra"] for s in all_stats]
    bars = ax.bar(range(len(frames)), mras, color=COLORS["frames"], alpha=0.85,
                  edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(frames)))
    ax.set_xticklabels([f"{f}f" for f in frames])
    ax.set_ylabel("MRA (%)")
    ax.set_title("Avg MRA (Numerical Questions)")
    ax.grid(axis="y", alpha=0.3)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)

    # (1,0) Success Rate
    ax = axes[1, 0]
    srs = [s["success_rate"] for s in all_stats]
    bars = ax.bar(range(len(frames)), srs, color=COLORS["frames"], alpha=0.85,
                  edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(frames)))
    ax.set_xticklabels([f"{f}f" for f in frames])
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate (Non-empty Answers)")
    ax.grid(axis="y", alpha=0.3)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)

    # (1,1) Avg Time
    ax = axes[1, 1]
    times = [s["avg_time"] for s in all_stats]
    bars = ax.bar(range(len(frames)), times, color=COLORS["frames"], alpha=0.85,
                  edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(frames)))
    ax.set_xticklabels([f"{f}f" for f in frames])
    ax.set_ylabel("Avg Time (s)")
    ax.set_title("Average Inference Time")
    ax.grid(axis="y", alpha=0.3)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)

    plt.suptitle("Embodied-R1-7B-Stage1 on VSI-Bench: Overview", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "overall_combined.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "overall_combined.pdf", dpi=600, bbox_inches="tight")
    plt.close()


def plot_comparison_with_qwen(all_stats, output_dir):
    """Compare Embodied-R1-7B-Stage1 against existing Qwen3-VL-8B Video and Sequential results."""
    # Load existing performance summary
    perf_csv = EXISTING_ANALYSIS / "performance_summary.csv"
    if not perf_csv.exists():
        print("  [WARN] performance_summary.csv not found, skipping comparison plot")
        return

    existing = pd.read_csv(perf_csv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Accuracy comparison ---
    # Plot Sequential 8B
    seq_8b = existing[(existing["Eval Type"] == "Sequential") & (existing["Model"] == "8B")]
    if len(seq_8b) > 0:
        ax1.plot(seq_8b["Frames"], seq_8b["Accuracy (%)"], "o--",
                 color="#443acf", linewidth=2, markersize=8, alpha=0.7,
                 label="Sequential Qwen3-VL-8B")

    # Plot Sequential 4B
    seq_4b = existing[(existing["Eval Type"] == "Sequential") & (existing["Model"] == "4B")]
    if len(seq_4b) > 0:
        ax1.plot(seq_4b["Frames"], seq_4b["Accuracy (%)"], "^--",
                 color="#443acf", linewidth=1.5, markersize=7, alpha=0.5,
                 label="Sequential Qwen3-VL-4B")

    # Plot Embodied-R1-7B-Stage1
    frames = [s["n_frames"] for s in all_stats]
    accs = [s["accuracy"] for s in all_stats]
    ax1.plot(frames, accs, "s-", color="#e15759", linewidth=2.5, markersize=10,
             label="Embodied-R1-7B-Stage1", zorder=3)
    for x, y in zip(frames, accs):
        ax1.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontsize=9, fontweight="bold")

    ax1.set_xlabel("Number of Frames/Steps")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Accuracy Comparison")
    ax1.set_xticks([4, 8, 16, 32])
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- MRA comparison ---
    seq_8b_mra = existing[(existing["Eval Type"] == "Sequential") & (existing["Model"] == "8B")]
    if len(seq_8b_mra) > 0 and "MRA (%)" in seq_8b_mra.columns:
        valid = seq_8b_mra[seq_8b_mra["MRA (%)"].notna() & (seq_8b_mra["MRA (%)"] > 0)]
        if len(valid) > 0:
            ax2.plot(valid["Frames"], valid["MRA (%)"], "o--",
                     color="#443acf", linewidth=2, markersize=8, alpha=0.7,
                     label="Sequential Qwen3-VL-8B")

    seq_4b_mra = existing[(existing["Eval Type"] == "Sequential") & (existing["Model"] == "4B")]
    if len(seq_4b_mra) > 0 and "MRA (%)" in seq_4b_mra.columns:
        valid = seq_4b_mra[seq_4b_mra["MRA (%)"].notna() & (seq_4b_mra["MRA (%)"] > 0)]
        if len(valid) > 0:
            ax2.plot(valid["Frames"], valid["MRA (%)"], "^--",
                     color="#443acf", linewidth=1.5, markersize=7, alpha=0.5,
                     label="Sequential Qwen3-VL-4B")

    mras = [s["avg_mra"] for s in all_stats]
    ax2.plot(frames, mras, "s-", color="#e15759", linewidth=2.5, markersize=10,
             label="Embodied-R1-7B-Stage1", zorder=3)
    for x, y in zip(frames, mras):
        ax2.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontsize=9, fontweight="bold")

    ax2.set_xlabel("Number of Frames/Steps")
    ax2.set_ylabel("MRA (%)")
    ax2.set_title("MRA Comparison (Numerical Questions)")
    ax2.set_xticks([4, 8, 16, 32])
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Embodied-R1-7B-Stage1 vs Qwen3-VL on VSI-Bench", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_with_qwen.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "comparison_with_qwen.pdf", dpi=600, bbox_inches="tight")
    plt.close()


def plot_answer_breakdown(all_dfs, output_dir):
    """Show distribution of answer correctness across frame configs."""
    fig, axes = plt.subplots(1, len(all_dfs), figsize=(4 * len(all_dfs), 5), sharey=True)
    if len(all_dfs) == 1:
        axes = [axes]

    for ax, (n_frames, df) in zip(axes, sorted(all_dfs.items())):
        no_answer = df["model_answer"].isin(["NO_ANSWER", "None", ""]).sum() + df["model_answer"].isna().sum()
        correct = df["is_correct"].sum()
        wrong = len(df) - correct - no_answer

        sizes = [correct, wrong, no_answer]
        labels_pie = [f"Correct\n({correct})", f"Wrong\n({wrong})", f"No Answer\n({no_answer})"]
        colors_pie = ["#2ca02c", "#d62728", "#7f7f7f"]

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels_pie, colors=colors_pie, autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 9}
        )
        ax.set_title(f"{n_frames} Frames", fontsize=12, fontweight="bold")

    plt.suptitle("Embodied-R1-7B-Stage1: Answer Breakdown", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "answer_breakdown.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "answer_breakdown.pdf", dpi=600, bbox_inches="tight")
    plt.close()


def plot_heatmap(all_stats, output_dir):
    """Heatmap: accuracy by frame count vs question type."""
    all_qtypes = set()
    for s in all_stats:
        all_qtypes.update(s["qtype_stats"].keys())
    all_qtypes = sorted(all_qtypes)

    display_names = {
        "route_planning": "Route Planning",
        "object_rel_distance": "Rel. Distance",
        "object_rel_direction_easy": "Rel. Dir. (Easy)",
        "object_rel_direction_medium": "Rel. Dir. (Med)",
        "object_rel_direction_hard": "Rel. Dir. (Hard)",
        "object_counting": "Object Counting",
        "object_abs_distance": "Abs. Distance",
        "object_size_estimation": "Size Estimation",
        "room_size_estimation": "Room Size",
        "obj_appearance_order": "Appearance Order",
    }

    data = []
    for stats in all_stats:
        row = {}
        for qt in all_qtypes:
            if qt in stats["qtype_stats"]:
                row[display_names.get(qt, qt)] = stats["qtype_stats"][qt]["accuracy"]
            else:
                row[display_names.get(qt, qt)] = 0
        data.append(row)

    heatmap_df = pd.DataFrame(data, index=[f'{s["n_frames"]} frames' for s in all_stats])

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(heatmap_df, annot=True, fmt=".1f", cmap="YlOrRd",
                cbar_kws={"label": "Accuracy (%)"},
                linewidths=0.5, ax=ax)
    ax.set_title("Embodied-R1-7B-Stage1: Accuracy Heatmap (Frames vs Question Type)")
    ax.set_ylabel("Frame Count")
    ax.set_xlabel("")
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_heatmap.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "accuracy_heatmap.pdf", dpi=600, bbox_inches="tight")
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("VIDEO-R1-7B ANALYSIS ON VSI-BENCH")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    # 1. Load and aggregate results
    all_dfs = {}
    all_stats = []

    for n_frames in FRAME_CONFIGS:
        print(f"\n--- {n_frames} Frames ---")
        df = aggregate_frame_config(VIDEO_R1_BASE, n_frames)
        if len(df) == 0:
            print(f"  [SKIP] No data for {n_frames} frames")
            continue

        # Compute metrics
        df = compute_metrics(df)
        print(f"  Total questions: {len(df)}")
        all_dfs[n_frames] = df

        # Save aggregated CSV
        agg_path = EXISTING_ANALYSIS / f"EmbodiedR1_7B_{n_frames}f_aggregated.csv"
        df.to_csv(agg_path, index=False)
        print(f"  Saved: {agg_path.name}")

        # Compute summary stats
        stats = compute_summary_stats(df, n_frames)
        all_stats.append(stats)

    if not all_stats:
        print("\n[ERROR] No data found!")
        return

    # 2. Print summary table
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"{'Frames':<8} {'Questions':<10} {'Accuracy':<12} {'MRA':<10} {'Success':<10} {'No-Ans':<10} {'Avg Time':<10}")
    print("-" * 70)
    for s in all_stats:
        print(f"{s['n_frames']:<8} {s['total']:<10} {s['accuracy']:<12.2f} {s['avg_mra']:<10.2f} "
              f"{s['success_rate']:<10.2f} {s['no_answer']:<10} {s['avg_time']:<10.1f}")

    # Per question type
    print("\n" + "=" * 70)
    print("ACCURACY BY QUESTION TYPE")
    print("=" * 70)
    all_qtypes = set()
    for s in all_stats:
        all_qtypes.update(s["qtype_stats"].keys())
    all_qtypes = sorted(all_qtypes)

    header = f"{'Question Type':<30}"
    for s in all_stats:
        header += f" {s['n_frames']}f     "
    print(header)
    print("-" * (30 + 10 * len(all_stats)))

    for qt in all_qtypes:
        row = f"{qt:<30}"
        for s in all_stats:
            if qt in s["qtype_stats"]:
                acc = s["qtype_stats"][qt]["accuracy"]
                n = s["qtype_stats"][qt]["total"]
                row += f" {acc:5.1f}% ({n:4d})"
            else:
                row += f"      ---    "
        print(row)

    # Per dataset
    print("\n" + "=" * 70)
    print("ACCURACY BY DATASET")
    print("=" * 70)
    all_datasets = set()
    for s in all_stats:
        all_datasets.update(s["dataset_stats"].keys())
    all_datasets = sorted(all_datasets)

    header = f"{'Dataset':<15}"
    for s in all_stats:
        header += f" {s['n_frames']}f     "
    print(header)
    print("-" * (15 + 10 * len(all_stats)))

    for ds in all_datasets:
        row = f"{ds:<15}"
        for s in all_stats:
            if ds in s["dataset_stats"]:
                acc = s["dataset_stats"][ds]["accuracy"]
                n = s["dataset_stats"][ds]["total"]
                row += f" {acc:5.1f}% ({n:4d})"
            else:
                row += f"      ---    "
        print(row)

    # 3. Save performance summary CSV (append Embodied-R1 rows)
    summary_rows = []
    for s in all_stats:
        ark_acc = s["dataset_stats"].get("arkitscenes", {}).get("accuracy", 0)
        scan_acc = s["dataset_stats"].get("scannet", {}).get("accuracy", 0)
        scanpp_acc = s["dataset_stats"].get("scannetpp", {}).get("accuracy", 0)
        summary_rows.append({
            "Eval Type": "Embodied-R1",
            "Model": "7B",
            "Frames": s["n_frames"],
            "Questions": s["total"],
            "Accuracy (%)": round(s["accuracy"], 2),
            "MRA (%)": round(s["avg_mra"], 2),
            "Success Rate (%)": round(s["success_rate"], 2),
            "No Answer": s["no_answer"],
            "Avg Time (s)": round(s["avg_time"], 1),
            "ARKit Acc (%)": round(ark_acc, 2),
            "ScanNet Acc (%)": round(scan_acc, 2),
            "ScanNetpp Acc (%)": round(scanpp_acc, 2),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "performance_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")

    # 4. Generate plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    plot_accuracy_vs_frames(all_stats, OUTPUT_DIR)
    print("  [OK] accuracy_vs_frames")

    plot_category_accuracy(all_stats, OUTPUT_DIR)
    print("  [OK] category_accuracy")

    plot_category_mra(all_stats, OUTPUT_DIR)
    print("  [OK] category_mra")

    plot_dataset_breakdown(all_stats, OUTPUT_DIR)
    print("  [OK] dataset_breakdown")

    plot_no_answer_rate(all_stats, OUTPUT_DIR)
    print("  [OK] no_answer_rate")

    plot_overall_combined(all_stats, OUTPUT_DIR)
    print("  [OK] overall_combined")

    plot_comparison_with_qwen(all_stats, OUTPUT_DIR)
    print("  [OK] comparison_with_qwen")

    plot_answer_breakdown(all_dfs, OUTPUT_DIR)
    print("  [OK] answer_breakdown")

    plot_heatmap(all_stats, OUTPUT_DIR)
    print("  [OK] accuracy_heatmap")

    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("Generated files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
