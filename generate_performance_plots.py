#!/usr/bin/env python3
"""
Generate performance comparison plots for Sequential vs Video vs Blind evaluation.
Aggregates results across all runs, handles duplicates, and creates comparison plots.

Scoring logic:
  - Numerical questions (object_counting, object_abs_distance, object_size_estimation,
    room_size_estimation): MRA score in [0,1]; NOT binarized — reported as continuous %.
  - MCA questions (object_rel_distance, object_rel_direction_*, route_planning,
    obj_appearance_order): correct if model_answer == gt_answer (exact match).

Color Palette:
  Red:        #BD0A0A
  Green:      #229E4A
  Blue:       #443ACF
  Violet:     #DEDAFF
  Light Blue: #C6DCFF
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import sys
import os
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from utils.evaluation import calculate_mra

# ─── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path("/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs")
BASE_DIR_SCRATCH = Path("/dss/mcmlscratch/06/di38riq/experiment_logs")
OUTPUT_DIR = Path("/dss/dsshome1/06/di38riq/rl_multi_turn/analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── Color Palette ──────────────────────────────────────────────────────────────
C_RED        = "#BD0A0A"
C_GREEN      = "#229E4A"
C_BLUE       = "#443ACF"
C_VIOLET     = "#DEDAFF"
C_LIGHT_BLUE = "#C6DCFF"

# Derived / complementary
C_RED_LIGHT    = "#E8A0A0"
C_GREEN_LIGHT  = "#A0D4B0"
C_GREY         = "#9E9E9E"
C_GREY_LIGHT   = "#E0E0E0"

# ─── Configuration ──────────────────────────────────────────────────────────────
EVAL_TYPES = ["Sequential", "Video"]
MODELS = ["4B", "8B"]
FRAMES = [4, 8, 16, 32]

# Question type display names
QUESTION_TYPE_NAMES = {
    'route_planning':              'Route Planning',
    'object_rel_distance':         'Obj Rel Dist',
    'object_rel_direction_easy':   'Obj Rel Dir (Easy)',
    'object_rel_direction_medium': 'Obj Rel Dir (Med)',
    'object_rel_direction_hard':   'Obj Rel Dir (Hard)',
    'object_counting':             'Obj Counting',
    'object_abs_distance':         'Obj Abs Distance',
    'object_size_estimation':      'Obj Size Est',
    'room_size_estimation':        'Room Size Est',
    'obj_appearance_order':        'Obj Appear Order',
}

COMBINED_QT_NAMES = {
    'route_planning':         'Route Planning',
    'object_rel_distance':    'Obj Rel Distance',
    'object_rel_direction':   'Obj Rel Direction',
    'object_counting':        'Obj Counting',
    'object_abs_distance':    'Obj Abs Distance',
    'object_size_estimation': 'Obj Size Estimation',
    'room_size_estimation':   'Room Size Estimation',
    'obj_appearance_order':   'Obj Appear Order',
}

COMBINED_QT_ORDER = [
    'route_planning',
    'object_rel_distance',
    'object_rel_direction',
    'object_counting',
    'object_abs_distance',
    'object_size_estimation',
    'room_size_estimation',
    'obj_appearance_order',
]

MCA_QUESTION_TYPES = [
    'object_rel_distance',
    'object_rel_direction_easy',
    'object_rel_direction_medium',
    'object_rel_direction_hard',
    'route_planning',
    'obj_appearance_order',
]
NUMERICAL_QUESTION_TYPES = [
    'object_counting',
    'object_abs_distance',
    'object_size_estimation',
    'room_size_estimation',
]
DIRECTION_TYPES = {
    'object_rel_direction_easy',
    'object_rel_direction_medium',
    'object_rel_direction_hard',
}


def _map_qt(qt):
    """Map raw question type to combined type."""
    return 'object_rel_direction' if qt in DIRECTION_TYPES else qt


# ─── Data loading ───────────────────────────────────────────────────────────────

def _find_csv_files(base_dir):
    """Walk directory tree finding results.csv, skipping step_ dirs."""
    found = []
    base_str = str(base_dir)
    if not os.path.isdir(base_str):
        return found
    for dirpath, dirnames, filenames in os.walk(base_str, followlinks=False):
        dirnames[:] = [d for d in dirnames if not d.startswith('step_')]
        if 'results.csv' in filenames:
            found.append(os.path.join(dirpath, 'results.csv'))
    return found


def aggregate_results_for_config(eval_type, model, frames):
    """
    Aggregate all results for a specific configuration.
    Deduplicates by (scene_id, question) keeping first occurrence.
    Recomputes MRA scores where missing.
    """
    frames_dir = f"{frames}_frames"
    config_paths = [
        BASE_DIR / eval_type / model / frames_dir,
        BASE_DIR / eval_type / eval_type / model / frames_dir,  # bug path
        BASE_DIR_SCRATCH / eval_type / model / frames_dir,
    ]

    all_rows = []
    seen_pairs = set()
    duplicate_count = 0
    csv_count = 0

    for config_path in config_paths:
        csv_files = _find_csv_files(config_path)
        csv_count += len(csv_files)
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    scene_id = str(row.get('scene_id', ''))
                    question = str(row.get('question', ''))
                    pair = (scene_id, question)
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        all_rows.append(row.to_dict())
                    else:
                        duplicate_count += 1
            except Exception as e:
                print(f"  ✗ Error reading {csv_path}: {e}")

    if not all_rows:
        print(f"[WARNING] No data for {eval_type}/{model}/{frames}f")
        return None

    result_df = pd.DataFrame(all_rows)
    result_df = _recompute_scores(result_df)

    print(f"  {eval_type:>10} {model} {frames:>2}f: {len(result_df):>5} unique ({csv_count} CSVs, {duplicate_count} dups)")

    csv_output = OUTPUT_DIR / f"{eval_type}_{model}_{frames}f_aggregated.csv"
    result_df.to_csv(csv_output, index=False)
    return result_df


def aggregate_blind_results(model):
    """Aggregate blind baseline results for a model."""
    blind_path = BASE_DIR_SCRATCH / "Blind" / model
    csv_files = _find_csv_files(blind_path)

    all_rows = []
    seen_pairs = set()

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                scene_id = str(row.get('scene_id', ''))
                question = str(row.get('question', ''))
                pair = (scene_id, question)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    all_rows.append(row.to_dict())
        except Exception as e:
            print(f"  ✗ Error reading {csv_path}: {e}")

    if not all_rows:
        print(f"[WARNING] No blind data for {model}")
        return None

    result_df = pd.DataFrame(all_rows)
    result_df = _recompute_scores(result_df)

    print(f"     Blind {model}:     {len(result_df):>5} unique ({len(csv_files)} CSVs)")

    csv_output = OUTPUT_DIR / f"Blind_{model}_aggregated.csv"
    result_df.to_csv(csv_output, index=False)
    return result_df


def _recompute_scores(df):
    """Recompute is_correct and mra_score columns."""
    is_correct_list = []
    mra_list = []

    for _, row in df.iterrows():
        gt = str(row['gt_answer']).strip()
        pred = str(row['model_answer']).strip()
        qtype = row['question_type']
        is_num = qtype in NUMERICAL_QUESTION_TYPES

        if is_num and pred != 'NO_ANSWER':
            try:
                pred_val = float(pred)
                gt_val = float(gt)
                mra = calculate_mra(pred_val, gt_val)
                correct = bool(mra > 0.5)
            except (ValueError, TypeError):
                mra = 0.0
                correct = False
        elif is_num:
            mra = 0.0
            correct = False
        else:
            mra = np.nan
            correct = (pred.upper() == gt.upper())

        is_correct_list.append(correct)
        existing_mra = row.get('mra_score')
        if is_num and pd.isna(existing_mra):
            mra_list.append(mra)
        else:
            mra_list.append(existing_mra if is_num else np.nan)

    df['is_correct'] = is_correct_list
    df['mra_score'] = mra_list
    return df


# ─── Metrics ────────────────────────────────────────────────────────────────────

def calc_metrics(df, question_type=None):
    """
    MCA Accuracy: fraction of MCA questions answered correctly.
    MRA: mean continuous mra_score over numerical questions (NOT binarized).
    Returns (mca_acc_pct, mra_pct).
    """
    if df is None or len(df) == 0:
        return None, None

    if question_type:
        df = df[df['question_type'] == question_type]
        if len(df) == 0:
            return None, None

    mca_df = df[df['question_type'].isin(MCA_QUESTION_TYPES)]
    num_df = df[df['question_type'].isin(NUMERICAL_QUESTION_TYPES)]

    if question_type:
        if question_type in MCA_QUESTION_TYPES:
            acc = mca_df['is_correct'].sum() / len(mca_df) * 100 if len(mca_df) > 0 else None
            return acc, None
        elif question_type in NUMERICAL_QUESTION_TYPES:
            scores = num_df['mra_score'].dropna()
            mra = scores.mean() * 100 if len(scores) > 0 else None
            return None, mra

    mca_acc = mca_df['is_correct'].sum() / len(mca_df) * 100 if len(mca_df) > 0 else None
    num_scores = num_df['mra_score'].dropna()
    mra = num_scores.mean() * 100 if len(num_scores) > 0 else None
    return mca_acc, mra


def calc_combined_type_metrics(df, combined_qt):
    """Calculate metrics for a combined question type (e.g., merged direction types)."""
    if df is None or len(df) == 0:
        return None, None
    mask = df['question_type'].map(_map_qt) == combined_qt
    sub = df[mask]
    if len(sub) == 0:
        return None, None

    if combined_qt in NUMERICAL_QUESTION_TYPES:
        scores = sub['mra_score'].dropna()
        return None, scores.mean() * 100 if len(scores) > 0 else None
    else:
        acc = sub['is_correct'].sum() / len(sub) * 100
        return acc, None


# ─── Global matplotlib style ───────────────────────────────────────────────────

def set_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linewidth': 0.5,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#cccccc',
    })


# ─── Plotting colors & style maps ──────────────────────────────────────────────
MODEL_COLORS = {'4B': C_GREEN, '8B': C_RED}
PIPELINE_LS = {'Sequential': '-', 'Video': '--', 'Blind': ':'}
PIPELINE_MK = {'Sequential': 'o', 'Video': 's', 'Blind': 'D'}


# ─── Plot Functions ─────────────────────────────────────────────────────────────

def plot_overall_accuracy(overall, blind_metrics, ax=None):
    """Plot 1: Overall MCA Accuracy comparison with blind baseline."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 6))

    for et in EVAL_TYPES:
        for mdl in MODELS:
            pts = [(f, v) for f, v in overall[(et, mdl)]['accuracy'] if v is not None]
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker=PIPELINE_MK[et], linewidth=2.5, markersize=8,
                        color=MODEL_COLORS[mdl], linestyle=PIPELINE_LS[et],
                        label=f"{mdl} {et}")

    for mdl in MODELS:
        ba, _ = blind_metrics.get(mdl, (None, None))
        if ba is not None:
            ax.axhline(y=ba, color=MODEL_COLORS[mdl], linestyle=':', linewidth=1.5, alpha=0.6)
            ax.text(FRAMES[-1] + 0.5, ba, f'{mdl} Blind\n{ba:.1f}%', fontsize=8,
                    color=MODEL_COLORS[mdl], va='center')

    ax.set_xlabel('Number of Frames', fontsize=13)
    ax.set_ylabel('MCA Accuracy (%)', fontsize=13)
    ax.set_title('MCA Accuracy: Sequential vs Video', fontsize=15, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.set_xticks(FRAMES)

    if own_fig:
        plt.tight_layout()
        p = OUTPUT_DIR / "overall_accuracy_comparison.png"
        plt.savefig(p, dpi=300, bbox_inches='tight'); plt.close()
        print(f"[SAVED] {p}")


def plot_overall_mra(overall, blind_metrics, ax=None):
    """Plot 2: Overall Mean MRA comparison with blind baseline."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 6))

    for et in EVAL_TYPES:
        for mdl in MODELS:
            pts = [(f, v) for f, v in overall[(et, mdl)]['mra'] if v is not None]
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker=PIPELINE_MK[et], linewidth=2.5, markersize=8,
                        color=MODEL_COLORS[mdl], linestyle=PIPELINE_LS[et],
                        label=f"{mdl} {et}")

    for mdl in MODELS:
        _, bm = blind_metrics.get(mdl, (None, None))
        if bm is not None:
            ax.axhline(y=bm, color=MODEL_COLORS[mdl], linestyle=':', linewidth=1.5, alpha=0.6)
            ax.text(FRAMES[-1] + 0.5, bm, f'{mdl} Blind\n{bm:.1f}%', fontsize=8,
                    color=MODEL_COLORS[mdl], va='center')

    ax.set_xlabel('Number of Frames', fontsize=13)
    ax.set_ylabel('Mean MRA (%)', fontsize=13)
    ax.set_title('Mean MRA: Sequential vs Video (Numerical Qs)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.set_xticks(FRAMES)

    if own_fig:
        plt.tight_layout()
        p = OUTPUT_DIR / "overall_mra_comparison.png"
        plt.savefig(p, dpi=300, bbox_inches='tight'); plt.close()
        print(f"[SAVED] {p}")


def plot_combined_overview(overall, blind_metrics):
    """Plot 3: Combined 1×2 — MCA Accuracy + MRA side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    plot_overall_accuracy(overall, blind_metrics, ax=ax1)
    plot_overall_mra(overall, blind_metrics, ax=ax2)
    fig.suptitle('Overall Performance: Sequential vs Video vs Blind', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    p = OUTPUT_DIR / "overall_combined.png"
    plt.savefig(p, dpi=300, bbox_inches='tight'); plt.close()
    print(f"[SAVED] {p}")


def plot_per_category(results, question_types, blind_results):
    """Plot 4: Per-category subplots (all question types)."""
    n = len(question_types); ncols = 3; nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
    axes_flat = axes.flatten()

    for i, qt in enumerate(question_types):
        ax = axes_flat[i]
        is_num = qt in NUMERICAL_QUESTION_TYPES

        for et in EVAL_TYPES:
            for mdl in MODELS:
                pts = []
                for frames in FRAMES:
                    df = results.get((et, mdl, frames))
                    a, m = calc_metrics(df, question_type=qt)
                    val = m if is_num else a
                    if val is not None:
                        pts.append((frames, val))
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, marker=PIPELINE_MK[et], linewidth=2,
                            color=MODEL_COLORS[mdl], linestyle=PIPELINE_LS[et],
                            label=f"{mdl} {et}")

        for mdl in MODELS:
            bdf = blind_results.get(mdl)
            if bdf is not None:
                a, m = calc_metrics(bdf, question_type=qt)
                val = m if is_num else a
                if val is not None:
                    ax.axhline(y=val, color=MODEL_COLORS[mdl], linestyle=':', linewidth=1.2, alpha=0.5)

        ylabel = 'Mean MRA (%)' if is_num else 'Accuracy (%)'
        ax.set_title(QUESTION_TYPE_NAMES.get(qt, qt), fontsize=12, fontweight='bold')
        ax.set_xlabel('Frames'); ax.set_ylabel(ylabel)
        ax.legend(fontsize=7); ax.set_xticks(FRAMES)

    for i in range(n, len(axes_flat)):
        axes_flat[i].axis('off')
    fig.suptitle('Per-Category Scores (MCA Acc / Numerical MRA)', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    p = OUTPUT_DIR / "category_accuracy_comparison.png"
    plt.savefig(p, dpi=300, bbox_inches='tight'); plt.close()
    print(f"[SAVED] {p}")


def plot_numerical_mra(results, question_types, blind_results):
    """Plot 5: Numerical question MRA subplots only."""
    num_types = [qt for qt in question_types if qt in NUMERICAL_QUESTION_TYPES]
    nn = len(num_types)
    if nn == 0:
        return
    ncols = min(nn, 2); nrows = (nn + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    if nn == 1:
        axes_flat = [axes]
    elif nn <= ncols:
        axes_flat = list(axes) if hasattr(axes, '__iter__') else [axes]
    else:
        axes_flat = axes.flatten()

    for i, qt in enumerate(num_types):
        ax = axes_flat[i]
        for et in EVAL_TYPES:
            for mdl in MODELS:
                pts = []
                for frames in FRAMES:
                    df = results.get((et, mdl, frames))
                    _, m = calc_metrics(df, question_type=qt)
                    if m is not None:
                        pts.append((frames, m))
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, marker=PIPELINE_MK[et], linewidth=2,
                            color=MODEL_COLORS[mdl], linestyle=PIPELINE_LS[et],
                            label=f"{mdl} {et}")

        for mdl in MODELS:
            bdf = blind_results.get(mdl)
            if bdf is not None:
                _, m = calc_metrics(bdf, question_type=qt)
                if m is not None:
                    ax.axhline(y=m, color=MODEL_COLORS[mdl], linestyle=':', linewidth=1.2, alpha=0.5)

        ax.set_title(QUESTION_TYPE_NAMES.get(qt, qt), fontsize=12, fontweight='bold')
        ax.set_xlabel('Frames'); ax.set_ylabel('Mean MRA (%)')
        ax.legend(fontsize=8); ax.set_xticks(FRAMES)

    for i in range(nn, len(axes_flat)):
        axes_flat[i].axis('off')
    fig.suptitle('MRA by Numerical Question Category', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    p = OUTPUT_DIR / "category_mra_comparison.png"
    plt.savefig(p, dpi=300, bbox_inches='tight'); plt.close()
    print(f"[SAVED] {p}")


def plot_stacked_bars(results, model, blind_results):
    """Plot 6/7: Stacked bar chart — correct/wrong/no_answer breakdown per question type."""
    present_types = set()
    for frames in FRAMES:
        df = results.get(('Sequential', model, frames))
        if df is not None:
            present_types.update(df['question_type'].map(_map_qt).unique())
    qt_order = [qt for qt in COMBINED_QT_ORDER if qt in present_types]

    nrows, ncols = 4, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 16))
    axes_flat = axes.flatten()

    for i, qt in enumerate(qt_order):
        ax = axes_flat[i]
        correct_pcts, wrong_pcts, na_pcts = [], [], []

        for frames in FRAMES:
            df = results.get(('Sequential', model, frames))
            if df is None:
                correct_pcts.append(0); wrong_pcts.append(0); na_pcts.append(0)
                continue
            mask = df['question_type'].map(_map_qt) == qt
            sub = df[mask]
            n_total = len(sub)
            if n_total == 0:
                correct_pcts.append(0); wrong_pcts.append(0); na_pcts.append(0)
                continue
            n_na = int((sub['model_answer'].astype(str).str.upper() == 'NO_ANSWER').sum())
            n_correct = int(sub['is_correct'].sum())
            n_wrong = n_total - n_correct - n_na
            correct_pcts.append(n_correct / n_total * 100)
            wrong_pcts.append(n_wrong / n_total * 100)
            na_pcts.append(n_na / n_total * 100)

        x = np.arange(len(FRAMES))
        w = 0.55
        ax.bar(x, correct_pcts, w, color=C_GREEN, label='Correct')
        ax.bar(x, wrong_pcts, w, bottom=correct_pcts, color=C_RED, label='Wrong')
        btm2 = [c + wr for c, wr in zip(correct_pcts, wrong_pcts)]
        ax.bar(x, na_pcts, w, bottom=btm2, color=C_GREY, label='No Answer')

        ax.set_xticks(x); ax.set_xticklabels([str(f) for f in FRAMES], fontsize=10)
        ax.set_xlabel('Frames', fontsize=10); ax.set_ylabel('%', fontsize=10)
        ax.set_ylim(0, 105)
        ax.set_title(COMBINED_QT_NAMES.get(qt, qt), fontsize=12, fontweight='bold')

        for j in range(len(FRAMES)):
            if correct_pcts[j] > 5:
                ax.text(x[j], correct_pcts[j] / 2, f'{correct_pcts[j]:.0f}%',
                        ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            if na_pcts[j] > 5:
                ax.text(x[j], btm2[j] + na_pcts[j] / 2, f'{na_pcts[j]:.0f}%',
                        ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        if i == 0:
            ax.legend(fontsize=8, loc='upper right')

    for i in range(len(qt_order), len(axes_flat)):
        axes_flat[i].axis('off')

    fig.suptitle(f'Sequential {model} — Answer Breakdown by Question Type',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    p = OUTPUT_DIR / f"answer_breakdown_{model}.png"
    plt.savefig(p, dpi=300, bbox_inches='tight'); plt.close()
    print(f"[SAVED] {p}")


def plot_seq_vs_video_grouped_bar(results, blind_metrics):
    """Plot 8: Grouped bar — Sequential vs Video at each frame count."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(FRAMES))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]
    bar_configs = [
        ('Sequential', '4B', C_GREEN,      'Seq 4B'),
        ('Sequential', '8B', C_RED,        'Seq 8B'),
        ('Video',      '4B', C_GREEN_LIGHT, 'Video 4B'),
        ('Video',      '8B', C_RED_LIGHT,   'Video 8B'),
    ]

    for (et, mdl, color, label), off in zip(bar_configs, offsets):
        vals = []
        for frames in FRAMES:
            df = results.get((et, mdl, frames))
            a, _ = calc_metrics(df)
            vals.append(a if a is not None else 0)
        bars = ax.bar(x + off * width, vals, width, color=color, label=label, edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f'{v:.1f}', ha='center', va='bottom', fontsize=7)

    for mdl in MODELS:
        ba, _ = blind_metrics.get(mdl, (None, None))
        if ba is not None:
            ax.axhline(y=ba, color=MODEL_COLORS[mdl], linestyle=':', linewidth=1.5, alpha=0.5,
                       label=f'Blind {mdl} ({ba:.1f}%)')

    ax.set_xlabel('Number of Frames', fontsize=13)
    ax.set_ylabel('MCA Accuracy (%)', fontsize=13)
    ax.set_title('MCA Accuracy: Sequential vs Video (Grouped)', fontsize=15, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([str(f) for f in FRAMES])
    ax.legend(fontsize=9, ncol=3, loc='upper left')
    ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
    plt.tight_layout()
    p = OUTPUT_DIR / "grouped_bar_accuracy.png"
    plt.savefig(p, dpi=300, bbox_inches='tight'); plt.close()
    print(f"[SAVED] {p}")


def plot_seq_vs_video_grouped_bar_mra(results, blind_metrics):
    """Plot 9: Grouped bar — Sequential vs Video MRA at each frame count."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(FRAMES))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]
    bar_configs = [
        ('Sequential', '4B', C_GREEN,      'Seq 4B'),
        ('Sequential', '8B', C_RED,        'Seq 8B'),
        ('Video',      '4B', C_GREEN_LIGHT, 'Video 4B'),
        ('Video',      '8B', C_RED_LIGHT,   'Video 8B'),
    ]

    for (et, mdl, color, label), off in zip(bar_configs, offsets):
        vals = []
        for frames in FRAMES:
            df = results.get((et, mdl, frames))
            _, m = calc_metrics(df)
            vals.append(m if m is not None else 0)
        bars = ax.bar(x + off * width, vals, width, color=color, label=label, edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f'{v:.1f}', ha='center', va='bottom', fontsize=7)

    for mdl in MODELS:
        _, bm = blind_metrics.get(mdl, (None, None))
        if bm is not None:
            ax.axhline(y=bm, color=MODEL_COLORS[mdl], linestyle=':', linewidth=1.5, alpha=0.5,
                       label=f'Blind {mdl} ({bm:.1f}%)')

    ax.set_xlabel('Number of Frames', fontsize=13)
    ax.set_ylabel('Mean MRA (%)', fontsize=13)
    ax.set_title('Mean MRA: Sequential vs Video (Grouped)', fontsize=15, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([str(f) for f in FRAMES])
    ax.legend(fontsize=9, ncol=3, loc='upper left')
    ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
    plt.tight_layout()
    p = OUTPUT_DIR / "grouped_bar_mra.png"
    plt.savefig(p, dpi=300, bbox_inches='tight'); plt.close()
    print(f"[SAVED] {p}")


def plot_blind_breakdown(blind_results):
    """Plot 10: Blind baseline breakdown — per-category accuracy for both models."""
    combined_types = COMBINED_QT_ORDER
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(combined_types))
    width = 0.35

    for mdl_idx, mdl in enumerate(MODELS):
        bdf = blind_results.get(mdl)
        vals = []
        for cqt in combined_types:
            if bdf is None:
                vals.append(0)
                continue
            is_num = cqt in NUMERICAL_QUESTION_TYPES
            mask = bdf['question_type'].map(_map_qt) == cqt
            sub = bdf[mask]
            if len(sub) == 0:
                vals.append(0)
                continue
            if is_num:
                scores = sub['mra_score'].dropna()
                vals.append(scores.mean() * 100 if len(scores) > 0 else 0)
            else:
                vals.append(sub['is_correct'].sum() / len(sub) * 100)

        offset = -width/2 if mdl_idx == 0 else width/2
        color = MODEL_COLORS[mdl]
        bars = ax.bar(x + offset, vals, width, color=color, label=f'{mdl} Blind',
                      edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 1:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{v:.1f}', ha='center', va='bottom', fontsize=8)

    ax.axhline(y=25, color=C_GREY, linestyle='--', linewidth=1, alpha=0.6, label='25% chance (MCQ)')

    ax.set_xticks(x)
    ax.set_xticklabels([COMBINED_QT_NAMES.get(qt, qt) for qt in combined_types],
                       rotation=35, ha='right', fontsize=10)
    ax.set_ylabel('Score (%)', fontsize=13)
    ax.set_title('Blind Baseline — Per-Category Performance (No Images)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(60, ax.get_ylim()[1] * 1.08))
    plt.tight_layout()
    p = OUTPUT_DIR / "blind_baseline_breakdown.png"
    plt.savefig(p, dpi=300, bbox_inches='tight'); plt.close()
    print(f"[SAVED] {p}")


def plot_delta_over_blind(results, blind_results):
    """Plot 11: Delta over blind baseline — improvement in pp."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for et in EVAL_TYPES:
        for mdl in MODELS:
            ba, _ = calc_metrics(blind_results.get(mdl))
            if ba is None:
                continue
            pts = []
            for frames in FRAMES:
                df = results.get((et, mdl, frames))
                a, _ = calc_metrics(df)
                if a is not None:
                    pts.append((frames, a - ba))
            if pts:
                xs, ys = zip(*pts)
                ax1.plot(xs, ys, marker=PIPELINE_MK[et], linewidth=2.5, markersize=8,
                         color=MODEL_COLORS[mdl], linestyle=PIPELINE_LS[et],
                         label=f"{mdl} {et}")

    ax1.axhline(y=0, color=C_GREY, linestyle='-', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Number of Frames', fontsize=13)
    ax1.set_ylabel('Δ MCA Accuracy (pp)', fontsize=13)
    ax1.set_title('MCA Accuracy Gain over Blind', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9); ax1.set_xticks(FRAMES)

    for et in EVAL_TYPES:
        for mdl in MODELS:
            _, bm = calc_metrics(blind_results.get(mdl))
            if bm is None:
                continue
            pts = []
            for frames in FRAMES:
                df = results.get((et, mdl, frames))
                _, m = calc_metrics(df)
                if m is not None:
                    pts.append((frames, m - bm))
            if pts:
                xs, ys = zip(*pts)
                ax2.plot(xs, ys, marker=PIPELINE_MK[et], linewidth=2.5, markersize=8,
                         color=MODEL_COLORS[mdl], linestyle=PIPELINE_LS[et],
                         label=f"{mdl} {et}")

    ax2.axhline(y=0, color=C_GREY, linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Number of Frames', fontsize=13)
    ax2.set_ylabel('Δ Mean MRA (pp)', fontsize=13)
    ax2.set_title('MRA Gain over Blind', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9); ax2.set_xticks(FRAMES)

    fig.suptitle('Improvement over Blind Baseline (percentage points)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    p = OUTPUT_DIR / "delta_over_blind.png"
    plt.savefig(p, dpi=300, bbox_inches='tight'); plt.close()
    print(f"[SAVED] {p}")


def plot_no_answer_rate(results):
    """Plot 12: NO_ANSWER rate across configs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for et in EVAL_TYPES:
        for mdl in MODELS:
            pts = []
            for frames in FRAMES:
                df = results.get((et, mdl, frames))
                if df is not None and len(df) > 0:
                    na_rate = (df['model_answer'].astype(str).str.upper() == 'NO_ANSWER').sum() / len(df) * 100
                    pts.append((frames, na_rate))
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker=PIPELINE_MK[et], linewidth=2.5, markersize=8,
                        color=MODEL_COLORS[mdl], linestyle=PIPELINE_LS[et],
                        label=f"{mdl} {et}")

    ax.set_xlabel('Number of Frames', fontsize=13)
    ax.set_ylabel('NO_ANSWER Rate (%)', fontsize=13)
    ax.set_title('NO_ANSWER Rate Across Configurations', fontsize=15, fontweight='bold')
    ax.legend(fontsize=10); ax.set_xticks(FRAMES)
    plt.tight_layout()
    p = OUTPUT_DIR / "no_answer_rate.png"
    plt.savefig(p, dpi=300, bbox_inches='tight'); plt.close()
    print(f"[SAVED] {p}")


def plot_per_category_grouped_bar(results, blind_results):
    """Plot 13: Per-category grouped bars — best Sequential vs best Video vs Blind."""
    combined_types = COMBINED_QT_ORDER
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    for mdl_idx, mdl in enumerate(MODELS):
        ax = axes[mdl_idx]
        x = np.arange(len(combined_types))
        width = 0.25

        seq_vals = []
        for cqt in combined_types:
            best = 0
            for frames in FRAMES:
                df = results.get(('Sequential', mdl, frames))
                a, m = calc_combined_type_metrics(df, cqt)
                v = m if cqt in NUMERICAL_QUESTION_TYPES else a
                if v is not None and v > best:
                    best = v
            seq_vals.append(best)

        vid_vals = []
        for cqt in combined_types:
            best = 0
            for frames in FRAMES:
                df = results.get(('Video', mdl, frames))
                a, m = calc_combined_type_metrics(df, cqt)
                v = m if cqt in NUMERICAL_QUESTION_TYPES else a
                if v is not None and v > best:
                    best = v
            vid_vals.append(best)

        bdf = blind_results.get(mdl)
        blind_vals = []
        for cqt in combined_types:
            a, m = calc_combined_type_metrics(bdf, cqt)
            v = m if cqt in NUMERICAL_QUESTION_TYPES else a
            blind_vals.append(v if v is not None else 0)

        ax.bar(x - width, seq_vals, width, color=C_BLUE, label='Best Sequential', edgecolor='white')
        ax.bar(x, vid_vals, width, color=C_GREEN, label='Best Video', edgecolor='white')
        ax.bar(x + width, blind_vals, width, color=C_GREY_LIGHT, label='Blind', edgecolor='white')

        ax.set_xticks(x)
        ax.set_xticklabels([COMBINED_QT_NAMES.get(qt, qt) for qt in combined_types],
                           rotation=35, ha='right', fontsize=9)
        ax.set_ylabel('Score (%)', fontsize=12)
        ax.set_title(f'{mdl} — Best Score per Category', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_ylim(0, max(60, ax.get_ylim()[1] * 1.1))

    fig.suptitle('Best Sequential vs Best Video vs Blind (per category)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    p = OUTPUT_DIR / "best_per_category.png"
    plt.savefig(p, dpi=300, bbox_inches='tight'); plt.close()
    print(f"[SAVED] {p}")


# ─── Main ───────────────────────────────────────────────────────────────────────

def generate_all_plots():
    set_style()
    print("=" * 80)
    print("GENERATING ALL PERFORMANCE PLOTS")
    print("=" * 80)

    # ── Step 1: Aggregate data ──
    print("\n[STEP 1] Aggregating results...")
    results = {}
    for eval_type in EVAL_TYPES:
        for model in MODELS:
            for frames in FRAMES:
                results[(eval_type, model, frames)] = aggregate_results_for_config(eval_type, model, frames)

    print()
    blind_results = {}
    for model in MODELS:
        blind_results[model] = aggregate_blind_results(model)

    # Collect question types
    all_qtypes = set()
    for df in results.values():
        if df is not None and 'question_type' in df.columns:
            all_qtypes.update(df['question_type'].unique())
    question_types = sorted(all_qtypes)

    # ── Summary table ──
    global blind_metrics
    blind_metrics = {}
    print("\n" + "=" * 90)
    print(f"  {'Config':<25} {'N':<7} {'MCA Acc (%)':<14} {'Mean MRA (%)':<14}")
    print("-" * 70)
    for eval_type in EVAL_TYPES:
        for model in MODELS:
            for frames in FRAMES:
                df = results.get((eval_type, model, frames))
                if df is not None:
                    a, m = calc_metrics(df)
                    a_s = f"{a:.2f}" if a is not None else "N/A"
                    m_s = f"{m:.2f}" if m is not None else "N/A"
                    print(f"  {eval_type:>10} {model} {frames:>2}f          {len(df):<7} {a_s:<14} {m_s:<14}")
        print()

    for model in MODELS:
        bdf = blind_results.get(model)
        if bdf is not None:
            a, m = calc_metrics(bdf)
            blind_metrics[model] = (a, m)
            a_s = f"{a:.2f}" if a is not None else "N/A"
            m_s = f"{m:.2f}" if m is not None else "N/A"
            print(f"     Blind {model}               {len(bdf):<7} {a_s:<14} {m_s:<14}")
    print("=" * 90)

    # ── Build overall metric dicts for line plots ──
    overall = defaultdict(lambda: {'accuracy': [], 'mra': []})
    for eval_type in EVAL_TYPES:
        for model in MODELS:
            for frames in FRAMES:
                df = results.get((eval_type, model, frames))
                a, m = calc_metrics(df)
                overall[(eval_type, model)]['accuracy'].append((frames, a))
                overall[(eval_type, model)]['mra'].append((frames, m))

    # ── Step 2: Generate plots ──
    print("\n[STEP 2] Generating plots...")

    plot_overall_accuracy(overall, blind_metrics)               # 1
    plot_overall_mra(overall, blind_metrics)                    # 2
    plot_combined_overview(overall, blind_metrics)              # 3
    plot_per_category(results, question_types, blind_results)  # 4
    plot_numerical_mra(results, question_types, blind_results) # 5
    for model in MODELS:
        plot_stacked_bars(results, model, blind_results)       # 6, 7
    plot_seq_vs_video_grouped_bar(results, blind_metrics)      # 8
    plot_seq_vs_video_grouped_bar_mra(results, blind_metrics)  # 9
    plot_blind_breakdown(blind_results)                        # 10
    plot_delta_over_blind(results, blind_results)              # 11
    plot_no_answer_rate(results)                               # 12
    plot_per_category_grouped_bar(results, blind_results)      # 13

    print("\n" + "=" * 80)
    print(f"DONE — 13 plots + aggregated CSVs saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    generate_all_plots()
