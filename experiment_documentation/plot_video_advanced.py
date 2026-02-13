#!/usr/bin/env python3
"""
Advanced plotting for video experiment results with question type breakdown.
Creates:
1. 2x4 grid showing performance by question category
2. Large plot showing overall accuracy with error bars
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

BASE_PATH = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/Video"
OUTPUT_DIR = Path("/dss/dsshome1/06/di38riq/rl_multi_turn/experiment_documentation/figures_advanced")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def calculate_mra(predicted, ground_truth):
    """Calculate MRA score for numerical predictions."""
    try:
        pred_val = float(predicted)
        gt_val = float(ground_truth)
    except (ValueError, TypeError):
        return 0.0
    
    if gt_val == 0:
        return 1.0 if pred_val == 0 else 0.0
    
    thresholds = [0.5 + 0.05 * i for i in range(10)]
    relative_error = abs(pred_val - gt_val) / abs(gt_val)
    
    score = 0.0
    for theta in thresholds:
        if relative_error < (1 - theta):
            score += 1.0
    
    return score / 10.0

def load_all_results():
    """Load all video experiment results."""
    all_data = []
    
    for model in ["4B", "8B"]:
        for frames in [4, 8, 16, 32]:
            pattern = f"{BASE_PATH}/{model}/{frames}_frames/2026-02-05/*/results.csv"
            files = glob.glob(pattern)
            
            for file_path in files:
                df = pd.read_csv(file_path)
                df['model'] = f"Qwen3-VL-{model}"
                df['frames'] = frames
                all_data.append(df)
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Calculate correctness (handle NaN in mra_score)
    combined['correct'] = (combined['gt_answer'] == combined['model_answer']).astype(int)
    
    # Calculate proper MRA scores for numerical questions
    combined['mra_calculated'] = combined.apply(
        lambda row: calculate_mra(row['model_answer'], row['gt_answer']) 
        if row['is_numerical'] else (1.0 if row['correct'] else 0.0),
        axis=1
    )
    
    # For hybrid metric: use MRA for numerical, exact match for MCQ
    combined['hybrid_score'] = combined['mra_calculated']
    
    print(f"Loaded {len(combined)} total results")
    print(f"Unique questions: {combined['question_id'].nunique()}")
    print(f"Models: {combined['model'].unique()}")
    print(f"Frame configs: {sorted(combined['frames'].unique())}")
    
    return combined

def group_question_types(df):
    """Group question types into meaningful categories."""
    # Combine directional questions by difficulty
    df['question_category'] = df['question_type']
    
    # Group directional questions
    df.loc[df['question_type'].str.contains('object_rel_direction'), 'question_category'] = \
        df.loc[df['question_type'].str.contains('object_rel_direction'), 'question_type'].apply(
            lambda x: 'obj_rel_direction_' + x.split('_')[-1]
        )
    
    return df

def calculate_confidence_interval(accuracy_series, confidence=0.95):
    """Calculate confidence interval for binomial proportion."""
    n = len(accuracy_series)
    if n == 0:
        return 0, 0
    
    p = accuracy_series.mean()
    # Wilson score interval
    z = stats.norm.ppf((1 + confidence) / 2)
    denominator = 1 + z**2/n
    center = (p + z**2/(2*n)) / denominator
    margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denominator
    
    return center - margin, center + margin

def plot_question_type_performance(df):
    """Create 4x2 grid of question type performance."""
    # Define question categories
    categories = {
        'object_counting': 'Object Counting',
        'object_size_estimation': 'Object Size',
        'room_size_estimation': 'Room Size',
        'object_abs_distance': 'Absolute Distance',
        'object_rel_distance': 'Relative Distance',
        'obj_rel_direction_easy': 'Direction (Easy)',
        'obj_rel_direction_medium': 'Direction (Medium)',
        'obj_rel_direction_hard': 'Direction (Hard)',
        'route_planning': 'Route Planning',
        'obj_appearance_order': 'Appearance Order',
    }
    
    # Combine directional categories
    combined_categories = {
        'object_counting': 'Object Counting',
        'object_size_estimation': 'Object Size Estimation',
        'room_size_estimation': 'Room Size Estimation',
        'object_abs_distance': 'Absolute Distance',
        'object_rel_distance': 'Relative Distance',
        'object_rel_direction': 'Spatial Direction\n(All Difficulties)',
        'route_planning': 'Route Planning',
        'obj_appearance_order': 'Temporal Order',
    }
    
    # Group directional questions
    df_grouped = df.copy()
    df_grouped.loc[df_grouped['question_type'].str.contains('object_rel_direction'), 'question_category'] = 'object_rel_direction'
    df_grouped.loc[~df_grouped['question_type'].str.contains('object_rel_direction'), 'question_category'] = df_grouped.loc[~df_grouped['question_type'].str.contains('object_rel_direction'), 'question_type']
    
    # Select 8 most common categories
    top_categories = df_grouped['question_category'].value_counts().head(8).index.tolist()
    
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    axes = axes.flatten()
    
    colors = {'Qwen3-VL-4B': '#2E86AB', 'Qwen3-VL-8B': '#A23B72'}
    frames_list = [4, 8, 16, 32]
    
    for idx, category in enumerate(top_categories[:8]):
        ax = axes[idx]
        
        category_data = df_grouped[df_grouped['question_category'] == category]
        n_questions = len(category_data['question_id'].unique())
        
        for model in ['Qwen3-VL-4B', 'Qwen3-VL-8B']:
            model_data = category_data[category_data['model'] == model]
            
            accuracies = []
            errors_lower = []
            errors_upper = []
            
            for frames in frames_list:
                frame_data = model_data[model_data['frames'] == frames]
                if len(frame_data) > 0:
                    # Use hybrid_score (MRA for numerical, exact match for MCQ)
                    acc = frame_data['hybrid_score'].mean() * 100
                    ci_lower, ci_upper = calculate_confidence_interval(frame_data['hybrid_score'])
                    accuracies.append(acc)
                    errors_lower.append(acc - ci_lower * 100)
                    errors_upper.append(ci_upper * 100 - acc)
                else:
                    accuracies.append(0)
                    errors_lower.append(0)
                    errors_upper.append(0)
            
            ax.errorbar(frames_list, accuracies, 
                       yerr=[errors_lower, errors_upper],
                       marker='o', label=model.replace('Qwen3-VL-', ''),
                       color=colors[model], linewidth=2, markersize=6,
                       capsize=4, capthick=1.5)
        
        # Formatting
        title = combined_categories.get(category, category.replace('_', ' ').title())
        
        # Determine if this is a numerical question type
        is_numerical = category in ['object_counting', 'object_size_estimation', 
                                    'room_size_estimation', 'object_abs_distance']
        metric_label = 'MRA Score (%)' if is_numerical else 'Accuracy (%)'
        
        ax.set_title(f'{title}\n(n={n_questions})', fontweight='bold', fontsize=10)
        ax.set_xlabel('Frames', fontweight='bold')
        ax.set_ylabel(metric_label, fontweight='bold', fontsize=9)
        ax.set_xticks(frames_list)
        
        # Set y-axis limits based on actual data range with padding
        all_accs = []
        for model in ['Qwen3-VL-4B', 'Qwen3-VL-8B']:
            model_data = category_data[category_data['model'] == model]
            for frames in frames_list:
                frame_data = model_data[model_data['frames'] == frames]
                if len(frame_data) > 0:
                    all_accs.append(frame_data['hybrid_score'].mean() * 100)
        
        if all_accs:
            y_min = max(0, min(all_accs) - 5)
            y_max = min(100, max(all_accs) + 5)
            ax.set_ylim(y_min, y_max)
        
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', framealpha=0.9)
    
    plt.suptitle('Video Baseline Performance by Question Category\n(MRA for Numerical | Exact Match for MCQ)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save
    plt.savefig(OUTPUT_DIR / 'video_question_categories_4x2.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'video_question_categories_4x2.pdf', bbox_inches='tight')
    print(f"✅ Saved: {OUTPUT_DIR / 'video_question_categories_4x2.png'}")
    plt.close()

def plot_overall_accuracy(df):
    """Create large plot showing overall accuracy with error bars."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {'Qwen3-VL-4B': '#2E86AB', 'Qwen3-VL-8B': '#A23B72'}
    frames_list = [4, 8, 16, 32]
    
    for model in ['Qwen3-VL-4B', 'Qwen3-VL-8B']:
        model_data = df[df['model'] == model]
        
        accuracies = []
        errors_lower = []
        errors_upper = []
        
        for frames in frames_list:
            frame_data = model_data[model_data['frames'] == frames]
            acc = frame_data['correct'].mean() * 100
            ci_lower, ci_upper = calculate_confidence_interval(frame_data['correct'])
            
            accuracies.append(acc)
            errors_lower.append(acc - ci_lower * 100)
            errors_upper.append(ci_upper * 100 - acc)
        
        # Plot with error bars
        ax.errorbar(frames_list, accuracies, 
                   yerr=[errors_lower, errors_upper],
                   marker='o', label=model.replace('Qwen3-VL-', ''),
                   color=colors[model], linewidth=3, markersize=12,
                   capsize=6, capthick=2, elinewidth=2)
        
        # Add value labels
        for x, y in zip(frames_list, accuracies):
            ax.text(x, y + 0.8, f'{y:.2f}%', ha='center', va='bottom',
                   fontweight='bold', fontsize=11, color=colors[model])
    
    # Formatting
    ax.set_xlabel('Number of Frames', fontweight='bold', fontsize=14)
    ax.set_ylabel('Exact Match Accuracy (%)', fontweight='bold', fontsize=14)
    ax.set_title('Video Baseline: Overall Performance Across Frame Configurations\n(Exact Match - Not MRA | 95% Confidence Intervals)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(frames_list)
    ax.set_xticklabels([f'{f} frames' for f in frames_list])
    ax.set_ylim(15, 30)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=13, framealpha=0.95, 
             title='Model', title_fontsize=13)
    
    # Add sample size annotation
    n_questions = len(df['question_id'].unique())
    ax.text(0.02, 0.98, f'n = {n_questions:,} questions', 
           transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=dict(boxstyle='round', 
           facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    plt.savefig(OUTPUT_DIR / 'video_overall_accuracy_large.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'video_overall_accuracy_large.pdf', bbox_inches='tight')
    print(f"✅ Saved: {OUTPUT_DIR / 'video_overall_accuracy_large.png'}")
    plt.close()

def create_summary_table(df):
    """Create detailed summary table."""
    summary_rows = []
    
    for model in ['Qwen3-VL-4B', 'Qwen3-VL-8B']:
        for frames in [4, 8, 16, 32]:
            data = df[(df['model'] == model) & (df['frames'] == frames)]
            
            if len(data) > 0:
                acc = data['correct'].mean() * 100
                ci_lower, ci_upper = calculate_confidence_interval(data['correct'])
                avg_time = data['time_seconds'].mean()
                
                summary_rows.append({
                    'Model': model.replace('Qwen3-VL-', ''),
                    'Frames': frames,
                    'Questions': len(data['question_id'].unique()),
                    'Accuracy (%)': f'{acc:.2f}',
                    'CI Lower (%)': f'{ci_lower*100:.2f}',
                    'CI Upper (%)': f'{ci_upper*100:.2f}',
                    'Avg Time (s)': f'{avg_time:.2f}',
                })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / 'video_detailed_summary.csv', index=False)
    print(f"✅ Saved: {OUTPUT_DIR / 'video_detailed_summary.csv'}")
    print("\nSummary Table:")
    print(summary_df.to_string(index=False))

def plot_hybrid_metric(df):
    """Create plot using MRA for numerical questions and exact match for MCQ."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {'Qwen3-VL-4B': '#2E86AB', 'Qwen3-VL-8B': '#A23B72'}
    frames_list = [4, 8, 16, 32]
    
    for model in ['Qwen3-VL-4B', 'Qwen3-VL-8B']:
        model_data = df[df['model'] == model]
        
        accuracies = []
        errors_lower = []
        errors_upper = []
        
        for frames in frames_list:
            frame_data = model_data[model_data['frames'] == frames]
            
            # Calculate hybrid accuracy (MRA for numerical, exact match for MCQ)
            hybrid_acc = frame_data['hybrid_score'].mean() * 100
            ci_lower, ci_upper = calculate_confidence_interval(frame_data['hybrid_score'])
            
            accuracies.append(hybrid_acc)
            errors_lower.append(hybrid_acc - ci_lower * 100)
            errors_upper.append(ci_upper * 100 - hybrid_acc)
        
        # Plot with error bars
        ax.errorbar(frames_list, accuracies, 
                   yerr=[errors_lower, errors_upper],
                   marker='o', label=model.replace('Qwen3-VL-', ''),
                   color=colors[model], linewidth=3, markersize=12,
                   capsize=6, capthick=2, elinewidth=2)
        
        # Add value labels
        for x, y in zip(frames_list, accuracies):
            ax.text(x, y + 0.8, f'{y:.2f}%', ha='center', va='bottom',
                   fontweight='bold', fontsize=11, color=colors[model])
    
    # Formatting
    ax.set_xlabel('Number of Frames', fontweight='bold', fontsize=14)
    ax.set_ylabel('Performance Score (%)', fontweight='bold', fontsize=14)
    ax.set_title('Video Baseline: Hybrid Evaluation Metric\n(MRA for Numerical Questions | Exact Match for MCQ | 95% CI)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(frames_list)
    ax.set_xticklabels([f'{f} frames' for f in frames_list])
    ax.set_ylim(15, 30)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=13, framealpha=0.95, 
             title='Model', title_fontsize=13)
    
    # Add detailed annotation
    n_questions = len(df['question_id'].unique())
    n_numerical = df[df['is_numerical']]['question_id'].nunique()
    n_mcq = n_questions - n_numerical
    
    annotation_text = (f'Total: {n_questions:,} questions\n'
                      f'Numerical (MRA): {n_numerical:,} questions\n'
                      f'MCQ (Exact Match): {n_mcq:,} questions')
    
    ax.text(0.02, 0.98, annotation_text, 
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', 
           facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    # Save
    plt.savefig(OUTPUT_DIR / 'video_hybrid_metric.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'video_hybrid_metric.pdf', bbox_inches='tight')
    print(f"✅ Saved: {OUTPUT_DIR / 'video_hybrid_metric.png'}")
    plt.close()

def plot_mra_breakdown(df):
    """Create plot showing MRA vs Exact Match breakdown."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = {'Qwen3-VL-4B': '#2E86AB', 'Qwen3-VL-8B': '#A23B72'}
    frames_list = [4, 8, 16, 32]
    
    # Left plot: Numerical questions (MRA)
    for model in ['Qwen3-VL-4B', 'Qwen3-VL-8B']:
        model_data = df[(df['model'] == model) & (df['is_numerical'] == True)]
        
        mra_scores = []
        for frames in frames_list:
            frame_data = model_data[model_data['frames'] == frames]
            if len(frame_data) > 0:
                mra_scores.append(frame_data['mra_calculated'].mean() * 100)
            else:
                mra_scores.append(0)
        
        ax1.plot(frames_list, mra_scores, marker='o', 
                label=model.replace('Qwen3-VL-', ''),
                color=colors[model], linewidth=2.5, markersize=10)
        
        # Add value labels
        for x, y in zip(frames_list, mra_scores):
            ax1.text(x, y + 0.5, f'{y:.1f}%', ha='center', va='bottom',
                    fontsize=9, color=colors[model])
    
    n_numerical = df[df['is_numerical']]['question_id'].nunique()
    ax1.set_title(f'Numerical Questions\n(MRA Metric, n={n_numerical:,})', 
                 fontweight='bold', fontsize=12)
    ax1.set_xlabel('Frames', fontweight='bold')
    ax1.set_ylabel('Mean MRA Score (%)', fontweight='bold')
    ax1.set_xticks(frames_list)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right')
    ax1.set_ylim(0, max(30, max(mra_scores) + 5))
    
    # Right plot: MCQ questions (Exact Match)
    for model in ['Qwen3-VL-4B', 'Qwen3-VL-8B']:
        model_data = df[(df['model'] == model) & (df['is_numerical'] == False)]
        
        accuracies = []
        for frames in frames_list:
            frame_data = model_data[model_data['frames'] == frames]
            if len(frame_data) > 0:
                accuracies.append(frame_data['correct'].mean() * 100)
            else:
                accuracies.append(0)
        
        ax2.plot(frames_list, accuracies, marker='o', 
                label=model.replace('Qwen3-VL-', ''),
                color=colors[model], linewidth=2.5, markersize=10)
        
        # Add value labels
        for x, y in zip(frames_list, accuracies):
            ax2.text(x, y + 0.5, f'{y:.1f}%', ha='center', va='bottom',
                    fontsize=9, color=colors[model])
    
    n_mcq = df[~df['is_numerical']]['question_id'].nunique()
    ax2.set_title(f'Multiple Choice Questions\n(Exact Match, n={n_mcq:,})', 
                 fontweight='bold', fontsize=12)
    ax2.set_xlabel('Frames', fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontweight='bold')
    ax2.set_xticks(frames_list)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right')
    ax2.set_ylim(0, max(40, max(accuracies) + 5))
    
    plt.suptitle('Video Baseline: Performance by Question Type', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    plt.savefig(OUTPUT_DIR / 'video_mra_vs_exact_match.png', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'video_mra_vs_exact_match.pdf', bbox_inches='tight')
    print(f"✅ Saved: {OUTPUT_DIR / 'video_mra_vs_exact_match.png'}")
    plt.close()

def main():
    print("="*80)
    print("Advanced Video Experiment Analysis")
    print("="*80)
    print()
    
    # Load data
    print("Loading results...")
    df = load_all_results()
    print()
    
    # Print statistics about MRA vs exact match
    print("Evaluation Metrics:")
    print(f"  Numerical questions (MRA): {df[df['is_numerical']]['question_id'].nunique():,}")
    print(f"  MCQ questions (Exact Match): {df[~df['is_numerical']]['question_id'].nunique():,}")
    print(f"  Mean MRA (numerical): {df[df['is_numerical']]['mra_calculated'].mean()*100:.2f}%")
    print(f"  Mean Accuracy (MCQ): {df[~df['is_numerical']]['correct'].mean()*100:.2f}%")
    print(f"  Hybrid Score (overall): {df['hybrid_score'].mean()*100:.2f}%")
    print()
    
    # Create plots
    print("Creating 4x2 question category plot...")
    plot_question_type_performance(df)
    print()
    
    print("Creating overall accuracy plot...")
    plot_overall_accuracy(df)
    print()
    
    print("Creating hybrid metric plot (MRA + Exact Match)...")
    plot_hybrid_metric(df)
    print()
    
    print("Creating MRA vs Exact Match breakdown plot...")
    plot_mra_breakdown(df)
    print()
    
    print("Creating summary table...")
    create_summary_table(df)
    print()
    
    print("="*80)
    print(f"✅ All plots saved to: {OUTPUT_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()
