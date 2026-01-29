#!/usr/bin/env python3
"""
Compare all 4 approaches on VSI-Bench:
1. Sequential 8 steps (mesh exploration)
2. Sequential 16 steps (mesh exploration)
3. Video baseline 8 frames
4. Video baseline 16 frames

Usage:
    python analysis/scripts/compare_all_approaches.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 11


# Experiment folder configurations
EXPERIMENTS = {
    'Sequential 8 Steps': [
        '20260122_163600_sequential_split1of4_8steps',
        '20260122_180418_sequential_split2of4_8steps',
        '20260122_182103_sequential_split3of4_8steps',
        '20260122_185604_sequential_split4of4_8steps',
    ],
    'Sequential 16 Steps': [
        '20260123_002949_sequential_split1of4_16steps',
        '20260123_003904_sequential_split2of4_16steps',
        '20260123_003904_sequential_split3of4_16steps',
        '20260123_003904_sequential_split4of4_16steps',
    ],
    'Video 8 Frames': [
        '20260124_003553_video_baseline_8frames_split1of4',
        '20260124_003552_video_baseline_8frames_split2of4',
        '20260124_004427_video_baseline_8frames_split3of4',
        '20260124_005714_video_baseline_8frames_split4of4',
    ],
    'Video 16 Frames': [
        '20260124_010750_video_baseline_16frames_split1of4',
        '20260124_012009_video_baseline_16frames_split2of4',
        '20260124_012000_video_baseline_16frames_split3of4',
        '20260124_012028_video_baseline_16frames_split4of4',
    ],
}

# Question type groupings
QUESTION_CATEGORIES = {
    'Relative Direction': ['object_rel_direction_easy', 'object_rel_direction_medium', 'object_rel_direction_hard'],
    'Relative Distance': ['object_rel_distance'],
    'Route Planning': ['route_planning'],
}


def load_experiment_results(folders: List[str], base_dir: Path) -> pd.DataFrame:
    """Load and combine results from experiment folders."""
    all_data = []
    
    for folder in folders:
        csv_file = base_dir / folder / 'results.csv'
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            all_data.append(df)
    
    if not all_data:
        return pd.DataFrame()
    
    return pd.concat(all_data, ignore_index=True)


def calculate_metrics(df: pd.DataFrame) -> Dict:
    """Calculate performance metrics."""
    if len(df) == 0:
        return {'total': 0, 'answered': 0, 'correct': 0, 'accuracy': 0, 'success_rate': 0}
    
    total = len(df)
    has_answer = df[df['model_answer'] != 'NO_ANSWER']
    correct = df[df['gt_answer'] == df['model_answer']]
    
    metrics = {
        'total': total,
        'answered': len(has_answer),
        'correct': len(correct),
        'no_answer': total - len(has_answer),
        'accuracy': 100 * len(correct) / total if total > 0 else 0,
        'accuracy_answered': 100 * len(correct) / len(has_answer) if len(has_answer) > 0 else 0,
        'success_rate': 100 * len(has_answer) / total if total > 0 else 0,
        'avg_time': df['time_seconds'].mean() if 'time_seconds' in df.columns else 0,
    }
    
    # Add step statistics if num_steps column exists
    if 'num_steps' in df.columns:
        steps_data = df['num_steps'][df['num_steps'] > 0]  # Exclude 0 steps (failed questions)
        if len(steps_data) > 0:
            metrics['avg_steps'] = steps_data.mean()
            metrics['median_steps'] = steps_data.median()
            metrics['min_steps'] = steps_data.min()
            metrics['max_steps'] = steps_data.max()
    
    return metrics


def group_question_types(df: pd.DataFrame) -> pd.DataFrame:
    """Add question category column."""
    df = df.copy()
    
    def get_category(qtype):
        for cat, types in QUESTION_CATEGORIES.items():
            if qtype in types:
                return cat
        return 'Other'
    
    df['question_category'] = df['question_type'].apply(get_category)
    return df


def analyze_all_experiments(base_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load and analyze all experiments."""
    results = {}
    
    print("=" * 80)
    print("📊 LOADING EXPERIMENT DATA")
    print("=" * 80)
    
    for exp_name, folders in EXPERIMENTS.items():
        df = load_experiment_results(folders, base_dir)
        if len(df) > 0:
            df = group_question_types(df)
            results[exp_name] = df
            print(f"  ✓ {exp_name}: {len(df)} questions loaded")
        else:
            print(f"  ✗ {exp_name}: No data found")
    
    return results


def print_overall_comparison(results: Dict[str, pd.DataFrame]):
    """Print overall comparison table."""
    print("\n" + "=" * 80)
    print("📈 OVERALL PERFORMANCE COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Approach':<25} {'Total':>8} {'Correct':>8} {'Acc (All)':>10} {'Acc (Ans)':>10} {'Success':>10} {'Avg Time':>10}")
    print("-" * 95)
    
    for exp_name, df in results.items():
        metrics = calculate_metrics(df)
        print(f"{exp_name:<25} {metrics['total']:>8} {metrics['correct']:>8} "
              f"{metrics['accuracy']:>9.2f}% {metrics['accuracy_answered']:>9.2f}% "
              f"{metrics['success_rate']:>9.2f}% {metrics['avg_time']:>9.2f}s")
    
    # Print step statistics for sequential approaches
    print("\n" + "=" * 80)
    print("📊 STEP STATISTICS (Sequential Approaches Only)")
    print("=" * 80)
    
    has_steps = False
    for exp_name, df in results.items():
        if 'Sequential' in exp_name:
            metrics = calculate_metrics(df)
            if 'avg_steps' in metrics:
                has_steps = True
                print(f"\n{exp_name}:")
                print(f"  Average Steps:   {metrics['avg_steps']:.2f}")
                print(f"  Median Steps:    {metrics['median_steps']:.1f}")
                print(f"  Min Steps:       {int(metrics['min_steps'])}")
                print(f"  Max Steps:       {int(metrics['max_steps'])}")
    
    if not has_steps:
        print("\n  No step data found in results.")


def print_category_comparison(results: Dict[str, pd.DataFrame]):
    """Print comparison by question category."""
    print("\n" + "=" * 80)
    print("📋 PERFORMANCE BY QUESTION CATEGORY")
    print("=" * 80)
    
    categories = list(QUESTION_CATEGORIES.keys())
    
    for category in categories:
        print(f"\n### {category}")
        print(f"{'Approach':<25} {'Total':>8} {'Correct':>8} {'Acc (All)':>10} {'Acc (Ans)':>10}")
        print("-" * 70)
        
        for exp_name, df in results.items():
            cat_df = df[df['question_category'] == category]
            metrics = calculate_metrics(cat_df)
            # Only show Acc (Ans) for Sequential approaches
            if 'Sequential' in exp_name:
                print(f"{exp_name:<25} {metrics['total']:>8} {metrics['correct']:>8} "
                      f"{metrics['accuracy']:>9.2f}% {metrics['accuracy_answered']:>9.2f}%")
            else:
                print(f"{exp_name:<25} {metrics['total']:>8} {metrics['correct']:>8} "
                      f"{metrics['accuracy']:>9.2f}% {metrics['accuracy_answered']:>9.2f}%")


def print_detailed_type_comparison(results: Dict[str, pd.DataFrame]):
    """Print comparison by detailed question type."""
    print("\n" + "=" * 80)
    print("📝 PERFORMANCE BY QUESTION TYPE (DETAILED)")
    print("=" * 80)
    
    # Get all question types
    all_types = set()
    for df in results.values():
        all_types.update(df['question_type'].unique())
    
    for qtype in sorted(all_types):
        print(f"\n### {qtype}")
        print(f"{'Approach':<25} {'Total':>8} {'Correct':>8} {'Accuracy':>10}")
        print("-" * 55)
        
        for exp_name, df in results.items():
            type_df = df[df['question_type'] == qtype]
            metrics = calculate_metrics(type_df)
            print(f"{exp_name:<25} {metrics['total']:>8} {metrics['correct']:>8} "
                  f"{metrics['accuracy']:>9.2f}%")


def create_visualizations(results: Dict[str, pd.DataFrame], output_dir: Path):
    """Create comparison visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for plotting
    exp_names = list(results.keys())
    categories = list(QUESTION_CATEGORIES.keys())
    
    # Colors for each approach
    colors = {
        'Sequential 8 Steps': '#3498db',
        'Sequential 16 Steps': '#2980b9',
        'Video 8 Frames': '#e74c3c',
        'Video 16 Frames': '#c0392b',
    }
    
    # Figure 1: Overall accuracy comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. Overall accuracy bar chart
    ax = axes[0, 0]
    accuracies = [calculate_metrics(results[exp])['accuracy'] for exp in exp_names]
    bars = ax.bar(range(len(exp_names)), accuracies, color=[colors[exp] for exp in exp_names], 
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels([exp.replace(' ', '\n') for exp in exp_names], fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Overall Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(accuracies) + 10])
    ax.grid(axis='y', alpha=0.3)
    
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 2. Accuracy by category (grouped bar chart)
    ax = axes[0, 1]
    x = np.arange(len(categories))
    width = 0.2
    
    for i, exp_name in enumerate(exp_names):
        df = results[exp_name]
        cat_accs = []
        for cat in categories:
            cat_df = df[df['question_category'] == cat]
            cat_accs.append(calculate_metrics(cat_df)['accuracy'])
        
        offset = (i - len(exp_names)/2 + 0.5) * width
        bars = ax.bar(x + offset, cat_accs, width, label=exp_name, 
                     color=colors[exp_name], alpha=0.8, edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy by Question Category', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Success rate comparison
    ax = axes[1, 0]
    success_rates = [calculate_metrics(results[exp])['success_rate'] for exp in exp_names]
    bars = ax.bar(range(len(exp_names)), success_rates, color=[colors[exp] for exp in exp_names],
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels([exp.replace(' ', '\n') for exp in exp_names], fontsize=10)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Rate (% Questions Answered)', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, rate in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 4. Average time comparison
    ax = axes[1, 1]
    times = [calculate_metrics(results[exp])['avg_time'] for exp in exp_names]
    bars = ax.bar(range(len(exp_names)), times, color=[colors[exp] for exp in exp_names],
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels([exp.replace(' ', '\n') for exp in exp_names], fontsize=10)
    ax.set_ylabel('Average Time (seconds)', fontsize=12)
    ax.set_title('Average Time per Question', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{t:.1f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'comparison_overview.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[SAVE] {output_path}")
    plt.close()
    
    # Figure 2: Detailed question type comparison
    all_types = sorted(set().union(*[set(df['question_type'].unique()) for df in results.values()]))
    
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(all_types))
    width = 0.2
    
    for i, exp_name in enumerate(exp_names):
        df = results[exp_name]
        type_accs = []
        for qtype in all_types:
            type_df = df[df['question_type'] == qtype]
            type_accs.append(calculate_metrics(type_df)['accuracy'])
        
        offset = (i - len(exp_names)/2 + 0.5) * width
        ax.bar(x + offset, type_accs, width, label=exp_name, 
               color=colors[exp_name], alpha=0.8, edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace('object_rel_', '').replace('_', '\n') for t in all_types], 
                       fontsize=9, rotation=0)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy by Question Type (Detailed)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'comparison_by_type.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[SAVE] {output_path}")
    plt.close()
    
    # Figure 3: Sequential vs Video comparison (heatmap style)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    data = []
    for exp_name in exp_names:
        df = results[exp_name]
        row = [calculate_metrics(df[df['question_category'] == cat])['accuracy'] 
               for cat in categories]
        row.append(calculate_metrics(df)['accuracy'])  # Overall
        data.append(row)
    
    data = np.array(data)
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=60)
    
    ax.set_xticks(range(len(categories) + 1))
    ax.set_xticklabels(categories + ['Overall'], fontsize=10, rotation=45, ha='right')
    ax.set_yticks(range(len(exp_names)))
    ax.set_yticklabels(exp_names, fontsize=10)
    
    # Add text annotations
    for i in range(len(exp_names)):
        for j in range(len(categories) + 1):
            text = ax.text(j, i, f'{data[i, j]:.1f}%', ha='center', va='center', 
                          fontsize=11, fontweight='bold',
                          color='white' if data[i, j] > 40 else 'black')
    
    ax.set_title('Accuracy Heatmap: All Approaches × Categories', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Accuracy (%)')
    
    plt.tight_layout()
    output_path = output_dir / 'comparison_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[SAVE] {output_path}")
    plt.close()


def save_summary_csv(results: Dict[str, pd.DataFrame], output_dir: Path):
    """Save summary CSV with all metrics."""
    rows = []
    
    for exp_name, df in results.items():
        # Overall metrics
        overall = calculate_metrics(df)
        row = {
            'Approach': exp_name,
            'Category': 'Overall',
            'Total': overall['total'],
            'Correct': overall['correct'],
            'No Answer': overall['no_answer'],
            'Accuracy All (%)': round(overall['accuracy'], 2),
            'Accuracy Answered (%)': round(overall['accuracy_answered'], 2),
            'Success Rate (%)': round(overall['success_rate'], 2),
            'Avg Time (s)': round(overall['avg_time'], 2),
        }
        
        # Add step statistics for sequential approaches
        if 'avg_steps' in overall:
            row['Avg Steps'] = round(overall['avg_steps'], 2)
            row['Median Steps'] = round(overall['median_steps'], 1)
            row['Min Steps'] = int(overall['min_steps'])
            row['Max Steps'] = int(overall['max_steps'])
        
        rows.append(row)
        
        # By category
        for cat in QUESTION_CATEGORIES.keys():
            cat_df = df[df['question_category'] == cat]
            cat_metrics = calculate_metrics(cat_df)
            row = {
                'Approach': exp_name,
                'Category': cat,
                'Total': cat_metrics['total'],
                'Correct': cat_metrics['correct'],
                'No Answer': cat_metrics['no_answer'],
                'Accuracy All (%)': round(cat_metrics['accuracy'], 2),
                'Accuracy Answered (%)': round(cat_metrics['accuracy_answered'], 2),
                'Success Rate (%)': round(cat_metrics['success_rate'], 2),
                'Avg Time (s)': round(cat_metrics['avg_time'], 2),
            }
            
            # Add step statistics for sequential approaches
            if 'avg_steps' in cat_metrics:
                row['Avg Steps'] = round(cat_metrics['avg_steps'], 2)
                row['Median Steps'] = round(cat_metrics['median_steps'], 1)
                row['Min Steps'] = int(cat_metrics['min_steps'])
                row['Max Steps'] = int(cat_metrics['max_steps'])
            
            rows.append(row)
    
    summary_df = pd.DataFrame(rows)
    output_path = output_dir / 'comparison_summary.csv'
    summary_df.to_csv(output_path, index=False)
    print(f"[SAVE] {output_path}")


def main():
    """Main analysis function."""
    print("\n" + "=" * 80)
    print("🔬 VSI-BENCH: COMPARING ALL 4 APPROACHES")
    print("=" * 80)
    print("\nApproaches:")
    print("  1. Sequential 8 Steps  - Mesh exploration with 8 rendered views")
    print("  2. Sequential 16 Steps - Mesh exploration with 16 rendered views")
    print("  3. Video 8 Frames      - 8 equally-spaced video frames")
    print("  4. Video 16 Frames     - 16 equally-spaced video frames")
    
    base_dir = Path("experiment_logs")
    output_dir = Path("experiment_logs/comparison_all_approaches")
    
    # Load all experiments
    results = analyze_all_experiments(base_dir)
    
    if len(results) < 4:
        print(f"\n[WARNING] Only {len(results)} experiments loaded, expected 4")
    
    # Print comparisons
    print_overall_comparison(results)
    print_category_comparison(results)
    print_detailed_type_comparison(results)
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("📊 GENERATING VISUALIZATIONS")
    print("=" * 80)
    create_visualizations(results, output_dir)
    
    # Save summary CSV
    save_summary_csv(results, output_dir)
    
    # Key findings
    print("\n" + "=" * 80)
    print("🎯 KEY FINDINGS")
    print("=" * 80)
    
    # Find best approach overall
    best_overall = max(results.items(), key=lambda x: calculate_metrics(x[1])['accuracy'])
    print(f"\n✓ Best Overall Accuracy: {best_overall[0]} ({calculate_metrics(best_overall[1])['accuracy']:.2f}%)")
    
    # Find best for each category
    for cat in QUESTION_CATEGORIES.keys():
        best_cat = max(results.items(), 
                       key=lambda x: calculate_metrics(x[1][x[1]['question_category'] == cat])['accuracy'])
        cat_acc = calculate_metrics(best_cat[1][best_cat[1]['question_category'] == cat])['accuracy']
        print(f"✓ Best for {cat}: {best_cat[0]} ({cat_acc:.2f}%)")
    
    # Sequential vs Video comparison
    seq_8 = calculate_metrics(results.get('Sequential 8 Steps', pd.DataFrame()))['accuracy']
    seq_16 = calculate_metrics(results.get('Sequential 16 Steps', pd.DataFrame()))['accuracy']
    vid_8 = calculate_metrics(results.get('Video 8 Frames', pd.DataFrame()))['accuracy']
    vid_16 = calculate_metrics(results.get('Video 16 Frames', pd.DataFrame()))['accuracy']
    
    print(f"\n📊 Step/Frame Count Impact:")
    print(f"   Sequential: 8→16 steps = {seq_16 - seq_8:+.2f}% accuracy change")
    print(f"   Video:      8→16 frames = {vid_16 - vid_8:+.2f}% accuracy change")
    
    print(f"\n📊 Approach Comparison (at 16 steps/frames):")
    print(f"   Sequential 16 Steps: {seq_16:.2f}%")
    print(f"   Video 16 Frames:     {vid_16:.2f}%")
    print(f"   Difference:          {seq_16 - vid_16:+.2f}%")
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
