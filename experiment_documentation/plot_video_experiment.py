#!/usr/bin/env python3
"""
Create plots for Video Baseline Experiment (February 5, 2026)

This script analyzes the video baseline results across:
- 2 models (4B, 8B)
- 4 frame configurations (4, 8, 16, 32)
- 5,130 questions total

Usage:
    python experiment_documentation/plot_video_experiment.py
    python experiment_documentation/plot_video_experiment.py --output experiment_documentation/figures/
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def setup_style():
    """Set up publication-quality matplotlib styling."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "lines.linewidth": 2.0,
        "lines.markersize": 8,
    })
    sns.set_theme(style="whitegrid", context="paper", palette="muted")
    

COLORS = {
    'model': {'4B': '#229e4a', '8B': '#659df2'},
    'frames': {4: '#229e4a', 8: '#443acf', 16: '#bd0a0a', 32: '#659df2'},
}


def load_experiment_results(base_path: Path) -> pd.DataFrame:
    """Load all video experiment results from split files."""
    all_results = []
    
    for model in ['4B', '8B']:
        for frames in [4, 8, 16, 32]:
            model_path = base_path / model / f'{frames}_frames' / '2026-02-05'
            
            if not model_path.exists():
                print(f"Warning: Path not found: {model_path}")
                continue
            
            # Find all split result files
            csv_files = list(model_path.glob('*/results.csv'))
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    df['model'] = model
                    df['frames'] = frames
                    df['split_file'] = csv_file.name
                    all_results.append(df)
                    print(f"Loaded {len(df)} questions from {model} {frames} frames ({csv_file.parent.name})")
                except Exception as e:
                    print(f"Error loading {csv_file}: {e}")
    
    if not all_results:
        raise ValueError("No result files found!")
    
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Calculate correctness
    combined_df['correct'] = combined_df['model_answer'] == combined_df['gt_answer']
    
    return combined_df


def plot_accuracy_vs_frames(df: pd.DataFrame, output_dir: Path):
    """Plot accuracy vs number of frames for both models."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for model in ['4B', '8B']:
        model_data = df[df['model'] == model]
        accuracy_by_frames = model_data.groupby('frames')['correct'].mean() * 100
        
        ax.plot(accuracy_by_frames.index, accuracy_by_frames.values, 
               marker='o', label=f'Qwen3-VL-{model}', 
               color=COLORS['model'][model], linewidth=2.5, markersize=10)
    
    ax.set_xlabel('Number of Frames', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Video Baseline: Accuracy vs Number of Frames', fontweight='bold', pad=15)
    ax.legend(frameon=True, shadow=True, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([4, 8, 16, 32])
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'video_accuracy_vs_frames.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'video_accuracy_vs_frames.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'video_accuracy_vs_frames.png'}")
    plt.close()


def plot_model_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare 4B vs 8B models across all frame configurations."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data
    comparison_data = []
    for model in ['4B', '8B']:
        for frames in [4, 8, 16, 32]:
            subset = df[(df['model'] == model) & (df['frames'] == frames)]
            accuracy = subset['correct'].mean() * 100
            count = len(subset)
            comparison_data.append({
                'Model': f'{model}',
                'Frames': frames,
                'Accuracy': accuracy,
                'Count': count
            })
    
    comp_df = pd.DataFrame(comparison_data)
    
    # Create grouped bar chart
    x = np.arange(len([4, 8, 16, 32]))
    width = 0.35
    
    acc_4b = comp_df[comp_df['Model'] == '4B']['Accuracy'].values
    acc_8b = comp_df[comp_df['Model'] == '8B']['Accuracy'].values
    
    bars1 = ax.bar(x - width/2, acc_4b, width, label='4B', color=COLORS['model']['4B'], alpha=0.8)
    bars2 = ax.bar(x + width/2, acc_8b, width, label='8B', color=COLORS['model']['8B'], alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Number of Frames', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Model Comparison: 4B vs 8B Across Frame Configurations', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([4, 8, 16, 32])
    ax.legend(frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'video_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'video_model_comparison.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'video_model_comparison.png'}")
    plt.close()


def plot_question_type_breakdown(df: pd.DataFrame, output_dir: Path):
    """Plot accuracy breakdown by question type."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    frame_configs = [4, 8, 16, 32]
    
    for idx, frames in enumerate(frame_configs):
        ax = axes[idx]
        subset = df[df['frames'] == frames]
        
        # Calculate accuracy by question type for each model
        accuracy_data = []
        for model in ['4B', '8B']:
            model_subset = subset[subset['model'] == model]
            by_type = model_subset.groupby('question_type')['correct'].mean() * 100
            
            for qtype, acc in by_type.items():
                accuracy_data.append({
                    'Model': model,
                    'Question Type': qtype,
                    'Accuracy': acc
                })
        
        acc_df = pd.DataFrame(accuracy_data)
        
        # Pivot for grouped bar chart
        pivot = acc_df.pivot(index='Question Type', columns='Model', values='Accuracy')
        
        # Plot
        pivot.plot(kind='barh', ax=ax, color=[COLORS['model']['4B'], COLORS['model']['8B']], 
                  alpha=0.8, width=0.7)
        
        ax.set_xlabel('Accuracy (%)', fontweight='bold')
        ax.set_ylabel('')
        ax.set_title(f'{frames} Frames', fontweight='bold', fontsize=12)
        ax.legend(title='Model', frameon=True, shadow=True, loc='lower right')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(0, 100)
    
    plt.suptitle('Accuracy by Question Type and Frame Count', 
                fontweight='bold', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'video_question_type_breakdown.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'video_question_type_breakdown.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'video_question_type_breakdown.png'}")
    plt.close()


def plot_dataset_performance(df: pd.DataFrame, output_dir: Path):
    """Plot performance across different datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, model in enumerate(['4B', '8B']):
        ax = axes[idx]
        model_data = df[df['model'] == model]
        
        # Extract dataset from scene_id
        def get_dataset(scene_id):
            if str(scene_id).startswith('scene'):
                return 'ScanNet'
            elif len(str(scene_id)) == 10 and not str(scene_id).isdigit():
                return 'ScanNet++'
            else:
                return 'ARKitScenes'
        
        model_data['dataset'] = model_data['scene_id'].apply(get_dataset)
        
        # Calculate accuracy by dataset and frames
        accuracy_data = []
        for frames in [4, 8, 16, 32]:
            for dataset in ['ARKitScenes', 'ScanNet', 'ScanNet++']:
                subset = model_data[(model_data['frames'] == frames) & 
                                   (model_data['dataset'] == dataset)]
                if len(subset) > 0:
                    acc = subset['correct'].mean() * 100
                    accuracy_data.append({
                        'Frames': frames,
                        'Dataset': dataset,
                        'Accuracy': acc
                    })
        
        acc_df = pd.DataFrame(accuracy_data)
        
        # Plot grouped bar chart
        pivot = acc_df.pivot(index='Frames', columns='Dataset', values='Accuracy')
        pivot.plot(kind='bar', ax=ax, alpha=0.8, width=0.7)
        
        ax.set_xlabel('Number of Frames', fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.set_title(f'Qwen3-VL-{model}', fontweight='bold', fontsize=12)
        ax.legend(title='Dataset', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 100)
        ax.set_xticklabels([4, 8, 16, 32], rotation=0)
    
    plt.suptitle('Performance Across Datasets', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'video_dataset_performance.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'video_dataset_performance.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'video_dataset_performance.png'}")
    plt.close()


def plot_inference_time(df: pd.DataFrame, output_dir: Path):
    """Plot average inference time vs number of frames."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for model in ['4B', '8B']:
        model_data = df[df['model'] == model]
        avg_time_by_frames = model_data.groupby('frames')['time_seconds'].mean()
        
        ax.plot(avg_time_by_frames.index, avg_time_by_frames.values,
               marker='s', label=f'Qwen3-VL-{model}',
               color=COLORS['model'][model], linewidth=2.5, markersize=10)
    
    ax.set_xlabel('Number of Frames', fontweight='bold')
    ax.set_ylabel('Average Inference Time (seconds)', fontweight='bold')
    ax.set_title('Inference Time vs Number of Frames', fontweight='bold', pad=15)
    ax.legend(frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([4, 8, 16, 32])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'video_inference_time.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'video_inference_time.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'video_inference_time.png'}")
    plt.close()


def create_summary_table(df: pd.DataFrame, output_dir: Path):
    """Create a summary table of all results."""
    summary_data = []
    
    for model in ['4B', '8B']:
        for frames in [4, 8, 16, 32]:
            subset = df[(df['model'] == model) & (df['frames'] == frames)]
            
            summary_data.append({
                'Model': f'Qwen3-VL-{model}',
                'Frames': frames,
                'Questions': len(subset),
                'Accuracy (%)': f"{subset['correct'].mean() * 100:.2f}",
                'Avg Time (s)': f"{subset['time_seconds'].mean():.2f}",
                'Total Time (h)': f"{subset['time_seconds'].sum() / 3600:.2f}",
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    output_file = output_dir / 'video_experiment_summary.csv'
    summary_df.to_csv(output_file, index=False)
    print(f"\nSummary table saved to: {output_file}")
    print("\n" + "="*80)
    print(summary_df.to_string(index=False))
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Plot video experiment results')
    parser.add_argument('--base-path', type=str,
                       default='/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/Video',
                       help='Base path to experiment results')
    parser.add_argument('--output', type=str,
                       default='experiment_documentation/figures',
                       help='Output directory for plots')
    args = parser.parse_args()
    
    # Setup
    setup_style()
    base_path = Path(args.base_path)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("VIDEO EXPERIMENT ANALYSIS - February 5, 2026")
    print("="*80)
    print(f"\nLoading results from: {base_path}")
    
    # Load data
    df = load_experiment_results(base_path)
    print(f"\nTotal questions loaded: {len(df)}")
    print(f"Unique questions: {df['question_id'].nunique()}")
    print(f"Models: {df['model'].unique()}")
    print(f"Frame configurations: {sorted(df['frames'].unique())}")
    
    # Create plots
    print("\n" + "="*80)
    print("Generating plots...")
    print("="*80 + "\n")
    
    plot_accuracy_vs_frames(df, output_dir)
    plot_model_comparison(df, output_dir)
    plot_question_type_breakdown(df, output_dir)
    plot_dataset_performance(df, output_dir)
    plot_inference_time(df, output_dir)
    create_summary_table(df, output_dir)
    
    print(f"\n✅ All plots saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
