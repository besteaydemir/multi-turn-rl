#!/usr/bin/env python3
"""
Generate publication-quality plots for VSI-Bench experiment comparison.
Follows academic paper standards with proper fonts, colors, and styling.

Usage:
    python analysis/plot_results.py
    python analysis/plot_results.py --input comparison_combined.csv --output figures/
    python analysis/plot_results.py --latex  # Use LaTeX fonts (requires LaTeX installation)
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# Publication-ready styling
def setup_publication_style(use_latex: bool = False):
    """Set up publication-quality matplotlib/seaborn styling."""
    
    # LaTeX fonts (academic standard)
    if use_latex:
        plt.rcParams["text.usetex"] = True
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times", "Computer Modern Roman"]
    else:
        # Fallback to system serif fonts
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "serif"]
    
    # Publication settings
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
        "patch.linewidth": 0.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "axes.edgecolor": "0.3",
        "axes.axisbelow": True,
    })
    
    # Seaborn theme for better defaults
    sns.set_theme(style="whitegrid", context="paper", palette="muted")
    sns.despine(left=True, bottom=True)

# Academic color palettes
COLORS = {
    # Custom color scheme
    'method': ['#443acf', '#bd0a0a'],  # Purple (Sequential), Red (Video)
    'model': ['#229e4a', '#659df2'],   # Green (4B), Blue (8B)
    'frames': ['#229e4a', '#443acf', '#bd0a0a', '#659df2'],  # Green, Purple, Red, Blue
    
    # Tableau colors (muted) - kept for backwards compatibility
    'tableau': ['#4e79a7', '#f28e2c', '#e15759', '#76b7b2', '#59a14f', 
                '#edc949', '#af7aa1', '#ff9da7', '#9c755f', '#bab0ab'],
}

def load_and_preprocess_data(csv_path: Path) -> pd.DataFrame:
    """Load and preprocess the comparison CSV data."""
    df = pd.read_csv(csv_path)
    
    # Clean up column names
    df.columns = df.columns.str.strip()
    
    # Extract numeric values from percentage strings
    for col in ['Accuracy (%)', 'Success Rate (%)', 'ARKit Acc (%)', 'ScanNet Acc (%)',
                'Accuracy Std', 'Accuracy SEM']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Extract frame/step numbers
    def extract_number(frame_step_str):
        if pd.isna(frame_step_str):
            return None
        import re
        match = re.search(r'(\d+)', str(frame_step_str))
        return int(match.group(1)) if match else None
    
    df['Frames/Steps_Num'] = df['Frames/Steps'].apply(extract_number)
    
    # Clean model names
    df['Model_Clean'] = df['Model'].str.replace('Qwen3-VL-', '', regex=False)
    
    # Sort by frames/steps for proper line connections
    df = df.sort_values('Frames/Steps_Num')
    
    return df

def plot_overall_accuracy(df: pd.DataFrame, output_dir: Path):
    """Plot overall accuracy comparison by method and model."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Prepare data for grouped bar chart
    df_plot = df.groupby(['Method', 'Model_Clean', 'Frames/Steps_Num'])['Accuracy (%)'].first().reset_index()
    
    # Create grouped bar plot
    methods = ['Sequential', 'Video']
    models = ['4B', '8B']
    x = np.arange(len(methods))
    width = 0.35
    
    for i, model in enumerate(models):
        model_data = []
        for method in methods:
            method_model_data = df_plot[(df_plot['Method'] == method) & 
                                       (df_plot['Model_Clean'] == model)]
            if len(method_model_data) > 0:
                # Take best performing configuration
                best_acc = method_model_data['Accuracy (%)'].max()
                model_data.append(best_acc)
            else:
                model_data.append(0)
        
        ax.bar(x + i*width, model_data, width, 
               label=f'{model} Model', 
               color=COLORS['model'][i],
               alpha=0.8,
               edgecolor='white',
               linewidth=0.8)
    
    ax.set_xlabel('Method')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('VSI-Bench: Best Accuracy by Method and Model Size')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(df['Accuracy (%)']) * 1.1)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_accuracy.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(output_dir / 'overall_accuracy.png', dpi=600, bbox_inches='tight')
    plt.close()

def plot_accuracy_vs_frames(df: pd.DataFrame, output_dir: Path):
    """Plot accuracy vs number of frames/steps with error bars."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = ['Sequential', 'Video']
    colors = COLORS['method']
    
    for i, method in enumerate(methods):
        ax = ax1 if method == 'Sequential' else ax2
        method_data = df[df['Method'] == method].copy()
        
        for j, model in enumerate(['4B', '8B']):
            model_data = method_data[method_data['Model_Clean'] == model].copy()
            if len(model_data) > 0:
                # Sort by frames/steps to ensure correct line connections
                model_data = model_data.sort_values('Frames/Steps_Num')
                
                # Get error bars (use SEM if available, otherwise use std)
                if 'Accuracy SEM' in model_data.columns and model_data['Accuracy SEM'].notna().any():
                    yerr = model_data['Accuracy SEM']
                    error_label = 'SEM'
                elif 'Accuracy Std' in model_data.columns and model_data['Accuracy Std'].notna().any():
                    yerr = model_data['Accuracy Std']
                    error_label = 'Std'
                else:
                    yerr = None
                    error_label = None
                
                ax.errorbar(model_data['Frames/Steps_Num'], 
                           model_data['Accuracy (%)'],
                           yerr=yerr,
                           fmt='o-', 
                           color=COLORS['model'][j],
                           label=f'{model} Model',
                           linewidth=2,
                           markersize=8,
                           alpha=0.8,
                           capsize=5,
                           capthick=1.5)
        
        ax.set_xlabel('Number of Steps' if method == 'Sequential' else 'Number of Frames')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{method} Method')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set y-axis based on data range
        if len(method_data) > 0:
            min_acc = method_data['Accuracy (%)'].min()
            max_acc = method_data['Accuracy (%)'].max()
            y_range = max_acc - min_acc
            ax.set_ylim(max(0, min_acc - y_range * 0.15), max_acc + y_range * 0.15)
        
        # Set x-axis to show actual frame numbers where data exists
        if len(method_data) > 0:
            frame_nums = sorted(method_data['Frames/Steps_Num'].unique())
            ax.set_xticks(frame_nums)
            ax.set_xticklabels([str(int(f)) for f in frame_nums])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_vs_frames.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(output_dir / 'accuracy_vs_frames.png', dpi=600, bbox_inches='tight')
    plt.close()

def plot_methods_by_model(df: pd.DataFrame, output_dir: Path):
    """Plot both methods together, separated by model size."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    models = ['4B', '8B']
    axes = [ax1, ax2]
    
    for ax, model in zip(axes, models):
        model_data = df[df['Model_Clean'] == model].copy()
        
        for i, method in enumerate(['Sequential', 'Video']):
            method_data = model_data[model_data['Method'] == method].copy()
            if len(method_data) > 0:
                # Sort by frames/steps to ensure correct line connections
                method_data = method_data.sort_values('Frames/Steps_Num')
                
                # Get error bars (use SEM if available, otherwise use std)
                if 'Accuracy SEM' in method_data.columns and method_data['Accuracy SEM'].notna().any():
                    yerr = method_data['Accuracy SEM']
                elif 'Accuracy Std' in method_data.columns and method_data['Accuracy Std'].notna().any():
                    yerr = method_data['Accuracy Std']
                else:
                    yerr = None
                
                ax.errorbar(method_data['Frames/Steps_Num'], 
                           method_data['Accuracy (%)'],
                           yerr=yerr,
                           fmt='o-', 
                           color=COLORS['method'][i],
                           label=method,
                           linewidth=2,
                           markersize=8,
                           alpha=0.8,
                           capsize=5,
                           capthick=1.5)
        
        ax.set_xlabel('Number of Frames/Steps')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{model} Model')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(df['Accuracy (%)']) * 1.05)
        
        # Set x-axis to show actual frame numbers where data exists
        if len(model_data) > 0:
            frame_nums = sorted(model_data['Frames/Steps_Num'].unique())
            ax.set_xticks(frame_nums)
            ax.set_xticklabels([str(int(f)) for f in frame_nums])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'methods_by_model.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(output_dir / 'methods_by_model.png', dpi=600, bbox_inches='tight')
    plt.close()

def plot_method_comparison_by_model(df: pd.DataFrame, output_dir: Path):
    """Plot 4B vs 8B comparison with two lines (Sequential vs Video) per subplot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    models = ['4B', '8B']
    axes = [ax1, ax2]
    
    for ax, model in zip(axes, models):
        model_data = df[df['Model_Clean'] == model].copy()
        
        for i, method in enumerate(['Sequential', 'Video']):
            method_data = model_data[model_data['Method'] == method].copy()
            if len(method_data) > 0:
                # Sort by frames/steps to ensure correct line connections
                method_data = method_data.sort_values('Frames/Steps_Num')
                
                # Get error bars (use SEM if available, otherwise use std)
                if 'Accuracy SEM' in method_data.columns and method_data['Accuracy SEM'].notna().any():
                    yerr = method_data['Accuracy SEM']
                elif 'Accuracy Std' in method_data.columns and method_data['Accuracy Std'].notna().any():
                    yerr = method_data['Accuracy Std']
                else:
                    yerr = None
                
                ax.errorbar(method_data['Frames/Steps_Num'], 
                           method_data['Accuracy (%)'],
                           yerr=yerr,
                           fmt='o-', 
                           color=COLORS['method'][i],
                           label=method,
                           linewidth=2.5,
                           markersize=9,
                           alpha=0.85,
                           capsize=5,
                           capthick=1.5)
        
        ax.set_xlabel('Number of Frames/Steps', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(f'Qwen3-VL-{model}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Set y-axis based on data range
        if len(model_data) > 0:
            min_acc = model_data['Accuracy (%)'].min()
            max_acc = model_data['Accuracy (%)'].max()
            y_range = max_acc - min_acc
            ax.set_ylim(max(0, min_acc - y_range * 0.15), max_acc + y_range * 0.15)
        
        # Set x-axis to show actual frame numbers where data exists
        if len(model_data) > 0:
            frame_nums = sorted(model_data['Frames/Steps_Num'].unique())
            ax.set_xticks(frame_nums)
            ax.set_xticklabels([str(int(f)) for f in frame_nums])
    
    plt.suptitle('Sequential vs Video Method Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'method_comparison_4b_vs_8b.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(output_dir / 'method_comparison_4b_vs_8b.png', dpi=600, bbox_inches='tight')
    plt.close()

def plot_dataset_breakdown(df: pd.DataFrame, output_dir: Path):
    """Plot accuracy breakdown by dataset (ARKit vs ScanNet)."""
    # Filter out rows with missing dataset breakdown
    df_valid = df[(df['ARKit Acc (%)'].notna()) & (df['ScanNet Acc (%)'].notna())]
    
    if len(df_valid) == 0:
        print("Warning: No dataset breakdown data available")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Prepare data
    configs = []
    arkit_acc = []
    scannet_acc = []
    
    for _, row in df_valid.iterrows():
        config = f"{row['Method']}\n{row['Model_Clean']} ({row['Frames/Steps_Num']})"
        configs.append(config)
        arkit_acc.append(row['ARKit Acc (%)'])
        scannet_acc.append(row['ScanNet Acc (%)'])
    
    x = np.arange(len(configs))
    width = 0.35
    
    ax.bar(x - width/2, arkit_acc, width, 
           label='ARKitScenes', 
           color='#2E8B57',  # Sea Green
           alpha=0.8,
           edgecolor='white',
           linewidth=0.8)
    ax.bar(x + width/2, scannet_acc, width, 
           label='ScanNet', 
           color='#4682B4',  # Steel Blue
           alpha=0.8,
           edgecolor='white',
           linewidth=0.8)
    
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy by Dataset: ARKitScenes vs ScanNet')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_breakdown.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(output_dir / 'dataset_breakdown.png', dpi=600, bbox_inches='tight')
    plt.close()

def plot_method_comparison_heatmap(df: pd.DataFrame, output_dir: Path):
    """Create a heatmap comparing methods across configurations."""
    # Pivot data for heatmap
    pivot_data = df.pivot_table(
        values='Accuracy (%)', 
        index=['Method', 'Model_Clean'], 
        columns='Frames/Steps_Num', 
        aggfunc='first'
    )
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Create heatmap with custom colormap
    sns.heatmap(pivot_data, 
                annot=True, 
                fmt='.1f', 
                cmap='YlOrRd',  # Yellow-Orange-Red colormap
                cbar_kws={'label': 'Accuracy (%)'},
                square=True,
                linewidths=0.5,
                ax=ax)
    
    ax.set_title('Accuracy Heatmap: Method vs Configuration')
    ax.set_xlabel('Frames/Steps')
    ax.set_ylabel('Method & Model')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'method_heatmap.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(output_dir / 'method_heatmap.png', dpi=600, bbox_inches='tight')
    plt.close()

def plot_success_rate_vs_accuracy(df: pd.DataFrame, output_dir: Path):
    """Scatter plot of success rate vs accuracy."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    methods = df['Method'].unique()
    colors = COLORS['method'][:len(methods)]
    
    for i, method in enumerate(methods):
        method_data = df[df['Method'] == method]
        
        for j, model in enumerate(['4B', '8B']):
            model_data = method_data[method_data['Model_Clean'] == model]
            if len(model_data) > 0:
                marker = 'o' if model == '4B' else 's'
                ax.scatter(model_data['Success Rate (%)'], 
                          model_data['Accuracy (%)'],
                          c=colors[i],
                          marker=marker,
                          s=100,
                          alpha=0.7,
                          edgecolors='black',
                          linewidth=0.5,
                          label=f'{method} {model}')
    
    ax.set_xlabel('Success Rate (%)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Success Rate vs Accuracy Trade-off')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add diagonal reference line
    min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
    max_val = min(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'success_vs_accuracy.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(output_dir / 'success_vs_accuracy.png', dpi=600, bbox_inches='tight')
    plt.close()

def create_summary_figure(df: pd.DataFrame, output_dir: Path):
    """Create a comprehensive summary figure with multiple subplots."""
    fig = plt.figure(figsize=(16, 12))
    
    # Define grid layout
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
    
    # Overall accuracy comparison
    ax1 = fig.add_subplot(gs[0, :2])
    methods = ['Sequential', 'Video']
    models = ['4B', '8B']
    
    df_best = df.loc[df.groupby(['Method', 'Model_Clean'])['Accuracy (%)'].idxmax()]
    
    x = np.arange(len(methods))
    width = 0.35
    
    for i, model in enumerate(models):
        model_data = []
        for method in methods:
            method_data = df_best[(df_best['Method'] == method) & 
                                 (df_best['Model_Clean'] == model)]
            if len(method_data) > 0:
                model_data.append(method_data['Accuracy (%)'].iloc[0])
            else:
                model_data.append(0)
        
        ax1.bar(x + i*width, model_data, width, 
               label=f'{model} Model', 
               color=COLORS['model'][i],
               alpha=0.8)
    
    ax1.set_title('Best Accuracy by Method')
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_xticks(x + width/2)
    ax1.set_xticklabels(methods)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Accuracy vs frames/steps
    ax2 = fig.add_subplot(gs[0, 2])
    for method in ['Sequential', 'Video']:
        method_data = df[df['Method'] == method]
        avg_acc = method_data.groupby('Frames/Steps_Num')['Accuracy (%)'].mean()
        ax2.plot(avg_acc.index, avg_acc.values, 'o-', 
                label=method, linewidth=2, markersize=6)
    
    ax2.set_title('Accuracy vs Steps/Frames')
    ax2.set_xlabel('Steps/Frames')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Dataset breakdown (if available)
    if df['ARKit Acc (%)'].notna().any():
        ax3 = fig.add_subplot(gs[1, :])
        df_valid = df[(df['ARKit Acc (%)'].notna()) & (df['ScanNet Acc (%)'].notna())]
        
        configs = [f"{row['Method']} {row['Model_Clean']} ({row['Frames/Steps_Num']})" 
                  for _, row in df_valid.iterrows()]
        
        x = np.arange(len(configs))
        width = 0.35
        
        ax3.bar(x - width/2, df_valid['ARKit Acc (%)'], width, 
               label='ARKitScenes', color='#2E8B57', alpha=0.8)
        ax3.bar(x + width/2, df_valid['ScanNet Acc (%)'], width, 
               label='ScanNet', color='#4682B4', alpha=0.8)
        
        ax3.set_title('Accuracy by Dataset')
        ax3.set_xlabel('Configuration')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(configs, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
    
    # Success rate vs accuracy
    ax4 = fig.add_subplot(gs[2, :])
    for i, method in enumerate(['Sequential', 'Video']):
        method_data = df[df['Method'] == method]
        ax4.scatter(method_data['Success Rate (%)'], 
                   method_data['Accuracy (%)'],
                   c=COLORS['method'][i],
                   s=80,
                   alpha=0.7,
                   label=method,
                   edgecolors='black',
                   linewidth=0.5)
    
    ax4.set_title('Success Rate vs Accuracy')
    ax4.set_xlabel('Success Rate (%)')
    ax4.set_ylabel('Accuracy (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_figure.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(output_dir / 'summary_figure.png', dpi=600, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate publication-quality plots for VSI-Bench results")
    parser.add_argument("--input", type=str, default="comparison_combined.csv",
                       help="Input CSV file with comparison results")
    parser.add_argument("--output", type=str, default="figures/",
                       help="Output directory for plots")
    parser.add_argument("--latex", action="store_true",
                       help="Use LaTeX fonts (requires LaTeX installation)")
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    setup_publication_style(args.latex)
    
    # Load data
    print(f"Loading data from: {args.input}")
    df = load_and_preprocess_data(Path(args.input))
    print(f"Loaded {len(df)} experiment configurations")
    
    # Generate plots
    print("Generating plots...")
    
    plot_overall_accuracy(df, output_dir)
    print("✓ Overall accuracy comparison")
    
    plot_accuracy_vs_frames(df, output_dir)
    print("✓ Accuracy vs frames/steps")
    
    plot_methods_by_model(df, output_dir)
    print("✓ Methods by model comparison")
    
    plot_method_comparison_by_model(df, output_dir)
    print("✓ Method comparison (4B vs 8B)")
    
    plot_dataset_breakdown(df, output_dir)
    print("✓ Dataset breakdown")
    
    plot_method_comparison_heatmap(df, output_dir)
    print("✓ Method comparison heatmap")
    
    plot_success_rate_vs_accuracy(df, output_dir)
    print("✓ Success rate vs accuracy")
    
    create_summary_figure(df, output_dir)
    print("✓ Summary figure")
    
    print(f"\nAll plots saved to: {output_dir}")
    print("Generated files:")
    for file in sorted(output_dir.glob("*.pdf")):
        print(f"  📊 {file.name}")

if __name__ == "__main__":
    main()