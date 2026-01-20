#!/usr/bin/env python3
"""
Analysis script for combining and visualizing results from experimental splits.
Automatically finds the latest experiment runs and reads question types from sequential.py.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from pathlib import Path
from datasets import load_dataset
from datetime import datetime

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# ----------------- Auto-detect latest experiments -----------------
def parse_question_types_from_sequential():
    """Parse question types from sequential.py to match what's being evaluated."""
    sequential_path = Path("/dss/dsshome1/06/di38riq/rl_multi_turn/evaluation/sequential.py")
    if not sequential_path.exists():
        print("[WARN] Could not find sequential.py, using default question types")
        return ["route_planning", "object_rel_distance", "object_rel_direction_easy", 
                "object_rel_direction_medium", "object_rel_direction_hard"]
    
    with open(sequential_path, "r") as f:
        content = f.read()
    
    # Find the mca_types list in load_vsi_bench_questions function
    pattern = r'mca_types\s*=\s*\[(.*?)\]'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        types_str = match.group(1)
        # Extract quoted strings
        types = re.findall(r'["\']([^"\']+)["\']', types_str)
        print(f"[INFO] 📋 Question types from sequential.py: {types}")
        return types
    
    print("[WARN] Could not parse question types, using defaults")
    return ["route_planning", "object_rel_distance", "object_rel_direction_easy", 
            "object_rel_direction_medium", "object_rel_direction_hard"]

def find_latest_experiment_splits(base_path, min_splits=2):
    """Find the latest set of experiment splits (e.g., split1of5, split2of5, etc.)."""
    exp_dirs = list(base_path.glob("*_sequential_split*"))
    
    # Group by date and total_splits (ignore time for grouping)
    experiments = {}
    for exp_dir in exp_dirs:
        match = re.match(r'(\d{8})_\d{6}_sequential_split(\d+)of(\d+)', exp_dir.name)
        if match:
            date, split_num, total_splits = match.groups()
            # Group by date + total_splits to handle job arrays starting at slightly different times
            group_key = f"{date}_split{total_splits}"
            if group_key not in experiments:
                experiments[group_key] = {'date': date, 'splits': {}, 'total_splits': int(total_splits)}
            experiments[group_key]['splits'][int(split_num)] = exp_dir
    
    # Find the most recent complete or in-progress experiment
    best_experiment = None
    best_score = ('', 0)  # (date, num_splits)
    
    for group_key in experiments.keys():
        exp_info = experiments[group_key]
        date = exp_info['date']
        splits = exp_info['splits']
        total_splits = exp_info['total_splits']
        num_splits = len(splits)
        
        # Score: prioritize more recent date, then more splits
        score = (date, num_splits)
        
        if num_splits >= min_splits and score > best_score:
            best_score = score
            best_experiment = (date, splits, total_splits)
    
    if best_experiment:
        date, splits, total_splits = best_experiment
        # Convert splits dict to the expected format
        splits_info = {}
        for split_num, exp_dir in splits.items():
            splits_info[split_num] = {
                'dir': exp_dir,
                'total_splits': total_splits
            }
        print(f"\n[INFO] 🔍 Found experiment: {date} with {len(splits_info)}/{total_splits} splits")
        return date, splits_info, total_splits
    
    print("[ERROR] No experiment splits found!")
    return None, {}, 0

# Find latest experiments
base_path = Path("/dss/dsshome1/06/di38riq/rl_multi_turn/experiment_logs")
exp_timestamp, splits_info, total_splits = find_latest_experiment_splits(base_path)

if not splits_info:
    print("No experiments found. Exiting.")
    exit(1)

# Load results from all available splits
print(f"\n[INFO] 📂 Loading results from {len(splits_info)} splits...")
files = []
for split_num in sorted(splits_info.keys()):
    csv_path = splits_info[split_num]['dir'] / "results.csv"
    if csv_path.exists():
        files.append(csv_path)
        print(f"  ✓ Split {split_num}: {csv_path}")
    else:
        print(f"  ✗ Split {split_num}: No results.csv yet")

if not files:
    print("[ERROR] No results.csv files found!")
    exit(1)

# Read and combine all CSV files
dfs = []
for i, file_path in enumerate(files, 1):
    df = pd.read_csv(file_path)
    df['split'] = f'split{i}'
    dfs.append(df)
    print(f"[INFO] Loaded {file_path.name}: {len(df)} rows")

# Combine all dataframes
combined_df = pd.concat(dfs, ignore_index=True)
print(f"\n[INFO] Combined dataset: {len(combined_df)} rows")
print(f"[INFO] Question types: {sorted(combined_df['question_type'].unique())}")

# Load VSI-Bench to get dataset information for each scene
print("\n[INFO] Loading VSI-Bench dataset to map scenes to datasets...")
vsi = load_dataset("nyu-visionx/VSI-Bench", split="test")
vsi_df = pd.DataFrame(vsi)
# Keep scene_name as string and convert to int only if possible
scene_to_dataset = dict(zip(vsi_df['scene_name'], vsi_df['dataset']))

# Add dataset column to combined_df (convert scene_id to string for matching)
combined_df['scene_id_str'] = combined_df['scene_id'].astype(str)
combined_df['dataset'] = combined_df['scene_id_str'].map(scene_to_dataset)
print(f"[INFO] Mapped {combined_df['dataset'].notna().sum()} / {len(combined_df)} scenes to datasets")
print(f"[INFO] Unique datasets: {sorted(combined_df['dataset'].dropna().unique())}")

# ----------------- Progress Tracking -----------------
# Get question types from sequential.py to calculate total expected questions
question_types_from_code = parse_question_types_from_sequential()

print("\n[INFO] 🔢 Calculating total expected questions...")
vsi = load_dataset("nyu-visionx/VSI-Bench", split="test")
filtered_vsi = vsi.filter(
    lambda x: x["dataset"] == "arkitscenes" and x["question_type"] in question_types_from_code
)
total_expected = len(filtered_vsi)
completed_count = len(combined_df)
remaining_count = total_expected - completed_count

print("\n" + "="*80)
print("📊 PROGRESS SUMMARY")
print("="*80)
print(f"Experiment: {exp_timestamp}")
print(f"Splits running: {len(splits_info)}/{total_splits}")
print(f"Questions completed: {completed_count} / {total_expected}")
print(f"Questions remaining: {remaining_count}")
print(f"Progress: {100*completed_count/total_expected:.1f}%")
print()
for split_num in sorted(splits_info.keys()):
    csv_path = splits_info[split_num]['dir'] / "results.csv"
    if csv_path.exists():
        split_df = pd.read_csv(csv_path)
        print(f"  Split {split_num}: {len(split_df)} questions completed")
    else:
        print(f"  Split {split_num}: Not started yet")

# Calculate accuracy for each question type
def calculate_accuracy(df):
    """Calculate accuracy, handling NO_ANSWER cases."""
    total = len(df)
    if total == 0:
        return 0.0
    correct = (df['gt_answer'] == df['model_answer']).sum()
    return (correct / total) * 100

# Overall accuracy by question type
print("\n" + "="*80)
print("CURRENT ACCURACY BY QUESTION TYPE")
print("="*80)
accuracy_by_type = combined_df.groupby('question_type').apply(
    lambda x: pd.Series({
        'total': len(x),
        'correct': (x['gt_answer'] == x['model_answer']).sum(),
        'no_answer': (x['model_answer'] == 'NO_ANSWER').sum(),
        'answered': (x['model_answer'] != 'NO_ANSWER').sum(),
        'accuracy_all': calculate_accuracy(x),
        'accuracy_answered_only': calculate_accuracy(x[x['model_answer'] != 'NO_ANSWER']) if (x['model_answer'] != 'NO_ANSWER').sum() > 0 else 0.0,
        'success_rate': 100 * (x['model_answer'] != 'NO_ANSWER').sum() / len(x) if len(x) > 0 else 0.0
    })
).round(2)

print(accuracy_by_type.to_string())

# Overall summary
print("\n" + "="*80)
print("OVERALL CURRENT PERFORMANCE")
print("="*80)
total_questions = len(combined_df)
total_correct = (combined_df['gt_answer'] == combined_df['model_answer']).sum()
total_answered = (combined_df['model_answer'] != 'NO_ANSWER').sum()
total_no_answer = (combined_df['model_answer'] == 'NO_ANSWER').sum()

overall_accuracy = calculate_accuracy(combined_df)
answered_only_df = combined_df[combined_df['model_answer'] != 'NO_ANSWER']
accuracy_answered_only = calculate_accuracy(answered_only_df) if len(answered_only_df) > 0 else 0.0
success_rate = 100 * total_answered / total_questions if total_questions > 0 else 0.0

print(f"Total evaluated: {total_questions}")
print(f"Correct answers: {total_correct}")
print(f"Questions answered (not NO_ANSWER): {total_answered} ({success_rate:.1f}%)")
print(f"NO_ANSWER count: {total_no_answer}")
print(f"\nOverall accuracy (all): {overall_accuracy:.2f}%")
print(f"Overall accuracy (answered only): {accuracy_answered_only:.2f}%")
print(f"Success rate (answered/total): {success_rate:.2f}%")

# Combined accuracy for object_rel_direction types
obj_rel_types = ['object_rel_direction_hard', 'object_rel_direction_medium', 'object_rel_direction_easy']
obj_rel_df = combined_df[combined_df['question_type'].isin(obj_rel_types)]
combined_obj_rel_accuracy = calculate_accuracy(obj_rel_df)
obj_rel_no_answer = (obj_rel_df['model_answer'] == 'NO_ANSWER').sum()
obj_rel_answered = obj_rel_df[obj_rel_df['model_answer'] != 'NO_ANSWER']
combined_obj_rel_accuracy_answered_only = calculate_accuracy(obj_rel_answered) if len(obj_rel_answered) > 0 else 0.0

print("\n" + "="*80)
print("COMBINED OBJECT_REL_DIRECTION ACCURACY")
print("="*80)
print(f"Combined accuracy (hard + medium + easy): {combined_obj_rel_accuracy:.2f}%")
print(f"Total questions: {len(obj_rel_df)}")
print(f"Correct answers: {(obj_rel_df['gt_answer'] == obj_rel_df['model_answer']).sum()}")
print(f"NO_ANSWER count: {obj_rel_no_answer}")
print(f"Answered questions: {len(obj_rel_answered)}")
print(f"Accuracy (answered only): {combined_obj_rel_accuracy_answered_only:.2f}%")

# Create a version with combined object_rel_direction category
combined_df['question_type_grouped'] = combined_df['question_type'].apply(
    lambda x: 'object_rel_direction_combined' if x in obj_rel_types else x
)

# Crosstab analysis by dataset and question type
print("\n" + "="*80)
print("QUESTION TYPES BY DATASET (from evaluated results)")
print("="*80)
crosstab = pd.crosstab(combined_df['dataset'], combined_df['question_type_grouped'], margins=True, margins_name='Total')
print(crosstab.to_string())

# Accuracy by dataset and question type
print("\n" + "="*80)
print("ACCURACY BY DATASET AND QUESTION TYPE")
print("="*80)
accuracy_by_dataset_type = combined_df.groupby(['dataset', 'question_type_grouped']).apply(
    lambda x: pd.Series({
        'total': len(x),
        'answered': (x['model_answer'] != 'NO_ANSWER').sum(),
        'no_answer': (x['model_answer'] == 'NO_ANSWER').sum(),
        'correct': (x['gt_answer'] == x['model_answer']).sum(),
        'accuracy_all': calculate_accuracy(x),
        'accuracy_answered_only': calculate_accuracy(x[x['model_answer'] != 'NO_ANSWER']) if (x['model_answer'] != 'NO_ANSWER').sum() > 0 else 0.0,
    })
).round(2)
print(accuracy_by_dataset_type.to_string())

# Overall accuracy by dataset
print("\n" + "="*80)
print("OVERALL ACCURACY BY DATASET")
print("="*80)
accuracy_by_dataset = combined_df.groupby('dataset').apply(
    lambda x: pd.Series({
        'total': len(x),
        'answered': (x['model_answer'] != 'NO_ANSWER').sum(),
        'no_answer': (x['model_answer'] == 'NO_ANSWER').sum(),
        'correct': (x['gt_answer'] == x['model_answer']).sum(),
        'accuracy_all': calculate_accuracy(x),
        'accuracy_answered_only': calculate_accuracy(x[x['model_answer'] != 'NO_ANSWER']) if (x['model_answer'] != 'NO_ANSWER').sum() > 0 else 0.0,
    })
).round(2)
print(accuracy_by_dataset.to_string())

# Overall summary
print("\n" + "="*80)
print("OVERALL SUMMARY")
print("="*80)
total_questions = len(combined_df)
total_answered = (combined_df['model_answer'] != 'NO_ANSWER').sum()
total_no_answer = (combined_df['model_answer'] == 'NO_ANSWER').sum()
total_correct = (combined_df['gt_answer'] == combined_df['model_answer']).sum()
overall_accuracy_all = calculate_accuracy(combined_df)
overall_accuracy_answered = calculate_accuracy(combined_df[combined_df['model_answer'] != 'NO_ANSWER']) if total_answered > 0 else 0.0

print(f"Total rows in combined CSVs: {total_questions}")
print(f"Questions answered: {total_answered} ({(total_answered/total_questions*100):.2f}%)")
print(f"Questions attempted but resulted in NO_ANSWER: {total_no_answer} ({(total_no_answer/total_questions*100):.2f}%)")
print(f"Questions answered correctly: {total_correct}")
print(f"\nAccuracy (all questions): {overall_accuracy_all:.2f}%")
print(f"Accuracy (answered only): {overall_accuracy_answered:.2f}%")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS BY QUESTION TYPE")
print("="*80)
summary = combined_df.groupby('question_type').agg({
    'num_steps': ['mean', 'std', 'min', 'max'],
    'time_seconds': ['mean', 'std', 'min', 'max']
}).round(2)
print(summary.to_string())

# Create visualizations
output_dir = Path("/dss/dsshome1/06/di38riq/rl_multi_turn/experiment_logs/combined_analysis")
output_dir.mkdir(exist_ok=True)

# Color palette for question types
unique_types = sorted(combined_df['question_type'].unique())
colors = sns.color_palette("husl", len(unique_types))
color_map = dict(zip(unique_types, colors))

# 1. Histogram of num_steps by question type
fig, ax = plt.subplots(figsize=(15, 8))
for q_type in unique_types:
    data = combined_df[combined_df['question_type'] == q_type]['num_steps']
    ax.hist(data, bins=range(1, 12), alpha=0.6, label=q_type, color=color_map[q_type], edgecolor='black')

ax.set_xlabel('Number of Steps', fontsize=14, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
ax.set_title('Distribution of Number of Steps by Question Type', fontsize=16, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'num_steps_histogram.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: {output_dir / 'num_steps_histogram.png'}")
plt.close()

# 2. Histogram of time_seconds by question type
fig, ax = plt.subplots(figsize=(15, 8))
for q_type in unique_types:
    data = combined_df[combined_df['question_type'] == q_type]['time_seconds']
    ax.hist(data, bins=30, alpha=0.6, label=q_type, color=color_map[q_type], edgecolor='black')

ax.set_xlabel('Time (seconds)', fontsize=14, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
ax.set_title('Distribution of Time (seconds) by Question Type', fontsize=16, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'time_seconds_histogram.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'time_seconds_histogram.png'}")
plt.close()

# 3. Histogram of num_steps with combined object_rel_direction categories
unique_grouped_types = sorted(combined_df['question_type_grouped'].unique())
colors_grouped = sns.color_palette("husl", len(unique_grouped_types))
color_map_grouped = dict(zip(unique_grouped_types, colors_grouped))

fig, ax = plt.subplots(figsize=(15, 8))
for q_type in unique_grouped_types:
    data = combined_df[combined_df['question_type_grouped'] == q_type]['num_steps']
    ax.hist(data, bins=range(1, 12), alpha=0.6, label=q_type, 
            color=color_map_grouped[q_type], edgecolor='black')

ax.set_xlabel('Number of Steps', fontsize=14, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
ax.set_title('Distribution of Number of Steps by Question Type (Combined Object Rel Direction)', 
             fontsize=16, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'num_steps_histogram_combined.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'num_steps_histogram_combined.png'}")
plt.close()

# 4. Histogram of time_seconds with combined object_rel_direction categories
fig, ax = plt.subplots(figsize=(15, 8))
for q_type in unique_grouped_types:
    data = combined_df[combined_df['question_type_grouped'] == q_type]['time_seconds']
    ax.hist(data, bins=30, alpha=0.6, label=q_type, 
            color=color_map_grouped[q_type], edgecolor='black')

ax.set_xlabel('Time (seconds)', fontsize=14, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
ax.set_title('Distribution of Time (seconds) by Question Type (Combined Object Rel Direction)', 
             fontsize=16, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'time_seconds_histogram_combined.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'time_seconds_histogram_combined.png'}")
plt.close()

# 5. Accuracy bar chart
fig, ax = plt.subplots(figsize=(12, 6))
accuracy_data = accuracy_by_type.sort_values('accuracy_all', ascending=False)
bars = ax.bar(range(len(accuracy_data)), accuracy_data['accuracy_all'], 
              color=[color_map[qt] for qt in accuracy_data.index], edgecolor='black', linewidth=1.5)

ax.set_xlabel('Question Type', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Accuracy by Question Type', fontsize=16, fontweight='bold')
ax.set_xticks(range(len(accuracy_data)))
ax.set_xticklabels(accuracy_data.index, rotation=45, ha='right')
ax.grid(True, axis='y', alpha=0.3)

# Add value labels on bars
for i, (idx, row) in enumerate(accuracy_data.iterrows()):
    ax.text(i, row['accuracy_all'] + 1, f"{row['accuracy_all']:.1f}%", 
            ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'accuracy_by_type.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'accuracy_by_type.png'}")
plt.close()

# 6. Combined accuracy bar chart (with object_rel_direction_combined)
accuracy_grouped = combined_df.groupby('question_type_grouped').apply(
    lambda x: pd.Series({
        'accuracy': calculate_accuracy(x),
        'count': len(x),
        'no_answer': (x['model_answer'] == 'NO_ANSWER').sum(),
        'answered': (x['model_answer'] != 'NO_ANSWER').sum(),
        'accuracy_answered_only': calculate_accuracy(x[x['model_answer'] != 'NO_ANSWER']) if (x['model_answer'] != 'NO_ANSWER').sum() > 0 else 0.0
    })
).round(2)

fig, ax = plt.subplots(figsize=(12, 6))
accuracy_grouped_sorted = accuracy_grouped.sort_values('accuracy', ascending=False)
bars = ax.bar(range(len(accuracy_grouped_sorted)), accuracy_grouped_sorted['accuracy'], 
              color=[color_map_grouped[qt] for qt in accuracy_grouped_sorted.index], 
              edgecolor='black', linewidth=1.5)

ax.set_xlabel('Question Type', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Accuracy by Question Type (Combined Object Rel Direction)', fontsize=16, fontweight='bold')
ax.set_xticks(range(len(accuracy_grouped_sorted)))
ax.set_xticklabels(accuracy_grouped_sorted.index, rotation=45, ha='right')
ax.grid(True, axis='y', alpha=0.3)

# Add value labels on bars
for i, (idx, row) in enumerate(accuracy_grouped_sorted.iterrows()):
    ax.text(i, row['accuracy'] + 1, f"{row['accuracy']:.1f}%", 
            ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'accuracy_by_type_combined.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'accuracy_by_type_combined.png'}")
plt.close()

# 7. Accuracy bar chart for answered questions only
fig, ax = plt.subplots(figsize=(12, 6))
accuracy_answered_sorted = accuracy_by_type.sort_values('accuracy_answered_only', ascending=False)
bars = ax.bar(range(len(accuracy_answered_sorted)), accuracy_answered_sorted['accuracy_answered_only'], 
              color=[color_map[qt] for qt in accuracy_answered_sorted.index], edgecolor='black', linewidth=1.5)

ax.set_xlabel('Question Type', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy (%) - Answered Only', fontsize=14, fontweight='bold')
ax.set_title('Accuracy by Question Type (Excluding NO_ANSWER)', fontsize=16, fontweight='bold')
ax.set_xticks(range(len(accuracy_answered_sorted)))
ax.set_xticklabels(accuracy_answered_sorted.index, rotation=45, ha='right')
ax.grid(True, axis='y', alpha=0.3)

# Add value labels on bars with answered count
for i, (idx, row) in enumerate(accuracy_answered_sorted.iterrows()):
    ax.text(i, row['accuracy_answered_only'] + 1, 
            f"{row['accuracy_answered_only']:.1f}%\n(n={int(row['answered'])})", 
            ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'accuracy_by_type_answered_only.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'accuracy_by_type_answered_only.png'}")
plt.close()

# 8. Combined accuracy bar chart for answered questions only
fig, ax = plt.subplots(figsize=(12, 6))
accuracy_grouped_answered_sorted = accuracy_grouped.sort_values('accuracy_answered_only', ascending=False)
bars = ax.bar(range(len(accuracy_grouped_answered_sorted)), accuracy_grouped_answered_sorted['accuracy_answered_only'], 
              color=[color_map_grouped[qt] for qt in accuracy_grouped_answered_sorted.index], 
              edgecolor='black', linewidth=1.5)

ax.set_xlabel('Question Type', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy (%) - Answered Only', fontsize=14, fontweight='bold')
ax.set_title('Accuracy by Question Type (Excluding NO_ANSWER, Combined Object Rel Direction)', fontsize=16, fontweight='bold')
ax.set_xticks(range(len(accuracy_grouped_answered_sorted)))
ax.set_xticklabels(accuracy_grouped_answered_sorted.index, rotation=45, ha='right')
ax.grid(True, axis='y', alpha=0.3)

# Add value labels on bars with answered count
for i, (idx, row) in enumerate(accuracy_grouped_answered_sorted.iterrows()):
    ax.text(i, row['accuracy_answered_only'] + 1, 
            f"{row['accuracy_answered_only']:.1f}%\n(n={int(row['answered'])})", 
            ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'accuracy_by_type_answered_only_combined.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'accuracy_by_type_answered_only_combined.png'}")
plt.close()

# Save crosstab by dataset and question type
crosstab_file = output_dir / 'question_types_by_dataset.csv'
crosstab.to_csv(crosstab_file)
print(f"\nSaved crosstab to: {crosstab_file}")

# Save accuracy by dataset and question type
accuracy_dataset_type_file = output_dir / 'accuracy_by_dataset_and_type.csv'
accuracy_by_dataset_type.to_csv(accuracy_dataset_type_file)
print(f"Saved accuracy by dataset and type to: {accuracy_dataset_type_file}")

# Save accuracy by dataset
accuracy_dataset_file = output_dir / 'accuracy_by_dataset.csv'
accuracy_by_dataset.to_csv(accuracy_dataset_file)
print(f"Saved accuracy by dataset to: {accuracy_dataset_file}")

# Save summary CSV
summary_output = output_dir / 'summary_statistics.csv'
accuracy_by_type.to_csv(summary_output)
print(f"\nSaved summary statistics to: {summary_output}")

# Save grouped summary CSV
grouped_summary_output = output_dir / 'summary_statistics_grouped.csv'
accuracy_grouped.to_csv(grouped_summary_output)
print(f"Saved grouped summary statistics to: {grouped_summary_output}")

# Save combined dataset
combined_output = output_dir / 'combined_results.csv'
combined_df.to_csv(combined_output, index=False)
print(f"Saved combined dataset to: {combined_output}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"All outputs saved to: {output_dir}")
print("\nGenerated files:")
print("  - num_steps_histogram.png")
print("  - time_seconds_histogram.png")
print("  - num_steps_histogram_combined.png")
print("  - time_seconds_histogram_combined.png")
print("  - accuracy_by_type.png")
print("  - accuracy_by_type_combined.png")
print("  - accuracy_by_type_answered_only.png")
print("  - accuracy_by_type_answered_only_combined.png")
print("  - summary_statistics.csv")
print("  - summary_statistics_grouped.csv")
print("  - combined_results.csv")
