#!/usr/bin/env python3
"""
Analysis script for combining and visualizing results from three experimental splits.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Define file paths
base_path = Path("/dss/dsshome1/06/di38riq/rl_multi_turn/experiment_logs")
files = [
    base_path / "20251225_002135_sequential_split1of3" / "results.csv",
    base_path / "20251225_002137_sequential_split2of3" / "results.csv",
    base_path / "20251225_002139_sequential_split3of3" / "results.csv",
    base_path / "20260101_235631_sequential_split1of4" / "results.csv",
    base_path / "20260101_235632_sequential_split2of4" / "results.csv",
    base_path / "20260101_235633_sequential_split3of4" / "results.csv",
    base_path / "20260101_235633_sequential_split4of4" / "results.csv",
]

# Read and combine all CSV files
dfs = []
for i, file_path in enumerate(files, 1):
    df = pd.read_csv(file_path)
    df['split'] = f'split{i}'
    dfs.append(df)
    print(f"Loaded {file_path.name}: {len(df)} rows")

# Combine all dataframes
combined_df = pd.concat(dfs, ignore_index=True)
print(f"\nCombined dataset: {len(combined_df)} rows")
print(f"Question types: {combined_df['question_type'].unique()}")

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
print("ACCURACY BY QUESTION TYPE")
print("="*80)
accuracy_by_type = combined_df.groupby('question_type').apply(
    lambda x: pd.Series({
        'accuracy': calculate_accuracy(x),
        'count': len(x),
        'correct': (x['gt_answer'] == x['model_answer']).sum(),
        'no_answer': (x['model_answer'] == 'NO_ANSWER').sum(),
        'answered': (x['model_answer'] != 'NO_ANSWER').sum(),
        'accuracy_answered_only': calculate_accuracy(x[x['model_answer'] != 'NO_ANSWER']) if (x['model_answer'] != 'NO_ANSWER').sum() > 0 else 0.0
    })
).round(2)

print(accuracy_by_type.to_string())

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
accuracy_data = accuracy_by_type.sort_values('accuracy', ascending=False)
bars = ax.bar(range(len(accuracy_data)), accuracy_data['accuracy'], 
              color=[color_map[qt] for qt in accuracy_data.index], edgecolor='black', linewidth=1.5)

ax.set_xlabel('Question Type', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Accuracy by Question Type', fontsize=16, fontweight='bold')
ax.set_xticks(range(len(accuracy_data)))
ax.set_xticklabels(accuracy_data.index, rotation=45, ha='right')
ax.grid(True, axis='y', alpha=0.3)

# Add value labels on bars
for i, (idx, row) in enumerate(accuracy_data.iterrows()):
    ax.text(i, row['accuracy'] + 1, f"{row['accuracy']:.1f}%", 
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
