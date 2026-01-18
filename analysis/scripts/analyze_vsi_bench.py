#!/usr/bin/env python3
"""
Analysis script for VSI-Bench dataset from HuggingFace.
Creates a table showing question types by source dataset.
"""

import pandas as pd
from datasets import load_dataset
from pathlib import Path

print("[INFO] 📥 Loading VSI-Bench dataset from HuggingFace...")
vsi = load_dataset("nyu-visionx/VSI-Bench", split="test")
print(f"[INFO] Total VSI-Bench rows: {len(vsi)}")

# Convert to pandas DataFrame for easier analysis
df = pd.DataFrame(vsi)

# Group object_rel_direction types together
df['question_type_grouped'] = df['question_type'].apply(
    lambda x: 'object_rel_direction' if x in ['object_rel_direction_easy', 'object_rel_direction_hard', 'object_rel_direction_medium'] else x
)

print("\n" + "="*80)
print("DATASET OVERVIEW")
print("="*80)
print(f"Total questions: {len(df)}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nUnique datasets: {sorted(df['dataset'].unique())}")
print(f"Unique question types: {sorted(df['question_type'].unique())}")
print(f"Unique question types (grouped): {sorted(df['question_type_grouped'].unique())}")

# Create a cross-tabulation of dataset vs question_type (grouped)
print("\n" + "="*80)
print("QUESTION TYPES BY SOURCE DATASET (GROUPED)")
print("="*80)
crosstab = pd.crosstab(df['dataset'], df['question_type_grouped'], margins=True, margins_name='Total')
print(crosstab.to_string())

# Calculate percentages
print("\n" + "="*80)
print("QUESTION TYPES BY SOURCE DATASET (PERCENTAGES, GROUPED)")
print("="*80)
crosstab_pct = pd.crosstab(df['dataset'], df['question_type_grouped'], normalize='index') * 100
crosstab_pct = crosstab_pct.round(2)
print(crosstab_pct.to_string())

# Summary by dataset
print("\n" + "="*80)
print("SUMMARY BY DATASET")
print("="*80)
dataset_summary = df.groupby('dataset').agg({
    'question_type': 'count',
}).rename(columns={'question_type': 'total_questions'})
dataset_summary = dataset_summary.sort_values('total_questions', ascending=False)
print(dataset_summary.to_string())

# Summary by question type (original)
print("\n" + "="*80)
print("SUMMARY BY QUESTION TYPE (ORIGINAL)")
print("="*80)
qtype_summary_orig = df.groupby('question_type').agg({
    'dataset': 'count',
}).rename(columns={'dataset': 'total_questions'})
qtype_summary_orig = qtype_summary_orig.sort_values('total_questions', ascending=False)
print(qtype_summary_orig.to_string())

# Summary by question type (grouped)
print("\n" + "="*80)
print("SUMMARY BY QUESTION TYPE (GROUPED)")
print("="*80)
qtype_summary = df.groupby('question_type_grouped').agg({
    'dataset': 'count',
}).rename(columns={'dataset': 'total_questions'})
qtype_summary = qtype_summary.sort_values('total_questions', ascending=False)
print(qtype_summary.to_string())

# Check which question types are multiple choice (have options)
print("\n" + "="*80)
print("MULTIPLE CHOICE ANALYSIS (based on 'options' field)")
print("="*80)
df['has_options'] = df['options'].notna()
mc_analysis = df.groupby('question_type').agg({
    'has_options': ['sum', 'count', 'mean']
})
mc_analysis.columns = ['with_options', 'total', 'percentage']
mc_analysis['percentage'] = (mc_analysis['percentage'] * 100).round(2)
mc_analysis['question_format'] = mc_analysis['percentage'].apply(
    lambda x: 'Multiple Choice' if x > 50 else ('Numerical/Free-form' if x < 50 else 'Mixed')
)
mc_analysis = mc_analysis.sort_values('percentage', ascending=False)
print(mc_analysis.to_string())

print("\n" + "="*80)
print("QUESTION TYPES BY FORMAT")
print("="*80)
format_summary = mc_analysis.groupby('question_format').agg({
    'total': 'sum'
})
print(format_summary.to_string())

# Dataset breakdown for each question type (grouped)
print("\n" + "="*80)
print("DATASET BREAKDOWN FOR EACH QUESTION TYPE (GROUPED)")
print("="*80)
for qtype in sorted(df['question_type_grouped'].unique()):
    qtype_df = df[df['question_type_grouped'] == qtype]
    dataset_counts = qtype_df['dataset'].value_counts().sort_values(ascending=False)
    print(f"\n{qtype}:")
    for dataset, count in dataset_counts.items():
        pct = (count / len(qtype_df)) * 100
        print(f"  {dataset}: {count} ({pct:.1f}%)")

# Save results
output_dir = Path("/dss/dsshome1/06/di38riq/rl_multi_turn/analysis/vsi_bench_analysis")
output_dir.mkdir(exist_ok=True, parents=True)

# Save crosstab to CSV
crosstab_file = output_dir / "question_types_by_dataset.csv"
crosstab.to_csv(crosstab_file)
print(f"\n[INFO] 💾 Saved crosstab to: {crosstab_file}")

# Save percentages to CSV
crosstab_pct_file = output_dir / "question_types_by_dataset_percentages.csv"
crosstab_pct.to_csv(crosstab_pct_file)
print(f"[INFO] 💾 Saved percentages to: {crosstab_pct_file}")

# Save multiple choice analysis
mc_analysis_file = output_dir / "multiple_choice_analysis.csv"
mc_analysis.to_csv(mc_analysis_file)
print(f"[INFO] 💾 Saved multiple choice analysis to: {mc_analysis_file}")

# Save full dataset summary
summary_file = output_dir / "vsi_bench_summary.txt"
with open(summary_file, 'w') as f:
    f.write("VSI-BENCH DATASET ANALYSIS\n")
    f.write("="*80 + "\n\n")
    f.write(f"Total questions: {len(df)}\n")
    f.write(f"Unique datasets: {len(df['dataset'].unique())}\n")
    f.write(f"Unique question types: {len(df['question_type'].unique())}\n\n")
    f.write("QUESTION TYPES BY SOURCE DATASET\n")
    f.write("="*80 + "\n")
    f.write(crosstab.to_string())
    f.write("\n\n")
    f.write("QUESTION TYPES BY SOURCE DATASET (Percentages)\n")
    f.write("="*80 + "\n")
    f.write(crosstab_pct.to_string())
    f.write("\n\n")
    f.write("MULTIPLE CHOICE ANALYSIS\n")
    f.write("="*80 + "\n")
    f.write(mc_analysis.to_string())

print(f"[INFO] 💾 Saved summary to: {summary_file}")

# Print sample questions for each question type of interest
print("\n" + "="*80)
print("SAMPLE QUESTIONS FOR MULTIPLE CHOICE TYPES")
print("="*80)
mc_types = ['route_planning', 'object_rel_distance', 'object_rel_direction_easy', 
            'object_rel_direction_hard', 'object_rel_direction_medium']
for qtype in mc_types:
    qtype_df = df[df['question_type'] == qtype]
    if len(qtype_df) > 0:
        print(f"\n{'='*60}")
        print(f"QUESTION TYPE: {qtype}")
        print(f"{'='*60}")
        # Show 2 sample questions
        for i, (_, row) in enumerate(qtype_df.head(2).iterrows()):
            print(f"\n--- Sample {i+1} ---")
            print(f"Question: {row['question']}")
            print(f"Options: {row['options']}")
            print(f"Ground Truth: {row['ground_truth']}")
            print(f"Dataset: {row['dataset']}")
            print(f"Scene: {row['scene_name']}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"All outputs saved to: {output_dir}")
