#!/usr/bin/env python3
"""
Generate a summary table of performance metrics from the aggregated CSVs.
"""

import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("/dss/dsshome1/06/di38riq/rl_multi_turn/analysis")

print("=" * 100)
print("PERFORMANCE SUMMARY: SEQUENTIAL vs VIDEO EVALUATION")
print("=" * 100)

# Load all aggregated CSVs and calculate metrics
results = []

for eval_type in ["Sequential", "Video"]:
    for model in ["4B", "8B"]:
        for frames in [4, 8, 16, 32]:
            csv_file = OUTPUT_DIR / f"{eval_type}_{model}_{frames}f_aggregated.csv"
            
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                
                # Overall metrics
                n_questions = len(df)
                # Accuracy: mra_score = 1.0 means correct
                accuracy = ((df['mra_score'] == 1.0).sum() / len(df)) * 100 if 'mra_score' in df.columns else None
                mra = df['mra_score'].mean() * 100 if 'mra_score' in df.columns else None
                
                results.append({
                    'Eval Type': eval_type,
                    'Model': model,
                    'Frames': frames,
                    'Questions': n_questions,
                    'Accuracy (%)': f"{accuracy:.2f}" if accuracy is not None else "N/A",
                    'MRA (%)': f"{mra:.2f}" if mra is not None else "N/A"
                })

# Create DataFrame
results_df = pd.DataFrame(results)

# Print by evaluation type
for eval_type in ["Sequential", "Video"]:
    print(f"\n{'='*100}")
    print(f"{eval_type.upper()} EVALUATION")
    print(f"{'='*100}")
    
    subset = results_df[results_df['Eval Type'] == eval_type]
    
    # Format for display
    for model in ["4B", "8B"]:
        model_data = subset[subset['Model'] == model]
        print(f"\n{model} Model:")
        print(f"{'Frames':<10} {'Questions':<12} {'Accuracy':<15} {'MRA':<15}")
        print("-" * 60)
        
        for _, row in model_data.iterrows():
            print(f"{row['Frames']:<10} {row['Questions']:<12} {row['Accuracy (%)']:<15} {row['MRA (%)']:<15}")

# Save to CSV
summary_file = OUTPUT_DIR / "performance_summary.csv"
results_df.to_csv(summary_file, index=False)
print(f"\n{'='*100}")
print(f"Summary saved to: {summary_file}")
print(f"{'='*100}")

# Compare Sequential completion status
print(f"\n{'='*100}")
print("SEQUENTIAL EVALUATION COMPLETION STATUS")
print(f"{'='*100}")
print(f"{'Config':<15} {'Questions':<12} {'Total Expected':<15} {'Complete %':<12}")
print("-" * 60)

for model in ["4B", "8B"]:
    for frames in [4, 8, 16, 32]:
        csv_file = OUTPUT_DIR / f"Sequential_{model}_{frames}f_aggregated.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            n_questions = len(df)
            expected = 4512  # VSI-Bench Sequential has 4512 questions
            pct = (n_questions / expected) * 100
            status = "✅" if pct >= 100 else "⚠️"
            print(f"{status} {model} {frames}f     {n_questions:<12} {expected:<15} {pct:>6.1f}%")

print(f"{'='*100}")
