#!/usr/bin/env python3
"""
Check sequential baseline progress by counting rows in CSV files.
Reports actual completed questions vs expected total (4512).

Structure: Sequential/{4B,8B}/{N}_frames/YYYY-MM-DD/run_folder/results.csv
Also checks Sequential/Sequential (bug from earlier runs)
"""

import os
import pandas as pd
from pathlib import Path

BASE_DIR = Path("/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/Sequential")
BUG_DIR = Path("/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/Sequential/Sequential")
EXPECTED_TOTAL = 4512

def count_csv_rows(csv_path):
    """Count rows in a CSV file (excluding header)."""
    try:
        df = pd.read_csv(csv_path)
        return len(df)
    except Exception as e:
        return 0

def find_results_csvs(folder):
    """Find results.csv files in run folders (not recursively into step folders)."""
    csv_files = []
    if not folder.exists():
        return csv_files
    # Structure: {folder}/{date}/{run_folder}/results.csv
    for date_folder in folder.iterdir():
        if date_folder.is_dir() and date_folder.name.startswith("20"):
            for run_folder in date_folder.iterdir():
                if run_folder.is_dir():
                    results_csv = run_folder / "results.csv"
                    if results_csv.exists():
                        csv_files.append(results_csv)
    return csv_files

def get_unique_completed(csv_files):
    """Get unique (scene_id, question) pairs from CSV files."""
    completed = set()
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                completed.add((row.get('scene_id', ''), row.get('question', '')))
        except:
            pass
    return completed

def main():
    print("=" * 80)
    print("SEQUENTIAL BASELINE - CSV ROW COUNT")
    print("=" * 80)
    print()
    
    results = []
    all_csv_details = {}
    
    for model in ["4B", "8B"]:
        for frames in [4, 8, 16, 32]:
            # Check both normal and bug paths
            folder = BASE_DIR / model / f"{frames}_frames"
            bug_folder = BUG_DIR / model / f"{frames}_frames"
            
            csv_files = find_results_csvs(folder) + find_results_csvs(bug_folder)
            
            if not csv_files:
                results.append({
                    "Model": model,
                    "Frames": frames,
                    "Completed": 0,
                    "Total": EXPECTED_TOTAL,
                    "Remaining": EXPECTED_TOTAL,
                    "Percent": 0.0,
                    "Status": "NOT STARTED"
                })
                continue
            
            # Count unique completed questions (by scene_id, question)
            completed = get_unique_completed(csv_files)
            total_rows = len(completed)
            
            # Also collect CSV details
            csv_details = []
            for csv_file in csv_files:
                rows = count_csv_rows(csv_file)
                csv_details.append((csv_file, rows))
            
            all_csv_details[f"{model}_{frames}f"] = csv_details
            
            remaining = EXPECTED_TOTAL - total_rows
            percent = (total_rows / EXPECTED_TOTAL) * 100
            
            if total_rows >= EXPECTED_TOTAL:
                status = "✅ COMPLETE"
            elif total_rows > 0:
                status = "⏳ RUNNING"
            else:
                status = "❌ EMPTY"
            
            results.append({
                "Model": model,
                "Frames": frames,
                "Completed": total_rows,
                "Total": EXPECTED_TOTAL,
                "Remaining": remaining,
                "Percent": percent,
                "Status": status,
                "CSV_Count": len(csv_files)
            })
    
    # Print table
    print(f"{'Model':<8} {'Frames':<8} {'Completed':<12} {'Remaining':<12} {'Percent':<10} {'Status':<15} {'CSVs':<6}")
    print("-" * 80)
    
    total_completed = 0
    total_remaining = 0
    
    for r in results:
        csv_count = r.get('CSV_Count', 0)
        print(f"{r['Model']:<8} {r['Frames']:<8} {r['Completed']:<12} {r['Remaining']:<12} {r['Percent']:>6.1f}%    {r['Status']:<15} {csv_count:<6}")
        total_completed += r['Completed']
        total_remaining += max(0, r['Remaining'])
    
    print("-" * 80)
    total_expected = EXPECTED_TOTAL * 8  # 8 configs (2 models x 4 frame settings)
    print(f"{'TOTAL':<8} {'':<8} {total_completed:<12} {total_remaining:<12} {(total_completed/total_expected)*100:>6.1f}%")
    print()
    
    # Detailed breakdown by CSV file
    print("=" * 80)
    print("DETAILED CSV FILE BREAKDOWN")
    print("=" * 80)
    
    for key, csv_details in all_csv_details.items():
        print(f"\n{key}:")
        for csv_file, rows in csv_details:
            # Extract run folder name for readability
            run_folder = csv_file.parent.name
            print(f"  {run_folder}: {rows} rows")

if __name__ == "__main__":
    main()
