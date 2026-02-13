#!/usr/bin/env python3
"""
Check progress of Sequential baseline experiments.
Usage: python3 check_sequential_progress.py
"""

import glob
import os
from datetime import datetime

def check_progress():
    base_path = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/Sequential"
    today = datetime.now().strftime("%Y-%m-%d")
    
    results = {}
    
    # Collect results from today's runs
    for model in ["4B", "8B"]:
        for frames in [4, 8, 16, 32]:
            frames_dir = f"{frames}_frames"
            pattern = f"{base_path}/{model}/{frames_dir}/{today}/*/results.csv"
            files = glob.glob(pattern)
            
            key = f"{model} {frames:2d} frames"
            results[key] = {"split1": 0, "split2": 0, "target": 2256, "status": "NOT STARTED"}
            
            for file_path in files:
                if "split1of2" in file_path:
                    split = "split1"
                elif "split2of2" in file_path:
                    split = "split2"
                else:
                    continue
                
                try:
                    # Use pandas to properly count data rows (excluding header)
                    import pandas as pd
                    df = pd.read_csv(file_path)
                    results[key][split] = len(df)
                    results[key]["status"] = "RUNNING"
                except:
                    pass
    
    # Check if configs are complete
    for key in results:
        total = results[key]["split1"] + results[key]["split2"]
        if total >= results[key]["target"] * 2:  # Both splits complete
            results[key]["status"] = "COMPLETE ✅"
        elif total > 0:
            results[key]["status"] = "RUNNING ⏳"
        else:
            results[key]["status"] = "PENDING 🕐"
    
    # Print summary
    print("=" * 85)
    print(f"SEQUENTIAL BASELINE PROGRESS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 85)
    print()
    
    total_completed = 0
    total_target = 0
    configs_complete = 0
    configs_running = 0
    configs_pending = 0
    
    # Sort by model then frames
    sorted_keys = sorted(results.keys(), key=lambda x: (x.split()[0], int(x.split()[1])))
    
    for key in sorted_keys:
        data = results[key]
        completed = data["split1"] + data["split2"]
        target = data["target"] * 2  # Both splits
        pct = (completed / target) * 100 if target > 0 else 0
        total_completed += completed
        total_target += target
        
        status = data["status"]
        if "COMPLETE" in status:
            configs_complete += 1
        elif "RUNNING" in status:
            configs_running += 1
        else:
            configs_pending += 1
        
        print(f"{status:15s} {key:15s} {completed:5d} / {target:4d} ({pct:5.1f}%)")
        if completed > 0 and "COMPLETE" not in status:
            print(f"                 Split 1: {data['split1']:4d}  |  Split 2: {data['split2']:4d}")
    
    print()
    print("-" * 85)
    overall_pct = (total_completed / total_target) * 100 if total_target > 0 else 0
    print(f"OVERALL: {total_completed:5d} / {total_target:5d} ({overall_pct:5.1f}%)")
    print(f"Complete: {configs_complete}/8  |  Running: {configs_running}/8  |  Pending: {configs_pending}/8")
    print("-" * 85)
    print()
    
    # Estimate remaining time based on current rates
    if configs_running > 0:
        print("Estimated completion:")
        for key in sorted_keys:
            data = results[key]
            if data["status"] == "RUNNING ⏳":
                completed = data["split1"] + data["split2"]
                remaining = (data["target"] * 2) - completed
                if completed > 100:  # Need some data for estimate
                    # Rough estimate based on model/frame complexity
                    model = key.split()[0]
                    frames = int(key.split()[1])
                    
                    # Time per question (rough estimates in seconds)
                    if model == "4B":
                        time_per_q = 1.5 + (frames * 0.5)  # ~2-3s for 4f, ~5-7s for 32f
                    else:
                        time_per_q = 2.5 + (frames * 0.7)  # Slower for 8B
                    
                    est_hours = (remaining * time_per_q) / 3600
                    print(f"  {key}: ~{est_hours:.1f} hours")
        print()
    
    print("=" * 85)
    
    return results

if __name__ == "__main__":
    check_progress()
