#!/usr/bin/env python3
"""
Check if duplicate questions (same scene_id, question) have the same outputs.
This validates whether re-runs are deterministic or if there are variations.
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict

# Check Sequential 4B 16f which had 286 duplicates
config_path = Path("/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/Sequential/4B/16_frames")

print("=" * 80)
print("CHECKING DUPLICATE QUESTION OUTPUTS")
print("=" * 80)
print(f"Config: Sequential/4B/16f (had 286 duplicates)")
print("=" * 80)

# Track all instances of each question
question_instances = defaultdict(list)

for date_dir in sorted(config_path.iterdir()):
    if not date_dir.is_dir() or not date_dir.name.startswith("20"):
        continue
    
    for run_dir in sorted(date_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        
        results_file = run_dir / "results.csv"
        if not results_file.exists():
            continue
        
        try:
            df = pd.read_csv(results_file)
            
            for _, row in df.iterrows():
                scene_id = str(row.get('scene_id', ''))
                question = str(row.get('question', ''))
                response = str(row.get('response', ''))
                
                question_instances[(scene_id, question)].append({
                    'run': run_dir.name,
                    'response': response,
                    'is_correct': row.get('is_correct'),
                    'relative_accuracy': row.get('relative_accuracy')
                })
        except Exception as e:
            print(f"Error reading {results_file}: {e}")

# Find questions with multiple instances
duplicates = {k: v for k, v in question_instances.items() if len(v) > 1}

print(f"\nTotal unique questions: {len(question_instances)}")
print(f"Questions with multiple runs: {len(duplicates)}")

if duplicates:
    print("\n" + "=" * 80)
    print("ANALYZING DUPLICATE INSTANCES")
    print("=" * 80)
    
    same_response = 0
    diff_response = 0
    
    for (scene_id, question), instances in list(duplicates.items())[:10]:  # Check first 10
        responses = [inst['response'] for inst in instances]
        unique_responses = set(responses)
        
        if len(unique_responses) == 1:
            same_response += 1
            status = "✓ SAME"
        else:
            diff_response += 1
            status = "✗ DIFFERENT"
        
        print(f"\n{status}")
        print(f"Scene: {scene_id[:20]}...")
        print(f"Question: {question[:60]}...")
        print(f"Instances: {len(instances)}")
        
        for i, inst in enumerate(instances, 1):
            print(f"  Run {i} ({inst['run'][:30]}...): {inst['response'][:50]}...")
    
    print("\n" + "=" * 80)
    print(f"Summary of {len(duplicates)} duplicate questions:")
    print(f"  Same response across runs: {same_response}/{min(10, len(duplicates))} sampled")
    print(f"  Different responses: {diff_response}/{min(10, len(duplicates))} sampled")
    print("=" * 80)
else:
    print("\nNo duplicates found in this configuration.")
