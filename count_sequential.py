#!/usr/bin/env python3
"""Count completed questions across all Sequential configs accurately."""
import csv
import os

OLD = '/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs'
SCRATCH = '/dss/mcmlscratch/06/di38riq/experiment_logs'

configs = [
    ('4B', '4_frames'),  ('4B', '8_frames'),  ('4B', '16_frames'),  ('4B', '32_frames'),
    ('8B', '4_frames'),  ('8B', '8_frames'),  ('8B', '16_frames'),  ('8B', '32_frames'),
]

TOTAL = 4512

def find_csv_files(base_dir):
    """Walk directory tree finding results.csv, skip step_ dirs."""
    found = []
    if not os.path.isdir(base_dir):
        return found
    for dirpath, dirnames, filenames in os.walk(base_dir, followlinks=False):
        # Prune step_ directories to avoid broken symlinks and speed up
        dirnames[:] = [d for d in dirnames if not d.startswith('step_')]
        if 'results.csv' in filenames:
            found.append(os.path.join(dirpath, 'results.csv'))
    return found

print("{:<18} {:>10} {:>10} {:>8}".format("Config", "Completed", "Remaining", "CSVs"))
print("-" * 50)

for model, frames in configs:
    completed = set()
    csv_count = 0
    search_dirs = [
        os.path.join(OLD, 'Sequential', model, frames),
        os.path.join(OLD, 'Sequential', 'Sequential', model, frames),
        os.path.join(SCRATCH, 'Sequential', model, frames),
    ]
    for base in search_dirs:
        csv_files = find_csv_files(base)
        csv_count += len(csv_files)
        for csv_path in csv_files:
            try:
                with open(csv_path) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        sid = row.get('scene_id', '')
                        q = row.get('question', '')
                        if sid and q:
                            completed.add((sid, q))
            except Exception as e:
                print(f"  [WARN] Failed to read {csv_path}: {e}")
    remaining = TOTAL - len(completed)
    print("{:<18} {:>10} {:>10} {:>8}".format(
        model + " " + frames, len(completed), remaining, csv_count))

print()
print("Note: 55 remaining = 2 ScanNet++ scenes without meshes (permanent)")
