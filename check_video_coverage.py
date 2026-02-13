#!/usr/bin/env python3
"""Check if all VSI-Bench scene IDs have corresponding video frames."""

from pathlib import Path
from collections import defaultdict
from datasets import load_dataset

# Load VSI-Bench dataset
cache_dir = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/datasets"
dataset = load_dataset("nyu-visionx/vsi-bench", cache_dir=cache_dir)

# Combine all splits
all_questions = []
for split in dataset.keys():
    all_questions.extend(dataset[split])

# Get unique scene IDs per dataset
scenes_by_dataset = defaultdict(set)
for q in all_questions:
    scenes_by_dataset[q['dataset']].add(q['scene_name'])

print('=== Scene counts by dataset ===')
for dataset_name, scenes in scenes_by_dataset.items():
    print(f'{dataset_name}: {len(scenes)} unique scenes')

# Check if frames exist
VIDEO_BASE_DIR = Path('/dss/mcmlscratch/06/di38riq/arkit_vsi/raw')
SCANNET_BASE = Path('/dss/mcmlscratch/06/di38riq/scans/scans')
SCANNETPP_BASE = Path('/dss/mcmlscratch/06/di38riq/data')

missing = []

print('\n=== Checking ARKitScenes ===')
for scene_id in sorted(scenes_by_dataset['arkitscenes']):
    found = False
    for split in ['Training', 'Validation']:
        frames_dir = VIDEO_BASE_DIR / split / scene_id / 'vga_wide'
        if frames_dir.exists():
            num_frames = len(list(frames_dir.glob('*.png')))
            if num_frames > 0:
                found = True
                break
    if not found:
        missing.append(('arkitscenes', scene_id))
        print(f'  ❌ {scene_id}')

print(f'\nARKitScenes: {len(scenes_by_dataset["arkitscenes"]) - len([m for m in missing if m[0]=="arkitscenes"])}/{len(scenes_by_dataset["arkitscenes"])} found')

print('\n=== Checking ScanNet ===')
for scene_id in sorted(scenes_by_dataset['scannet']):
    frames_dir = SCANNET_BASE / scene_id / 'frames' / 'color'
    if not frames_dir.exists() or len(list(frames_dir.glob('*.jpg'))) == 0:
        missing.append(('scannet', scene_id))
        print(f'  ❌ {scene_id}')

print(f'\nScanNet: {len(scenes_by_dataset["scannet"]) - len([m for m in missing if m[0]=="scannet"])}/{len(scenes_by_dataset["scannet"])} found')

print('\n=== Checking ScanNet++ ===')
for scene_id in sorted(scenes_by_dataset['scannetpp']):
    frames_dir = SCANNETPP_BASE / scene_id / 'dslr' / 'resized_undistorted_images'
    if not frames_dir.exists() or (len(list(frames_dir.glob('*.JPG'))) == 0 and len(list(frames_dir.glob('*.jpg'))) == 0):
        missing.append(('scannetpp', scene_id))
        print(f'  ❌ {scene_id}')

print(f'\nScanNet++: {len(scenes_by_dataset["scannetpp"]) - len([m for m in missing if m[0]=="scannetpp"])}/{len(scenes_by_dataset["scannetpp"])} found')

print(f'\n=== SUMMARY ===')
print(f'Total missing: {len(missing)} scenes')
if missing:
    print('\nMissing scenes:')
    for dataset_name, scene_id in missing:
        print(f'  {dataset_name}: {scene_id}')
else:
    print('✅ All scenes have video frames!')
