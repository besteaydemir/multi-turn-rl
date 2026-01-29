#!/usr/bin/env python3
"""Extract ScanNet scene list from VSI-Bench."""

from datasets import load_dataset

print("Loading VSI-Bench...")
vsi = load_dataset('nyu-visionx/VSI-Bench', split='test')

print("Filtering ScanNet scenes...")
scannet = [x for x in vsi if x['dataset'] == 'scannet']
scenes = sorted(set(x['scene_name'] for x in scannet))

print(f'Total unique ScanNet scenes: {len(scenes)}')

output_file = 'vsi_bench_scannet_scenes.txt'
with open(output_file, 'w') as f:
    f.write('\n'.join(scenes))

print(f'Saved to {output_file}')
print(f'\nFirst 10 scenes:')
print('\n'.join(scenes[:10]))
