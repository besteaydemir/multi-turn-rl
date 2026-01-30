# Downloading ScanNet++ for VSI-Bench

This guide shows how to download **only the 50 ScanNet++ scenes** needed for VSI-Bench evaluation (instead of the full 10TB+ dataset).

## Quick Start

```bash
# 1. Generate download configuration
python scripts/download_scannetpp_vsibench.py --generate-config \
  --data-root /dss/mcmlscratch/06/di38riq/scannetpp

# 2. Add your token to scannetpp_vsibench_config.yml
#    Get token from: https://kaldir.vc.in.tum.de/scannetpp/

# 3. Download (requires ~30GB disk space)
python data_download/download_scannetpp.py scannetpp_vsibench_config.yml
```

## What Gets Downloaded

For each of the 50 VSI-Bench scenes:

- **`mesh_aligned_0.05.ply`** (~100MB/scene) - Textured 3D mesh at 5cm resolution
- **`dslr/resized_images/`** - RGB frames from DSLR camera
- **`dslr/resized_depth/`** - Depth maps aligned to RGB
- **`dslr/resized_anon_masks/`** - Privacy masks for faces

**Total size:** ~30GB (50 scenes × ~600MB/scene)

## Scene List

VSI-Bench uses 50 ScanNet++ scenes with 1,458 total questions:

```
09c1414f1b  0d2ee665be  13c3e046d7  1ada7a0617  21d970d8de
25f3b7a318  27dd4da69e  286b55a2bf  31a2c91c43  3864514494
... (45 more)
```

Full list in: `vsi_bench_scannetpp_scenes.txt`

## Usage Examples

### List all scenes
```bash
python scripts/download_scannetpp_vsibench.py --list-scenes
```

### Generate config with custom location
```bash
python scripts/download_scannetpp_vsibench.py --generate-config \
  --data-root /path/to/your/data \
  --token YOUR_TOKEN
```

### Dry run (check what would be downloaded)
```bash
# Edit scannetpp_vsibench_config.yml and set: dry_run: true
python data_download/download_scannetpp.py scannetpp_vsibench_config.yml
```

## VSI-Bench Statistics

- **Total questions:** 1,458
- **Unique scenes:** 50
- **Questions per scene:** 3-77 (avg: 29.2)
- **Question types:** All 10 types (object counting, distance, direction, etc.)

## Comparison

| Dataset | Scenes | Questions | Download Size |
|---------|--------|-----------|---------------|
| **Full ScanNet++** | 230 | N/A | >10TB |
| **VSI-Bench subset** | 50 | 1,458 | ~30GB |

## Directory Structure After Download

```
/dss/mcmlscratch/06/di38riq/scannetpp/
├── data/
│   ├── 09c1414f1b/
│   │   ├── mesh_aligned_0.05.ply
│   │   └── dslr/
│   │       ├── resized_images/
│   │       ├── resized_depth/
│   │       └── resized_anon_masks/
│   ├── 0d2ee665be/
│   │   └── ...
│   └── ... (48 more scenes)
└── splits/
    ├── nvs_sem_train.txt
    └── nvs_sem_val.txt
```

## Troubleshooting

### "Invalid token" error
- Get a new token from: https://kaldir.vc.in.tum.de/scannetpp/
- Make sure there are no spaces in the token

### Download interrupted
- The script will skip already-downloaded files
- Just re-run the same command to resume

### Out of disk space
- Each scene is ~600MB
- You can edit `download_scenes` in the config to download fewer scenes
- Or download in batches (edit config, download, repeat)

## Alternative: Download Specific Scenes Only

If you only want scenes for specific question types:

```python
from datasets import load_dataset

# Get scenes for route planning only
vsi = load_dataset('nyu-visionx/VSI-Bench', split='test')
route = [x for x in vsi if x['dataset'] == 'scannetpp' and x['question_type'] == 'route_planning']
scenes = sorted(set(x['scene_name'] for x in route))

# Edit scannetpp_vsibench_config.yml and update download_scenes list
```

## Related Scripts

- **`create_scannet_scene_list.py`** - Extract ScanNet scene list (similar for ScanNet v2)
- **`check_video_download.py`** - Check ARKitScenes download progress
- **`download_scannetpp_vsibench.py`** - This script (for ScanNet++)

## Get ScanNet++ Token

1. Go to: https://kaldir.vc.in.tum.de/scannetpp/
2. Sign the terms of use
3. Register with your institutional email
4. You'll receive a download token
5. Copy the token to `scannetpp_vsibench_config.yml`
