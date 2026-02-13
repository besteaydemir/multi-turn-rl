# Dataset Status Report

**Generated**: February 4, 2026

## Summary

All three datasets (ARKitScenes, ScanNet, ScanNet++) are available with high coverage. Total 4512 questions across 288 unique scenes.

| Dataset | Scenes | Questions | Meshes | Video Frames |
|---------|--------|-----------|--------|--------------|
| **ARKitScenes** | 150 | 1601 | ✅ 150/150 (100%) | ✅ 150/150 (100%) |
| **ScanNet** | 88 | 1681 | ✅ 88/88 (100%) | ⚠️ 86/88 (97.7%) |
| **ScanNet++** | 50 | 1230 | ⚠️ 48/50 (96%) | ⚠️ 48/50 (96%) |
| **Total** | 288 | 4512 | 286/288 (99.3%) | 284/288 (98.6%) |

---

## Dataset Paths & Naming Conventions

### ARKitScenes (150 scenes, 1601 questions)

**Scene ID Format**: Pure numeric (e.g., `41069025`, `47333462`)

**Paths**:
```
Meshes:    /dss/mcmlscratch/06/di38riq/arkit_vsi/raw/{Training,Validation}/<scene_id>/<scene_id>_3dod_mesh.ply
Video:     /dss/mcmlscratch/06/di38riq/arkit_vsi/raw/{Training,Validation}/<scene_id>/vga_wide/
Frames:    <scene_id>_<timestamp>.png (e.g., 41069025_589.277.png)
```

**Status**: ✅ Complete (all 150 scenes have both meshes and video frames)

---

### ScanNet (88 scenes, 1681 questions)

**Scene ID Format**: `scene<XXXX>_<XX>` (e.g., `scene0025_01`)

**Paths**:
```
Meshes:    /dss/mcmlscratch/06/di38riq/scans/<scene_id>/<scene_id>_vh_clean_2.ply
Video:     /dss/mcmlscratch/06/di38riq/scans/scans/<scene_id>/frames/color/
Frames:    <number>.jpg (e.g., 0.jpg, 1000.jpg, 1001.jpg)
```

**Status**: ⚠️ Mostly complete with 2 missing video frame directories
- Meshes: ✅ 88/88 (100%)
- Video frames: ⚠️ 86/88 (97.7%)

**Missing Video Frames** (questions affected):
| Scene | Questions | Issue |
|-------|-----------|-------|
| `scene0645_00` | 40 | Frames directory is empty (extraction incomplete) |
| `scene0704_01` | 7 | No frames directory (not extracted yet) |

---

### ScanNet++ (50 scenes, 1230 questions)

**Scene ID Format**: 10-character alphanumeric hash (e.g., `09c1414f1b`) 
**Exception**: 2 scenes have numeric IDs (see below)

**Paths**:
```
Meshes:    /dss/mcmlscratch/06/di38riq/data/<scene_id>/scans/mesh_aligned_0.05.ply
Video:     /dss/mcmlscratch/06/di38riq/data/<scene_id>/dslr/resized_undistorted_images/
Frames:    DSC<XXXXX>.JPG (e.g., DSC04432.JPG, DSC04433.JPG)
```

**Status**: ⚠️ Mostly complete with 2 mislabeled scenes missing entirely
- Meshes: ⚠️ 48/50 (96%)
- Video frames: ⚠️ 48/50 (96%)

**Missing Scenes** (root cause: data labeling errors):
| Scene | Questions | Dataset Field | Issue |
|-------|-----------|---|-------|
| `3864514494` | 18 | `scannetpp` | Numeric ID doesn't match ScanNet++ naming. Not found in any dataset. |
| `5942004064` | 37 | `scannetpp` | Numeric ID doesn't match ScanNet++ naming. Not found in any dataset. |

**Analysis**: These 2 scenes appear to be mislabeled in VSI-Bench. They have numeric IDs like ARKitScenes but are tagged as ScanNet++. The IDs don't exist in either dataset, indicating a data labeling error in VSI-Bench.

---

## Impact on Evaluation

### Sequential Pipeline (Mesh-based Rendering)
**Questions Skipped**: 55/4512 (1.2%)
```
3864514494 (scannetpp):  18 questions
5942004064 (scannetpp):  37 questions
```

### Video Baseline Pipeline (Real Video Frames)
**Questions Skipped**: 102/4512 (2.3%)
```
3864514494 (scannetpp):  18 questions
5942004064 (scannetpp):  37 questions
scene0645_00 (scannet):  40 questions
scene0704_01 (scannet):   7 questions
```

**Remaining Evaluable Questions**:
- Sequential: 4457/4512 (98.8%)
- Video: 4410/4512 (97.7%)

---

## Code Implementation Status

### Mesh Loading (`utils/mesh.py`) ✅ CORRECT
Function uses dataset-specific base directories:
- `find_mesh_file(scene_id, dataset="scannetpp")` → uses `/dss/mcmlscratch/06/di38riq/data`
- `find_mesh_file(scene_id, dataset="scannet")` → uses `/dss/mcmlscratch/06/di38riq/scans`
- `find_mesh_file(scene_id, dataset="arkitscenes")` → uses `/dss/mcmlscratch/06/di38riq/arkit_vsi/raw`

**Fix Applied**: Changed function to use dataset-specific defaults when `mesh_base_dir=None`

### Sequential Pipeline (`evaluation/sequential.py`) ✅ CORRECT
Correctly passes dataset-specific base directories:
```python
if q_dataset == "scannetpp":
    q_mesh_base_dir = SCANNETPP_MESH_BASE_DIR  # /dss/mcmlscratch/06/di38riq/data
elif q_dataset == "scannet":
    q_mesh_base_dir = SCANNET_MESH_BASE_DIR    # /dss/mcmlscratch/06/di38riq/scans
else:
    q_mesh_base_dir = MESH_BASE_DIR            # /dss/mcmlscratch/06/di38riq/arkit_vsi/raw

mesh_file = find_mesh_file(scene_id, q_mesh_base_dir, dataset=q_dataset)
```

**Status**: ✅ Paths are correct and properly implemented

### Video Pipeline (`evaluation/video_baseline.py`) ✅ CORRECT
Uses dataset-specific frame directories:
- `get_video_frames_dir()` for ARKitScenes → `/dss/mcmlscratch/06/di38riq/arkit_vsi/raw/{Training,Validation}/<id>/vga_wide/`
- `get_scannet_frames_dir()` for ScanNet → `/dss/mcmlscratch/06/di38riq/scans/scans/<id>/frames/color/`
- `get_scannetpp_frames_dir()` for ScanNet++ → `/dss/mcmlscratch/06/di38riq/data/<id>/dslr/resized_undistorted_images/`

**Status**: ✅ Paths are correct and properly implemented

---

## Recommendations

1. **For Sequential Evaluation**: Proceed with 4457 evaluable questions (55 skipped due to missing ScanNet++ meshes)

2. **For Video Evaluation**: Proceed with 4410 evaluable questions (102 skipped due to missing meshes/frames)

3. **For Missing ScanNet Frames**: Consider extracting frames from `.sens` files for `scene0645_00` and `scene0704_01` if needed:
   - Both have `.sens` raw sensor data files available: `/dss/mcmlscratch/06/di38riq/scans/scans/<scene_id>/<scene_id>.sens`
   - Only need to run frame extraction on these 2 scenes (47 questions total)

4. **For Missing ScanNet++ Scenes**: Cannot be recovered
   - These scenes don't exist in any dataset
   - VSI-Bench has mislabeled/corrupted entries for scenes `3864514494` and `5942004064`
   - Flag as invalid in VSI-Bench metadata

---

## Verification Commands

Check all mesh availability:
```bash
cd /dss/dsshome1/06/di38riq/rl_multi_turn
python3 << 'EOF'
from utils.data import load_vsi_bench_questions, ALL_SEQUENTIAL_QUESTION_TYPES
from utils import find_mesh_file
from collections import defaultdict

questions = load_vsi_bench_questions(question_types=ALL_SEQUENTIAL_QUESTION_TYPES, dataset='combined')
scenes_by_dataset = defaultdict(set)
for q in questions:
    scenes_by_dataset[q.get('dataset', 'unknown')].add(q['scene_name'])

for dataset in ['arkitscenes', 'scannet', 'scannetpp']:
    found = sum(1 for scene in scenes_by_dataset[dataset] if find_mesh_file(scene, dataset=dataset) is not None)
    total = len(scenes_by_dataset[dataset])
    print(f'{dataset}: {found}/{total}')
