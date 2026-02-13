# Trajectory Visualization Guide

## Overview

This guide documents the procedure for preparing trajectory visualization data from sequential evaluation runs for local analysis on your computer. The system extracts camera trajectories and consolidates mesh files for easy download and 3D visualization.

---

## Preparation Procedure (Server Side)

### 1. Run Sequential Evaluation

First, run sequential evaluation with the desired configuration:

```bash
# Example: 20 random questions with 4B model, 8 frames
sbatch evaluation/test_trajectory_4B_8frames.sh
```

This creates output in `trajectory_test_run/Sequential/{MODEL}/{FRAMES}_frames/` with:
- Individual question folders (q001, q002, ...)
- `trajectory.json` - Camera poses for each question
- `render_XX.png` - Rendered images at each step
- `cam_pose_XX.npy` - Raw camera pose matrices
- `results.csv` - Performance metrics

### 2. Prepare Visualization Data

Run the preparation script to consolidate meshes:

```bash
cd /dss/dsshome1/06/di38riq/rl_multi_turn/trajectory_test_run
python prepare_trajectory_visualization.py
```

**What this script does:**
1. Scans all experiment runs in `Sequential/` folders
2. Extracts unique scene IDs from `results.csv` files
3. Finds corresponding mesh files from dataset-specific locations:
   - ARKitScenes: `/dss/mcmlscratch/06/di38riq/arkit_vsi/raw/{Training,Validation}/<id>/<id>_3dod_mesh.ply`
   - ScanNet: `/dss/mcmlscratch/06/di38riq/scans/<id>/<id>_vh_clean_2.ply`
   - ScanNet++: `/dss/mcmlscratch/06/di38riq/data/<id>/scans/mesh_aligned_0.05.ply`
4. Copies meshes to `meshes_for_visualization/` with naming: `{scene_id}_{dataset}.ply`
5. Creates `mesh_manifest.json` with scene-to-mesh mapping

### 3. Manual Mesh Addition (If Needed)

If some meshes are missing (e.g., ScanNet++ scenes), manually copy them:

```bash
cd /dss/dsshome1/06/di38riq/rl_multi_turn/trajectory_test_run

# Example: Copy missing ScanNet++ scenes
for scene in 25f3b7a318 5eb31827b7 5f99900f09; do
  mesh_path="/dss/mcmlscratch/06/di38riq/data/${scene}/scans/mesh_aligned_0.05.ply"
  if [ -f "$mesh_path" ]; then
    cp "$mesh_path" "meshes_for_visualization/${scene}_scannetpp.ply"
    echo "✅ Copied $scene"
  else
    echo "❌ Not found: $scene"
  fi
done
```

### 4. Verify Completeness

Check that all required scenes have meshes:

```bash
cd /dss/dsshome1/06/di38riq/rl_multi_turn/trajectory_test_run

# List required scenes from results
cat Sequential/4B/8_frames/*/results.csv | cut -d',' -f2,3 | tail -n +2 | sort -u

# List available meshes
ls -1 meshes_for_visualization/ | sed 's/_[^_]*\.ply$//' | sort

# Compare the two lists
```

---

## Download Package Structure

After preparation, download the entire `trajectory_test_run/` folder (~900 MB typical):

```
trajectory_test_run/
├── Sequential/
│   ├── 4B/
│   │   ├── 8_frames/2026-02-04/{timestamp}_sequential_*/
│   │   │   ├── q001/
│   │   │   │   ├── trajectory.json  ← Camera poses
│   │   │   │   ├── render_00.png through render_0N.png
│   │   │   │   ├── cam_pose_00.npy through cam_pose_0N.npy
│   │   │   │   ├── results.json
│   │   │   │   └── initial_view_selection.json
│   │   │   ├── q002/
│   │   │   ├── ...
│   │   │   └── results.csv  ← Summary of all questions
│   │   └── 16_frames/...
│   └── 8B/
│       ├── 8_frames/...
│       └── 16_frames/...
├── meshes_for_visualization/
│   ├── 42446049_arkitscenes.ply
│   ├── scene0193_01_scannet.ply
│   ├── 5eb31827b7_scannetpp.ply
│   └── ... (all mesh files)
├── mesh_manifest.json
├── prepare_trajectory_visualization.py
├── visualize_question.py
├── VISUALIZATION_README.md
└── DOWNLOAD_INSTRUCTIONS.md
```

---

## Local Visualization (Your Computer)

### Prerequisites

Install required packages:
```bash
pip install open3d numpy pandas
```

### Method 1: Quick Visualization Helper

```bash
cd trajectory_test_run/

# Visualize specific questions
python visualize_question.py q001          # 4B model, 8 frames (default)
python visualize_question.py q005 8B 16    # 8B model, 16 frames
```

The helper script automatically:
1. Finds the trajectory.json for the specified question
2. Loads scene_id and dataset from trajectory metadata
3. Looks up the mesh file in `meshes_for_visualization/`
4. Launches the visualization

### Method 2: Direct Visualization

```bash
cd trajectory_test_run/

# Manual path specification
python ../visualize_trajectory.py \
    meshes_for_visualization/47332899_arkitscenes.ply \
    Sequential/4B/8_frames/2026-02-04/*/q001/trajectory.json
```

### Visualization Controls (Open3D)

- **Mouse left drag**: Rotate view
- **Mouse right drag**: Pan
- **Mouse scroll**: Zoom in/out
- **R**: Reset view
- **Q**: Quit

### Visualization Legend

- **Red spheres**: Camera positions (bright = start, dark = end)
- **Green cylinders**: Viewing direction at each step
- **Blue sphere**: Starting position marker
- **Gray mesh**: 3D room reconstruction

---

## Trajectory JSON Format

Each `trajectory.json` contains:

```json
{
  "question_id": "q001",
  "scene_id": "47332899",
  "dataset": "arkitscenes",
  "num_steps": 8,
  "poses": [
    {
      "step": 0,
      "position": [x, y, z],
      "rotation": [
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22]
      ]
    },
    ...
  ],
  "metadata": {
    "question": "What is the size of this room (in square meters)?",
    "answer": "15.8",
    "choices": [...],
    "question_type": "room_size_estimation",
    ...
  }
}
```

**Position**: 3D coordinates in scene space (meters)  
**Rotation**: 3×3 rotation matrix (camera orientation)  
**Metadata**: Original question data from VSI-Bench

---

## Use Cases

### 1. Model Comparison

Compare how 4B vs 8B models explore the same scene:

```bash
python visualize_question.py q001 4B 8   # 4B exploration
python visualize_question.py q001 8B 8   # 8B exploration
```

**Analysis questions:**
- Do they choose similar starting positions?
- Do they explore more/less before answering?
- Are exploration patterns similar?

### 2. Frame Count Impact

Compare 8 frames vs 16 frames for the same model:

```bash
python visualize_question.py q001 4B 8    # 8 frames (7 steps)
python visualize_question.py q001 4B 16   # 16 frames (15 steps)
```

**Analysis questions:**
- Does more frames lead to better coverage?
- Is there redundant exploration with 16 frames?

### 3. Question Type Analysis

Group questions by type and analyze exploration patterns:

```python
import pandas as pd

# Load results
df = pd.read_csv('Sequential/4B/8_frames/*/results.csv')

# Group by question type
for qtype in df['question_type'].unique():
    print(f"\n{qtype}:")
    questions = df[df['question_type'] == qtype]['question_id'].tolist()
    print(f"  Questions: {questions}")
```

Then visualize specific question types to see if exploration differs (e.g., object_counting vs room_size_estimation).

### 4. Error Analysis

Find questions the model got wrong and visualize:

```python
# Find incorrect answers
df = pd.read_csv('Sequential/4B/8_frames/*/results.csv')
df['correct'] = df['model_answer'] == df['gt_answer']
incorrect = df[~df['correct']]['question_id'].tolist()

print(f"Incorrect: {incorrect}")
```

Then visualize to understand if poor exploration caused errors.

---

## Troubleshooting

### "Mesh not found" Error

**Cause**: Scene mesh file missing from `meshes_for_visualization/`

**Solution**:
1. Check `mesh_manifest.json` for which scenes have meshes
2. Manually copy missing mesh from scratch storage:
   ```bash
   # Find mesh on server
   ls /dss/mcmlscratch/06/di38riq/arkit_vsi/raw/*/SCENE_ID/
   ls /dss/mcmlscratch/06/di38riq/scans/SCENE_ID/
   ls /dss/mcmlscratch/06/di38riq/data/SCENE_ID/scans/
   
   # Copy to visualization folder
   cp <mesh_path> trajectory_test_run/meshes_for_visualization/SCENE_ID_DATASET.ply
   ```

### "trajectory.json not found" Error

**Cause**: Incomplete download or wrong question ID

**Solution**:
1. Verify question folder exists: `ls Sequential/4B/8_frames/*/q001/`
2. Check if trajectory.json was created during evaluation
3. Re-download `Sequential/` folder if corrupted

### Old Runs with "unknown" Dataset

**Cause**: Older runs before dataset column was added to results.csv

**Solution**: Use Method 2 (direct visualization) and manually specify mesh path.

---

## Performance Notes

### Download Size Estimates

- **Sequential folders**: ~50-100 MB per run (depends on question count and frame count)
- **Meshes**: 
  - ARKitScenes: 10-50 MB per scene
  - ScanNet: 3-20 MB per scene
  - ScanNet++: 20-100 MB per scene (high detail)
- **Total**: 500-1000 MB for 20 questions

### Visualization Performance

- Open3D renders smoothly on most modern GPUs
- Large meshes (>100 MB) may load slowly
- Consider downsampling very large meshes if needed

---

## Example Workflow

Complete example from evaluation to visualization:

```bash
# ===== SERVER SIDE =====

# 1. Run trajectory test
cd /dss/dsshome1/06/di38riq/rl_multi_turn
sbatch evaluation/test_trajectory_4B_8frames.sh

# 2. Wait for completion
squeue -u $USER

# 3. Prepare visualization data
cd trajectory_test_run
python prepare_trajectory_visualization.py

# 4. Verify all meshes copied
ls -1 meshes_for_visualization/ | wc -l

# 5. Download trajectory_test_run/ folder to local machine

# ===== LOCAL MACHINE =====

# 6. Navigate to downloaded folder
cd ~/Downloads/trajectory_test_run/

# 7. Quick visualization
python visualize_question.py q001

# 8. Compare models
python visualize_question.py q001 4B 8
python visualize_question.py q001 8B 8

# 9. Analyze results
python -c "
import pandas as pd
df = pd.read_csv('Sequential/4B/8_frames/*/results.csv')
print(df[['question_id', 'question_type', 'model_answer', 'gt_answer']].head(10))
"
```

---

## Related Documentation

- **EXPERIMENT_LOG.md**: Overall experiment tracking and issues
- **DATASET_STATUS.md**: Dataset availability and mesh locations
- **DOWNLOAD_INSTRUCTIONS.md**: Quick guide for end users (included in download)
- **VISUALIZATION_README.md**: Technical details on trajectory format (included in download)

---

**Last Updated**: 2026-02-04  
**Maintainer**: Experiment tracking system  
**Related Scripts**:
- `trajectory_test_run/prepare_trajectory_visualization.py`
- `trajectory_test_run/visualize_question.py`
- `visualize_trajectory.py` (main visualization script)
