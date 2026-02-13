# Habitat-Sim Environment — Reference Guide

## Table of Contents

0. [Runtime Setup for `habitat_source` (MCML DGX)](#0-runtime-setup-for-habitat_source-mcml-dgx)
1. [EGL Error Fix (DGX / Multi-GPU Nodes)](#1-egl-error-fix-dgx--multi-gpu-nodes)
2. [Action Mapping: Continuous → Discrete Habitat Actions](#2-action-mapping-continuous--discrete-habitat-actions)
3. [Trajectory Output Format](#3-trajectory-output-format)
4. [Available Scenes](#4-available-scenes)
5. [Quick Start](#5-quick-start)

---

## 0. Runtime Setup for `habitat_source` (MCML DGX)

**Every time you open a new srun session**, run these commands:

```bash
# 1. Get GPU node
srun --partition=mcml-dgx-a100-40x8 --qos=mcml --gres=gpu:1 --mem=32G --time=01:10:00 --pty bash

# 2. Activate the environment
conda activate habitat_source

# 3. CRITICAL: Set GL library path (conda's libglvnd before system's)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 4. EGL fix for multi-GPU nodes
export __EGL_VENDOR_LIBRARY_FILENAMES=/dss/dsshome1/06/di38riq/habitat-sim/10_nvidia.json

# 5. Force headless mode
unset DISPLAY
```

### One-liner version

```bash
conda activate habitat_source && \
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH && \
export __EGL_VENDOR_LIBRARY_FILENAMES=/dss/dsshome1/06/di38riq/habitat-sim/10_nvidia.json && \
unset DISPLAY
```

### Why the GL fix is needed

The `habitat_source` env was built from source with Python 3.12. Its habitat-sim
binary links to the **system's** GLVND (`/usr/lib/x86_64-linux-gnu/libEGL.so.1`),
which is ABI-incompatible with the NVIDIA driver on DGX nodes.

By prepending `$CONDA_PREFIX/lib` to `LD_LIBRARY_PATH`, the dynamic loader finds
conda-forge's `libglvnd` first, which works correctly with the NVIDIA driver.

Without this fix, you get:
```
GL::Context: cannot retrieve OpenGL version: GL::Renderer::Error::InvalidValue
```

### Verify it works

```bash
cd ~/rl_multi_turn
python -c "from vagen_vsi_rl.env.habitat_env import HabitatEnv; print('OK')"
```

### Environment contents (for reference)

- Python 3.12.12
- habitat-sim 0.3.3 (source build)
- vLLM 0.15.1 (supports Qwen3-VL!)
- torch 2.9.1+cu128
- transformers 4.57.6
- qwen-vl-utils 0.0.14

---

## 1. EGL Error Fix (DGX / Multi-GPU Nodes)

### The Error

When running on DGX or other multi-GPU headless nodes, habitat-sim fails with:

```
Platform::WindowlessEglApplication::tryCreateContext():
  unable to find CUDA device 0 among 18 EGL devices
WindowlessContext: Unable to create windowless context
```

### Root Cause

On DGX nodes with many GPUs (e.g. 8× A100), the system exposes multiple EGL
device files.  By default, the EGL loader picks up **all** vendor ICDs
(including non-NVIDIA ones like `libEGL_mesa.so`), and habitat-sim can't
match CUDA device 0 to the correct EGL device among 18 candidates.

### The Fix

Force EGL to use **only** the NVIDIA ICD by setting the environment variable
`__EGL_VENDOR_LIBRARY_FILENAMES` to point at a JSON file that references
`libEGL_nvidia.so.0`.

**The JSON file** already exists at:
```
/dss/dsshome1/06/di38riq/habitat-sim/10_nvidia.json
```

Contents:
```json
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
```

#### Option A — Set the env var before running Python

```bash
export __EGL_VENDOR_LIBRARY_FILENAMES=/dss/dsshome1/06/di38riq/habitat-sim/10_nvidia.json
unset DISPLAY
python my_script.py
```

#### Option B — `HabitatEnvConfig.egl_nvidia_json` (automatic)

`HabitatEnv._make_sim()` **auto-detects** this fix.  It checks:

1. Whether `__EGL_VENDOR_LIBRARY_FILENAMES` is already set → skips if so.
2. Whether `cfg.egl_nvidia_json` was explicitly set → uses that path.
3. Auto-detects `<workspace>/habitat-sim/10_nvidia.json` relative to the
   `habitat_env.py` source file.

So in most cases you don't need to do anything — just make sure
`10_nvidia.json` is present at the path above.  If you move it, pass the
new path via:

```python
cfg = HabitatEnvConfig(
    egl_nvidia_json="/path/to/10_nvidia.json",
)
```

#### Option C — SLURM job script

```bash
#!/bin/bash
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
export __EGL_VENDOR_LIBRARY_FILENAMES=/dss/dsshome1/06/di38riq/habitat-sim/10_nvidia.json
unset DISPLAY
conda activate habitat
python train.py
```

### Also important

- Always `unset DISPLAY` (or `os.environ.pop("DISPLAY", None)` in Python).
  If `$DISPLAY` is set, habitat-sim tries GLX instead of EGL and fails
  on headless nodes.
- `HabitatEnv._make_sim()` already does `os.environ.pop("DISPLAY", None)`
  automatically.

---

## 2. Action Mapping: Continuous → Discrete Habitat Actions

### Our RL Policy Actions (continuous, same as VSIEnv)

The VLM policy outputs a JSON dict per step:

```json
{
    "rotation_angle_degrees": 45,
    "forward_meters": 0.5,
    "left_meters": 0.3,
    "z_delta_meters": 0,
    "answer": null,
    "done": false
}
```

| Field | Type | Meaning |
|-------|------|---------|
| `rotation_angle_degrees` | float | Yaw rotation. Positive = turn left, negative = turn right |
| `forward_meters` | float | Distance to walk forward (≥ 0) |
| `left_meters` | float | Lateral strafe. Positive = move left, negative = move right |
| `z_delta_meters` | float | Vertical movement (ignored by habitat — navmesh is 2D) |
| `answer` | str \| null | Final answer (e.g. `"C"` or `"3.5"`) |
| `done` | bool | Whether the agent is submitting its answer |

### Habitat-Sim's Discrete Actions

Habitat-sim only supports **three** discrete movement primitives:

| Action | Effect |
|--------|--------|
| `"move_forward"` | Move 0.25 m in the facing direction |
| `"turn_left"` | Rotate 10° counter-clockwise |
| `"turn_right"` | Rotate 10° clockwise |

These amounts are configurable via `HabitatEnvConfig`:
- `move_forward_amount` (default: 0.25 m)
- `turn_amount` (default: 10°)

### How the Mapping Works

`HabitatEnv._execute_movement()` converts one continuous action into a
**sequence** of discrete habitat actions, executed in order:

```
Step 1 — ROTATION
  rotation_angle_degrees = 45
  → 45 / 10 = 4.5 → round to 5
  → ["turn_left", "turn_left", "turn_left", "turn_left", "turn_left"]

Step 2 — LATERAL MOVEMENT (strafe)
  left_meters = 0.3
  → rotate 90° left: ["turn_left"] (one 10° turn only — special-cased
    as a single 90° rotate-move-unrotate macro)
  → Actually: turn_left → N × move_forward → turn_right
  → 0.3 / 0.25 = 1.2 → round to 1
  → ["turn_left", "move_forward", "turn_right"]

Step 3 — FORWARD MOVEMENT
  forward_meters = 0.5
  → 0.5 / 0.25 = 2
  → ["move_forward", "move_forward"]
```

**Full discrete sequence for one policy step:**
```
turn_left × 5, turn_left, move_forward, turn_right, move_forward × 2
```
(10 discrete habitat actions for one policy action)

#### Lateral Movement Detail

Since habitat-sim has no `"strafe_left"` action, lateral movement is
implemented as a **rotate-move-unrotate** macro:

```
left_meters > 0 (move left):
    turn_left → move_forward × N → turn_right   (undo the rotation)

left_meters < 0 (move right):
    turn_right → move_forward × N → turn_left   (undo the rotation)
```

Note: the lateral turn is a single `turn_left`/`turn_right` (10°), not a
full 90°.  This is a simplification — the agent strafes at a slight angle
rather than perfectly sideways.  For small lateral movements this is adequate.

#### z_delta (vertical movement)

Habitat-sim's navmesh is 2D — there is no "fly up" action.  The
`z_delta_meters` field is **silently ignored**.  If the scene has stairs or
ramps, the navmesh handles elevation changes automatically during
`move_forward`.

#### Collisions

Habitat-sim's navmesh prevents the agent from walking through walls.  If a
`move_forward` would collide, the agent simply stays in place (the action
is a no-op).  No error is raised.

---

## 3. Trajectory Output Format

Each episode saves to `{output_dir}/q{NNN}/`:

```
q000/
├── render_00.png         ← initial observation (at reset)
├── render_01.png         ← after step 1
├── render_02.png         ← after step 2
└── trajectory.json       ← full episode metadata
```

### Example `trajectory.json`

```json
{
  "question_id": 0,
  "scene_id": "skokloster-castle",
  "dataset": "habitat",
  "question": "What color is the ceiling?",
  "choices": ["Red", "Blue", "White", "Brown"],
  "question_type": "color",
  "is_numerical": false,
  "final_answer": "C",
  "ground_truth": "C",
  "is_correct": true,
  "reward": 1.0,
  "elapsed_time": 3.995,
  "num_steps": 2,
  "num_images": 3,
  "image_paths": [
    ".../q000/render_00.png",
    ".../q000/render_01.png",
    ".../q000/render_02.png"
  ],
  "movement_history": [
    {
      "rotation": 0,
      "forward": 0.5,
      "left": 0,
      "z_delta": 0,
      "position": "X=-3.38m, Y=0.04m, Z=6.97m"
    },
    {
      "rotation": 45,
      "forward": 0.25,
      "left": 0,
      "z_delta": 0,
      "position": "X=-3.63m, Y=0.04m, Z=6.96m"
    }
  ],
  "env_type": "habitat"
}
```

The format is identical to VSIEnv's `trajectory.json`, with the addition of
`"env_type": "habitat"` (vs `"vsi"` for the Open3D backend).

### Saved Test Trajectories

The verified GPU test output is saved at:
```
vagen_vsi_rl/test_outputs/habitat_test/q000/
```

---

## 4. Available Scenes

### Two Separate Scene Sets

There are **two independent** scene inventories. They have **zero overlap**:

#### A) VSI-Bench Scenes (`.ply` meshes — for VSIEnv / Open3D)

The VSI-Bench `test_pruned` split has **2,768 questions** across **282 scenes**
from three source datasets.  These are `.ply` meshes rendered by Open3D in
`VSIEnv`.  They are **not** `.glb` and are **not** used by `HabitatEnv`.

| Source Dataset | Scenes | Questions | Mesh Path Pattern |
|---------------|--------|-----------|-------------------|
| **ARKitScenes** | 146 | 926 | `/dss/mcmlscratch/06/di38riq/arkit_vsi/raw/{Training,Validation}/{id}/{id}_3dod_mesh.ply` |
| **ScanNet** | 86 | 1,136 | `/dss/mcmlscratch/06/di38riq/scans/{scene_id}/{scene_id}_vh_clean_2.ply` |
| **ScanNet++** | 50 | 706 | `/dss/mcmlscratch/06/di38riq/data/{hash}/scans/mesh_aligned_0.05.ply` |

These paths are defined in `utils/mesh.py::find_mesh_file()` and used by
`evaluation/sequential.py`.

#### B) HM3D Scenes (`.glb` — for HabitatEnv)

**900 HM3D (Habitat-Matterport 3D) scenes** are available as `.glb` files:

| Split | Count | Path | Layout |
|-------|-------|------|--------|
| **Train** | 800 | `/dss/mcmlscratch/06/di38riq/habitat/` | `NNNNN-hashID/hashID.glb` |
| **Val** | 100 | `/dss/mcmlscratch/06/di38riq/habitat_val/` | `NNNNN-hashID/hashID.glb` |

Directory structure example:
```
/dss/mcmlscratch/06/di38riq/habitat/
├── 00000-kfPV7w3FaU5/
│   └── kfPV7w3FaU5.glb          (27 MB)
├── 00001-UVdNNRcVyV1/
│   └── UVdNNRcVyV1.glb
├── ...
└── 00799-.../

/dss/mcmlscratch/06/di38riq/habitat_val/
├── 00800-TEEsavR23oF/
│   └── TEEsavR23oF.glb
├── ...
└── 00899-.../
```

You can reference scenes by either the **hash** (`kfPV7w3FaU5`) or the
**full directory name** (`00000-kfPV7w3FaU5`).

#### C) Demo Scenes (3 small `.glb` — for testing)

| Scene | Path |
|-------|------|
| `skokloster-castle` | `.../habitat-test-scenes/skokloster-castle.glb` (38 MB) |
| `apartment_1` | `.../habitat-test-scenes/apartment_1.glb` (50 MB) |
| `van-gogh-room` | `.../habitat-test-scenes/van-gogh-room.glb` (22 MB) |

### Scene Search Order

`_find_scene_glb()` searches these directories in order:

```python
DEFAULT_SCENE_DIRS = [
    "/dss/mcmlscratch/06/di38riq/habitat",         # HM3D train (800)
    "/dss/mcmlscratch/06/di38riq/habitat_val",      # HM3D val   (100)
    ".../aydemir/scene_datasets",                    # demo scenes
    ".../habitat-sim/data/scene_datasets",           # habitat-sim bundled
]
```

And for each directory it tries (in order):
```
{dir}/NNNNN-{name}/{name}.glb          ← HM3D local layout (our data)
{dir}/habitat-test-scenes/{name}.glb   ← demo scenes
{dir}/hm3d/{name}/{name}.basis.glb     ← standard HM3D layout
{dir}/mp3d/{name}/{name}.glb           ← Matterport3D
{dir}/{name}.glb                       ← flat layout
```

### Using VSI-Bench Scenes with Habitat?

The VSI-Bench `.ply` meshes can be converted to `.glb` for use with
`HabitatEnv`:

```python
import trimesh

mesh = trimesh.load("scene0025_01_vh_clean_2.ply")
mesh.export("scene0025_01.glb", file_type="glb")
```

But this is **optional** — the primary use case for `HabitatEnv` is the
900 HM3D scenes, and the primary use case for VSI-Bench is `VSIEnv`.

---

## 5. Quick Start

### Prerequisites

```bash
conda activate habitat
# habitat-sim should already be installed:
#   conda install habitat-sim headless -c conda-forge -c aihabitat
pip install pandas pyarrow   # for dataset loading
```

### Minimal Test (on GPU node)

```bash
srun --partition=mcml-dgx-a100-40x8 --qos=mcml --gres=gpu:1 \
     --mem=32G --time=00:30:00 --pty bash

conda activate habitat
export __EGL_VENDOR_LIBRARY_FILENAMES=/dss/dsshome1/06/di38riq/habitat-sim/10_nvidia.json
unset DISPLAY

python << 'PYEOF'
from vagen_vsi_rl.env.habitat_env import HabitatEnv, HabitatEnvConfig

cfg = HabitatEnvConfig(max_steps=3)
env = HabitatEnv(output_dir="/tmp/habitat_test", config=cfg)

obs = env.reset({
    "scene_name": "skokloster-castle",
    "question": "What color is the ceiling?",
    "choices": ["Red", "Blue", "White", "Brown"],
    "answer_id": "C",
    "question_type": "color",
    "is_numerical": False,
    "dataset": "habitat",
})
print(f"Step {obs.step}: {len(obs.image_paths)} images")

obs, r, done, info = env.step({
    "rotation_angle_degrees": 30, "forward_meters": 0.5,
    "left_meters": 0, "z_delta_meters": 0,
    "answer": None, "done": False,
})
print(f"Step {obs.step}: moved, done={done}")

obs, r, done, info = env.step({
    "rotation_angle_degrees": 0, "forward_meters": 0,
    "left_meters": 0, "z_delta_meters": 0,
    "answer": "C", "done": True,
})
print(f"Done! reward={r}, correct={info['is_correct']}")
env.close()
PYEOF
```

### Using with RolloutCollector

```python
from vagen_vsi_rl.rollout.collector import RolloutCollector

collector = RolloutCollector(
    env_type="habitat",       # ← switches to HabitatEnv
    output_dir="./outputs",
)
```
