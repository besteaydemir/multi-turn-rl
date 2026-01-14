# RL Multi-Turn VSI-Bench Evaluation

Multi-turn spatial reasoning evaluation on VSI-Bench using Qwen3-VL.

## Quick Start

**Installation:**
```bash
pip install -r requirements.txt
```

**Run Evaluation:**
```bash
# Submit all 4 splits in parallel
sbatch run_evaluation_sequential_continue.sh

# Run single split for testing
sbatch --array=1 run_evaluation_sequential_continue.sh

# Test mode (5 questions locally)
python render_point_cloud_qwen_sequential_split.py --test --steps 8
```

**Analyze Results:**
```bash
python analysis/scripts/analyze_results.py
```

## Entry Points

### Main Evaluation Scripts
- `render_point_cloud_qwen_sequential_split.py` - Primary evaluation script with split support
- `render_point_cloud_qwen_angle.py` - Angle-based evaluation variant
- `render_point_cloud_qwen_angle_batched.py` - Batched inference version

### SLURM Submission
- `run_evaluation_sequential_continue.sh` - Submit parallel jobs (4 splits)

### Analysis
- `analysis/scripts/analyze_results.py` - Generate plots and statistics

### Training
- `train_rl.py` - RL training script
- `rl_environment/` - Environment implementation
- `rl_trainer/` - Training utilities

### Utilities
- `scripts/utilities/vsi_download.py` - Download VSI-Bench dataset
- `scripts/utilities/starting_point.py` - Initialization helper
- `utils/` - Shared utility modules (camera, rendering, parsing)

## Repository Structure

```
rl_multi_turn/
├── render_point_cloud_qwen_sequential_split.py  # Main script
├── run_evaluation_sequential_continue.sh         # SLURM job
├── train_rl.py                                   # RL training
├── config.py                                     # Configuration
├── example_config.yaml                           # Config template
├── requirements.txt                              # Dependencies
│
├── analysis/scripts/                             # Result analysis
├── rl_environment/                               # RL environment
├── rl_trainer/                                   # Training modules  
├── utils/                                        # Utilities
├── tests/                                        # Test suite
├── scripts/                                      # Helper scripts
│   ├── deprecated/                               # Old scripts
│   └── utilities/                                # Utility scripts
└── docs/                                         # Documentation
```

## Question Types & Metrics

**Multiple Choice (Accuracy):**
- `object_rel_distance` - Relative distance relationships

**Numerical (MRA = 1/10 × Σ 1[|y_pred - y|/y < 1 - θ], θ ∈ {0.5, 0.55, ..., 0.95}):**
- `object_size_estimation` - Object dimensions  
- `room_size_estimation` - Room measurements
- `object_counting` - Object counts
- `object_abs_distance` - Absolute distances

## Command Line Options

```bash
python render_point_cloud_qwen_sequential_split.py \
    --split 1 \                    # Which split (1-4)
    --num-splits 4 \               # Total splits
    --steps 8 \                    # Max reasoning steps
    --continue /path/to/exp \      # Resume from checkpoint
    --test \                       # Test mode (5 questions)
    --max-questions 10             # Limit questions
```

## Output Structure

```
experiment_logs/YYYYMMDD_HHMMSS_sequential_splitXofY/
├── results.json                   # Full results
├── results.csv                    # CSV format
└── q001/, q002/, ...              # Per-question outputs
    ├── render_00.png              # Initial view
    ├── step_XX/                   # Reasoning steps
    │   ├── render.png
    │   ├── prompt.txt
    │   └── response.txt
    └── birds_eye_view_path_*.png  # Trajectory visualizations
```

## Configuration

Key parameters in `render_point_cloud_qwen_sequential_split.py`:
```python
NUM_STEPS = 8                                    # Max reasoning steps
IMAGE_WH = (640, 480)                            # Render resolution
CAM_HEIGHT = 1.6                                 # Camera height (meters)
INITIAL_VIEW_SELECTION_METRIC = "visibility"    # "visibility" or "laplacian"
```

## Testing

```bash
pytest tests/                     # Run all tests
bash scripts/validate.sh          # Validate setup
```
