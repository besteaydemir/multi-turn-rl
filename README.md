# RL Multi-Turn VSI-Bench Evaluation

Multi-turn spatial reasoning evaluation on VSI-Bench using Qwen3-VL.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run evaluation (4 parallel splits)
sbatch run_evaluation_sequential_continue.sh

# Analyze results
python analysis/scripts/analyze_results.py
```

## 📊 Question Types & Metrics

**Multiple Choice (MCA):** Exact match accuracy
- `object_rel_distance`

**Numerical (NA):** Mean Relative Accuracy (MRA = 1/10 × Σ 𝟙[|ŷ-y|/y < 1-θ], θ ∈ {0.5...0.95})
- `object_size_estimation`, `room_size_estimation`, `object_counting`, `object_abs_distance`

## 📁 Structure

```
render_point_cloud_qwen_sequential_split.py  # Main evaluation script
run_evaluation_sequential_continue.sh         # SLURM submission
analysis/scripts/analyze_results.py           # Result analysis
docs/                                         # Documentation
scripts/deprecated/                           # Old scripts
```

## 🔧 Key Options

```bash
# Test mode (5 questions)
python render_point_cloud_qwen_sequential_split.py --test --steps 8

# Single split
sbatch --array=1 run_evaluation_sequential_continue.sh

# Continue from checkpoint
python render_point_cloud_qwen_sequential_split.py --continue /path/to/experiment --split 1 --num-splits 4
```

## 📝 Outputs

Results saved to `experiment_logs/YYYYMMDD_HHMMSS_sequential_splitXofY/`:
- `results.json` / `results.csv` - Evaluation metrics
- `q001/`, `q002/`, ... - Per-question renders and trajectories
