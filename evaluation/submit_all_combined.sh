#!/bin/bash

# ============================================================
# Submit all 12 experiments (2 models × 2 methods × 3 frame counts)
# Dataset: combined (ARKitScenes + ScanNet = ~2716 questions)
# ============================================================
# 
# Experiments:
#   - Video 8B: 4, 8, 16 frames
#   - Video 4B: 4, 8, 16 frames
#   - Sequential 8B: 4, 8, 16 steps
#   - Sequential 4B: 4, 8, 16 steps
#
# Time estimates (combined ~2716 questions):
#   - Video: ~3-4s/question → 4 splits @ ~1hr each → 1hr walltime
#   - Sequential: ~45s/question → 4 splits @ ~8hr each → 10hr walltime
#
# Logs: /dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/
# ============================================================

set -e
cd /dss/dsshome1/06/di38riq/rl_multi_turn

echo "=============================================="
echo "Submitting all 12 combined experiments"
echo "=============================================="
echo "Date: $(date)"
echo ""

# Model IDs
MODEL_8B="Qwen/Qwen3-VL-8B-Instruct"
MODEL_4B="Qwen/Qwen3-VL-4B-Instruct"

# Frame/step counts
COUNTS=(4 8 16)

# Submit Video experiments (faster, 1.5 hr walltime each)
echo "--- VIDEO EXPERIMENTS ---"
for count in "${COUNTS[@]}"; do
    echo "Submitting Video 8B ${count} frames..."
    MODEL_ID="${MODEL_8B}" NUM_FRAMES=${count} DATASET=combined sbatch evaluation/submit_video_baseline.sbatch
    
    echo "Submitting Video 4B ${count} frames..."
    MODEL_ID="${MODEL_4B}" NUM_FRAMES=${count} DATASET=combined sbatch evaluation/submit_video_baseline.sbatch
done

echo ""

# Submit Sequential experiments (slower, 10 hr walltime each)
echo "--- SEQUENTIAL EXPERIMENTS ---"
for count in "${COUNTS[@]}"; do
    echo "Submitting Sequential 8B ${count} steps..."
    MODEL_ID="${MODEL_8B}" STEPS=${count} DATASET=combined sbatch evaluation/submit_vllm.sbatch
    
    echo "Submitting Sequential 4B ${count} steps..."
    MODEL_ID="${MODEL_4B}" STEPS=${count} DATASET=combined sbatch evaluation/submit_vllm.sbatch
done

echo ""
echo "=============================================="
echo "All 12 experiments submitted!"
echo "=============================================="
echo ""
echo "Check queue with: squeue -u \$USER"
echo "Logs will be in: /dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/"
