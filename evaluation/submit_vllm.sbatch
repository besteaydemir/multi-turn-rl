#!/bin/bash
#SBATCH --job-name=vsibench_vllm
#SBATCH --output=logs/vsibench_vllm_%A_%a.out
#SBATCH --error=logs/vsibench_vllm_%A_%a.err
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --array=1-4

# ============================================================
# VSI-Bench Evaluation with vLLM Backend (FASTER)
# ============================================================
# This script runs the VSI-Bench evaluation using the vLLM
# backend which provides ~4-8x speedup through:
#   - Continuous batching
#   - Optimized KV cache management
#   - PagedAttention
#
# NOTE: vLLM requires additional installation:
#   pip install vllm
#
# Usage:
#   sbatch submit_vllm.sbatch          # Run all 4 splits
#   sbatch --array=1 submit_vllm.sbatch  # Run only split 1
#
# Expected speedup: ~4-8x faster than HuggingFace backend
# ============================================================

echo "=============================================="
echo "VSI-Bench Evaluation (vLLM Backend - FAST)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=============================================="

# Create logs directory
mkdir -p logs

# Set up environment
cd /dss/dsshome1/06/di38riq/rl_multi_turn

# Activate conda environment (adjust as needed)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate reasoning  # or your environment name

# Set cache directories
export HF_HOME="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir"
export TRANSFORMERS_CACHE="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir"
export TORCH_HOME="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/torch"

# vLLM specific settings
export VLLM_USE_MODELSCOPE=False
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Run the evaluation with vLLM backend
SPLIT=${SLURM_ARRAY_TASK_ID}
NUM_SPLITS=4

echo ""
echo "Running split ${SPLIT} of ${NUM_SPLITS} with vLLM..."
echo ""

python evaluation/sequential.py \
    --backend vllm \
    --split ${SPLIT} \
    --num-splits ${NUM_SPLITS} \
    --steps 8

echo ""
echo "=============================================="
echo "Split ${SPLIT} completed at $(date)"
echo "=============================================="
