#!/bin/bash
#SBATCH --job-name=video_baseline
#SBATCH --output=logs/video_baseline_%A_%a.out
#SBATCH --error=logs/video_baseline_%A_%a.err
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:45:00
#SBATCH --array=1-4

# ============================================================
# Video Baseline Evaluation for VSI-Bench
# ============================================================
# This script evaluates Qwen3-VL on VSI-Bench using actual video
# frames from ARKitScenes/ScanNet.
#
# Usage:
#   sbatch submit_video_baseline.sbatch          # 8 frames, arkitscenes
#   NUM_FRAMES=16 sbatch submit_video_baseline.sbatch
#   DATASET=combined sbatch submit_video_baseline.sbatch
#   MODEL_ID="Qwen/Qwen3-VL-4B-Instruct" sbatch submit_video_baseline.sbatch
# ============================================================

echo "=============================================="
echo "Video Baseline Evaluation"
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

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env

# Set cache directories
export HF_HOME="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir"
export TRANSFORMERS_CACHE="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir"
export TORCH_HOME="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/torch"

# vLLM specific settings
export VLLM_USE_MODELSCOPE=False
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# CRITICAL: Fix CUDA multiprocessing issue
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Configuration (can override with environment variables)
NUM_FRAMES=${NUM_FRAMES:-8}
DATASET=${DATASET:-arkitscenes}
MODEL_ID_PARAM=${MODEL_ID:-"Qwen/Qwen3-VL-8B-Instruct"}

SPLIT=${SLURM_ARRAY_TASK_ID}
NUM_SPLITS=4

echo ""
echo "Model: ${MODEL_ID_PARAM}"
echo "Dataset: ${DATASET}"
echo "Running split ${SPLIT} of ${NUM_SPLITS} with ${NUM_FRAMES} frames..."
echo ""

MODEL_ID="${MODEL_ID_PARAM}" python evaluation/video_baseline.py \
    --dataset ${DATASET} \
    --num-frames ${NUM_FRAMES} \
    --split ${SPLIT} \
    --num-splits ${NUM_SPLITS}

echo ""
echo "=============================================="
echo "Split ${SPLIT} completed at $(date)"
echo "=============================================="
