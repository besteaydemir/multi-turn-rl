#!/bin/bash
#SBATCH --job-name=test_32frames
#SBATCH --output=logs/test_32frames_%j.out
#SBATCH --error=logs/test_32frames_%j.err
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:20:00

# ============================================================
# Quick Test: 32 Frames Video Backend
# ============================================================
# This tests just 5 questions with 32 frames to verify
# the video backend works before submitting full jobs.
# ============================================================

echo "=============================================="
echo "Test 32 Frames Video Backend"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
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
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export OMP_NUM_THREADS=1

# Test with 4B model, combined dataset, 32 frames, first 5 questions only
echo ""
echo "Testing with Qwen3-VL-4B, 32 frames, 5 questions..."
echo ""

MODEL_ID="Qwen/Qwen3-VL-4B-Instruct" \
python evaluation/video_baseline.py \
    --num-frames 32 \
    --split 1 \
    --num-splits 1 \
    --max-questions 5 \
    --dataset combined

echo ""
echo "=============================================="
echo "Test Complete: $(date)"
echo "=============================================="
