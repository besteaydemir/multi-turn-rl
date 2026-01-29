#!/bin/bash
#SBATCH --job-name=test_video_hf
#SBATCH --output=logs/test_video_hf_%j.out
#SBATCH --error=logs/test_video_hf_%j.err
#SBATCH --partition=mcml-hgx-a100-80x4-mig
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:30:00

# ============================================================
# Test Video Backend using HuggingFace (MIG-friendly)
# ============================================================
# This script tests video processing with HuggingFace, which is
# more compatible with MIG devices than vLLM. Use this first to
# verify the video format works before testing vLLM.
#
# Usage:
#   sbatch submit_test_video_hf.sbatch
#   NUM_FRAMES=32 sbatch submit_test_video_hf.sbatch
# ============================================================

echo "=============================================="
echo "Test HuggingFace Video Backend on MIG"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
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

# Get parameters from environment or use defaults
NUM_FRAMES=${NUM_FRAMES:-32}
MODEL=${MODEL:-4B}

echo ""
echo "Parameters:"
echo "  NUM_FRAMES: $NUM_FRAMES"
echo "  MODEL: $MODEL"
echo ""

# Run test
python evaluation/test_video_backend_hf.py --num-frames $NUM_FRAMES --model $MODEL

echo ""
echo "=============================================="
echo "Test Complete: $(date)"
echo "=============================================="
