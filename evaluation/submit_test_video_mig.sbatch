#!/bin/bash
#SBATCH --job-name=test_video_mig
#SBATCH --output=logs/test_video_mig_%j.out
#SBATCH --error=logs/test_video_mig_%j.err
#SBATCH --partition=mcml-hgx-a100-80x4-mig
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:30:00

# ============================================================
# Test vLLM Video Backend on MIG Device
# ============================================================
# This script tests the video backend on MIG (Multi-Instance GPU)
# partitions which have smaller memory but are easier to get.
#
# Usage:
#   sbatch submit_test_video_mig.sbatch
#   NUM_FRAMES=16 sbatch submit_test_video_mig.sbatch
#   MODEL=8B sbatch submit_test_video_mig.sbatch
# ============================================================

echo "=============================================="
echo "Test vLLM Video Backend on MIG"
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

# MIG-specific settings
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export OMP_NUM_THREADS=1

# vLLM settings for MIG
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Get parameters from environment or use defaults
NUM_FRAMES=${NUM_FRAMES:-32}
MODEL=${MODEL:-4B}

echo ""
echo "Parameters:"
echo "  NUM_FRAMES: $NUM_FRAMES"
echo "  MODEL: $MODEL"
echo ""

# Run test
python evaluation/test_video_backend_mig.py --num-frames $NUM_FRAMES --model $MODEL

echo ""
echo "=============================================="
echo "Test Complete: $(date)"
echo "=============================================="
