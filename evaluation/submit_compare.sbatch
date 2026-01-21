#!/bin/bash
#SBATCH --job-name=compare_backends
#SBATCH --output=logs/compare_backends_%j.out
#SBATCH --error=logs/compare_backends_%j.err
#SBATCH --partition=mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# ============================================================
# Backend Comparison Script
# ============================================================
# Compares HuggingFace vs vLLM backends for speedup analysis.
#
# Usage:
#   sbatch submit_compare.sbatch
# ============================================================

echo "=============================================="
echo "Backend Comparison: HuggingFace vs vLLM"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=============================================="

# Create logs directory
mkdir -p logs

# Set up environment
cd /dss/dsshome1/06/di38riq/rl_multi_turn

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate reasoning  # or your environment name

# Set cache directories
export HF_HOME="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir"
export TRANSFORMERS_CACHE="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir"
export TORCH_HOME="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/torch"

# Run comparison
python evaluation/compare_backends.py --num-questions 5 --num-steps 3

echo ""
echo "=============================================="
echo "Comparison completed at $(date)"
echo "=============================================="
