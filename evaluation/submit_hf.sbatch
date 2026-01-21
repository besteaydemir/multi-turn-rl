#!/bin/bash
#SBATCH --job-name=vsibench_hf
#SBATCH --output=logs/vsibench_hf_%A_%a.out
#SBATCH --error=logs/vsibench_hf_%A_%a.err
#SBATCH --partition=mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=1-4

# ============================================================
# VSI-Bench Evaluation with HuggingFace Backend
# ============================================================
# This script runs the VSI-Bench evaluation using the original
# HuggingFace transformers backend (sequential inference).
#
# Usage:
#   sbatch submit_hf.sbatch          # Run all 4 splits
#   sbatch --array=1 submit_hf.sbatch  # Run only split 1
#
# To check status:
#   squeue -u $USER
#
# To cancel:
#   scancel <job_id>
# ============================================================

echo "=============================================="
echo "VSI-Bench Evaluation (HuggingFace Backend)"
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

# Run the evaluation
SPLIT=${SLURM_ARRAY_TASK_ID}
NUM_SPLITS=4

echo ""
echo "Running split ${SPLIT} of ${NUM_SPLITS}..."
echo ""

python evaluation/sequential.py \
    --backend hf \
    --split ${SPLIT} \
    --num-splits ${NUM_SPLITS} \
    --steps 8

echo ""
echo "=============================================="
echo "Split ${SPLIT} completed at $(date)"
echo "=============================================="
