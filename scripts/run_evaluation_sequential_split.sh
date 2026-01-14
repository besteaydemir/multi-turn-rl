#!/bin/bash
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:30:00
#SBATCH --qos=mcml
#SBATCH --job-name=vsi_seq_split
#SBATCH --output=/dss/dsshome1/06/di38riq/rl_multi_turn/logs/vsi_seq_split%a_%j.out
#SBATCH --error=/dss/dsshome1/06/di38riq/rl_multi_turn/logs/vsi_seq_split%a_%j.err
#SBATCH --array=1-4  # Run splits 1 through 6 (adjust based on --num-splits)

# =============================================================================
# VSI-Bench Sequential Evaluation with Job Splitting
# 
# This script uses SLURM job arrays to run different splits in parallel.
# Each job processes a subset of questions sequentially.
#
# USAGE:
#   # Submit all 6 splits at once (edit --array=1-6 to match NUM_SPLITS):
#   sbatch run_evaluation_sequential_split.sh
#
#   # Submit a specific split only:
#   sbatch --array=3 run_evaluation_sequential_split.sh  # Only run split 3
#
#   # Submit a range of splits:
#   sbatch --array=1-3 run_evaluation_sequential_split.sh  # Only splits 1-3
#
# CONFIGURATION:
#   - NUM_SPLITS: Total number of splits (must match --array range)
#   - STEPS: Number of reasoning steps per question
#   - Adjust memory/time based on your split size
# =============================================================================

# Configuration
NUM_SPLITS=4  # Total number of splits (MUST match --array range above!)
STEPS=8       # Number of reasoning steps per question

# The current split number comes from SLURM_ARRAY_TASK_ID
SPLIT_NUM=$SLURM_ARRAY_TASK_ID

# Create logs directory if it doesn't exist
mkdir -p /dss/dsshome1/06/di38riq/rl_multi_turn/logs

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Job Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Split: $SPLIT_NUM of $NUM_SPLITS"
echo "Starting time: $(date)"
echo "=========================================="

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env

# Print environment info
echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")')"
echo "=========================================="

# Run the sequential split evaluation
cd /dss/dsshome1/06/di38riq/rl_multi_turn
python evaluation/sequential.py \
    --split $SPLIT_NUM \
    --num-splits $NUM_SPLITS \
    --steps $STEPS

# Print completion info
echo "=========================================="
echo "Split $SPLIT_NUM/$NUM_SPLITS completed at: $(date)"
echo "=========================================="
