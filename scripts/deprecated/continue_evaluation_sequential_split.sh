#!/bin/bash
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --qos=mcml
#SBATCH --job-name=vsi_seq_split_continue
#SBATCH --output=/dss/dsshome1/06/di38riq/rl_multi_turn/logs/vsi_seq_split_continue_%a_%j.out
#SBATCH --error=/dss/dsshome1/06/di38riq/rl_multi_turn/logs/vsi_seq_split_continue_%a_%j.err
#SBATCH --array=1-3  # Run splits 1 through 3

# =============================================================================
# VSI-Bench Sequential Evaluation Continuation with Job Splitting
# 
# This script continues evaluation from existing experiment directories.
# Each job continues processing its split from where it left off.
#
# USAGE:
#   # Continue all 3 splits at once:
#   sbatch continue_evaluation_sequential_split.sh
#
#   # Continue a specific split only:
#   sbatch --array=2 continue_evaluation_sequential_split.sh  # Only continue split 2
#
#   # Continue a range of splits:
#   sbatch --array=1-2 continue_evaluation_sequential_split.sh  # Only splits 1-2
# =============================================================================

# Configuration
NUM_SPLITS=3  # Total number of splits
STEPS=8       # Number of reasoning steps per question

# The current split number comes from SLURM_ARRAY_TASK_ID
SPLIT_NUM=$SLURM_ARRAY_TASK_ID

# Map split numbers to their experiment directories
case $SPLIT_NUM in
    1)
        EXPERIMENT_DIR="/dss/dsshome1/06/di38riq/rl_multi_turn/experiment_logs/20251225_002135_sequential_split1of3"
        ;;
    2)
        EXPERIMENT_DIR="/dss/dsshome1/06/di38riq/rl_multi_turn/experiment_logs/20251225_002137_sequential_split2of3"
        ;;
    3)
        EXPERIMENT_DIR="/dss/dsshome1/06/di38riq/rl_multi_turn/experiment_logs/20251225_002139_sequential_split3of3"
        ;;
    *)
        echo "ERROR: Invalid split number: $SPLIT_NUM"
        exit 1
        ;;
esac

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
echo "Continuing from: $EXPERIMENT_DIR"
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

# Run the sequential split evaluation with continuation
cd /dss/dsshome1/06/di38riq/rl_multi_turn
python render_point_cloud_qwen_sequential_split.py \
    --split $SPLIT_NUM \
    --num-splits $NUM_SPLITS \
    --steps $STEPS \
    --continue "$EXPERIMENT_DIR"

# Print completion info
echo "=========================================="
echo "Split $SPLIT_NUM/$NUM_SPLITS completed at: $(date)"
echo "=========================================="
