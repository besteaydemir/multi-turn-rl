#!/bin/bash
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:30:00
#SBATCH --qos=mcml
#SBATCH --job-name=vsi_seq_continue
#SBATCH --output=/dss/dsshome1/06/di38riq/rl_multi_turn/logs/vsi_seq_continue%a_%j.out
#SBATCH --error=/dss/dsshome1/06/di38riq/rl_multi_turn/logs/vsi_seq_continue%a_%j.err
#SBATCH --array=1-4  # Run splits 1 through 4

# =============================================================================
# VSI-Bench Sequential Evaluation - New Run for Numerical + MCA Questions
# 
# This script runs a fresh evaluation with updated question types:
#   - Numerical (NA): object_size_estimation, room_size_estimation, 
#                      object_counting, object_abs_distance
#   - Multiple Choice (MCA): object_rel_distance
#
# Uses Mean Relative Accuracy (MRA) for numerical questions.
#
# USAGE:
#   # Submit all 4 splits at once:
#   sbatch run_evaluation_sequential_continue.sh
#
#   # Submit a specific split only:
#   sbatch --array=2 run_evaluation_sequential_continue.sh  # Only run split 2
# =============================================================================

# Configuration
NUM_SPLITS=4  # Total number of splits
STEPS=8       # Number of reasoning steps per question

# Map SLURM_ARRAY_TASK_ID to experiment folders
case $SLURM_ARRAY_TASK_ID in
    1)
        CONTINUE_DIR=""
        SPLIT_NUM=1
        ;;
    2)
        CONTINUE_DIR=""
        SPLIT_NUM=2
        ;;
    3)
        CONTINUE_DIR=""
        SPLIT_NUM=3
        ;;
    4)
        CONTINUE_DIR=""
        SPLIT_NUM=4
        ;;
    *)
        echo "ERROR: Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
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
echo "Mode: Fresh run with numerical + MCA questions"
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
if [ -z "$CONTINUE_DIR" ]; then
    # Fresh run
    python render_point_cloud_qwen_sequential_split.py \
        --split $SPLIT_NUM \
        --num-splits $NUM_SPLITS \
        --steps $STEPS
else
    # Continue from existing
    python render_point_cloud_qwen_sequential_split.py \
        --split $SPLIT_NUM \
        --num-splits $NUM_SPLITS \
        --steps $STEPS \
        --continue "$CONTINUE_DIR"
fi

# Print completion info
echo "=========================================="
echo "Split $SPLIT_NUM/$NUM_SPLITS completed at: $(date)"
echo "=========================================="
