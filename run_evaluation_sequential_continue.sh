#!/bin/bash
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:30:00
#SBATCH --qos=mcml
#SBATCH --job-name=vsi_seq_continue
#SBATCH --output=/dss/dsshome1/06/di38riq/rl_multi_turn/logs/vsi_seq_continue%a_%j.out
#SBATCH --error=/dss/dsshome1/06/di38riq/rl_multi_turn/logs/vsi_seq_continue%a_%j.err
#SBATCH --array=1-5  # Run splits 1 through 5

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
NUM_SPLITS=5  # Total number of splits
STEPS=8       # Number of reasoning steps per question
AUTO_CONTINUE=true  # Set to true to automatically continue from latest experiment

# Automatically find the latest experiment folder for this split
SPLIT_NUM=$SLURM_ARRAY_TASK_ID

if [ "$AUTO_CONTINUE" = true ]; then
    # Find the most recent experiment folder for this split (any date, matching NUM_SPLITS)
    # This will find folders like: 20260119_013550_sequential_split1of5
    LATEST_FOLDER=$(ls -td /dss/dsshome1/06/di38riq/rl_multi_turn/experiment_logs/*_sequential_split${SPLIT_NUM}of${NUM_SPLITS} 2>/dev/null | head -1)
    
    if [ -n "$LATEST_FOLDER" ] && [ -d "$LATEST_FOLDER" ]; then
        CONTINUE_DIR="$LATEST_FOLDER"
        echo "Auto-detected continue folder: $CONTINUE_DIR"
    else
        CONTINUE_DIR=""
        echo "No existing folder found for split $SPLIT_NUM of $NUM_SPLITS, starting fresh"
    fi
else
    # Manual specification (legacy mode) - update these paths as needed
    case $SLURM_ARRAY_TASK_ID in
        1)
            CONTINUE_DIR="experiment_logs/20260119_013550_sequential_split1of5"
            ;;
        2)
            CONTINUE_DIR="experiment_logs/20260119_013638_sequential_split2of5"
            ;;
        3)
            CONTINUE_DIR="experiment_logs/20260119_013653_sequential_split3of5"
            ;;
        4)
            CONTINUE_DIR="experiment_logs/20260119_015245_sequential_split4of5"
            ;;
        5)
            CONTINUE_DIR="experiment_logs/20260119_015847_sequential_split5of5"
            ;;
        *)
            echo "ERROR: Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
            exit 1
            ;;
    esac
fi

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
if [ -n "$CONTINUE_DIR" ]; then
    echo "Mode: Continue from $CONTINUE_DIR"
else
    echo "Mode: Fresh run"
fi
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
    python evaluation/sequential.py \
        --split $SPLIT_NUM \
        --num-splits $NUM_SPLITS \
        --steps $STEPS
else
    # Continue from existing
    python evaluation/sequential.py \
        --split $SPLIT_NUM \
        --num-splits $NUM_SPLITS \
        --steps $STEPS \
        --continue "$CONTINUE_DIR"
fi

# Print completion info
echo "=========================================="
echo "Split $SPLIT_NUM/$NUM_SPLITS completed at: $(date)"
echo "=========================================="
