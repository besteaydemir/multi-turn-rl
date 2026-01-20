#!/bin/bash
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --qos=mcml
#SBATCH --job-name=vsi_test_50q
#SBATCH --output=/dss/dsshome1/06/di38riq/rl_multi_turn/logs/vsi_test_50q_%j.out
#SBATCH --error=/dss/dsshome1/06/di38riq/rl_multi_turn/logs/vsi_test_50q_%j.err

# =============================================================================
# VSI-Bench Test Run - 50 Random Questions with Full Visualizations
# 
# This script runs 50 randomly selected questions with:
# - Fixed initial view selection (labeled images)
# - Fixed exploration history (labeled images) 
# - All BEV trajectory visualizations enabled
# - Fixed dollhouse views (proper orientation + wall removal)
#
# USAGE:
#   sbatch run_test_50_questions.sh
# =============================================================================

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Test: 50 random questions with full visualizations"
echo "Starting time: $(date)"
echo "=========================================="

# Create logs directory if it doesn't exist
mkdir -p /dss/dsshome1/06/di38riq/rl_multi_turn/logs

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env

# Print environment info
echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")')"
echo "=========================================="

# Change to working directory
cd /dss/dsshome1/06/di38riq/rl_multi_turn

# Run 50 random questions from the full dataset
# This will create a new experiment folder with timestamp
python evaluation/sequential.py \
    --split 1 \
    --num-splits 1 \
    --steps 8 \
    --max-questions 50

echo "=========================================="
echo "Test completed at: $(date)"
echo "Results will be in experiment_logs/ folder"
echo "Check for:"
echo "  - initial_view_selection.json files (view selection fix)"
echo "  - birds_eye_view_path_*.png files (dollhouse fix)" 
echo "  - step_XX/ folders with labeled images (exploration fix)"
echo "========================================"