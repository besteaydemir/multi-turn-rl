#!/bin/bash
#SBATCH --job-name=video_baseline
#SBATCH --output=logs/video_baseline_%A_%a.out
#SBATCH --error=logs/video_baseline_%A_%a.err
#SBATCH --partition=mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=1-4

# Video Baseline Evaluation - with corrected sky direction rotation
# Usage: sbatch scripts/submit_video_baseline.sbatch 8   # for 8 frames
#        sbatch scripts/submit_video_baseline.sbatch 16  # for 16 frames

NUM_FRAMES=${1:-8}

echo "=============================================="
echo "Video Baseline: ${NUM_FRAMES} frames, Split ${SLURM_ARRAY_TASK_ID}/4"
echo "Started at $(date)"
echo "=============================================="

cd /dss/dsshome1/06/di38riq/rl_multi_turn

# Activate environment
source /dss/dsshome1/06/di38riq/miniconda3/etc/profile.d/conda.sh
conda activate env

# Set environment variables
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TRANSFORMERS_CACHE="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir"
export HF_HOME="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir"

# Run the evaluation
python evaluation/video_baseline.py \
    --num-frames ${NUM_FRAMES} \
    --split ${SLURM_ARRAY_TASK_ID} \
    --num-splits 4

echo "=============================================="
echo "Split ${SLURM_ARRAY_TASK_ID} completed at $(date)"
echo "=============================================="
