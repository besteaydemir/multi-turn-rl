#!/bin/bash
#SBATCH --job-name=traj_4B_8f
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/dss/dsshome1/06/di38riq/rl_multi_turn/logs/trajectory_test_4B_8frames_%j.log
#SBATCH --error=/dss/dsshome1/06/di38riq/rl_multi_turn/logs/trajectory_test_4B_8frames_%j.log

# Trajectory test: 4B model, 8 frames, 20 random questions
# Output: trajectory_test_run/4B/8_frames/

# Environment setup (matching submit_full_vsi_bench.sh)
source /dss/dsshome1/06/di38riq/miniconda3/etc/profile.d/conda.sh
conda activate env

export HF_HOME="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TORCH_HOME="${HF_HOME}"
export MODEL_ID="Qwen/Qwen3-VL-4B-Instruct"
export RANDOM_SEED=42

# CRITICAL: Fix CUDA multiprocessing issue
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Debug: Check CUDA visibility
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi -L

cd /dss/dsshome1/06/di38riq/rl_multi_turn

# Run with 7 steps (= 8 frames) on 20 random questions
# Use --output-base to save to trajectory_test_run instead of experiment_logs
python evaluation/sequential.py \
    --backend vllm \
    --dataset combined \
    --steps 7 \
    --split 1 \
    --num-splits 1 \
    --max-questions 20 \
    --question-types all \
    --output-base /dss/dsshome1/06/di38riq/rl_multi_turn/trajectory_test_run

echo "Job completed at $(date)"
