#!/bin/bash
#SBATCH --job-name=vagen_ppo
#SBATCH --partition=mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/dss/dsshome1/06/di38riq/rl_multi_turn/logs/train_ppo_%j.out
#SBATCH --error=/dss/dsshome1/06/di38riq/rl_multi_turn/logs/train_ppo_%j.err
#SBATCH --mail-type=END,FAIL

# Create logs directory if it doesn't exist
mkdir -p /dss/dsshome1/06/di38riq/rl_multi_turn/logs

# Activate environment
source ~/.bashrc
eval "$(micromamba shell hook --shell bash)"
conda activate habitat_source

# Change to project directory
cd /dss/dsshome1/06/di38riq/rl_multi_turn

# Set environment variable for vLLM multiprocessing
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Run training - 4 episodes/update, 3 max-steps for faster iteration
python -m vagen_vsi_rl.scripts.train \
    --num-updates 1 \
    --episodes-per-update 4 \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --lora --lora-r 16 --lora-alpha 32 \
    --gpu-memory-utilization 0.40 \
    --max-model-len 8192 \
    --no-ref \
    --share-critic-backbone \
    --dataset arkitscenes \
    --max-steps 3 \
    --gamma 0.99 \
    --ppo-epochs 4 \
    --lr 1e-5 \
    --critic-lr 1e-4 \
    --save-every 10 \
    --val-every 10 --val-episodes 4 \
    --wandb --wandb-project vagen-vsi-rl \
    --output-dir ./checkpoints/ppo_run_$(date +%Y%m%d)

echo "Training completed at $(date)"
