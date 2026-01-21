#!/bin/bash
#SBATCH --job-name=vllm_memory
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/vllm_memory_%j.out
#SBATCH --error=logs/vllm_memory_%j.err

# Load conda environment
source /dss/dsshome1/06/di38riq/miniconda3/etc/profile.d/conda.sh
conda activate env

# Set cache directory
export HF_HOME=/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir

# Set OpenMP threads to 1 (vLLM recommendation)
export OMP_NUM_THREADS=1

# Run vLLM backend with memory profiling
echo "Starting vLLM memory profiling..."
python compare_backends.py --backend vllm --num-questions 20 --num-steps 8 --use-wandb

echo "vLLM memory profiling complete!"
