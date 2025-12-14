#!/bin/bash
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:30:00
#SBATCH --qos=mcml
#SBATCH --job-name=vsi_bench_continue
#SBATCH --output=/dss/dsshome1/06/di38riq/rl_multi_turn/logs/vsi_bench_continue_%j.out
#SBATCH --error=/dss/dsshome1/06/di38riq/rl_multi_turn/logs/vsi_bench_continue_%j.err

# Create logs directory if it doesn't exist
mkdir -p /dss/dsshome1/06/di38riq/rl_multi_turn/logs

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
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

# Run the evaluation (continuing from most recent folder)
cd /dss/dsshome1/06/di38riq/rl_multi_turn
python render_point_cloud_qwen_angle.py --batch --steps 8 --continue recent

# Print completion info
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
