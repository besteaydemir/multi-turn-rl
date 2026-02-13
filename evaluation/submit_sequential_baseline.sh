#!/bin/bash
# Submit Sequential Baseline Experiments for VSI-Bench
# Date: February 5, 2026
# 
# This script submits 16 SLURM jobs for sequential baseline evaluation:
# - 2 models: Qwen3-VL-4B-Instruct, Qwen3-VL-8B-Instruct
# - 4 step configurations: 3, 7, 15, 31 steps (equivalent to 4, 8, 16, 32 frames)
# - 2 splits per configuration
# - Dataset: combined (ARKitScenes + ScanNet + ScanNet++)
# - Question types: all (excluding temporal - not supported in sequential)
# - Total: 4,512 questions per configuration (2,256 per split)
#
# Output location: /dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/Sequential/
#
# Jobs will be named: seq_4B_4f_s1, seq_4B_4f_s2, seq_4B_8f_s1, ..., seq_8B_32f_s2

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_ROOT}/logs/$(date +%Y-%m-%d)"
CONDA_PATH="/dss/dsshome1/06/di38riq/miniconda3/etc/profile.d/conda.sh"
CACHE_DIR="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir"

# Create log directory
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "Sequential Baseline Submission"
echo "=============================================="
echo "Date: $(date)"
echo "Log directory: $LOG_DIR"
echo ""

# Job counter
job_count=0
declare -a job_ids

# Models and configurations
models=("4B" "8B")
steps=(3 7 15 31)  # 3 steps = 4 frames, 7 steps = 8 frames, etc.
frame_labels=(4 8 16 32)

# Submit jobs for each model and step configuration
for model_size in "${models[@]}"; do
    if [ "$model_size" == "4B" ]; then
        model_id="Qwen/Qwen3-VL-4B-Instruct"
        time_limit="06:00:00"  # 6 hours for 4B
    else
        model_id="Qwen/Qwen3-VL-8B-Instruct"
        time_limit="08:00:00"  # 8 hours for 8B
    fi
    
    for idx in "${!steps[@]}"; do
        step_count="${steps[$idx]}"
        frame_label="${frame_labels[$idx]}"
        
        # Submit 2 splits for each configuration
        for split in 1 2; do
            job_name="seq_${model_size}_${frame_label}f_s${split}"
            log_file="${LOG_DIR}/${job_name}_%j.log"
            
            echo "Submitting: $job_name (${model_size}, ${step_count} steps, split ${split}/2)"
            
            job_id=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --time=${time_limit}
#SBATCH --output=${log_file}
#SBATCH --error=${log_file}

# Environment setup
source ${CONDA_PATH}
conda activate env

export HF_HOME="${CACHE_DIR}"
export TRANSFORMERS_CACHE="${CACHE_DIR}"
export TORCH_HOME="${CACHE_DIR}"
export MODEL_ID="${model_id}"
export RANDOM_SEED=42

# CRITICAL: Fix CUDA multiprocessing issue
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Debug info
echo "=============================================="
echo "Job: ${job_name}"
echo "Model: ${model_id}"
echo "Steps: ${step_count} (equivalent to ${frame_label} frames)"
echo "Split: ${split}/2"
echo "Started: \$(date)"
echo "Node: \$SLURMD_NODENAME"
echo "=============================================="
echo ""
echo "CUDA_VISIBLE_DEVICES: \$CUDA_VISIBLE_DEVICES"
nvidia-smi -L
echo ""

cd ${PROJECT_ROOT}

# Run sequential baseline
python evaluation/sequential.py \\
    --backend vllm \\
    --dataset combined \\
    --steps ${step_count} \\
    --split ${split} \\
    --num-splits 2 \\
    --question-types all \\
    --output-base ${CACHE_DIR}/experiment_logs/Sequential

echo ""
echo "=============================================="
echo "Job completed: \$(date)"
echo "=============================================="
EOF
)
            
            job_ids+=("$job_id")
            ((job_count++))
            echo "  → Job ID: $job_id"
        done
    done
done

echo ""
echo "=============================================="
echo "Submission Summary"
echo "=============================================="
echo "Total jobs submitted: $job_count"
echo "Job IDs: ${job_ids[*]}"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER -o '%.18i %.12P %.20j %.8T %.10M %.6D %R'"
echo ""
echo "Check logs:"
echo "  tail -f ${LOG_DIR}/*.log"
echo ""
echo "Expected output structure:"
echo "  ${CACHE_DIR}/experiment_logs/Sequential/"
echo "    ├── 4B/"
echo "    │   ├── 4_steps/"
echo "    │   ├── 8_steps/"
echo "    │   ├── 16_steps/"
echo "    │   └── 32_steps/"
echo "    └── 8B/"
echo "        ├── 4_steps/"
echo "        ├── 8_steps/"
echo "        ├── 16_steps/"
echo "        └── 32_steps/"
echo ""
echo "Results will be in: *_sequential_*_combined_*_split*of2/"
echo "  - results.csv: evaluation results"
echo "  - config.json: experiment configuration"
echo "  - trajectories/: saved trajectories (if enabled)"
echo "=============================================="
