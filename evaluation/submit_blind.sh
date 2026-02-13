#!/bin/bash

# Blind Baseline Evaluation — No images, text-only
# Submits 2 jobs: one for 4B, one for 8B
# Output goes to /dss/mcmlscratch/06/di38riq/experiment_logs/Blind/
#
# Usage: ./submit_blind.sh

DATE_FOLDER=$(date +%Y-%m-%d)
LOG_DIR="/dss/dsshome1/06/di38riq/rl_multi_turn/logs/${DATE_FOLDER}"
mkdir -p "$LOG_DIR"

echo "======================================================================"
echo "BLIND BASELINE EVALUATION (NO IMAGES)"
echo "======================================================================"
echo "Date: ${DATE_FOLDER}"
echo "Dataset: COMBINED (arkitscenes + scannet + scannetpp)"
echo "Models: 4B, 8B"
echo "Question types: ALL (MCQ + numerical + temporal)"
echo "Images: NONE (blind)"
echo "Output: /dss/mcmlscratch/06/di38riq/experiment_logs/Blind/"
echo "======================================================================"
echo ""

declare -a JOB_IDS
declare -a JOB_NAMES

submit_blind() {
    local MODEL=$1

    if [ "$MODEL" = "4B" ]; then
        MODEL_ID="Qwen/Qwen3-VL-4B-Instruct"
        TIME_LIMIT="04:00:00"
    else
        MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
        TIME_LIMIT="06:00:00"
    fi

    JOB_NAME="blind_${MODEL}"
    TIMESTAMP=$(date +%H-%M-%S)

    JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=${TIME_LIMIT}
#SBATCH --qos=mcml
#SBATCH --exclude=mcml-dgx-002,mcml-dgx-003
#SBATCH --output=${LOG_DIR}/${TIMESTAMP}_${JOB_NAME}_%j.log

# Environment setup
source /dss/dsshome1/06/di38riq/miniconda3/etc/profile.d/conda.sh
conda activate env

export HF_HOME="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir"
export TRANSFORMERS_CACHE="\${HF_HOME}"
export TORCH_HOME="\${HF_HOME}"
export MODEL_ID="${MODEL_ID}"

# CRITICAL: Fix CUDA multiprocessing issue
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Debug: Check CUDA visibility
echo "CUDA_VISIBLE_DEVICES: \$CUDA_VISIBLE_DEVICES"
nvidia-smi -L

# Run blind evaluation
cd /dss/dsshome1/06/di38riq/rl_multi_turn
python3 evaluation/blind_baseline.py --model ${MODEL} --backend vllm

echo ""
echo "✅ Blind ${MODEL} evaluation complete"
EOF
)

    JOB_IDS+=("$JOB_ID")
    JOB_NAMES+=("$JOB_NAME")
    echo "  ✅ $JOB_NAME: $JOB_ID"
}

echo "Submitting BLIND jobs..."
echo "────────────────────────────────────────"
submit_blind "4B"
submit_blind "8B"

echo ""
echo "======================================================================"
echo "SUBMISSION SUMMARY"
echo "======================================================================"
echo "Total jobs submitted: ${#JOB_IDS[@]}"
echo ""
echo "  blind_4B: ${JOB_IDS[0]}"
echo "  blind_8B: ${JOB_IDS[1]}"
echo ""
echo "Time allocations:"
echo "  4B: 4 hours (text-only is fast)"
echo "  8B: 6 hours"
echo ""
echo "Logs directory: ${LOG_DIR}"
echo "Monitor: squeue -u \$USER"
echo "======================================================================"
