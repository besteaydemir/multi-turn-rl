#!/bin/bash

# Resume Sequential VSI-Bench Evaluation
# This script resumes incomplete runs by using --continue with the original folder
# It maintains the SAME split assignments as the original run

DATE_FOLDER=$(date +%Y-%m-%d)
LOG_DIR="/dss/dsshome1/06/di38riq/rl_multi_turn/logs/${DATE_FOLDER}"
mkdir -p "$LOG_DIR"

EXPERIMENT_BASE="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/Sequential"

echo "======================================================================"
echo "RESUME SEQUENTIAL VSI-BENCH EVALUATION"
echo "======================================================================"
echo "Date: ${DATE_FOLDER}"
echo "This script resumes runs with --continue flag"
echo "======================================================================"
echo ""

# Arrays to track submitted jobs
declare -a JOB_IDS
declare -a JOB_NAMES

# Function to submit a sequential resume job
submit_resume() {
    local MODEL=$1
    local FRAMES=$2
    local SPLIT=$3
    local NUM_SPLITS=$4
    local CONTINUE_DIR=$5
    local TIME_LIMIT=$6
    local STEPS=$((FRAMES - 1))
    
    if [ "$MODEL" = "4B" ]; then
        MODEL_ID="Qwen/Qwen3-VL-4B-Instruct"
    else
        MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
    fi
    
    JOB_NAME="res_${MODEL}_${FRAMES}f_s${SPLIT}"
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

echo "CUDA_VISIBLE_DEVICES: \$CUDA_VISIBLE_DEVICES"
nvidia-smi -L

echo ""
echo "Resuming from: ${CONTINUE_DIR}"
echo ""

# Run sequential evaluation with --continue
cd /dss/dsshome1/06/di38riq/rl_multi_turn
python3 evaluation/sequential.py --dataset combined --steps ${STEPS} --split ${SPLIT} --num-splits ${NUM_SPLITS} --continue "${CONTINUE_DIR}" --question-types all

echo ""
echo "✅ Resume ${MODEL} ${FRAMES} frames split ${SPLIT}/${NUM_SPLITS} complete"
EOF
)
    
    JOB_IDS+=("$JOB_ID")
    JOB_NAMES+=("$JOB_NAME")
    echo "  ✅ $JOB_NAME: $JOB_ID (continuing from $(basename $CONTINUE_DIR))"
}

# ============================================================================
# RESUME 4B 16 FRAMES (was 2 splits, remaining ~1523 questions)
# ============================================================================
echo "Resuming 4B 16-frame jobs..."
echo "────────────────────────────────────────"

# Find the existing run directories
SPLIT1_DIR="${EXPERIMENT_BASE}/4B/16_frames/2026-02-05/20260205_110525_sequential_Qwen3-VL-4B_combined_split1of2_15steps"
SPLIT2_DIR="${EXPERIMENT_BASE}/4B/16_frames/2026-02-05/20260205_110523_sequential_Qwen3-VL-4B_combined_split2of2_15steps"

if [ -d "$SPLIT1_DIR" ]; then
    submit_resume "4B" 16 1 2 "$SPLIT1_DIR" "08:00:00"
else
    echo "  ⚠️ Split 1 directory not found: $SPLIT1_DIR"
fi

if [ -d "$SPLIT2_DIR" ]; then
    submit_resume "4B" 16 2 2 "$SPLIT2_DIR" "08:00:00"
else
    echo "  ⚠️ Split 2 directory not found: $SPLIT2_DIR"
fi

echo ""
echo "======================================================================"
echo "SUBMISSION SUMMARY"
echo "======================================================================"
echo "Total jobs submitted: ${#JOB_IDS[@]}"
echo ""
for i in "${!JOB_IDS[@]}"; do
    echo "  ${JOB_NAMES[$i]}: ${JOB_IDS[$i]}"
done
echo ""
echo "Logs directory: ${LOG_DIR}"
echo ""
echo "Monitor jobs: squeue -u \$USER"
echo "======================================================================"
