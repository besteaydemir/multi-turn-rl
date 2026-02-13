#!/bin/bash

# Resume VSI-Bench Sequential Evaluation
# For jobs that timed out and need to continue from where they left off
# 
# Usage: ./submit_resume.sh <MODEL> <FRAMES> <NUM_SPLITS>
# Example: ./submit_resume.sh 4B 16 4

MODEL=${1:-4B}
FRAMES=${2:-16}
NUM_SPLITS=${3:-4}

DATE_FOLDER=$(date +%Y-%m-%d)
LOG_DIR="/dss/dsshome1/06/di38riq/rl_multi_turn/logs/${DATE_FOLDER}"
mkdir -p "$LOG_DIR"

STEPS=$((FRAMES - 1))

# Set time limits based on model
if [ "$MODEL" = "4B" ]; then
    MODEL_ID="Qwen/Qwen3-VL-4B-Instruct"
    TIME_LIMIT="08:00:00"
else
    MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
    TIME_LIMIT="12:00:00"
fi

echo "======================================================================"
echo "RESUME VSI-BENCH SEQUENTIAL EVALUATION"
echo "======================================================================"
echo "Date: ${DATE_FOLDER}"
echo "Model: ${MODEL} (${MODEL_ID})"
echo "Frames: ${FRAMES} (${STEPS} steps)"
echo "Splits: ${NUM_SPLITS}"
echo "Time limit: ${TIME_LIMIT}"
echo "======================================================================"
echo ""

# Arrays to track submitted jobs
declare -a JOB_IDS
declare -a JOB_NAMES

for SPLIT in $(seq 1 $NUM_SPLITS); do
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

# Run resume evaluation
cd /dss/dsshome1/06/di38riq/rl_multi_turn
python3 evaluation/sequential_resume.py --model ${MODEL} --frames ${FRAMES} --split ${SPLIT} --num-splits ${NUM_SPLITS} --dataset combined --question-types all

echo ""
echo "✅ Resume ${MODEL} ${FRAMES} frames split ${SPLIT}/${NUM_SPLITS} complete"
EOF
)
    
    JOB_IDS+=("$JOB_ID")
    JOB_NAMES+=("$JOB_NAME")
    echo "  ✅ $JOB_NAME: $JOB_ID"
done

echo ""
echo "======================================================================"
echo "SUBMISSION SUMMARY"
echo "======================================================================"
echo "Total jobs submitted: ${#JOB_IDS[@]}"
echo ""
echo "Logs directory: ${LOG_DIR}"
echo ""
echo "Monitor jobs: squeue -u \$USER"
echo "======================================================================"

# Save job list to file
JOB_LIST_FILE="${LOG_DIR}/resume_jobs_${MODEL}_${FRAMES}f.txt"
echo "Job ID, Job Name" > "$JOB_LIST_FILE"
for i in "${!JOB_IDS[@]}"; do
    echo "${JOB_IDS[$i]}, ${JOB_NAMES[$i]}" >> "$JOB_LIST_FILE"
done
echo ""
echo "Job list saved to: ${JOB_LIST_FILE}"
