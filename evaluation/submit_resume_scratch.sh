#!/bin/bash

# Resume VSI-Bench Sequential Evaluation — writing to SCRATCH
# Submits with a configurable delay (default 5 hours) to avoid quota issues.
#
# Output goes to /dss/mcmlscratch/06/di38riq/experiment_logs/
# Completed-question lookup reads from BOTH old and scratch locations.
#
# Usage: ./submit_resume_scratch.sh <MODEL> <FRAMES> <NUM_SPLITS> [DELAY_HOURS]
# Example: ./submit_resume_scratch.sh 4B 32 2 5
#          ./submit_resume_scratch.sh 8B 32 12 5

MODEL=${1:-4B}
FRAMES=${2:-32}
NUM_SPLITS=${3:-4}
DELAY_HOURS=${4:-5}

DATE_FOLDER=$(date +%Y-%m-%d)
LOG_DIR="/dss/dsshome1/06/di38riq/rl_multi_turn/logs/${DATE_FOLDER}"
mkdir -p "$LOG_DIR"

STEPS=$((FRAMES - 1))

# Compute the --begin time
BEGIN_TIME=$(date -d "+${DELAY_HOURS} hours" +"%Y-%m-%dT%H:%M:%S")

# Set time limits based on model
if [ "$MODEL" = "4B" ]; then
    MODEL_ID="Qwen/Qwen3-VL-4B-Instruct"
    TIME_LIMIT="08:00:00"
else
    MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
    TIME_LIMIT="12:00:00"
fi

echo "======================================================================"
echo "RESUME VSI-BENCH SEQUENTIAL EVALUATION (SCRATCH OUTPUT)"
echo "======================================================================"
echo "Date: ${DATE_FOLDER}"
echo "Model: ${MODEL} (${MODEL_ID})"
echo "Frames: ${FRAMES} (${STEPS} steps)"
echo "Splits: ${NUM_SPLITS}"
echo "Time limit: ${TIME_LIMIT}"
echo "Delay: ${DELAY_HOURS} hours (begin at ${BEGIN_TIME})"
echo ""
echo "Output: /dss/mcmlscratch/06/di38riq/experiment_logs/"
echo "Reads completed from:"
echo "  - /dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/"
echo "  - /dss/mcmlscratch/06/di38riq/experiment_logs/"
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
#SBATCH --begin=${BEGIN_TIME}
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

# Run resume evaluation (writes to scratch, reads from both locations)
cd /dss/dsshome1/06/di38riq/rl_multi_turn
python3 evaluation/sequential_resume.py --model ${MODEL} --frames ${FRAMES} --split ${SPLIT} --num-splits ${NUM_SPLITS} --dataset combined --question-types all

echo ""
echo "✅ Resume ${MODEL} ${FRAMES} frames split ${SPLIT}/${NUM_SPLITS} complete"
EOF
)
    
    JOB_IDS+=("$JOB_ID")
    JOB_NAMES+=("$JOB_NAME")
    echo "  ✅ $JOB_NAME: $JOB_ID (begins at ${BEGIN_TIME})"
done

echo ""
echo "======================================================================"
echo "SUBMISSION SUMMARY"
echo "======================================================================"
echo "Total jobs submitted: ${#JOB_IDS[@]}"
echo "All jobs begin at: ${BEGIN_TIME}"
echo ""
echo "Logs directory: ${LOG_DIR}"
echo ""
echo "Monitor jobs: squeue -u \$USER"
echo "======================================================================"

# Save job list to file
JOB_LIST_FILE="${LOG_DIR}/resume_scratch_jobs_${MODEL}_${FRAMES}f.txt"
echo "Job ID, Job Name, Begin Time" > "$JOB_LIST_FILE"
for i in "${!JOB_IDS[@]}"; do
    echo "${JOB_IDS[$i]}, ${JOB_NAMES[$i]}, ${BEGIN_TIME}" >> "$JOB_LIST_FILE"
done
echo ""
echo "Job list saved to: ${JOB_LIST_FILE}"
