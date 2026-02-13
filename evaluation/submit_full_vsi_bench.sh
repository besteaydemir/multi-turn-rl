#!/bin/bash

# Full VSI-Bench Evaluation
# Submits jobs for all configurations:
# - Models: 4B, 8B
# - Pipelines: Sequential, Video
# - Frames: 4, 8
# - Dataset: combined (ALL of VSI-Bench: arkitscenes + scannet + scannetpp = ~5130 questions)
# - Splits: 2 splits per configuration for parallelization

DATE_FOLDER=$(date +%Y-%m-%d)
LOG_DIR="/dss/dsshome1/06/di38riq/rl_multi_turn/logs/${DATE_FOLDER}"
mkdir -p "$LOG_DIR"

NUM_SPLITS=2

echo "======================================================================"
echo "FULL VSI-BENCH EVALUATION SUBMISSION"
echo "======================================================================"
echo "Date: ${DATE_FOLDER}"
echo "Dataset: COMBINED (arkitscenes + scannet + scannetpp = ~4512 questions)"
echo "Models: 4B, 8B"
echo "Pipelines: Sequential, Video"
echo "Frames: 4, 8, 16, 32"
echo "Splits: ${NUM_SPLITS} per configuration (~2256 questions per split)"
echo "======================================================================"
echo ""

# Arrays to track submitted jobs
declare -a JOB_IDS
declare -a JOB_NAMES

# Function to submit a sequential job
submit_sequential() {
    local MODEL=$1
    local FRAMES=$2
    local SPLIT=$3
    local STEPS=$((FRAMES - 1))
    
    if [ "$MODEL" = "4B" ]; then
        MODEL_ID="Qwen/Qwen3-VL-4B-Instruct"
        TIME_LIMIT="08:00:00"  # 8 hours for 4B
    else
        MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
        TIME_LIMIT="12:00:00"  # 12 hours for 8B (slower)
    fi
    
    JOB_NAME="seq_${MODEL}_${FRAMES}f_s${SPLIT}"
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

# Debug: Check CUDA visibility
echo "CUDA_VISIBLE_DEVICES: \$CUDA_VISIBLE_DEVICES"
nvidia-smi -L

# Run sequential evaluation
cd /dss/dsshome1/06/di38riq/rl_multi_turn
python3 evaluation/sequential.py --dataset combined --steps ${STEPS} --split ${SPLIT} --num-splits ${NUM_SPLITS} --question-types all

echo ""
echo "✅ Sequential ${MODEL} ${FRAMES} frames split ${SPLIT}/${NUM_SPLITS} complete"
EOF
)
    
    JOB_IDS+=("$JOB_ID")
    JOB_NAMES+=("$JOB_NAME")
    echo "  ✅ $JOB_NAME: $JOB_ID"
}

# Function to submit a video job
submit_video() {
    local MODEL=$1
    local FRAMES=$2
    local SPLIT=$3
    
    if [ "$MODEL" = "4B" ]; then
        MODEL_ID="Qwen/Qwen3-VL-4B-Instruct"
        TIME_LIMIT="08:00:00"  # 4 hours for 4B (video is faster)
    else
        MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
        TIME_LIMIT="08:00:00"  # 6 hours for 8B
    fi
    
    JOB_NAME="vid_${MODEL}_${FRAMES}f_s${SPLIT}"
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

# Debug: Check CUDA visibility
echo "CUDA_VISIBLE_DEVICES: \$CUDA_VISIBLE_DEVICES"
nvidia-smi -L

# Run video evaluation
cd /dss/dsshome1/06/di38riq/rl_multi_turn
python3 evaluation/video_baseline.py --dataset combined --num-frames ${FRAMES} --split ${SPLIT} --num-splits ${NUM_SPLITS} --question-types all

echo ""
echo "✅ Video ${MODEL} ${FRAMES} frames split ${SPLIT}/${NUM_SPLITS} complete"
EOF
)
    
    JOB_IDS+=("$JOB_ID")
    JOB_NAMES+=("$JOB_NAME")
    echo "  ✅ $JOB_NAME: $JOB_ID"
}

echo "Submitting SEQUENTIAL jobs..."
echo "────────────────────────────────────────"
for MODEL in 4B 8B; do
    for FRAMES in 4 8 16 32; do
        for SPLIT in $(seq 1 $NUM_SPLITS); do
            submit_sequential "$MODEL" "$FRAMES" "$SPLIT"
        done
    done
done

# echo ""
# echo "Submitting VIDEO jobs..."
# echo "────────────────────────────────────────"
# for MODEL in 4B 8B; do
#     for FRAMES in 4 8 16 32; do
#         for SPLIT in $(seq 1 $NUM_SPLITS); do
#             submit_video "$MODEL" "$FRAMES" "$SPLIT"
#         done
#     done
# done

echo ""
echo "======================================================================"
echo "SUBMISSION SUMMARY"
echo "======================================================================"
echo "Total jobs submitted: ${#JOB_IDS[@]}"
echo ""
echo "Sequential: 16 jobs (2 models × 4 frame configs × 2 splits)"
echo "Video: 16 jobs (2 models × 4 frame configs × 2 splits)"
echo ""
echo "Time allocations:"
echo "  Sequential 4B:  8 hours per split"
echo "  Sequential 8B: 12 hours per split"
echo "  Video 4B:  8 hours per split"
echo "  Video 8B:  8 hours per split"
echo ""
echo "Logs directory: ${LOG_DIR}"
echo ""
echo "Monitor all jobs: squeue -u \$USER"
echo "Check specific config: ls -lth ${LOG_DIR}/vid_4B_8f_*"
echo "======================================================================"

# Save job list to file
JOB_LIST_FILE="${LOG_DIR}/submitted_jobs.txt"
echo "Job ID, Job Name" > "$JOB_LIST_FILE"
for i in "${!JOB_IDS[@]}"; do
    echo "${JOB_IDS[$i]}, ${JOB_NAMES[$i]}" >> "$JOB_LIST_FILE"
done
echo ""
echo "Job list saved to: ${JOB_LIST_FILE}"
