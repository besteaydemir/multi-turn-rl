#!/bin/bash

# Submit Embodied-R1 evaluation jobs for VSI-Bench
# Model: IffYuan/Embodied-R1-7B-Stage1
# Frames: 4, 8, 16, 32
# Dataset: combined (arkitscenes + scannet + scannetpp)
# Question types: all

DATE_FOLDER=$(date +%Y-%m-%d)
LOG_DIR="/dss/dsshome1/06/di38riq/rl_multi_turn/logs/${DATE_FOLDER}"
mkdir -p "$LOG_DIR"

NUM_SPLITS=4
FRAME_CONFIGS=(4 8 16 32)
DATASET="combined"
QUESTION_TYPES="all"
MODEL_ID="IffYuan/Embodied-R1-7B-Stage1"
MODEL_SHORT="Embodied-R1-7B"

echo "======================================================================"
echo "EMBODIED-R1 EVALUATION SUBMISSION"
echo "======================================================================"
echo "Date: ${DATE_FOLDER}"
echo "Model: ${MODEL_ID}"
echo "Dataset: ${DATASET}"
echo "Frames: 4, 8, 16, 32"
echo "Question types: ${QUESTION_TYPES}"
echo "Splits: ${NUM_SPLITS} per frame config"
echo "======================================================================"
echo ""

declare -a JOB_IDS
declare -a JOB_NAMES

# Step 1: Submit a download-only job first to avoid race conditions
echo "Step 1: Submitting model download job..."
echo "────────────────────────────────────────"
TIMESTAMP=$(date +%H-%M-%S)
DOWNLOAD_JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=er1_download
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --qos=mcml
#SBATCH --output=${LOG_DIR}/${TIMESTAMP}_er1_download_%j.log

source /dss/dsshome1/06/di38riq/miniconda3/etc/profile.d/conda.sh
conda activate env

export HF_HOME="/dss/mcmlscratch/06/di38riq/hf_cache"
export TRANSFORMERS_CACHE="\${HF_HOME}"
export TORCH_HOME="\${HF_HOME}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "Downloading ${MODEL_ID} model..."
python3 -c "
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
import os, json, glob

cache = os.environ['HF_HOME']
print('Downloading model weights...')
snapshot_download('${MODEL_ID}', cache_dir=cache)

# Patch preprocessor_config.json: Qwen2_5_VL* -> Qwen2VL*
# (this transformers version doesn't have Qwen2_5_VLImageProcessor as a standalone module)
# The model cache folder name is like: models--IffYuan--Embodied-R1-7B-Stage1
model_cache_name = 'models--' + '${MODEL_ID}'.replace('/', '--')
pattern = cache + '/' + model_cache_name + '/snapshots/*/preprocessor_config.json'
for p in glob.glob(pattern):
    with open(p, 'r') as f:
        cfg = json.load(f)
    changed = False
    if cfg.get('image_processor_type') == 'Qwen2_5_VLImageProcessor':
        cfg['image_processor_type'] = 'Qwen2VLImageProcessor'
        changed = True
    if cfg.get('processor_class') == 'Qwen2_5_VLProcessor':
        cfg['processor_class'] = 'Qwen2VLProcessor'
        changed = True
    if changed:
        with open(p, 'w') as f:
            json.dump(cfg, f, indent=2)
        print(f'Patched {p}: Qwen2_5_VL -> Qwen2VL')

from transformers import AutoProcessor
print('Loading processor...')
AutoProcessor.from_pretrained('${MODEL_ID}')
print('Loading tokenizer...')
AutoTokenizer.from_pretrained('${MODEL_ID}')
print('Download and patch complete!')
"

echo "✅ Model download complete"
EOF
)
echo "  ✅ er1_download: $DOWNLOAD_JOB_ID"
echo ""

# Step 2: Submit evaluation jobs dependent on download
echo "Step 2: Submitting evaluation jobs (depend on download)..."
echo "────────────────────────────────────────"

submit_embodied_r1() {
    local NUM_FRAMES=$1
    local SPLIT=$2
    
    JOB_NAME="er1_${NUM_FRAMES}f_s${SPLIT}"
    TIMESTAMP=$(date +%H-%M-%S)
    
    JOB_ID=$(sbatch --parsable --dependency=afterok:${DOWNLOAD_JOB_ID} <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --qos=mcml
#SBATCH --output=${LOG_DIR}/${TIMESTAMP}_${JOB_NAME}_%j.log

# Environment setup
source /dss/dsshome1/06/di38riq/miniconda3/etc/profile.d/conda.sh
conda activate env

export HF_HOME="/dss/mcmlscratch/06/di38riq/hf_cache"
export TRANSFORMERS_CACHE="\${HF_HOME}"
export TORCH_HOME="\${HF_HOME}"
export HF_DATASETS_CACHE="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/datasets"
export MODEL_ID="${MODEL_ID}"

# CRITICAL: Fix CUDA multiprocessing issue
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Debug
echo "CUDA_VISIBLE_DEVICES: \$CUDA_VISIBLE_DEVICES"
nvidia-smi -L
echo "Model: ${MODEL_ID}"
echo "Split: ${SPLIT}/${NUM_SPLITS}"
echo "Frames: ${NUM_FRAMES}"

# Run Embodied-R1 evaluation (using the same script as Video-R1)
cd /dss/dsshome1/06/di38riq/rl_multi_turn
python3 evaluation/video_baseline_r1.py --dataset ${DATASET} --num-frames ${NUM_FRAMES} --split ${SPLIT} --num-splits ${NUM_SPLITS} --question-types ${QUESTION_TYPES}

echo ""
echo "✅ Embodied-R1 ${NUM_FRAMES} frames split ${SPLIT}/${NUM_SPLITS} complete"
EOF
)
    
    JOB_IDS+=("$JOB_ID")
    JOB_NAMES+=("$JOB_NAME")
    echo "  ✅ $JOB_NAME: $JOB_ID"
}

echo "Submitting Embodied-R1 jobs..."
echo "────────────────────────────────────────"
for NUM_FRAMES in "${FRAME_CONFIGS[@]}"; do
    echo "Frame config: ${NUM_FRAMES}"
    for SPLIT in $(seq 1 $NUM_SPLITS); do
        submit_embodied_r1 "$NUM_FRAMES" "$SPLIT"
    done
done

echo ""
echo "======================================================================"
echo "SUBMISSION SUMMARY"
echo "======================================================================"
echo "Download job: ${DOWNLOAD_JOB_ID} (all eval jobs depend on this)"
echo "Eval jobs submitted: ${#JOB_IDS[@]}"
echo "  Model: ${MODEL_ID}"
echo "  Frame configs: 4, 8, 16, 32"
echo "  Splits per config: ${NUM_SPLITS}"
echo "  Dataset: ${DATASET}"
echo "  Question types: ${QUESTION_TYPES}"
echo "  Time limit: 12 hours per split"
echo ""
echo "Jobs:"
for i in "${!JOB_IDS[@]}"; do
    echo "  ${JOB_NAMES[$i]}: ${JOB_IDS[$i]}"
done
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs: ${LOG_DIR}"
echo "======================================================================"

# Save job list
JOB_LIST_FILE="${LOG_DIR}/embodied_r1_jobs.txt"
echo "Job ID, Job Name" > "$JOB_LIST_FILE"
for i in "${!JOB_IDS[@]}"; do
    echo "${JOB_IDS[$i]}, ${JOB_NAMES[$i]}" >> "$JOB_LIST_FILE"
done
echo "Job list saved to: ${JOB_LIST_FILE}"
