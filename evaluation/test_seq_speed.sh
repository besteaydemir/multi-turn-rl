#!/bin/bash
#SBATCH --job-name=seq_speed_test
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --qos=mcml
#SBATCH --output=/dss/dsshome1/06/di38riq/rl_multi_turn/logs/seq_speed_test_%j.log

# Sequential Speed Test
# Tests 100 random questions with 16 frames (15 steps), 8B model, vLLM backend

echo "======================================================================"
echo "SEQUENTIAL SPEED TEST"
echo "======================================================================"
echo "Questions: 100 random from combined dataset"
echo "Frames: 16 (15 steps)"
echo "Model: Qwen3-VL-8B-Instruct"
echo "Backend: vLLM (default)"
echo "======================================================================"

# Environment setup
source /dss/dsshome1/06/di38riq/miniconda3/etc/profile.d/conda.sh
conda activate env

export HF_HOME="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TORCH_HOME="${HF_HOME}"
export MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"

# CRITICAL: Fix CUDA multiprocessing issue
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Debug
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi -L

cd /dss/dsshome1/06/di38riq/rl_multi_turn

# First, sample 100 random question IDs
echo ""
echo "Sampling 100 random questions..."
python3 -c "
import random
import json
import sys
sys.path.insert(0, '.')
from utils.data import load_vsi_bench_questions, ALL_SEQUENTIAL_QUESTION_TYPES

# Load all questions
questions = load_vsi_bench_questions(question_types=ALL_SEQUENTIAL_QUESTION_TYPES, dataset='combined')
print(f'Total questions: {len(questions)}')

# Sample 100 random
random.seed(42)  # Reproducible
sampled = random.sample(questions, min(10, len(questions)))

# Save as (scene_name, question_text) pairs for filtering
question_ids = [[q['scene_name'], q['question']] for q in sampled]
with open('/tmp/speed_test_100_questions.json', 'w') as f:
    json.dump(question_ids, f)

print(f'Sampled {len(question_ids)} questions')

# Show distribution by type
from collections import Counter
type_counts = Counter(q['question_type'] for q in sampled)
for qt, count in sorted(type_counts.items()):
    print(f'  {qt}: {count}')
"

# Run sequential with the sampled questions
export QUESTION_IDS_FILE="/tmp/speed_test_100_questions.json"

echo ""
echo "Starting sequential evaluation..."
echo "Start time: $(date)"
START_TIME=$(date +%s)

python3 evaluation/sequential.py --dataset combined --steps 15 --question-types all

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "======================================================================"
echo "SPEED TEST COMPLETE"
echo "======================================================================"
echo "End time: $(date)"
echo "Total elapsed: ${ELAPSED} seconds ($(echo "scale=2; $ELAPSED/60" | bc) minutes)"
echo "Average per question: $(echo "scale=2; $ELAPSED/10" | bc) seconds"
echo "======================================================================"
