#!/bin/bash
#SBATCH --job-name=seq_kv_test
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --qos=mcml
#SBATCH --output=/dss/dsshome1/06/di38riq/rl_multi_turn/logs/seq_kv_cache_test_%j.log

# Sequential KV-Cache Speed Test
# Tests 10 random questions with 16 frames (15 steps), 8B model
# Compares: Standard vLLM vs vLLM with KV-Cache optimization

echo "======================================================================"
echo "SEQUENTIAL KV-CACHE COMPARISON TEST"
echo "======================================================================"
echo "Questions: 10 random from combined dataset"
echo "Frames: 16 (15 steps)"
echo "Model: Qwen3-VL-8B-Instruct"
echo "Comparison: Standard vLLM vs KV-Cache Optimized vLLM"
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

# First, sample 10 random question IDs (reproducible)
echo ""
echo "Sampling 10 random questions..."
python3 -c "
import random
import json
import sys
sys.path.insert(0, '.')
from utils.data import load_vsi_bench_questions, ALL_SEQUENTIAL_QUESTION_TYPES

# Load all questions
questions = load_vsi_bench_questions(question_types=ALL_SEQUENTIAL_QUESTION_TYPES, dataset='combined')
print(f'Total questions: {len(questions)}')

# Sample 10 random with fixed seed
random.seed(42)
sampled = random.sample(questions, min(10, len(questions)))

# Save as (scene_name, question_text) pairs
question_ids = [[q['scene_name'], q['question']] for q in sampled]
with open('/tmp/kv_cache_test_questions.json', 'w') as f:
    json.dump(question_ids, f)

print(f'Sampled {len(question_ids)} questions')

from collections import Counter
type_counts = Counter(q['question_type'] for q in sampled)
for qt, count in sorted(type_counts.items()):
    print(f'  {qt}: {count}')
"

export QUESTION_IDS_FILE="/tmp/kv_cache_test_questions.json"

# ===========================================
# TEST 1: KV-CACHE OPTIMIZED VERSION
# ===========================================
echo ""
echo "======================================================================"
echo "TEST 1: KV-CACHE OPTIMIZED VERSION"
echo "======================================================================"
echo "Start time: $(date)"
START_TIME_KV=$(date +%s)

python3 evaluation/sequential_kv_cache.py --dataset combined --steps 15 --question-types all --test

END_TIME_KV=$(date +%s)
ELAPSED_KV=$((END_TIME_KV - START_TIME_KV))

echo ""
echo "KV-Cache version completed in ${ELAPSED_KV} seconds"

# Save results for comparison
cp test/results.json /tmp/kv_cache_results.json 2>/dev/null || echo "No test results to copy"
cp test/results.csv /tmp/kv_cache_results.csv 2>/dev/null || echo "No test results to copy"

# Clear test folder for next run
rm -rf test/q*

# ===========================================
# TEST 2: STANDARD VLLM VERSION
# ===========================================
echo ""
echo "======================================================================"
echo "TEST 2: STANDARD VLLM VERSION (for comparison)"
echo "======================================================================"
echo "Start time: $(date)"
START_TIME_STD=$(date +%s)

python3 evaluation/sequential.py --dataset combined --steps 15 --question-types all --test

END_TIME_STD=$(date +%s)
ELAPSED_STD=$((END_TIME_STD - START_TIME_STD))

echo ""
echo "Standard vLLM version completed in ${ELAPSED_STD} seconds"

# Save results
cp test/results.json /tmp/standard_results.json 2>/dev/null || echo "No test results to copy"
cp test/results.csv /tmp/standard_results.csv 2>/dev/null || echo "No test results to copy"

# ===========================================
# COMPARISON
# ===========================================
echo ""
echo "======================================================================"
echo "COMPARISON SUMMARY"
echo "======================================================================"
echo "KV-Cache Version: ${ELAPSED_KV} seconds"
echo "Standard vLLM:    ${ELAPSED_STD} seconds"

if [ $ELAPSED_STD -gt 0 ]; then
    SPEEDUP=$(echo "scale=2; $ELAPSED_STD / $ELAPSED_KV" | bc)
    echo "Speedup: ${SPEEDUP}x"
fi

echo ""
echo "Per question times:"
echo "  KV-Cache: $(echo "scale=1; $ELAPSED_KV / 10" | bc) seconds/question"
echo "  Standard: $(echo "scale=1; $ELAPSED_STD / 10" | bc) seconds/question"

echo ""
echo "Comparing answers..."
python3 -c "
import json

try:
    with open('/tmp/kv_cache_results.json') as f:
        kv_results = json.load(f)
    with open('/tmp/standard_results.json') as f:
        std_results = json.load(f)
    
    print(f'KV-Cache results: {len(kv_results)} questions')
    print(f'Standard results: {len(std_results)} questions')
    
    # Compare answers
    matches = 0
    for kv, std in zip(kv_results, std_results):
        if kv.get('model_answer') == std.get('model_answer'):
            matches += 1
        else:
            print(f'  Mismatch: KV={kv.get(\"model_answer\")} vs Std={std.get(\"model_answer\")}')
    
    print(f'Answer match rate: {matches}/{len(kv_results)} = {100*matches/len(kv_results):.1f}%')
    
    # Accuracy comparison
    kv_correct = sum(1 for r in kv_results if r.get('correct'))
    std_correct = sum(1 for r in std_results if r.get('correct'))
    print(f'KV-Cache accuracy: {kv_correct}/{len(kv_results)} = {100*kv_correct/len(kv_results):.1f}%')
    print(f'Standard accuracy: {std_correct}/{len(std_results)} = {100*std_correct/len(std_results):.1f}%')
except Exception as e:
    print(f'Could not compare results: {e}')
"

echo ""
echo "======================================================================"
echo "TEST COMPLETE"
echo "======================================================================"
