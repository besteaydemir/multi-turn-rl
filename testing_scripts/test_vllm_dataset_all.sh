#!/bin/bash
# Test script for validating dataset=all with vLLM

source ~/miniconda3/etc/profile.d/conda.sh
conda activate env
export HF_HOME=/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir
export TRANSFORMERS_CACHE=/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir
export TORCH_HOME=/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/torch
export VLLM_USE_MODELSCOPE=False
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd /dss/dsshome1/06/di38riq/rl_multi_turn

echo "=== Testing Video Baseline with dataset=all ===" 
MODEL_ID=Qwen/Qwen3-VL-4B-Instruct python evaluation/video_baseline.py --backend vllm --dataset all --question-types all --test --max-questions 2

echo ""
echo "=== Testing Sequential with dataset=all ==="
MODEL_ID=Qwen/Qwen3-VL-4B-Instruct python evaluation/sequential.py --backend vllm --dataset all --question-types all --test --max-questions 2

echo ""
echo "=== Test Complete ==="
