#!/bin/bash
# Test video_baseline.py with ScanNet++ dataset
# This runs on 3 questions and saves output to test/video_scannetpp

cd /dss/dsshome1/06/di38riq/rl_multi_turn

# CRITICAL: Disable vLLM v1 engine - it has CUDA issues on SLURM
export VLLM_USE_V1=0

echo "=========================================="
echo "Testing video_baseline.py with ScanNet++"
echo "=========================================="
echo ""
echo "Dataset: scannetpp"
echo "Questions: 3"
echo "Frames per video: 16"
echo "Backend: hf (HuggingFace)"
echo "Output: test/video_scannetpp"
echo ""

python evaluation/video_baseline.py \
    --backend hf \
    --dataset scannetpp \
    --num-frames 16 \
    --max-questions 3 \
    --split 1 \
    --num-splits 1 \
    --test

echo ""
echo "=========================================="
echo "Test complete!"
echo "=========================================="
echo "Results saved to: test/video_scannetpp"
echo ""
echo "Check the following:"
echo "  - test/q001/, q002/, q003/ (question outputs)"
echo "  - Each has frame_*.png (sampled video frames)"
echo "  - prompt.txt (input prompt)"
echo "  - output.txt (model response)"
echo "  - results.json (answers and accuracy)"
