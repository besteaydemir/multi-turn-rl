#!/bin/bash
# Test sequential.py with ScanNet++ dataset
# This runs on 3 questions and saves output to test/sequential_scannetpp

cd /dss/dsshome1/06/di38riq/rl_multi_turn

# CRITICAL: Disable vLLM v1 engine - it has CUDA issues on SLURM
export VLLM_USE_V1=0

echo "=========================================="
echo "Testing sequential.py with ScanNet++"
echo "=========================================="
echo ""
echo "Dataset: scannetpp"
echo "Questions: 3"
echo "Steps per question: 5"
echo "Backend: hf (HuggingFace)"
echo "Output: test/sequential_scannetpp"
echo ""

python evaluation/sequential.py \
    --backend hf \
    --dataset scannetpp \
    --steps 5 \
    --max-questions 3 \
    --split 1 \
    --num-splits 1 \
    --test

echo ""
echo "=========================================="
echo "Test complete!"
echo "=========================================="
echo "Results saved to: test/sequential_scannetpp"
echo ""
echo "Check the following:"
echo "  - test/q001/, q002/, q003/ (question outputs)"
echo "  - Each has render_*.png (rendered images)"
echo "  - cam_pose_*.npy (camera poses)"
echo "  - trajectory.json (full trajectory)"
echo "  - results.json (answers and accuracy)"
