#!/bin/bash
# End-to-End Test Script for RL Training Pipeline
# Tests forward pass (rollout) and backward pass (training) with weight sync

set -e  # Exit on error

echo "=================================="
echo "RL Training Pipeline E2E Test"
echo "=================================="
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env

cd /dss/dsshome1/06/di38riq/rl_multi_turn

echo "Running with config: configs/train_rl_test.yaml"
echo ""
echo "This will:"
echo "1. Initialize Qwen3-VL-4B model (2 instances: vLLM for rollout, HF for training)"
echo "2. Collect 2 trajectories using vLLM (forward pass)"
echo "3. Train on collected data using HuggingFace model (backward pass)"
echo "4. Sync weights from HF model back to vLLM (weight synchronization)"
echo ""
echo "Expected output:"
echo "- Rollout: 2 trajectories with 3 turns each"
echo "- Training: Policy loss, gradient updates"
echo "- Weight sync: Checkpoint save + vLLM reload"
echo ""
read -p "Press Enter to start..."

python rl_multiturn_v2/train_v2.py --config configs/train_rl_test.yaml

echo ""
echo "=================================="
echo "Test Complete!"
echo "=================================="
