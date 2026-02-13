#!/bin/bash
# Test RL training with scene rendering enabled

cd /dss/dsshome1/06/di38riq/rl_multi_turn

echo "Testing RL training with scene rendering..."
echo "This will:"
echo "  - Load a ScanNet++ scene"
echo "  - Render images during trajectory collection"
echo "  - Save trajectories with image paths"
echo ""

python rl_multiturn_v2/train_v2.py --config configs/train_rl_test.yaml

echo ""
echo "Check the following directories for results:"
echo "  - checkpoints/run_*/trajectories/  (saved trajectory data)"
echo "  - test_renders/  (rendered images)"
