#!/bin/bash
# Run pipeline tests in background, save results
cd ~/rl_multi_turn
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export __EGL_VENDOR_LIBRARY_FILENAMES=/dss/dsshome1/06/di38riq/habitat-sim/10_nvidia.json
unset DISPLAY

echo "=== Starting habitat test at $(date) ==="
python test_pipeline.py --habitat-only 2>&1
HABITAT_EXIT=$?
echo "=== Habitat test finished at $(date) with exit code $HABITAT_EXIT ==="

echo ""
echo "=== Starting vLLM test at $(date) ==="
python test_pipeline.py --vllm-only 2>&1
VLLM_EXIT=$?
echo "=== vLLM test finished at $(date) with exit code $VLLM_EXIT ==="

echo ""
echo "FINAL: habitat=$HABITAT_EXIT vllm=$VLLM_EXIT"
