#!/bin/bash
# Combined test for both sequential.py and video_baseline.py with ScanNet++

cd /dss/dsshome1/06/di38riq/rl_multi_turn

echo "=========================================="
echo "TESTING SCANNET++ SUPPORT"
echo "=========================================="
echo ""
echo "This will test both sequential.py and video_baseline.py"
echo "with ScanNet++ dataset (3 questions each)"
echo ""
echo "Dataset: scannetpp"
echo "Output: test/ directory"
echo ""
echo "=========================================="
echo ""

# Test 1: Sequential with mesh rendering
echo "█ TEST 1/2: Sequential (mesh rendering)"
echo "=========================================="
./test_sequential_scannetpp.sh
echo ""

# Test 2: Video baseline with actual frames
echo "█ TEST 2/2: Video baseline (real frames)"
echo "=========================================="
./test_video_scannetpp.sh
echo ""

echo "=========================================="
echo "✅ ALL TESTS COMPLETE"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  📁 test/sequential_scannetpp/"
echo "  📁 test/video_scannetpp/"
echo ""
echo "To inspect results:"
echo "  ls -R test/"
echo "  cat test/sequential_scannetpp/results.json"
echo "  cat test/video_scannetpp/results.json"
