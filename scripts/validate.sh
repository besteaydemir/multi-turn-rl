#!/bin/bash
# Quick validation script - runs fast tests before full training

set -e  # Exit on error

echo "========================================"
echo "RL Training Pipeline Validation"
echo "========================================"
echo ""

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if pytest is installed


echo ""
echo "Running quick validation tests..."
echo "========================================"
echo ""

# Run quick tests (no model loading)
python tests/run_tests.py --mode quick

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ Quick validation passed!"
    echo "========================================"
    echo ""
    echo "To run full test suite (requires model download):"
    echo "  python tests/run_tests.py"
    echo ""
    echo "To start training:"
    echo "  python train.py --config configs/my_config.yaml"
else
    echo ""
    echo "========================================"
    echo "✗ Validation failed"
    echo "========================================"
    echo ""
    echo "Please fix failing tests before training."
    exit 1
fi
