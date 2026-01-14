# RL Training Pipeline - Test Suite Summary

## Overview

Created a comprehensive test suite to validate the entire RL training pipeline before running full-scale training. The tests catch bugs early and ensure correctness of critical components.

## Test Suite Structure

```
tests/
├── __init__.py                      # Test package initialization
├── README.md                        # Comprehensive test documentation
├── pytest.ini                       # Pytest configuration
├── run_tests.py                     # Main test runner with multiple modes
├── run_validation_workflow.py      # Interactive validation workflow
├── test_alignment.py                # Forward pass and token alignment tests
├── test_advantages.py               # LOO baseline and gradient tests
├── test_reference_model.py          # Reference model behavior tests
└── test_integration.py              # End-to-end integration tests
```

## Test Categories

### 1. Forward Pass Alignment (`test_alignment.py`)
**Purpose:** Verify token log probabilities are computed correctly.

**Tests:**
- ✅ `test_logprob_computation_vs_manual_nll` - Log probs match manual NLL computation
- ✅ `test_logprob_computation_with_action_mask` - Action masking isolates correct tokens
- ✅ `test_json_start_detection` - JSON start token detection works
- ✅ `test_action_mask_shape` - Masks have correct shape and values
- ✅ `test_teacher_forcing_consistency` - Input/label alignment is correct

**What it catches:**
- Off-by-one errors in token alignment
- Incorrect log softmax computation
- Misaligned logits and labels
- Action mask selecting wrong tokens

### 2. Advantage Computation (`test_advantages.py`)
**Purpose:** Verify LOO baseline and gradient flow correctness.

**Tests:**
- ✅ `test_loo_baseline_manual_example` - Known rewards [1,2,3] → baselines [2.5, 2.0, 1.5]
- ✅ `test_loo_baseline_batch` - Batch LOO computation
- ✅ `test_advantage_normalized` - Normalization produces mean=0, std=1
- ✅ `test_positive_advantage_increases_logprob` - Positive advantage → increase logprob
- ✅ `test_negative_advantage_decreases_logprob` - Negative advantage → decrease logprob
- ✅ `test_gradient_magnitude_scales_with_advantage` - Larger advantage → larger gradient

**What it catches:**
- Incorrect LOO baseline computation
- Wrong gradient direction (critical bug!)
- Advantage normalization errors
- Numerical instability

### 3. Reference Model Behavior (`test_reference_model.py`)
**Purpose:** Ensure reference model stays frozen and computes correct KL.

**Tests:**
- ✅ `test_frozen_reference_matches_policy` - Fresh copy matches exactly
- ✅ `test_reference_differs_after_policy_update` - Reference stays frozen during training
- ✅ `test_no_gradients_in_reference` - No gradients flow into reference
- ✅ `test_ema_update` - EMA formula works correctly
- ✅ `test_kl_is_zero_when_models_match` - KL=0 for identical models

**What it catches:**
- Reference model accidentally updating
- Gradients leaking into frozen model
- Incorrect EMA update
- Wrong KL divergence computation

### 4. Integration Tests (`test_integration.py`)
**Purpose:** End-to-end validation with synthetic data.

**Tests:**
- ✅ `test_episode_creation` - Synthetic episodes are valid
- ✅ `test_invalid_json_detection` - Malformed JSON is caught
- ✅ `test_dataloader_with_toy_episodes` - Batching works
- ✅ `test_episode_filtering` - Invalid episodes filtered correctly
- ✅ `test_loss_components` - All loss components computed
- ✅ `test_reward_statistics` - Metrics computed correctly

**What it catches:**
- Episode structure errors
- JSON parsing bugs
- Dataloader issues
- Metrics computation errors

## Usage

### Quick Start (Recommended)

Run quick tests (no model loading, ~10 seconds):
```bash
python tests/run_tests.py --mode quick
```

Or use the bash script:
```bash
./validate.sh
```

### Full Test Suite

Run all tests including model loading (~2-5 minutes):
```bash
python tests/run_tests.py --include-slow
```

### Interactive Workflow

Run the interactive validation workflow:
```bash
python tests/run_validation_workflow.py
```

This will:
1. Run quick tests first
2. Ask if you want to run full tests (with model download)
3. Show next steps for training

### Specific Test Files

Run individual test files:
```bash
# Alignment tests
pytest tests/test_alignment.py -v -s

# Advantage tests
pytest tests/test_advantages.py -v -s

# Reference model tests
pytest tests/test_reference_model.py -v -s

# Integration tests
pytest tests/test_integration.py -v -s
```

### Run Specific Test

```bash
pytest tests/test_advantages.py::TestLOOBaseline::test_loo_baseline_manual_example -v
```

## Test Modes

The test runner supports multiple modes:

```bash
# All tests (default)
python tests/run_tests.py

# Quick tests only (fast, no model loading)
python tests/run_tests.py --mode quick

# Alignment tests only
python tests/run_tests.py --mode alignment

# With coverage report
python tests/run_tests.py --mode coverage

# Stop on first failure
python tests/run_tests.py --stop-on-fail

# Include slow tests (model loading)
python tests/run_tests.py --include-slow
```

## Expected Output

When all tests pass:

```
================================================================================
RUNNING COMPREHENSIVE TEST SUITE
================================================================================

Test directory: /path/to/tests
Test files: 4
Arguments: -v -m not slow ...

tests/test_alignment.py::TestForwardPassAlignment::test_logprob_computation_vs_manual_nll PASSED
tests/test_alignment.py::TestActionMaskCorrectness::test_json_start_detection PASSED
tests/test_advantages.py::TestLOOBaseline::test_loo_baseline_manual_example PASSED
tests/test_advantages.py::TestGradientSanity::test_positive_advantage_increases_logprob PASSED
...

================================ 35 passed in 12.34s =================================

================================================================================
✓ ALL TESTS PASSED
================================================================================

Your RL training pipeline is ready!

Next steps:
1. Collect real episodes with your simulator
2. Create a config file for your experiment
3. Run training with: python train.py --config your_config.yaml
```

## What Each Test Validates

### Critical Tests (Must Pass Before Training)

1. **Forward Pass Alignment** - Ensures log probabilities are computed correctly
   - **Why critical:** Wrong log probs → wrong gradients → model won't learn
   
2. **LOO Baseline** - Ensures advantages are computed correctly
   - **Why critical:** Wrong advantages → wrong learning signal → poor performance
   
3. **Gradient Direction** - Ensures gradients increase good actions
   - **Why critical:** Wrong gradient sign → model learns opposite of what you want!
   
4. **Reference Model Frozen** - Ensures reference doesn't update
   - **Why critical:** Updating reference → KL penalty becomes meaningless

### Important Tests (Highly Recommended)

5. **Action Masking** - Ensures only JSON tokens are supervised
   - **Why important:** Wrong mask → training on context tokens → poor efficiency
   
6. **Teacher Forcing Alignment** - Ensures input/label alignment
   - **Why important:** Misalignment → training on wrong tokens → degraded performance
   
7. **JSON Validation** - Ensures invalid episodes are filtered
   - **Why important:** Training on garbage → model learns to output garbage

## Common Issues and Solutions

### Issue: Test fails with "Module not found"
**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: Test fails with "CUDA out of memory"
**Solution:** Tests use CPU by default. Check test fixtures use `device_map="cpu"`

### Issue: Forward pass alignment test fails
**Problem:** Log probability computation is wrong
**Solution:** Check token alignment, logits shifting, log softmax

### Issue: LOO baseline test fails
**Problem:** Advantage computation is wrong
**Solution:** Verify baseline excludes correct episode, check mean computation

### Issue: Gradient sanity test fails
**Problem:** Gradients flowing in wrong direction
**Solution:** Check loss sign, advantage sign, ensure using gradient descent correctly

## Integration with Training

The test suite is designed to catch bugs before they cause training failures:

```python
# Before training
python tests/run_tests.py  # Must pass!

# Create config
python -c 'from config import get_small_scale_config; config = get_small_scale_config(); config.to_yaml("config.yaml")'

# Run training
python train.py --config config.yaml
```

## Continuous Integration

To run in CI/CD:

```yaml
# .github/workflows/tests.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v -m "not slow"
```

## Test Coverage

The test suite covers:
- ✅ Token log probability computation
- ✅ Action mask creation
- ✅ Teacher forcing alignment
- ✅ LOO baseline computation
- ✅ Advantage normalization
- ✅ Gradient flow direction
- ✅ Gradient magnitude scaling
- ✅ Reference model freezing
- ✅ EMA update strategy
- ✅ KL divergence computation
- ✅ Episode creation and validation
- ✅ JSON parsing and filtering
- ✅ Dataloader batching
- ✅ Loss component computation
- ✅ Metrics logging

## Next Steps After Tests Pass

1. **Collect Episodes**
   ```bash
   python scripts/collect_episodes.py --config config.yaml
   ```

2. **Start Training**
   ```bash
   python train.py --config config.yaml
   ```

3. **Monitor Training**
   - View logs: `tail -f checkpoints/training.log`
   - WandB dashboard: https://wandb.ai/your-project
   - TensorBoard: `tensorboard --logdir checkpoints/`

4. **Evaluate Model**
   ```bash
   python scripts/evaluate.py --checkpoint checkpoints/best_model
   ```

## Summary

The test suite provides:
- ✅ **Confidence** - Know your pipeline works before expensive training
- ✅ **Speed** - Catch bugs in seconds, not hours
- ✅ **Documentation** - Tests show how components should work
- ✅ **Regression Prevention** - Future changes won't break working code
- ✅ **Debugging** - Failing tests pinpoint exact issue

**Always run tests before training!** It takes 10 seconds and saves hours of debugging.
