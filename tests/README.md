# Test Suite Documentation

## Overview

This test suite validates the entire RL training pipeline before running full training. It ensures correctness of:

1. **Forward pass alignment** - Token log probabilities match manual NLL computation
2. **Mask correctness** - Action masks correctly select JSON response tokens
3. **Reference model behavior** - Reference model stays frozen and computes correct KL divergence
4. **LOO baseline** - Leave-one-out baseline is computed correctly
5. **Gradient sanity** - Gradients flow in the correct direction
6. **End-to-end integration** - Complete pipeline works on synthetic data
7. **JSON validity** - Invalid JSON responses are handled correctly

## Quick Start

Run all tests:
```bash
cd /dss/dsshome1/06/di38riq/rl_multi_turn
python tests/run_tests.py
```

Run quick tests only (fast):
```bash
python tests/run_tests.py --mode quick
```

Run specific test category:
```bash
# Alignment tests
pytest tests/test_alignment.py -v

# Advantage and gradient tests
pytest tests/test_advantages.py -v

# Reference model tests
pytest tests/test_reference_model.py -v

# Integration tests
pytest tests/test_integration.py -v
```

## Test Files

### 1. `test_alignment.py`
Tests forward pass and token alignment correctness.

**Key Tests:**
- `test_logprob_computation_vs_manual_nll` - Verifies log prob computation matches manual NLL
- `test_logprob_computation_with_action_mask` - Verifies action masking isolates correct tokens
- `test_json_start_detection` - Tests JSON start token detection
- `test_teacher_forcing_consistency` - Verifies input/label alignment

**What it validates:**
- Token log probabilities are computed correctly
- Action masks select only JSON response tokens
- Teacher forcing inputs are properly aligned
- No off-by-one errors in logits-to-labels mapping

### 2. `test_advantages.py`
Tests advantage computation and gradient flow.

**Key Tests:**
- `test_loo_baseline_manual_example` - Validates LOO baseline with known values [1,2,3] → [2.5, 2.0, 1.5]
- `test_loo_baseline_batch` - Tests batch LOO computation
- `test_advantage_normalized` - Verifies normalization produces mean=0, std=1
- `test_positive_advantage_increases_logprob` - Tests gradient direction for positive advantages
- `test_negative_advantage_decreases_logprob` - Tests gradient direction for negative advantages

**What it validates:**
- LOO baseline correctly excludes each episode's own reward
- Advantage normalization is stable and correct
- Gradients point in the correct direction (increase good actions, decrease bad actions)
- Gradient magnitude scales with advantage magnitude

### 3. `test_reference_model.py`
Tests reference model behavior and KL divergence.

**Key Tests:**
- `test_frozen_reference_matches_policy` - Fresh reference copy matches policy exactly
- `test_reference_differs_after_policy_update` - Reference stays frozen during policy updates
- `test_no_gradients_in_reference` - No gradients flow into reference model
- `test_ema_update` - EMA update formula works correctly
- `test_kl_is_zero_when_models_match` - KL divergence is ~0 for identical models

**What it validates:**
- Reference model is truly frozen (no gradient flow)
- Reference model produces correct log probabilities
- EMA update strategy works as expected
- KL divergence computation is correct

### 4. `test_integration.py`
End-to-end integration tests with synthetic data.

**Key Tests:**
- `test_episode_creation` - Creates valid synthetic episodes
- `test_invalid_json_detection` - Detects malformed JSON responses
- `test_dataloader_with_toy_episodes` - Dataloader batching works
- `test_mini_training_run` - Complete training loop (marked as slow)
- `test_episode_filtering` - Filters out invalid episodes
- `test_loss_components` - Loss components are computed correctly

**What it validates:**
- Episode structure is correct
- JSON parsing and validation works
- Dataloader batching works
- Invalid episodes can be filtered
- All metrics are computed correctly

## Running Tests

### Run all tests (recommended before training)
```bash
python tests/run_tests.py
```

This runs all tests except those marked as `slow` (which require model loading).

### Run with slow tests (requires GPU/model access)
```bash
python tests/run_tests.py --include-slow
```

### Stop on first failure
```bash
python tests/run_tests.py --stop-on-fail
```

### Run with coverage report
```bash
python tests/run_tests.py --mode coverage
```

This generates an HTML coverage report in `htmlcov/index.html`.

### Run specific test class
```bash
pytest tests/test_advantages.py::TestLOOBaseline -v
```

### Run specific test method
```bash
pytest tests/test_advantages.py::TestLOOBaseline::test_loo_baseline_manual_example -v
```

## Test Markers

Tests are organized with markers:

- `@pytest.mark.slow` - Tests that require model loading (skipped by default)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.alignment` - Alignment tests
- `@pytest.mark.gradients` - Gradient tests

Skip slow tests:
```bash
pytest -m "not slow"
```

Run only unit tests:
```bash
pytest -m unit
```

## Expected Output

When all tests pass, you should see:

```
================================ test session starts =================================
collected 35 items

tests/test_alignment.py::TestForwardPassAlignment::test_logprob_computation_vs_manual_nll PASSED
tests/test_alignment.py::TestForwardPassAlignment::test_logprob_computation_with_action_mask PASSED
tests/test_alignment.py::TestActionMaskCorrectness::test_json_start_detection PASSED
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

## Troubleshooting

### Test fails: "Module not found"
Make sure you're in the project root and have installed dependencies:
```bash
pip install -r requirements.txt
```

### Test fails: "CUDA out of memory"
The tests use CPU by default. If you see this error, check that `device_map="cpu"` is set in test fixtures.

### Test fails: Forward pass alignment test
This indicates a bug in log probability computation. Check:
- Token alignment between input_ids and labels
- Logits shifting (logits[t] should predict labels[t])
- Log softmax computation

### Test fails: LOO baseline test
This indicates a bug in advantage computation. Verify:
- Baseline correctly excludes each episode's own reward
- Mean computation is correct
- Batch dimensions are correct

### Test fails: Gradient sanity test
This indicates gradients are flowing incorrectly. Check:
- Loss sign (should be negative of policy gradient for gradient descent)
- Advantage sign
- No accidental gradient blocking

## Adding New Tests

To add a new test:

1. Create a new test method in the appropriate test file:
```python
def test_my_new_feature(self):
    """Test description."""
    # Arrange
    input_data = ...
    
    # Act
    result = my_function(input_data)
    
    # Assert
    assert result == expected_value
```

2. Use descriptive test names: `test_<what>_<condition>`

3. Add docstrings explaining what the test validates

4. Use appropriate assertions:
   - `assert x == y` for exact equality
   - `torch.testing.assert_close()` for tensor equality with tolerance
   - `pytest.approx()` for floating point comparison
   - `pytest.raises()` for exception testing

5. Mark slow tests:
```python
@pytest.mark.slow
def test_expensive_operation(self):
    ...
```

## CI/CD Integration

To run tests in CI:

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
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v -m "not slow"
```

## Additional Resources

- [Pytest documentation](https://docs.pytest.org/)
- [Pytest fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Pytest markers](https://docs.pytest.org/en/stable/mark.html)
- [Testing PyTorch models](https://pytorch.org/docs/stable/testing.html)
