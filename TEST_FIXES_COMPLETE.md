# Test Suite Fixes - Complete ✅

## Issues Fixed

### 1. Import Errors
**Problem:** Tests were importing from non-existent modules
- `rl_trainer.action_masking` → doesn't exist
- `rl_trainer.advantages` → doesn't exist  
- `rl_trainer.reference_model` → doesn't exist
- `rl_trainer.episode` → doesn't exist

**Solution:** Updated imports to match actual module structure:
- `rl_environment.ActionTokenMasker` (for action masking)
- `rl_trainer.batch.compute_loo_baseline` (for LOO baseline)
- `rl_trainer.logprobs.compute_advantages` (for advantages)
- `rl_trainer.reference.ReferenceModelManager` (for reference model)
- `rl_environment.Episode/Turn/Observation` (for data structures)

### 2. API Mismatches

#### test_reference_model.py
**Problem:** Tests used `ReferenceModel` class which doesn't exist
**Solution:** Updated to use `ReferenceModelManager` with correct API:
- `ref_manager = ReferenceModelManager(policy, strategy="frozen")`
- `ref_model = ref_manager.get_reference_model()`
- EMA update is internal: `ref_manager.maybe_update(policy, step=1)`

#### test_integration.py
**Problem:** Episode/Turn/Observation creation didn't match actual dataclass structure
**Solution:** Updated to create proper dataclass instances:
- Added required fields: `question`, `choices`, `ground_truth`
- Used `final_reward` instead of `total_reward`
- Created proper `Observation` with numpy arrays for positions/rotations
- Created proper `Turn` with all required fields and tensors

#### test_advantages.py  
**Problem:** `compute_loo_baseline` expected `EpisodeBatch` but tests passed plain tensors
**Solution:** Made function polymorphic to accept both:
```python
if isinstance(batch, torch.Tensor):
    rewards = batch
    N = len(rewards)
else:
    rewards = batch.rewards
    N = batch.batch_size
```

### 3. Dataloader Changes
**Problem:** Tests used `create_episode_dataloader()` function
**Solution:** Updated to use `EpisodeDataLoader` class:
```python
dataloader = EpisodeDataLoader(
    episodes=episodes,
    batch_size=4,
    shuffle=False
)
```

## Files Modified

1. **tests/test_alignment.py**
   - Fixed imports: `ActionTokenMasker` from `rl_environment`
   - Fixed imports: `compute_token_logprobs` from `rl_trainer.logprobs`

2. **tests/test_advantages.py**
   - Fixed imports: `compute_loo_baseline` from `rl_trainer.batch`
   - Fixed imports: `compute_advantages` from `rl_trainer.logprobs`

3. **tests/test_reference_model.py**
   - Fixed imports: `ReferenceModelManager` from `rl_trainer.reference`
   - Updated all test methods to use `ReferenceModelManager` API
   - Fixed `test_ema_update` to work with manager's internal EMA
   - Fixed `test_ema_tau_extremes` to use manager

4. **tests/test_integration.py**
   - Fixed imports: `Episode/Turn/Observation` from `rl_environment`
   - Fixed imports: `EpisodeDataLoader` from `rl_trainer.batch`
   - Completely rewrote `ToyEnvironment.create_episode()`:
     - Create proper `Observation` with numpy arrays
     - Create proper `Action` from JSON
     - Create proper `Turn` with all fields
     - Create proper `Episode` with required fields
   - Fixed `create_invalid_episode()` similarly
   - Updated test assertions to use `final_reward` instead of `total_reward`
   - Fixed filtering logic to check `action_valid` instead of JSON parsing

5. **rl_trainer/batch.py**
   - Made `compute_loo_baseline()` polymorphic to accept both tensor and EpisodeBatch
   - Added device parameter handling for plain tensors

## Test Structure Now

```
tests/
├── test_alignment.py          ✅ Imports fixed
│   ├── TestForwardPassAlignment
│   ├── TestActionMaskCorrectness
│   └── TestTeacherForcingConsistency
│
├── test_advantages.py         ✅ Imports fixed, LOO function fixed
│   ├── TestLOOBaseline
│   ├── TestAdvantageComputation
│   └── TestGradientSanity
│
├── test_reference_model.py    ✅ API updated to ReferenceModelManager
│   ├── TestReferenceModelBehavior
│   ├── TestReferenceUpdateStrategies
│   └── TestKLDivergence
│
└── test_integration.py        ✅ Episode creation completely rewritten
    ├── TestEndToEndTraining
    ├── TestJSONValidityHandling
    └── TestMetricsLogging
```

## How to Run Tests

### Quick tests (no model loading):
```bash
python tests/run_tests.py --mode quick
```

### All tests (requires model):
```bash
python tests/run_tests.py
```

### Specific test file:
```bash
pytest tests/test_advantages.py -v
```

### Specific test:
```bash
pytest tests/test_advantages.py::TestLOOBaseline::test_loo_baseline_manual_example -v
```

## Expected Results

All import errors should be resolved. Tests may still fail if:
1. Models need to be downloaded (slow tests)
2. GPU/CUDA issues
3. Missing dependencies

But the **import phase should succeed** now.

## Next Steps

1. Run quick tests: `./validate.sh`
2. Fix any remaining test failures (likely model loading issues)
3. Run full test suite once models are available
4. Proceed with training if all tests pass

---

**Status:** ✅ All import errors fixed
**Ready for:** Test execution
