# SimulatorConfig Migration Summary

## What Changed

The `EpisodeSimulator` initialization arguments have been moved to a centralized `SimulatorConfig` dataclass in `config.py`. This provides better configuration management, easier serialization, and cleaner code organization.

## Changes Made

### 1. New `SimulatorConfig` in `config.py`

Added a new configuration dataclass that contains all simulator parameters:

```python
@dataclass
class SimulatorConfig:
    """Episode simulator settings."""
    # Episode parameters
    max_steps: int = 2
    
    # Token tracking
    track_action_tokens: bool = True
    min_action_tokens: int = 10
    max_action_tokens: int = 100
    
    # Generation sampling
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
```

### 2. Updated `EpisodeSimulator.__init__()` in `simulator.py`

**Before:**
```python
def __init__(
    self,
    model,
    processor,
    device: str = "cuda",
    max_steps: int = 2,
    track_action_tokens: bool = True,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    min_action_tokens: int = 10,
    max_action_tokens: int = 100
):
    ...
```

**After:**
```python
def __init__(
    self,
    model,
    processor,
    config: SimulatorConfig,
    device: str = "cuda"
):
    self.config = config
    # Extract config values for convenience
    self.max_steps = config.max_steps
    self.temperature = config.temperature
    # ... etc
```

### 3. Updated All Usage Sites

Updated the following files to use `SimulatorConfig`:
- `train_rl.py` - Main training script
- `rl_environment/example_batch_collection.py` - Example batch collection
- `rl_environment/test_environment.py` - Environment test
- `rl_environment/test_step3_masking.py` - Masking test

### 4. Integration with `RLTrainingConfig`

The `SimulatorConfig` is now part of the main config system:

```python
@dataclass
class RLTrainingConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)  # NEW!
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
```

## Usage Examples

### Example 1: Load from YAML
```python
from config import RLTrainingConfig

# Load full config including simulator settings
config = RLTrainingConfig.from_yaml("example_config.yaml")

# Create simulator with config
simulator = EpisodeSimulator(
    model=model,
    processor=processor,
    config=config.simulator,
    device=config.training.device
)
```

### Example 2: Create Programmatically
```python
from config import SimulatorConfig

# Create custom simulator config
sim_config = SimulatorConfig(
    max_steps=5,
    track_action_tokens=True,
    do_sample=True,
    temperature=0.8,
    top_p=0.95,
    top_k=40
)

# Use with simulator
simulator = EpisodeSimulator(
    model=model,
    processor=processor,
    config=sim_config,
    device="cuda"
)
```

### Example 3: Modify Existing Config
```python
# Load base config
config = RLTrainingConfig.from_yaml("example_config.yaml")

# Modify simulator settings
config.simulator.temperature = 0.7
config.simulator.max_steps = 3

# Save modified config
config.to_yaml("modified_config.yaml")
```

## Benefits

1. **Centralized Configuration**: All hyperparameters in one place
2. **Easy Serialization**: Save/load configs as YAML/JSON
3. **Version Control Friendly**: Track config changes in git
4. **Hyperparameter Sweeps**: Easy to generate multiple configs programmatically
5. **Cleaner API**: Fewer arguments to `EpisodeSimulator.__init__()`
6. **Type Safety**: Dataclass validation and type hints
7. **Documentation**: Self-documenting through config files

## Files Created

1. `example_config.yaml` - Complete example configuration file
2. `example_config_usage.py` - Usage examples and documentation
3. `SIMULATOR_CONFIG_MIGRATION.md` - This document

## Backward Compatibility

The old initialization style is **not** supported. All code must be updated to use `SimulatorConfig`. The changes are straightforward and improve code organization.

## Configuration File Structure

The YAML config now includes a `simulator` section:

```yaml
simulator:
  max_steps: 2
  track_action_tokens: true
  min_action_tokens: 10
  max_action_tokens: 100
  do_sample: true
  temperature: 1.0
  top_p: 0.9
  top_k: 50
```

This can be loaded and used directly:

```python
config = RLTrainingConfig.from_yaml("config.yaml")
simulator = EpisodeSimulator(model, processor, config.simulator, device)
```

## Next Steps

To use this in your training pipeline:

1. Create a YAML config file (or use `example_config.yaml` as template)
2. Load it: `config = RLTrainingConfig.from_yaml("your_config.yaml")`
3. Create simulator: `simulator = EpisodeSimulator(model, processor, config.simulator, device)`
4. All parameters are now managed through the config file!

## Questions?

See `example_config_usage.py` for more examples, or check the docstrings in `config.py` and `simulator.py`.
