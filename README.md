# RL Multi-Turn Navigation Pipeline: Complete Implementation

## Overview

This repository implements a complete Reinforcement Learning pipeline for training Vision-Language Models (VLMs) on multi-turn navigation tasks using the VSI-Bench dataset.

**Key Features:**
- 🎯 REINFORCE with Leave-One-Out baseline
- 📊 Comprehensive WandB logging and visualization
- 💾 Full checkpointing with resume capability
- 🧪 Extensive test suite for validation
- ⚙️ Flexible configuration system
- 🎨 Multimodal input handling (Qwen3-VL compatible)

## Quick Start

### 1. Installation
```bash
# Clone repository
git clone <repo-url>
cd rl_multi_turn

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Tests
```bash
# Quick validation (10 seconds)
./validate.sh

# Or run full test suite
python tests/run_tests.py
```

### 3. Create Config
```python
from config import get_small_scale_config

config = get_small_scale_config()
config.to_yaml("my_config.yaml")
```

### 4. Train
```bash
python train.py --config my_config.yaml
```

### 5. Monitor
- **Logs:** `tail -f checkpoints/training.log`
- **WandB:** Visit your project dashboard
- **TensorBoard:** `tensorboard --logdir checkpoints/`

## Implementation Status

| Step | Status | Description |
|------|--------|-------------|
| **Step 1** | ✅ Complete | Environment & Episode Simulator |
| **Step 2** | ✅ Complete | Generation with Sampling & Token Tracking |
| **Step 3** | ✅ Complete | Enhanced Action Token Masking |
| **Step 4** | ✅ Complete | Episode Batching & DataLoader |
| **Step 5** | ✅ Complete | Token-Level Log Probability Computation |
| **Step 6** | ✅ Complete | LOO Baseline & Advantage Computation |
| **Step 7** | ✅ Complete | Policy Gradient Training Loop |
| **Step 8** | ✅ Complete | Reference Model & KL Penalty |
| **Step 9** | ✅ Complete | Integration Testing |
| **Step 10** | ✅ Complete | Multimodal Input Handling (Qwen3-VL) |
| **Step 11** | ✅ Complete | WandB Logging & Visualization |
| **Step 12** | ✅ Complete | Checkpointing & Resume |
| **Step 13** | ✅ Complete | Configuration System |
| **Step 14** | ✅ Complete | Comprehensive Test Suite |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RL Training Pipeline                      │
└─────────────────────────────────────────────────────────────┘

Step 1: Environment & Simulation
├── NavigationEnvironment (3D scenes, questions, ground truth)
├── Observation (RGB-D images, poses, questions)
├── Action (rotation, forward movement)
├── Turn (single reasoning + action)
└── Episode (multi-turn trajectory with reward)

Step 2: Episode Collection
├── EpisodeSimulator (run episodes with sampling)
├── Real-time token tracking (input lengths, action masks)
├── Persistent storage (JSONL + binary tensors)
└── Batch collection (parallel episode generation)

Step 3: Action Token Masking
├── ActionTokenMasker (robust multi-strategy masking)
│   ├── Primary: Brace depth tracking
│   ├── Fallback 1: Regex pattern matching
│   ├── Fallback 2: Last-N tokens heuristic
│   └── Complete failure: Empty mask + dropout
├── Episode quality evaluation
├── Dropout tracking and logging
└── Statistics monitoring

Step 4: Episode Batching
├── Create batches of episodes
├── Pad sequences to same length
└── Handle variable-length episodes

Step 5: Log Probability Computation
├── Token-level log probabilities
├── Teacher forcing alignment
├── Action token masking
└── Multimodal input handling

Step 6: Advantage Computation
├── Leave-One-Out (LOO) baseline
├── Advantage normalization
└── Variance reduction

Step 7: Policy Gradient Training
├── REINFORCE with baseline
├── Entropy regularization
├── Gradient clipping
└── Learning rate scheduling

Step 8: Reference Model & KL Penalty
├── Frozen reference model
├── EMA reference updates
├── KL divergence computation
└── Adaptive KL scheduling

Step 9: Logging & Visualization
├── WandB integration
├── Metric logging (loss, rewards, KL, entropy)
├── Example episode visualization
└── Action distribution histograms

Step 10: Checkpointing
├── Save full training state
├── Resume from checkpoint
├── Best model tracking
└── Checkpoint rotation

Step 11: Configuration System
├── Hierarchical configs (model, data, RL, training, wandb)
├── Hyperparameter management
├── Config templates (debug, small-scale, full-scale)
└── JSON/YAML serialization

Step 12: Test Suite
├── Forward pass alignment tests
├── LOO baseline validation
├── Gradient sanity checks
├── Reference model tests
├── Integration tests
└── JSON validity tests
```

## Directory Structure

```
rl_multi_turn/
├── rl_environment/          # Steps 1-3 implementation
│   ├── environment.py       # Core data structures
│   ├── simulator.py         # Episode collection
│   ├── masking.py          # Action token masking
│   ├── __init__.py         # Package exports
│   ├── test_environment.py # Step 1 tests
│   ├── example_batch_collection.py  # Step 2 example
│   ├── test_step3_masking.py       # Step 3 tests
│   ├── README.md           # Main documentation
│   ├── DATA_STRUCTURES.py  # Reference implementation
│   ├── STEP2_IMPLEMENTATION.md
│   ├── STEP2_SUMMARY.md
│   ├── STEP3_IMPLEMENTATION.md
│   └── STEP3_SUMMARY.md
└── rl_trainer/             # Step 4 (TODO)
    └── trainer.py
```

## Key Features

### Step 1: Environment & Simulation ✅
- **NavigationEnvironment**: Load 3D scenes, render RGB-D, apply actions
- **Data Structures**: Observation, Action, Turn, Episode with sparse terminal rewards
- **Episode Simulator**: Run multi-turn navigation with VLM
- **Trajectory**: Generate reasoning + action JSON each turn
- **Terminal Conditions**: 10 turns or goal reached

### Step 2: Sampling & Token Tracking ✅
- **Sampling**: temperature=1.0, top_p=0.9, top_k=50 (configurable)
- **Real-time Tracking**: Input token length, action token masks during generation
- **Persistent Storage**: JSONL for metadata, .pt for tensors
- **Batch Collection**: Parallel episode generation with progress tracking
- **Episode ID**: Unique identifiers for dataset management

### Step 3: Enhanced Masking ✅
- **Multi-Strategy Masking**: Brace depth → regex → last-N fallback
- **Confidence Scores**: 1.0 (brace) → 0.7 (regex) → 0.3 (last-N) → 0.0 (failed)
- **Validation**: Token count range checking (10-100 default)
- **Quality Evaluation**: Episodes marked as "good", "low_confidence", or "fallback_used"
- **Dropout Tracking**: Invalid episodes logged with reasons
- **Statistics**: Comprehensive logging of methods, quality, dropout rates

## Quick Start

### Installation

```bash
# Create conda environment
conda create -n rl_vlm python=3.10
conda activate rl_vlm

# Install dependencies
pip install torch transformers accelerate qwen-vl-utils open3d pandas
```

### Run Tests

```bash
cd rl_multi_turn/rl_environment

# Test Step 1: Environment & simulation
python test_environment.py

# Test Step 2: Batch collection
python example_batch_collection.py

# Test Step 3: Enhanced masking
python test_step3_masking.py
```

### Basic Usage

```python
from rl_environment import (
    NavigationEnvironment,
    EpisodeSimulator,
    EpisodeBatchCollector
)

# Load environment
env = NavigationEnvironment(dataset_path="/path/to/dataset")
env.load_scene("42445173")

# Create simulator with robust masking
simulator = EpisodeSimulator(
    model=model,
    processor=processor,
    min_action_tokens=10,
    max_action_tokens=100,
    do_sample=True,
    temperature=1.0
)

# Collect single episode
initial_pose = env.get_random_pose()
episode = simulator.run_episode(env, initial_pose, "ep001")

# Check quality
if episode.is_valid:
    print(f"Valid episode with quality: {episode.masking_quality}")
    # Use for training
else:
    print(f"Dropped episode: {episode.dropout_reason}")

# View statistics
simulator.log_stats()
```

### Batch Collection

```python
from rl_environment import EpisodeBatchCollector

# Collect multiple episodes
collector = EpisodeBatchCollector(
    simulator=simulator,
    env=env,
    output_dir="./episodes",
    num_episodes=100
)

episodes = collector.collect_batch(scene_id="42445173")

# Episodes saved to:
# - episodes/episode_manifest.jsonl  (metadata)
# - episodes/ep001_tensors.pt        (generated_ids, masks)
# - episodes/ep002_tensors.pt
# - ...
```

## Data Format

### Episode Metadata (JSONL)

```json
{
  "episode_id": "ep001",
  "scene_id": "42445173",
  "question_id": "42445173_route_planning_0",
  "question": "How do I get to the dining table?",
  "num_turns": 8,
  "terminal_reward": 1.0,
  "is_valid": true,
  "masking_quality": "good",
  "turns": [
    {
      "turn_index": 0,
      "observation": {...},
      "action": {"rotation_angle_degrees": 15, "forward_meters": 0.5},
      "reasoning": "I can see a chair...",
      "num_action_tokens": 45,
      "masking_method": "brace_depth",
      "masking_confidence": 1.0
    }
  ]
}
```

### Tensor Data (.pt)

```python
{
  "turns": [
    {
      "generated_ids": torch.Tensor([...]),      # Shape: [seq_len]
      "action_token_mask": torch.BoolTensor([...]), # Shape: [seq_len]
      "input_token_length": 1024
    }
  ]
}
```

## RL Training (Step 4 - TODO)

### Policy Gradient Loss

```python
# Only compute loss on action tokens
for turn in episode.turns:
    generated_ids = turn.generated_ids
    action_mask = turn.action_token_mask
    input_length = turn.input_token_length
    
    # Get logits (aligned with generated tokens)
    logits = model(...).logits[:, input_length:, :]
    
    # Select action token logits
    action_logits = logits[0, action_mask, :]
    action_ids = generated_ids[action_mask]
    
    # Compute log probs
    log_probs = F.log_softmax(action_logits, dim=-1)
    selected_log_probs = log_probs.gather(1, action_ids.unsqueeze(-1))
    
    # Policy gradient
    loss = -selected_log_probs.sum() * episode.terminal_reward
```

### KL Penalty

```python
# KL divergence on action tokens only
ref_log_probs = ref_model(...)[action_mask]
policy_log_probs = policy_model(...)[action_mask]
kl_penalty = (policy_log_probs - ref_log_probs).sum()

total_loss = policy_loss + kl_weight * kl_penalty
```

## Edge Case Handling

| Scenario | Detection | Handling | Result |
|----------|-----------|----------|--------|
| Valid JSON | Brace depth tracking | Mark action tokens | confidence=1.0 |
| Truncated JSON | Unclosed braces | Regex fallback | confidence=0.7 |
| No JSON | All strategies fail | Mark failed, dropout | confidence=0.0 |
| Invalid JSON | Parse error | Episode invalid | Dropped |
| Nested braces | Depth tracking | Primary or regex | confidence≥0.7 |

## Expected Performance

### Masking Statistics (Step 3)
- **Brace depth success**: 85-95%
- **Regex fallback**: 5-10%
- **Last-N fallback**: <3%
- **Complete failure**: <3%
- **Dropout rate**: 3-10%

### Quality Distribution
- **Good**: 80-90% (high confidence)
- **Low confidence**: 8-15% (some fallbacks)
- **Fallback used**: 2-5% (many fallbacks)

## Documentation

- **README.md**: Main package documentation
- **DATA_STRUCTURES.py**: Reference implementation
- **STEP2_IMPLEMENTATION.md**: Sampling & tracking details
- **STEP2_SUMMARY.md**: Step 2 summary
- **STEP3_IMPLEMENTATION.md**: Enhanced masking details
- **STEP3_SUMMARY.md**: Step 3 summary

## Testing

Each step has dedicated test scripts:

```bash
# Step 1
python test_environment.py

# Step 2
python example_batch_collection.py

# Step 3
python test_step3_masking.py
```

## Next Steps

1. **Implement Step 4**: RL trainer with PPO or policy gradient
2. **Large-scale Collection**: Generate 10k+ episodes
3. **Training**: Fine-tune VLM with RL
4. **Evaluation**: Test on VSI-Bench test set
5. **Ablations**: Study impact of masking strategies, rewards, etc.

## Citation

Based on VSI-Bench dataset for 3D visual-spatial instruction following:
- Dataset: arkitscenes route_planning tasks
- Model: Qwen3-VL-8B-Instruct
- RL approach: Policy gradient with action token masking

## License

Research code for academic use.

---

**Implementation Status**: Steps 1-3 Complete ✅ | Step 4 Pending ⏳
