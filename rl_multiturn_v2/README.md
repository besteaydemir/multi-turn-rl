# RL Multi-Turn View Selection v2

**The simplest correct RL implementation for multi-turn view selection with a VLM.**

This is a minimal, stable implementation designed to be extended toward the full VAGEN framework.

## Overview

This implementation provides:
- **Fixed T-turn episodes** with multi-view observation
- **Bracketed marker output format**: `[STATE]`, `[PLAN]`, `[PREDICT]`, `[ACTION]`, `[FINAL_ANSWER]`
- **vLLM rollout** (inference-only, no gradients)
- **REINFORCE/PPO training** with action-token-only optimization
- **Zero rewards** (placeholder - tests pipeline correctness)
- **Three-level logging** to Weights & Biases

## Architecture

```
┌─────────────────┐
│ VLLMRolloutEngine│  ← vLLM for fast inference (no gradients)
└────────┬────────┘
         │ Trajectories
         ↓
┌─────────────────┐
│   RLTrainer     │  ← REINFORCE/PPO on action tokens only
└────────┬────────┘
         │ Updated weights
         ↓
┌─────────────────┐
│  HF Model       │  ← Trainable VLM (frozen vision encoder)
└─────────────────┘
```

## Components

### 1. Data Structures ([data_structures.py](data_structures.py))
- `CameraPose`: Camera transform (4x4 matrix or position+rotation)
- `Action`: Parsed camera action from `[ACTION]` block
- `Turn`: Complete turn data with tokens, logprobs, action mask
- `Trajectory`: Full episode with all turns and metadata

### 2. Output Parser ([output_parser.py](output_parser.py))
- Parses bracketed format: `[STATE]`, `[PLAN]`, `[PREDICT]`, `[ACTION]`, `[FINAL_ANSWER]`
- `ActionTokenMasker`: Maps action JSON to token positions
- Prompt templates for each turn

### 3. Rollout Engine ([rollout.py](rollout.py))
- `VLLMRolloutEngine`: Fast trajectory collection with vLLM
- `MockRolloutEngine`: Testing without GPU
- No gradient computation during rollout

### 4. Trainer ([trainer.py](trainer.py))
- `RLTrainer`: REINFORCE/PPO policy gradient
- Action-token-only loss (only JSON tokens contribute)
- Optional: KL penalty, entropy bonus, PPO clipping
- `OnlineRLTrainer`: Full training loop (rollout → train → repeat)

### 5. Logging ([logging_utils.py](logging_utils.py))
Three logging levels:
- **Per-Turn**: Token counts, action JSON, parse success
- **Per-Trajectory**: Number of views, final answer, correctness
- **Per-Update**: Policy loss, KL divergence, entropy, accuracy

Logs to both Weights & Biases and local JSONL files.

## Usage

### Testing

```bash
# Test all components (no GPU needed)
python test_pipeline.py
```

### Training with Mock Rollout

```bash
# Quick test of training loop
python train_v2.py --mock --num_updates 10 --no_wandb
```

### Full Training

```bash
# Train with real model
python train_v2.py \
  --model_id Qwen/Qwen3-VL-4B-Instruct \
  --num_updates 100 \
  --episodes_per_update 4 \
  --learning_rate 1e-5 \
  --max_turns 5 \
  --wandb_project rl-view-selection
```

### Key Arguments

**Episode Settings:**
- `--max_turns`: Fixed number of views per episode (default: 5)
- `--max_new_tokens`: Maximum tokens per turn (default: 512)

**Training:**
- `--num_updates`: Number of policy updates (default: 100)
- `--episodes_per_update`: Trajectories per update (default: 4)
- `--learning_rate`: AdamW learning rate (default: 1e-5)
- `--batch_size`: Training batch size (default: 4)

**Algorithm:**
- `--use_ppo`: Use PPO instead of REINFORCE
- `--kl_coef`: KL penalty coefficient (default: 0.0)
- `--entropy_coef`: Entropy bonus coefficient (default: 0.0)

**Output:**
- `--output_dir`: Checkpoint and log directory
- `--wandb_project`: W&B project name
- `--no_wandb`: Disable W&B logging

**Development:**
- `--mock`: Use mock rollout (no GPU)
- `--debug`: Enable verbose logging
- `--seed`: Random seed (default: 42)

## Output Format

The model generates responses in this format:

```
[STATE]
Description of what is visible in the current view(s).
The scene appears to show an indoor environment...

[PLAN]
Reasoning about what information is still needed.
To answer the question, I need to see...

[PREDICT]
Prediction of what the next view will reveal.
If I move the camera to the left, I expect to see...

[ACTION]
{
  "camera_pose": [
    [1.0, 0.0, 0.0, 2.5],
    [0.0, 1.0, 0.0, 1.5],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
  ],
  "fov": 60.0
}

[FINAL_ANSWER]  (last turn only)
Based on all views, the answer is: A
```

## Extension Points (for VAGEN)

The implementation is designed to be extended without refactoring:

1. **Rewards**: Currently all zero. Replace with:
   - Intermediate rewards for information gain
   - Terminal reward for correctness

2. **Turn-level Advantages**: Currently simple returns. Add:
   - GAE (Generalized Advantage Estimation)
   - Critic network for value estimates

3. **Multi-level Optimization**: Currently action tokens only. Add:
   - Separate policies for planning vs execution
   - Hierarchical value functions

4. **Rendering**: Currently mocked. Integrate:
   - Real 3D rendering engine
   - Camera trajectory execution

## File Structure

```
rl_multiturn_v2/
├── __init__.py              # Package exports
├── data_structures.py       # Core data types
├── output_parser.py         # Bracketed format parser
├── rollout.py              # vLLM trajectory collection
├── trainer.py              # REINFORCE/PPO training
├── logging_utils.py        # W&B and file logging
├── train_v2.py             # Main training script
├── test_pipeline.py        # Unit tests
└── README.md               # This file
```

## Design Principles

1. **Minimal**: Only essential components
2. **Stable**: No moving parts, clear interfaces
3. **Testable**: Mock components for fast iteration
4. **Extensible**: Clear extension points for VAGEN
5. **Correct**: Zero rewards test pipeline correctness

## Current Limitations

- **Rewards**: All zero (placeholder)
- **Rendering**: Uses mock images
- **Critic**: Not implemented (optional)
- **Reference model**: Not implemented (optional for KL)

These are intentional - they can be added without modifying existing code.

## Testing Strategy

All tests pass without GPU:
- Data structure serialization
- Output parsing with various formats
- Action token masking
- Mock rollout engine
- Advantage computation
- Trainer initialization
- Logging infrastructure
- End-to-end mock training

Run tests: `python test_pipeline.py`

## Performance

**Rollout (vLLM):**
- Fast inference without gradient computation
- Batch processing for efficiency
- Automatic tensor management

**Training:**
- Action-token-only loss reduces memory
- Gradient accumulation for large batches
- Optional PPO clipping for stability

## Citation

This implementation follows the specification from the VAGEN project:
"The simplest correct RL implementation for multi-turn view selection with a VLM."
