# RL Multi-Turn View Selection - Architecture Explanation

## Overview

This is a **two-model RL training system** for training vision-language models to select better camera views:

```
┌─────────────────────────────────────────────────────────────┐
│                    RL Training Loop                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. ROLLOUT (Forward Pass) - vLLM                          │
│     ↓                                                       │
│  2. TRAINING (Backward Pass) - HuggingFace                 │
│     ↓                                                       │
│  3. WEIGHT SYNC - Copy HF weights → vLLM                   │
│     ↓                                                       │
│  Repeat                                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Why Two Models?

### Problem
- **vLLM**: Fast inference, but doesn't support gradients
- **HuggingFace**: Supports gradients, but slow for inference

### Solution
Use both:
- **Rollout (data collection)**: vLLM for speed
- **Training (gradient updates)**: HuggingFace for gradients
- **Synchronization**: Copy HF weights → vLLM after each update

## Architecture Components

### 1. **Rollout Engine** (`rollout.py`)

**Purpose**: Collect trajectories (episodes) using the current policy

**Process**:
```python
for turn in range(max_turns):
    # Generate action from current view
    prompt = build_prompt(question, images, turn)
    output = vllm.generate(prompt)  # Fast inference
    
    # Parse action (camera pose)
    action = parse_output(output)
    
    # Render new view (if scene loader enabled)
    new_image = render(action.camera_pose)
    images.append(new_image)
    
    # Store data: tokens, logprobs, action mask
    trajectory.add_turn(output, action)
```

**Key Features**:
- Uses vLLM for fast parallel inference
- Collects token IDs and log probabilities
- Masks action tokens (only these get gradients)
- No gradient computation during rollout

### 2. **Trainer** (`trainer.py`)

**Purpose**: Update policy using collected trajectories

**Process**:
```python
for trajectory in batch:
    for turn in trajectory.turns:
        # Get action tokens (masked)
        action_tokens = turn.tokens[turn.action_mask]
        action_logprobs = turn.logprobs[turn.action_mask]
        
        # Compute policy loss (REINFORCE)
        advantage = compute_advantage(trajectory.reward)
        loss = -(action_logprobs * advantage).mean()
        
        # Backprop and update
        loss.backward()
        optimizer.step()
```

**Key Features**:
- Only action tokens receive gradients
- Observation tokens (images, reasoning) are frozen
- REINFORCE algorithm (PPO optional)
- Advantage normalization

### 3. **Weight Synchronization**

**Two Options**:

**Option A (Checkpoint)** - Most Reliable:
```python
# After training step
trainer.save_hf_checkpoint("temp_checkpoint/")
rollout_engine.reload_from_checkpoint("temp_checkpoint/")
```

**Option B (Direct)** - Faster but experimental:
```python
# After training step  
state_dict = trainer.model.state_dict()
rollout_engine.vllm.update_weights(state_dict)
```

### 4. **Scene Loader** (`scene_loader.py`)

**Purpose**: Load and render 3D scenes

**Datasets**:
- **ScanNet**: Indoor scenes from RGB-D sensors
- **ScanNet++**: High-quality reconstructions
- **ARKitScenes**: iPhone LiDAR scans

**Process**:
```python
# Load scene
scene_loader.load_scene(scene_id, dataset="scannetpp")

# Select initial view (4 candidates, pick best)
pose = scene_loader.compute_initial_pose()

# Render from action
image = scene_loader.render_from_action(camera_pose)
```

## Data Flow

### 1. **Trajectory Collection** (Forward Pass)

```
Scene → Initial View → Question
                ↓
        ┌──────────────┐
        │  vLLM Model  │
        └──────────────┘
                ↓
    Generated Text + Tokens
                ↓
        Parse Action
                ↓
    Render New View → Add to History
                ↓
        Repeat for N turns
                ↓
    Final Answer → Trajectory
```

**Data Stored**:
- `Turn`: prompt, generated_text, token_ids, logprobs, action_mask
- `Trajectory`: all turns, question, final_answer, reward

### 2. **Training** (Backward Pass)

```
Trajectories (batch)
        ↓
Extract Action Tokens (masked)
        ↓
Compute Advantages (reward - baseline)
        ↓
Policy Loss = -log_prob * advantage
        ↓
Backpropagation
        ↓
Update Weights
```

**Gradient Flow**:
- ✅ Action tokens: Full gradients
- ❌ Reasoning tokens: No gradients
- ❌ Image tokens: No gradients

### 3. **Weight Sync**

```
HuggingFace Model (trained)
        ↓
Save to disk
        ↓
vLLM reloads from disk
        ↓
Next rollout uses updated policy
```

## Key Design Decisions

### 1. **Action-Token-Only Training**

**Why?**: We only want to train the policy over camera movements, not language.

**How?**: Mask all tokens except those inside `[ACTION]...[/ACTION]` markers.

### 2. **Turn-Level Structure**

**Why?**: Maintains compatibility with VAGEN framework.

**How?**: Each turn is a complete reasoning step with state/plan/action.

### 3. **Fixed Horizon (T turns)**

**Why?**: Simplifies credit assignment, easier to implement.

**How?**: Always take exactly T views, then answer.

### 4. **Binary Reward**

**Why?**: Simple baseline, interpretable.

**How?**: 
- `reward = 1` if final answer correct
- `reward = 0` if incorrect

## Configuration Options

### Essential:
- `model_id`: Path to Qwen3-VL-4B model
- `num_updates`: Number of policy updates
- `episodes_per_update`: Trajectories per update
- `learning_rate`: AdamW learning rate

### Weight Sync:
- `weight_sync`: "checkpoint" (reliable) or "direct" (fast)
- `weight_sync_interval`: How often to sync (default: every update)

### Scene Loading:
- `dataset`: "scannetpp", "scannet", "arkitscenes", or "combined"
- `use_scene_loader`: Enable 3D rendering

### Algorithm:
- `use_ppo`: Enable PPO clipping
- `kl_coef`: KL penalty coefficient
- `entropy_coef`: Entropy bonus

## Memory Requirements

**For Qwen3-VL-4B on A100-80GB**:

| Component | Memory |
|-----------|--------|
| vLLM model | ~9 GB |
| HF model | ~9 GB |
| Forward pass | ~1 GB |
| Backward pass (batch=4) | ~25 GB |
| **Total** | **~45 GB** |

Safe batch size: **4 episodes**

## Running the Test

```bash
# End-to-end test (1 update, 2 episodes)
./test_e2e.sh

# Or directly:
python rl_multiturn_v2/train_v2.py --config configs/train_rl_test.yaml
```

**What happens**:
1. ✅ Load Qwen3-VL-4B twice (vLLM + HuggingFace)
2. ✅ Collect 2 trajectories (3 turns each) using vLLM
3. ✅ Train on batch using HuggingFace model
4. ✅ Save HF checkpoint and reload vLLM
5. ✅ Ready for next update

## Code Structure

```
rl_multiturn_v2/
├── data_structures.py    # Turn, Trajectory, Action, CameraPose
├── output_parser.py      # Parse [ACTION]...[/ACTION] markers
├── rollout.py            # VLLMRolloutEngine (forward pass)
├── trainer.py            # RLTrainer (backward pass)
├── scene_loader.py       # 3D scene loading and rendering
├── train_v2.py           # Main training loop
└── load_config.py        # YAML config loading

configs/
└── train_rl_test.yaml    # Test configuration

utils/
├── mesh.py               # Mesh loading with caching
└── rendering.py          # Open3D headless rendering
```

## Next Steps

1. **Run test**: `./test_e2e.sh`
2. **Check logs**: Look for:
   - "Collecting trajectories..." (rollout)
   - "Policy loss: X.XXXX" (training)
   - "Syncing weights..." (weight sync)
3. **Scale up**: Increase `num_updates` and `episodes_per_update`
4. **Enable rendering**: Set `use_scene_loader: true`
5. **Tune rewards**: Replace binary reward with shaped rewards
