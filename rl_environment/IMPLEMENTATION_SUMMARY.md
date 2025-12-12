# RL Environment Implementation Summary

## What Was Created

A complete environment and episode simulator for multi-turn reinforcement learning on the VSI-Bench navigation pipeline.

## File Structure

```
rl_multi_turn/
├── render_point_cloud_qwen_angle.py    # [UNCHANGED] Baseline pipeline
└── rl_environment/                      # [NEW] RL components
    ├── __init__.py                      # Package exports
    ├── environment.py                   # Environment & data structures
    ├── simulator.py                     # Episode simulation
    ├── test_environment.py              # Test script
    └── README.md                        # Documentation
```

## Key Components

### 1. Data Structures (environment.py)

#### `Observation`
- Complete state at timestep t
- Contains: images, camera poses, question, choices, movement history
- JSON-serializable for storage

#### `Action`
- Parsed JSON action (no reasoning text)
- Fields: `rotation_angle_degrees`, `forward_meters`, `left_meters`, `z_delta_meters`, `answer`, `done`
- Includes validation logic (bounds checking, consistency)

#### `Turn`
- Complete data for one turn
- **Critical fields for RL**:
  - `generated_ids`: Token IDs (Tensor)
  - `action_token_mask`: Boolean mask marking action tokens (Tensor)
  - `observation`: State at turn start
  - `action`: Parsed action
  - `next_observation`: State after action

#### `Episode`
- Complete trajectory
- Contains: list of turns, final reward, metadata
- Methods: `compute_final_reward()`, `save_full()`

### 2. Environment (environment.py)

#### `NavigationEnvironment`
- Manages 3D scene state
- **API**:
  - `reset(initial_pose) -> Observation`
  - `step(action) -> (Observation, done)`
- Handles rendering, camera updates, state tracking
- Reuses baseline pipeline functions (no modifications needed)

### 3. Simulator (simulator.py)

#### `EpisodeSimulator`
- Orchestrates complete episodes
- **API**:
  - `run_episode(env, initial_pose, episode_id) -> Episode`
- Handles:
  - Model generation with token tracking
  - Action parsing and validation
  - Turn-by-turn execution
  - Data collection and storage

#### Token Tracking (`_identify_action_tokens`)
- **Problem**: Need to identify which tokens are action (JSON) vs reasoning
- **Solution**: 
  1. Find JSON in decoded text (regex)
  2. Map character positions to token positions
  3. Create boolean mask over `generated_ids`
- **Result**: `action_token_mask[i] = True` if token i is part of JSON

## Episode Data Format

Each episode saves:

```
output_dir/
├── episode_metadata.json       # Episode summary
├── turn_00/
│   ├── turn_metadata.json      # Turn info
│   ├── generated_ids.pt        # Tokens (Tensor)
│   ├── action_token_mask.pt    # Action mask (Tensor)
│   ├── full_prompt.txt         # Complete prompt
│   ├── context_text.txt        # Instruction
│   ├── generated_text.txt      # Model output
│   └── messages.json           # Processor input
├── turn_01/
└── env_renders/
    ├── render_00.png
    └── ...
```

## How It Works

### Episode Execution Flow

```
1. Initialize environment: env.reset(initial_pose)
   └─> Returns: obs_0

2. For each turn t:
   a. Build prompt from obs_t (with all images so far)
   b. Generate: model(prompt) -> (generated_ids, generated_text)
   c. Track action tokens: create action_token_mask
   d. Parse action from generated_text
   e. Validate action
   f. Execute: env.step(action) -> (obs_{t+1}, done)
   g. Record Turn with all data
   h. If done or max_steps: break

3. Compute final reward:
   - reward = 1.0 if final_answer == ground_truth
   - reward = 0.0 otherwise

4. Save complete Episode to disk
```

### Integration with Baseline

**Reused Functions** (no changes to baseline):
- `build_instruction_text()`: Create prompt
- `parse_rotation_angle()`: Apply camera rotation
- `apply_movement_in_camera_frame()`: Apply translation
- `render_mesh_from_pose()`: Render view
- `extract_first_json()`: Parse JSON

**New Functions** (in rl_environment):
- Environment state management
- Episode simulation
- Token tracking
- Data storage

## Usage Example

```python
from rl_environment import NavigationEnvironment, EpisodeSimulator

# 1. Setup
model = load_model()
mesh = load_mesh("scene.ply")
initial_pose = compute_initial_pose(mesh)

# 2. Create environment
env = NavigationEnvironment(
    mesh=mesh,
    scene_id="42444953",
    question="Navigate from X to Y",
    choices=["A", "B", "C", "D"],
    ground_truth="A",
    max_steps=10
)

# 3. Create simulator
sim = EpisodeSimulator(model, processor, track_action_tokens=True)

# 4. Run episode
episode = sim.run_episode(env, initial_pose, "ep001", output_dir)

# 5. Access data for RL
for turn in episode.turns:
    generated_ids = turn.generated_ids        # Shape: [seq_len]
    action_mask = turn.action_token_mask      # Shape: [seq_len], boolean
    obs = turn.observation
    action = turn.action
    next_obs = turn.next_observation

final_reward = episode.final_reward  # 1.0 or 0.0
```

## Testing

```bash
cd rl_environment
python test_environment.py --test all
```

This runs a 3-step test episode and verifies data storage/loading.

## Ready for RL Training

The environment provides everything needed for policy gradient methods:

### What You Have
- ✅ Observations: `obs_k` with images and context
- ✅ Actions: `action_k` with movement parameters
- ✅ Token-level tracking: `action_token_mask_k` for loss computation
- ✅ Transitions: `(obs_k, action_k, obs_{k+1})`
- ✅ Terminal reward: `R` at episode end
- ✅ Complete trajectories: All turns stored

### Next Steps for RL
1. **Collect episodes**: Run many episodes to gather data
2. **Compute log probs**: Use `generated_ids` and `action_token_mask` to get π(a|s)
3. **Train policy**: REINFORCE, PPO, or other policy gradient method
4. **Iterate**: Collect more data with updated policy

## Design Highlights

1. **Clean Separation**: Environment (state/rendering) vs Simulator (data collection)
2. **Non-Invasive**: Baseline pipeline unchanged
3. **Complete Storage**: All trajectory data saved with tensors
4. **Token Precision**: Action tokens identified via mask (not string matching)
5. **Reproducible**: Full prompts and metadata for debugging
6. **Validated Actions**: Bounds checking and consistency verification
7. **Extensible**: Easy to add intermediate rewards, value functions, etc.

## Action Token Mask Example

```python
# Generated sequence (example):
# "I can see a chair. I should move forward. {\"rotation_angle_degrees\": 15, ...}"
#
# Token IDs:        [101, 245, 892, ...., 1234, 5678, 9012, ....]
# Action Mask:      [  F,   F,   F, ....,    T,    T,    T, ....]
#                    ↑ reasoning tokens      ↑ action (JSON) tokens
#
# Only action tokens will contribute to RL loss
```

This allows the model to keep generating reasoning (which helps it think) while only the action tokens are trained via RL.
