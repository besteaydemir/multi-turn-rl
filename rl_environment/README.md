# RL Environment for Multi-Turn Navigation

This module provides a clean interface for reinforcement learning on the VSI-Bench navigation pipeline.

## Architecture Overview

### Core Components

#### 1. `environment.py` - NavigationEnvironment

**Purpose**: Manages the 3D scene state and action execution.

**Key Classes**:
- `Observation`: Complete observation at timestep t
  - Images accumulated so far
  - Camera positions/rotations
  - Scene bounding box
  - Question and choices
  - Movement history
  - Whether this is the final step

- `Action`: Parsed action from model (JSON only, no reasoning)
  - `rotation_angle_degrees`: Camera rotation
  - `forward_meters`: Forward/backward movement
  - `left_meters`: Left/right strafe
  - `z_delta_meters`: Up/down movement
  - `answer`: Final answer (only if done=True)
  - `done`: Termination flag
  - Includes validation logic

- `NavigationEnvironment`: The environment itself
  - `reset(initial_pose)`: Reset to initial state, returns initial observation
  - `step(action)`: Execute action, returns (next_observation, done)
  - Handles mesh rendering and camera pose updates
  - Maintains episode state (images, poses, movement history)

#### 2. `simulator.py` - EpisodeSimulator

**Purpose**: Runs complete episodes and collects trajectory data for RL training.

**Key Classes**:
- `Turn`: Complete data for a single turn
  - `observation`: Observation at this turn
  - `full_prompt`: Complete prompt with images
  - `context_text`: Instruction text portion
  - `generated_ids`: Token IDs of entire generated sequence (Tensor)
  - `generated_text`: Human-readable text
  - `action_token_mask`: **Boolean mask** marking which tokens are part of the JSON action
  - `action`: Parsed Action object
  - `action_valid`: Whether action passed validation
  - `next_observation`: Observation after executing action (None if terminal)

- `Episode`: Complete episode data
  - `episode_id`: Unique identifier
  - `scene_id`, `question`, `choices`, `ground_truth`
  - `turns`: List of Turn objects
  - `final_reward`: Terminal reward (1.0 if correct, 0.0 if wrong)
  - `final_answer`: Model's answer
  - `metadata`: Timestamps, duration, etc.
  - `save_full(output_dir)`: Save all data including tensors

- `EpisodeSimulator`: Orchestrates episode execution
  - `run_episode(env, initial_pose, episode_id)`: Run complete episode
  - Handles model generation with token tracking
  - Parses actions from generated text
  - Collects all trajectory data

### Token Tracking Strategy

**Critical for RL**: We need to identify which tokens in the generated sequence correspond to the action (JSON) vs. reasoning.

**Implementation** (`_identify_action_tokens` in simulator.py):
1. Generate full sequence (reasoning + JSON)
2. Find JSON pattern in decoded text using regex
3. Map character positions back to token positions
4. Create boolean mask over `generated_ids`

**Result**: `action_token_mask[i] = True` if token `i` is part of the JSON action.

This allows the RL loss to focus on action tokens while preserving reasoning tokens.

## Episode Data Format

Each episode is saved with the following structure:

```
episode_output_dir/
├── episode_metadata.json          # High-level episode info
├── turn_00/
│   ├── turn_metadata.json         # Turn summary
│   ├── generated_ids.pt           # Tensor: token IDs [seq_len]
│   ├── action_token_mask.pt       # Tensor: boolean [seq_len]
│   ├── full_prompt.txt            # Complete prompt with context
│   ├── context_text.txt           # Instruction portion
│   ├── generated_text.txt         # Human-readable output
│   └── messages.json              # Messages passed to processor
├── turn_01/
│   └── ...
└── env_renders/
    ├── render_00.png
    ├── render_01.png
    └── ...
```

### Key Data for RL Training

For each turn `k`, you have:
- **Observation**: `obs_k` (images + text context)
- **Full prompt**: `full_prompt_k` (for reproducibility)
- **Generated tokens**: `generated_ids_k` (shape: `[seq_len]`)
- **Action mask**: `action_token_mask_k` (shape: `[seq_len]`, boolean)
- **Parsed action**: `action_k` (validated Action object)
- **Next observation**: `obs_{k+1}` (from next turn)

At episode end:
- **Final reward**: `final_reward` (scalar: 1.0 or 0.0)

## Usage Example

```python
from rl_environment import NavigationEnvironment, EpisodeSimulator
import open3d as o3d

# Load model
model = Qwen3VLForConditionalGeneration.from_pretrained(...)
processor = AutoProcessor.from_pretrained(...)

# Load scene
mesh = o3d.io.read_triangle_mesh("scene.ply")
initial_pose = compute_initial_camera_pose(mesh)

# Create environment
env = NavigationEnvironment(
    mesh=mesh,
    scene_id="42444953",
    question="Navigate from X to Y",
    choices=["A", "B", "C", "D"],
    ground_truth="A",
    max_steps=10
)

# Create simulator
simulator = EpisodeSimulator(
    model=model,
    processor=processor,
    max_steps=10,
    track_action_tokens=True
)

# Run episode
episode = simulator.run_episode(
    env=env,
    initial_pose=initial_pose,
    episode_id="episode_001",
    output_dir=Path("output/episode_001")
)

# Access trajectory data
for turn in episode.turns:
    obs = turn.observation
    action = turn.action
    generated_ids = turn.generated_ids
    action_mask = turn.action_token_mask
    next_obs = turn.next_observation
    
    # Use for RL training...

reward = episode.final_reward  # Terminal reward
```

## Testing

Run the test script:

```bash
cd rl_environment
python test_environment.py --test all
```

This will:
1. Load a test scene
2. Run a 3-step episode
3. Save all trajectory data
4. Reload and verify the saved data

## Integration with Baseline Pipeline

The environment reuses these functions from `render_point_cloud_qwen_angle.py`:
- `build_instruction_text`: Build instruction prompt
- `parse_rotation_angle`: Apply rotation to camera
- `apply_movement_in_camera_frame`: Apply translation
- `render_mesh_from_pose`: Render view from pose
- `extract_first_json`: Parse JSON from text

**No modifications to the baseline pipeline are required**.

## Next Steps for RL Implementation

With this environment, you can now implement:

1. **Policy Gradient Training**:
   - Use `action_token_mask` to compute log probabilities of action tokens
   - Apply REINFORCE or PPO with `final_reward` as return
   
2. **Reward Shaping** (optional):
   - Add intermediate rewards (e.g., progress toward goal)
   - Modify `Episode.compute_final_reward()`

3. **Batch Collection**:
   - Run multiple episodes in parallel
   - Aggregate trajectory data for training

4. **Value Function**:
   - Train critic to predict expected reward
   - Use for variance reduction in policy gradient

## Design Principles

1. **Separation of Concerns**: Environment handles state/rendering, simulator handles data collection
2. **Minimal Invasiveness**: Reuses baseline pipeline code without modification
3. **Complete Trajectory Storage**: All data needed for RL is recorded
4. **Token-Level Tracking**: Action tokens are precisely identified for loss computation
5. **Reproducibility**: Full prompts and metadata saved for debugging
