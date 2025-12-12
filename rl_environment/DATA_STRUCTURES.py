"""
Quick Reference: Data Structures for RL

This file shows the exact structure of the data you'll work with for RL training.
"""

# ============================================================================
# OBSERVATION (State at timestep t)
# ============================================================================

observation = {
    "step": 0,                                    # Current step number
    "images": [                                   # List of image paths
        "/path/to/render_00.png",
        "/path/to/render_01.png"
    ],
    "camera_positions": [                         # 4x4 pose matrices (numpy)
        np.array([[...], [...], [...], [...]]),   # Shape: (4, 4)
        np.array([[...], [...], [...], [...]])
    ],
    "current_position": np.array([x, y, z]),      # Shape: (3,)
    "current_rotation": np.array([                # Shape: (3, 3)
        [r11, r12, r13],
        [r21, r22, r23],
        [r31, r32, r33]
    ]),
    "bbox_mins": [x_min, y_min, z_min],          # Scene bounds
    "bbox_maxs": [x_max, y_max, z_max],
    "question": "Navigate from X to Y",
    "choices": ["A", "B", "C", "D"],
    "movement_history": [                         # Previous movements
        {"rotation": 15.0, "forward": 0.2, "left": 0.0, "z_delta": 0.0},
        {"rotation": -30.0, "forward": 0.3, "left": -0.1, "z_delta": 0.05}
    ],
    "is_final_step": False                        # Whether this is the last allowed step
}

# ============================================================================
# ACTION (Model output - JSON only)
# ============================================================================

action = {
    "rotation_angle_degrees": 15.0,    # Rotate camera (+ = left, - = right)
    "forward_meters": 0.3,             # Move forward/back (range: -0.5 to 0.5)
    "left_meters": -0.1,               # Strafe left/right (range: -0.5 to 0.5)
    "z_delta_meters": 0.0,             # Move up/down (range: -0.3 to 0.3)
    "answer": "A",                     # Only if done=True
    "done": False                      # Terminate episode
}

# Validation rules:
# - forward_meters in [-0.5, 0.5]
# - left_meters in [-0.5, 0.5]
# - z_delta_meters in [-0.3, 0.3]
# - If done=True, answer must be present (A/B/C/D)
# - If done=False, answer must be None

# ============================================================================
# TURN (Complete data for one turn)
# ============================================================================

turn = {
    "turn_index": 0,
    
    # State
    "observation": Observation(...),              # See above
    
    # Generation inputs
    "full_prompt": "## Image History...\n...",    # Complete prompt string
    "context_text": "You are given...",           # Instruction portion
    
    # Generation outputs (THE CRITICAL DATA FOR RL)
    "generated_ids": torch.tensor([              # Token IDs - Shape: [seq_len]
        101, 245, 892, 1234, 5678, 9012, ...
    ]),
    "generated_text": "I can see... {\"rotation_angle_degrees\": 15, ...}",
    
    # ACTION TOKEN TRACKING (boolean mask)
    "action_token_mask": torch.tensor([          # Shape: [seq_len], dtype=bool
        False, False, False, ..., True, True, True, ...
    ]),
    # action_token_mask[i] = True if generated_ids[i] is part of the JSON action
    # This allows you to compute log_prob only for action tokens
    
    # Alternative: You can also use these if you prefer indices
    "action_token_start_index": 45,              # Start of JSON in token sequence
    "action_token_end_index": 60,                # End of JSON in token sequence
    
    # Parsed action
    "action": Action(...),                       # See above
    "action_valid": True,                        # Passed validation
    "action_error": "",                          # Error message if invalid
    
    # Next state
    "next_observation": Observation(...),        # State after executing action (None if terminal)
    
    # Metadata
    "attempt_count": 1,                          # How many generation attempts
    "timestamp": 1670000000.0
}

# ============================================================================
# EPISODE (Complete trajectory)
# ============================================================================

episode = {
    "episode_id": "episode_001",
    "scene_id": "42444953",
    "question": "Navigate from X to Y",
    "choices": ["A", "B", "C", "D"],
    "ground_truth": "A",
    
    # Trajectory
    "turns": [Turn(...), Turn(...), ...],       # List of Turn objects
    
    # Terminal reward
    "final_reward": 1.0,                        # 1.0 if correct, 0.0 if wrong
    "final_answer": "A",                        # Model's answer
    "is_correct": True,                         # Whether answer matches ground_truth
    
    # Metadata
    "metadata": {
        "start_time": 1670000000.0,
        "end_time": 1670000030.0,
        "duration_seconds": 30.0,
        "max_steps": 10,
        "num_turns": 5,
        "model_device": "cuda"
    }
}

# ============================================================================
# RL TRAINING DATA EXTRACTION
# ============================================================================

# For each turn k in an episode:

# 1. Get observation (state)
obs_k = turn.observation

# 2. Get generated tokens
token_ids = turn.generated_ids              # Shape: [seq_len]

# 3. Get action token mask
action_mask = turn.action_token_mask        # Shape: [seq_len], boolean

# 4. Compute log probabilities (pseudo-code)
logits = model(inputs)                      # Shape: [seq_len, vocab_size]
log_probs = F.log_softmax(logits, dim=-1)
action_log_probs = log_probs[action_mask]   # Only action tokens
selected_log_probs = action_log_probs.gather(1, token_ids[action_mask])

# 5. Get reward (only at episode end)
if turn_k == last_turn:
    reward = episode.final_reward           # 1.0 or 0.0

# 6. Compute loss (REINFORCE example)
loss = -selected_log_probs.sum() * reward

# ============================================================================
# FILE STRUCTURE ON DISK
# ============================================================================

"""
output_dir/
├── episode_metadata.json           # Episode summary (JSON)
├── turn_00/
│   ├── turn_metadata.json          # Turn info (JSON)
│   ├── generated_ids.pt            # torch.tensor, shape: [seq_len]
│   ├── action_token_mask.pt        # torch.tensor, shape: [seq_len], dtype=bool
│   ├── full_prompt.txt             # String
│   ├── context_text.txt            # String
│   ├── generated_text.txt          # String
│   └── messages.json               # List of dicts
├── turn_01/
│   └── ...
└── env_renders/
    ├── render_00.png
    ├── render_01.png
    └── ...
"""

# ============================================================================
# LOADING DATA FOR TRAINING
# ============================================================================

import torch
import json
from pathlib import Path

episode_dir = Path("output/episode_001")

# Load episode metadata
with open(episode_dir / "episode_metadata.json") as f:
    metadata = json.load(f)

final_reward = metadata["final_reward"]
num_turns = metadata["num_turns"]

# Load each turn
for turn_idx in range(num_turns):
    turn_dir = episode_dir / f"turn_{turn_idx:02d}"
    
    # Load tensors
    generated_ids = torch.load(turn_dir / "generated_ids.pt")
    action_mask = torch.load(turn_dir / "action_token_mask.pt")
    
    # Load metadata
    with open(turn_dir / "turn_metadata.json") as f:
        turn_meta = json.load(f)
    
    # Use for training...
    # compute_loss(generated_ids, action_mask, final_reward)

# ============================================================================
# KEY INSIGHTS FOR RL
# ============================================================================

"""
1. ACTION TOKEN MASK is the key innovation:
   - Allows you to compute loss ONLY on action tokens
   - Reasoning tokens are preserved but not trained via RL
   - This is crucial because:
     a) Reasoning helps the model think
     b) But only actions affect the environment
     c) RL should only optimize action selection

2. MULTI-TURN STRUCTURE:
   - Each turn is a (state, action) pair
   - Reward is only at the end (sparse reward)
   - You can add intermediate rewards if needed
   - Policy gradient methods handle this naturally

3. TOKEN-LEVEL PRECISION:
   - We track exact token IDs (not string matching)
   - This is critical for computing gradients correctly
   - action_token_mask is created during generation

4. REPRODUCIBILITY:
   - Full prompts saved
   - All random seeds should be in metadata
   - Can re-run exact episode for debugging
"""
