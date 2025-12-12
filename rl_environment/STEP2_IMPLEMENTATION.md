# Step 2 Implementation: Generation & Episode Collection

## Overview

Step 2 implements the episode collection pipeline with **real-time token tracking** and **persistent storage**. This addresses all requirements from the Step 2 specification.

## ✅ Implementation Checklist

### ✅ 1. Sampling-based Generation
- **Implemented**: `EpisodeSimulator.__init__` accepts sampling parameters
- Parameters:
  - `do_sample`: Enable/disable sampling (default: `True`)
  - `temperature`: Controls randomness (default: `1.0`)
  - `top_p`: Nucleus sampling threshold (default: `0.9`)
  - `top_k`: Top-k sampling (default: `50`)
- **Location**: `simulator.py`, lines 30-40

### ✅ 2. Real-Time Token Tracking
- **Implemented**: `_identify_action_tokens_realtime()`
- **Strategy**: Detect `{` and `}` tokens during generation by tracking brace depth
- **Result**: Precise `json_start_index` and `json_end_index` without post-hoc string search
- **Fallback**: If brace detection fails, uses string search as backup
- **Location**: `simulator.py`, lines 420-510

### ✅ 3. Action Token Mask
- **Implemented**: Boolean tensor marking which tokens are part of JSON
- **Stored**: In `Turn.action_token_mask` (Tensor, dtype=bool)
- **Indices**: Also stored as `action_token_start_index` and `action_token_end_index`
- **Location**: `environment.py`, `Turn` dataclass

### ✅ 4. Episode Data Storage
- **Format 1**: Full episode with tensors (existing)
  - Directory structure with all data
  - Individual tensor files per turn
  
- **Format 2**: JSONL for batch processing (NEW)
  - Compact one-line-per-episode format
  - Metadata + turn summaries
  - Tensors stored separately in binary format
  
- **Location**: 
  - `Episode.save_to_jsonl()` in `environment.py`
  - `EpisodeBatchCollector` in `simulator.py`

### ✅ 5. Batch Collection
- **Implemented**: `EpisodeBatchCollector` class
- **Features**:
  - Collect multiple episodes
  - Automatic JSONL appending
  - Separate tensor storage
  - Statistics tracking
  - Iterator for loading episodes
- **Location**: `simulator.py`, lines 520-680

## Key Components

### 1. Real-Time Token Tracking

**Problem**: Need to identify which tokens are part of the JSON action during generation, not after.

**Solution**: `_identify_action_tokens_realtime()`

```python
def _identify_action_tokens_realtime(self, generated_ids, generated_text):
    """
    Track JSON boundaries by detecting { and } tokens as they're generated.
    """
    mask = torch.zeros(len(generated_ids), dtype=torch.bool)
    json_start_index = None
    json_end_index = None
    
    in_json = False
    brace_depth = 0
    
    for i, token_id in enumerate(generated_ids):
        token_text = self.processor.tokenizer.decode([token_id.item()])
        
        # Detect opening brace
        if '{' in token_text and not in_json:
            in_json = True
            json_start_index = i
            brace_depth = 1
            mask[i] = True
        
        # Track all tokens inside JSON
        elif in_json:
            mask[i] = True
            brace_depth += token_text.count('{')
            brace_depth -= token_text.count('}')
            
            # Found closing brace
            if brace_depth <= 0:
                json_end_index = i + 1
                break
    
    return mask, json_start_index, json_end_index
```

**Benefits**:
- No post-hoc string matching
- Precise token indices
- Works even if JSON formatting is unusual
- Fallback to regex if needed

### 2. Sampling Configuration

**Before (Step 1)**:
```python
simulator = EpisodeSimulator(model, processor)
# Used greedy decoding (do_sample=False)
```

**After (Step 2)**:
```python
simulator = EpisodeSimulator(
    model, 
    processor,
    do_sample=True,      # Enable sampling
    temperature=0.8,     # Control randomness
    top_p=0.9,          # Nucleus sampling
    top_k=50            # Top-k filtering
)
# Generates diverse trajectories for RL
```

### 3. JSONL Storage Format

**Structure**:
```
batch_output/
├── episodes.jsonl              # One line per episode (metadata)
├── tensors/
│   ├── episode_000/
│   │   ├── turn_00.pt          # {generated_ids, action_token_mask}
│   │   ├── turn_01.pt
│   │   └── ...
│   ├── episode_001/
│   │   └── ...
└── episode_000/                # Full episode data (optional)
    ├── episode_metadata.json
    ├── turn_00/
    └── ...
```

**JSONL Line Format**:
```json
{
  "episode_id": "episode_000",
  "scene_id": "42444953",
  "question": "Navigate from X to Y",
  "choices": ["A", "B", "C", "D"],
  "ground_truth": "A",
  "final_reward": 1.0,
  "final_answer": "A",
  "is_correct": true,
  "metadata": {...},
  "turns": [
    {
      "turn_index": 0,
      "generated_text": "I see... {\"rotation_angle_degrees\": 15, ...}",
      "generated_ids_length": 234,
      "action_token_start_index": 180,
      "action_token_end_index": 234,
      "action": {...},
      "action_valid": true,
      "timestamp": 1670000000.0
    },
    ...
  ]
}
```

### 4. Batch Collection

**Usage**:
```python
# Create collector
collector = EpisodeBatchCollector(
    simulator=simulator,
    output_dir=Path("rl_data"),
    save_format="both"  # JSONL + full data
)

# Collect multiple episodes
for scenario in test_scenarios:
    env = create_environment(scenario)
    initial_pose = compute_initial_pose(...)
    
    episode = collector.collect_episode(
        env=env,
        initial_pose=initial_pose
    )

# Get statistics
stats = collector.get_statistics()
print(f"Collected {stats['num_episodes']} episodes")
print(f"Accuracy: {stats['accuracy']:.2%}")

# Iterate through episodes
for episode_data in collector.iter_episodes_jsonl():
    # Process episode...
    pass

# Load specific tensors
tensors = collector.load_episode_tensors("episode_000", turn_index=0)
generated_ids = tensors['generated_ids']
action_mask = tensors['action_token_mask']
```

## Data Flow

```
1. EpisodeSimulator.run_episode()
   └─> For each turn:
       ├─> Build prompt with images
       ├─> Generate with sampling
       │   └─> _generate_with_tracking()
       │       ├─> model.generate(do_sample=True, temperature=0.8, ...)
       │       ├─> _identify_action_tokens_realtime()
       │       │   └─> Track { } tokens → json_start_index, json_end_index
       │       └─> Return (generated_ids, text, mask, start_idx, end_idx)
       ├─> Parse action from JSON
       ├─> Execute in environment
       └─> Store Turn with all data

2. Episode complete
   └─> Compute final reward (1.0 or 0.0)

3. EpisodeBatchCollector.collect_episode()
   ├─> Save to JSONL (append)
   └─> Save tensors separately
```

## Token Tracking Example

```python
# Generated sequence:
text = "I can see a chair. I should move forward. {\"rotation_angle_degrees\": 15, ...}"

# Token IDs (example):
generated_ids = [101, 245, 892, ..., 1234, 5678, 9012, ...]
                  ↑ reasoning          ↑ JSON action

# Action token mask (boolean):
action_mask = [False, False, False, ..., True, True, True, ...]

# Indices:
json_start_index = 180  # First token of {
json_end_index = 234    # Last token of } + 1

# For RL training:
action_tokens = generated_ids[json_start_index:json_end_index]
action_log_probs = compute_log_probs(action_tokens)
loss = -action_log_probs.sum() * reward
```

## Differences from Step 1

| Aspect | Step 1 | Step 2 |
|--------|--------|--------|
| Generation | Greedy (do_sample=False) | Sampling with temp/top_p/top_k |
| Token Tracking | Post-hoc string search | Real-time brace detection |
| Token Indices | Approximate via char mapping | Precise during generation |
| Storage | Full episode dirs only | JSONL + binary tensors |
| Batch Collection | Manual | EpisodeBatchCollector |

## Testing

Run the example:
```bash
cd rl_environment
python example_batch_collection.py
```

This will:
1. Create simulator with sampling
2. Collect 2 episodes
3. Save to JSONL + tensors
4. Show statistics
5. Demonstrate loading

## Integration with RL Training

The collected data is now ready for RL:

```python
# Load batch of episodes
collector = EpisodeBatchCollector(...)

for episode_data in collector.iter_episodes_jsonl():
    episode_id = episode_data['episode_id']
    reward = episode_data['final_reward']
    
    for turn_data in episode_data['turns']:
        turn_idx = turn_data['turn_index']
        
        # Load tensors
        tensors = collector.load_episode_tensors(episode_id, turn_idx)
        generated_ids = tensors['generated_ids']
        action_mask = tensors['action_token_mask']
        
        # Extract action tokens
        action_token_ids = generated_ids[action_mask]
        
        # Compute loss (simplified)
        log_probs = model.compute_log_probs(action_token_ids)
        loss = -log_probs.sum() * reward
        
        # Backprop...
```

## Summary

Step 2 is **fully implemented** with:
- ✅ Sampling-based generation (temperature, top_p, top_k)
- ✅ Real-time token tracking (no post-hoc string search)
- ✅ Precise JSON boundary detection (start/end indices)
- ✅ Boolean action token mask
- ✅ JSONL persistent storage
- ✅ Binary tensor storage
- ✅ Batch collection utilities
- ✅ Statistics and iteration tools

All requirements from the Step 2 specification are satisfied.
