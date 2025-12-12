# Step 3 Implementation: Enhanced Action Token Masking

## Overview

Step 3 implements robust action token masking with comprehensive edge case handling, episode quality evaluation, and dropout tracking as specified.

## ✅ Implementation Checklist

### 1. **What to Mark as Action Tokens** ✅
- ✅ JSON substring from opening `{` to closing `}` inclusive
- ✅ Handle nested braces via depth tracking
- ✅ Generation-time tracking (not post-hoc)
- ✅ Special marker detection (brace tokens)

### 2. **Data Storage** ✅
- ✅ `generated_ids`: Token tensor
- ✅ `action_token_mask`: Boolean tensor (same length)
- ✅ `action_token_start_index`, `action_token_end_index`: Alternative indices
- ✅ `input_token_length`: Token offset for aligning logits
- ✅ Masking diagnostics: method, confidence, counts

### 3. **Masking Usage** ✅
- ✅ Compute log-prob only for `action_token_mask == True`
- ✅ KL penalty on action tokens only
- ✅ Entropy computed over action tokens only

### 4. **Edge Cases & Robustness** ✅
- ✅ Truncated JSON: Detected and handled
- ✅ No JSON found: Multiple fallback strategies
- ✅ Episode dropout: Invalid episodes marked and tracked
- ✅ Statistics logging: Dropout reasons and counts

## Architecture

### New Components

#### 1. `masking.py` - ActionTokenMasker

**Purpose**: Robust action token identification with fallback strategies.

**Strategy Hierarchy**:
1. **Primary**: Brace depth tracking (confidence: 1.0)
   - Detects `{` and `}` tokens during decoding
   - Tracks nesting depth
   - Marks all tokens between matching braces
   
2. **Fallback 1**: Regex pattern matching (confidence: 0.7)
   - Finds JSON via regex in decoded text
   - Maps character positions to tokens
   - Used when brace tracking fails
   
3. **Fallback 2**: Last-N tokens heuristic (confidence: 0.3)
   - Assumes last 30% of tokens are action
   - Minimum 15 tokens
   - **Marked as low confidence**
   
4. **Complete Failure**: Return empty mask (confidence: 0.0)
   - Episode marked for dropout

**Validation**:
- Check `min_action_tokens <= num_tokens <= max_action_tokens`
- Default: 10-100 tokens expected

#### 2. Enhanced Turn Dataclass

**New Fields**:
```python
@dataclass
class Turn:
    # ... existing fields ...
    
    # Input offset (Step 3)
    input_token_length: int = 0  # For aligning logits
    
    # Masking diagnostics (Step 3)
    num_action_tokens: int = 0
    num_reasoning_tokens: int = 0
    masking_method: str = "unknown"  # "brace_depth", "regex_fallback", "last_n_fallback", "failed"
    masking_confidence: float = 1.0  # 0.0-1.0
```

#### 3. Enhanced Episode Dataclass

**New Fields**:
```python
@dataclass
class Episode:
    # ... existing fields ...
    
    # Quality flags (Step 3)
    is_valid: bool = True  # False if should be dropped
    dropout_reason: Optional[str] = None
    masking_quality: str = "good"  # "good", "low_confidence", "fallback_used"
```

#### 4. Episode Quality Evaluation

**Method**: `_evaluate_episode_quality(episode)`

**Dropout Criteria**:
1. Masking failed completely (`method == "failed"`)
2. All turns used `last_n_fallback` (very low confidence)
3. Invalid action that couldn't be parsed

**Quality Levels**:
- **good**: avg_confidence >= 0.9 (brace depth for all turns)
- **low_confidence**: 0.6 <= avg_confidence < 0.9 (some regex fallbacks)
- **fallback_used**: avg_confidence < 0.6 (many fallbacks)

#### 5. Statistics Tracking

**Simulator Stats**:
```python
{
    "episodes_total": 150,
    "episodes_valid": 142,
    "episodes_dropped": 8,
    "dropout_reasons": {
        "masking_failed": 3,
        "invalid_action: No JSON found": 2,
        "all_turns_low_confidence_masking": 3
    },
    "masking_methods": {
        "brace_depth": 380,  # Most reliable
        "regex_fallback": 45,
        "last_n_fallback": 5,
        "failed": 3
    },
    "masking_quality": {
        "good": 120,
        "low_confidence": 18,
        "fallback_used": 4
    },
    "dropout_rate": 0.053  # 5.3%
}
```

## Usage

### Basic Usage

```python
from rl_environment import EpisodeSimulator

# Create simulator with masking parameters
simulator = EpisodeSimulator(
    model, processor,
    min_action_tokens=10,   # Minimum expected
    max_action_tokens=100,  # Maximum expected
    do_sample=True
)

# Run episode
episode = simulator.run_episode(env, initial_pose, "ep001")

# Check quality
if episode.is_valid:
    print(f"Valid episode, quality: {episode.masking_quality}")
    for turn in episode.turns:
        print(f"Turn {turn.turn_index}:")
        print(f"  Method: {turn.masking_method}")
        print(f"  Confidence: {turn.masking_confidence:.2f}")
        print(f"  Action tokens: {turn.num_action_tokens}")
else:
    print(f"Episode dropped: {episode.dropout_reason}")

# View statistics
simulator.log_stats()
```

### RL Training Usage

```python
# Load episode
for turn in episode.turns:
    if not episode.is_valid:
        continue  # Skip invalid episodes
    
    generated_ids = turn.generated_ids
    action_mask = turn.action_token_mask
    input_length = turn.input_token_length
    
    # Extract action tokens only
    action_token_ids = generated_ids[action_mask]
    
    # Compute log probabilities (only for action tokens)
    with torch.no_grad():
        # Re-run model to get logits
        outputs = model(input_ids=..., ...)
        logits = outputs.logits[:, input_length:, :]  # Align with generated tokens
        
        # Select action token logits
        action_logits = logits[0, action_mask, :]
        
        # Compute log probs
        log_probs = F.log_softmax(action_logits, dim=-1)
        selected_log_probs = log_probs.gather(1, action_token_ids.unsqueeze(-1))
        
        # Policy gradient loss
        loss = -selected_log_probs.sum() * reward
```

### Masking Diagnostics

Each turn saves detailed masking diagnostics:

```json
{
  "method": "brace_depth",
  "confidence": 1.0,
  "num_action_tokens": 45,
  "num_reasoning_tokens": 180,
  "validation_passed": true,
  "issues": []
}
```

Or if fallback was used:

```json
{
  "method": "regex_fallback",
  "confidence": 0.7,
  "num_action_tokens": 38,
  "num_reasoning_tokens": 195,
  "validation_passed": true,
  "issues": ["Used regex fallback instead of brace tracking"]
}
```

## Edge Case Handling

### Case 1: Truncated JSON

**Scenario**: Generation stops mid-JSON
```
"I can see a chair. {\"rotation_angle_degrees\": 15, \"forward_mete
```

**Handling**:
- Brace depth tracking detects unclosed JSON
- Adds diagnostic: "JSON not closed (truncated generation)"
- Falls back to regex (likely fails)
- Falls back to last-N tokens
- Marks episode with low confidence

### Case 2: No JSON Found

**Scenario**: Model generates pure reasoning, no action
```
"I should probably look around first to understand the scene better."
```

**Handling**:
- All strategies fail
- `masking_method = "failed"`
- `masking_confidence = 0.0`
- Episode marked `is_valid = False`
- `dropout_reason = "masking_failed"`

### Case 3: Malformed JSON

**Scenario**: JSON present but invalid
```
"Here's my action: {rotation: 15, forward: 0.5}"  # Missing quotes
```

**Handling**:
- Masking may succeed (finds braces)
- Parsing fails: `action_valid = False`
- Episode marked `is_valid = False`
- `dropout_reason = "invalid_action: Failed to parse action"`

### Case 4: Nested Braces in Reasoning

**Scenario**: Braces in reasoning text
```
"The room has {furniture, decorations}. I'll move: {\"rotation_angle_degrees\": 15, ...}"
```

**Handling**:
- Brace depth tracking finds FIRST `{`
- Tracks depth through nested braces
- May capture too many tokens
- Validation checks token count
- If out of range, falls back to regex (which looks for `rotation_angle_degrees`)

## Statistics & Monitoring

### Logging Statistics

```python
simulator.log_stats()
```

Output:
```
================================================================================
EPISODE SIMULATOR STATISTICS
================================================================================
Total episodes: 150
Valid episodes: 142
Dropped episodes: 8 (5.3%)

Dropout reasons:
  masking_failed: 3
  invalid_action: No JSON found: 2
  all_turns_low_confidence_masking: 3

Masking methods:
  brace_depth: 380
  regex_fallback: 45
  last_n_fallback: 5
  failed: 3

Masking quality:
  good: 120
  low_confidence: 18
  fallback_used: 4
================================================================================
```

### Per-Episode Quality Check

```python
if episode.is_valid:
    # Use for training
    training_data.append(episode)
else:
    # Log and skip
    print(f"Dropped {episode.episode_id}: {episode.dropout_reason}")
    dropped_episodes.append(episode)
```

## Key Improvements Over Step 2

| Aspect | Step 2 | Step 3 |
|--------|--------|--------|
| Masking | Basic brace tracking + regex | Multi-strategy hierarchy |
| Validation | None | Token count validation |
| Edge Cases | Minimal handling | Comprehensive fallbacks |
| Dropout | No tracking | Full tracking + reasons |
| Confidence | Not tracked | Per-turn confidence scores |
| Diagnostics | Minimal | Detailed per-turn diagnostics |
| Statistics | Basic counts | Comprehensive logging |

## Files Modified/Created

**Created**:
- `masking.py`: ActionTokenMasker class with robust strategies

**Modified**:
- `environment.py`: Added masking diagnostics to Turn, quality flags to Episode
- `simulator.py`: Integrated ActionTokenMasker, added quality evaluation, dropout tracking

## Testing

The masking system can be tested with edge cases:

```python
# Test truncated JSON
test_ids = tokenize("I see... {\"rotation_angle_degrees\": 15, \"forw")
mask, start, end, diag = masker.identify_action_tokens(test_ids, text)
assert diag["method"] in ["regex_fallback", "last_n_fallback", "failed"]

# Test no JSON
test_ids = tokenize("Just some reasoning text")
mask, start, end, diag = masker.identify_action_tokens(test_ids, text)
assert diag["method"] == "failed"
assert diag["confidence"] == 0.0

# Test valid JSON
test_ids = tokenize("Reasoning... {\"rotation_angle_degrees\": 15, ...}")
mask, start, end, diag = masker.identify_action_tokens(test_ids, text)
assert diag["method"] == "brace_depth"
assert diag["confidence"] == 1.0
```

## Ready for RL

Step 3 provides production-ready masking:
- ✅ Robust multi-strategy approach
- ✅ Comprehensive edge case handling
- ✅ Episode quality evaluation
- ✅ Dropout tracking and logging
- ✅ Per-token confidence scores
- ✅ Detailed diagnostics for debugging

All requirements from Step 3 specification are satisfied!
