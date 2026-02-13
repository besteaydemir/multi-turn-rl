# vagen_vsi_rl — RL training for VSI-Bench multi-turn spatial reasoning

## Overview

This package turns the existing `evaluation/sequential.py` question loop into a
proper RL training pipeline.  The core idea is:

| **Inference (sequential.py)**          | **RL wrapper (this package)**           |
|----------------------------------------|----------------------------------------|
| Hard-coded loop in `run_single_question` | `VSIEnv.reset()` + `VSIEnv.step()`   |
| Prompt constructed inline              | Observation returned by `step()`       |
| Ground-truth evaluated after the loop  | Reward returned on `done=True`         |

## Directory layout

```
vagen_vsi_rl/
├── __init__.py
├── README.md
├── configs/
│   └── base.yaml              # default hyper-parameters
├── env/
│   ├── __init__.py
│   └── vsi_env.py             # POMDP environment  ← Step 1 deliverable
├── rollout/
│   ├── __init__.py
│   ├── trajectory.py          # Turn / Trajectory dataclasses
│   └── collector.py           # Runs policy + env → trajectories
├── models/
│   ├── __init__.py
│   ├── actor_vllm.py          # vLLM-backed actor (rollout)
│   ├── actor_hf.py            # HuggingFace actor (gradient)
│   ├── critic_hf.py           # Value-function head
│   └── ref_hf.py              # Frozen reference model
├── rl/
│   ├── __init__.py
│   ├── rewards.py             # Reward shaping
│   ├── advantage.py           # GAE / Monte-Carlo returns
│   ├── ppo.py                 # Clipped PPO loss
│   └── sync.py                # Actor ↔ vLLM weight sync
└── scripts/
    ├── train.py               # Main training loop
    └── eval.py                # Greedy evaluation
```

## Quick start — verify the environment (Step 1)

```bash
cd /dss/dsshome1/06/di38riq/rl_multi_turn
python -m vagen_vsi_rl.scripts.eval --dummy --max-questions 2
```

This runs two questions with a **dummy random policy** through the POMDP
environment and saves rendered images + trajectory JSON under `test_env_output/`.

## ⚠️ Critical Implementation Gotchas

### 1. Token alignment between vLLM and HuggingFace

**Problem**: vLLM and HF must use *identical* tokenizers, special tokens, and chat
templates. If they differ, token IDs won't match and PPO ratio computation will be
completely wrong.

**Symptoms**:
- NaN losses
- Exploding/collapsing KL
- Training doesn't improve

**Solution**: The training script validates alignment automatically:

```python
from vagen_vsi_rl.utils.token_utils import validate_tokenizer_alignment

report = validate_tokenizer_alignment(
    actor_vllm.tokenizer, 
    actor_hf.tokenizer,
    strict=True,  # Raises if mismatch
)
```

**Best practice**: Always use the same `model_id` and `cache_dir` for both actors.

---

### 2. Vision tokens must NOT contribute to loss

**Problem**: In multimodal RL, observation tokens (images + prompt text) should be
**masked out** from the PPO loss. Only the *generated action tokens* should be trained.

This is a known failure mode explicitly called out in the VAGEN paper:
> "Observation tokens dominating the loss leads to degenerate policies."

**How we handle it**:

1. **vLLM's `generate_with_logprobs`** returns ONLY generated token IDs (prompt excluded)
2. **`action_token_mask`** in each `Turn` can exclude specific tokens
3. **`ppo_step`** applies the mask before computing loss

```python
from vagen_vsi_rl.utils.token_utils import create_action_mask

# For full sequence (prompt + generated):
mask = create_action_mask(
    input_ids=full_sequence,
    prompt_length=100,  # First 100 tokens are prompt → masked out
    tokenizer=tokenizer,
    exclude_vision_tokens=True,  # Also mask <|vision_start|> etc.
)
```

**Key insight**: Since `GenerateOutput.token_ids` contains ONLY generated tokens
(no prompt), and PPO trains only on those, we're already correct by construction.
The prompt/images never appear in the training loss.

