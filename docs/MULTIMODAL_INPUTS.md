# Multimodal Input Handling for Qwen3-VL

## Overview

This document explains how multimodal inputs (images + text) are handled throughout the RL pipeline to ensure consistency between episode generation and training.

## Key Principle

**CRITICAL**: The teacher-forcing pass used for computing log-probabilities must use the **exact same** multimodal formatting as was used during episode generation. Otherwise, we compute log p(action | different_context) which is incorrect.

## Qwen3-VL Multimodal Architecture

Qwen3-VL processes multimodal inputs as follows:

1. **Vision Token Insertion**: Special tokens `<|vision_start|>` and `<|vision_end|>` mark where image embeddings are inserted
2. **Image Encoding**: Images are encoded to `pixel_values` or `image_embeds` tensors
3. **Token Concatenation**: Text tokens and vision tokens are concatenated into a single sequence
4. **Grid Dimensions**: `image_grid_thw` specifies image grid dimensions (tiles, height, width)

## Pipeline Flow

### 1. Episode Generation (simulator.py)

```python
# During episode collection
inputs = processor.apply_chat_template(
    messages=[
        {"role": "user", "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": question}
        ]}
    ],
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)

# inputs contains:
# - input_ids: [1, seq_len] with vision tokens inserted
# - attention_mask: [1, seq_len]
# - pixel_values or image_embeds: Image tensor
# - image_grid_thw: Grid dimensions

generated_ids = model.generate(**inputs, max_new_tokens=100)
```

**Important**: We must save the context portion of these inputs for later training:
```python
context_data = {
    "context_input_ids": inputs["input_ids"][:, :input_length],
    "pixel_values": inputs.get("pixel_values"),
    "image_embeds": inputs.get("image_embeds"),
    "image_grid_thw": inputs.get("image_grid_thw")
}
```

### 2. Batch Preparation (batch.py)

When batching episodes, we need to:
1. Pad context_input_ids to max_context_len
2. Collect images from all episodes
3. Batch image tensors

```python
# Extract images from episodes
images_per_episode = []
for episode in episodes:
    episode_images = []
    for turn in episode.turns:
        images = load_images_from_paths(turn.observation.images)
        episode_images.extend(images)
    images_per_episode.append(episode_images)
```

### 3. Log-Probability Computation (logprobs.py)

During teacher-forcing, we reconstruct the full input with images:

```python
# Concatenate context + generated
full_input_ids = torch.cat([context_input_ids, generated_ids], dim=1)

# Build multimodal inputs (MUST match generation format)
input_builder = MultimodalInputBuilder(processor, device)
model_inputs = input_builder.prepare_teacher_forcing_inputs(
    context_input_ids=context_input_ids,
    generated_ids=generated_ids,
    attention_mask=full_attention_mask,
    images=images  # Same images as during generation
)

# Forward pass (same as generation)
outputs = model(**model_inputs)
logits = outputs.logits
```

## Implementation Components

### MultimodalInputBuilder (`rl_trainer/multimodal.py`)

This class handles:
- `prepare_generation_inputs()`: Format inputs for model.generate()
- `prepare_teacher_forcing_inputs()`: Format inputs for training forward pass
- `validate_multimodal_consistency()`: Verify generation and training use same format
- `get_vision_token_positions()`: Locate vision tokens in sequence

### Key Functions

**`batch_images_for_episodes(episodes)`**
- Extracts all images from episode turns
- Loads PIL images from paths
- Returns list of image lists (one per episode)

**`load_images_from_paths(image_paths)`**
- Loads PIL images from file paths
- Handles errors gracefully with fallback blank images

## Token Alignment

### Context Token Offset

When computing log-probabilities, we need to know where the generated tokens start:

```
Full sequence: [<vision_start> ... <vision_end> ... context_text ... generated_tokens]
               |<------- context_input_ids ------>|<-- generated_ids -->|
               
Logits:        [logit[0], logit[1], ..., logit[context_len-1], ..., logit[context_len+gen_len-1]]
                                                   ^
                                                   Predicts generated_ids[0]
```

The vision tokens are PART of context_input_ids, so they don't affect the offset calculation.

### Causal LM Shift

For causal language models:
- `logits[t]` predicts `token[t+1]`
- To get log p(generated_ids[i]), use `logits[context_len + i - 1]`

```python
# Extract logits that predict generated tokens
pred_logits = logits[:, context_len-1:-1, :]  # Shape: [batch, gen_len, vocab]

# These logits predict generated_ids
labels = generated_ids  # Shape: [batch, gen_len]
```

## Validation Checks

The `validate_multimodal_consistency()` function checks:

1. **Same keys**: Generation and training inputs have same keys (input_ids, pixel_values, etc.)
2. **Shape consistency**: Image tensors have same shapes (except batch dimension)
3. **Vision token positions**: Vision tokens appear at same positions in context

## Example Usage

### In Simulator (rl_environment/simulator.py)

```python
class EpisodeSimulator:
    def _generate_with_tracking(self, messages, images):
        # Build inputs with images
        input_builder = MultimodalInputBuilder(self.processor, self.device)
        inputs = input_builder.prepare_generation_inputs(messages, images)
        
        # Generate
        generated_ids = self.model.generate(**inputs, max_new_tokens=100)
        
        # Save context data for training
        context_data = input_builder.extract_context_from_generation(
            inputs, 
            input_length=inputs["input_ids"].shape[1]
        )
        
        return generated_ids, context_data
```

### In Trainer (rl_trainer/trainer.py)

```python
class RLTrainer:
    def _train_batch(self, batch_episodes):
        # Prepare batch
        batch = prepare_batch(batch_episodes, self.processor)
        
        # Extract images
        images, _ = batch_images_for_episodes(batch_episodes)
        
        # Compute log-probs with images
        logprobs_result = compute_sequence_logprobs(
            model=self.model,
            batch=batch,
            processor=self.processor,
            images=images,  # Images passed through
            ref_model=self.ref_model
        )
        
        # Compute loss
        loss = policy_gradient_loss(logprobs_result, advantages, kl_coef, entropy_coef)
```

## Testing

To verify multimodal consistency:

```python
# During training, validate first batch
if self.global_step == 0:
    input_builder = MultimodalInputBuilder(self.processor, self.device)
    is_valid, error = input_builder.validate_multimodal_consistency(
        generation_inputs=saved_generation_inputs,
        training_inputs=model_inputs
    )
    if not is_valid:
        raise ValueError(f"Multimodal consistency check failed: {error}")
```

## Troubleshooting

### Issue: Different log-probs during generation vs training

**Cause**: Multimodal inputs formatted differently
**Solution**: 
1. Check vision token positions match
2. Verify image tensors have same values
3. Ensure processor settings identical (same image preprocessing)

### Issue: CUDA out of memory with images

**Cause**: Large batch size with high-resolution images
**Solution**:
1. Reduce batch size
2. Downscale images during preprocessing
3. Use gradient accumulation instead of large batches

### Issue: Vision tokens not found

**Cause**: Processor doesn't insert vision tokens
**Solution**:
1. Check processor version compatibility
2. Verify messages format includes image content
3. Ensure images are PIL Image objects, not paths

## Future Improvements

1. **Caching**: Cache encoded images to avoid recomputing during training
2. **Efficient batching**: Batch images separately from text tokens
3. **Mixed precision**: Use FP16 for image encoders to save memory
4. **Lazy loading**: Load images on-demand rather than loading all at batch creation
