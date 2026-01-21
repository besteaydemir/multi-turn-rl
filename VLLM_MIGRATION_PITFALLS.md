# vLLM Migration Pitfalls Checklist

This document tracks common pitfalls when migrating from HuggingFace Transformers to vLLM, with our implementation status.

## Pitfall Analysis for Our Implementation

### ✅ 1. Assuming HF-style generate() semantics still apply

**Status: ADDRESSED**

- ✅ Use `SamplingParams` instead of HF's `generate()` kwargs
- ✅ Map `max_new_tokens` → `max_tokens`
- ✅ Use `temperature=0` for deterministic (greedy) decoding
- ✅ Don't use unsupported params like `do_sample`, `return_dict_in_generate`, `output_scores`

**Where:** [utils/inference.py](../utils/inference.py#L295)

```python
sampling_params = SamplingParams(
    temperature=0,  # Deterministic (greedy decoding)
    max_tokens=max_new_tokens,
)
```

---

### ✅ 2. Incorrect multimodal input formatting

**Status: FIXED (was a critical bug)**

**Original bug:** Manually constructed prompts with hardcoded tokens like `<|image|>`

**Fix:**
- ✅ Use `processor.apply_chat_template()` (same as HF)
- ✅ Use `process_vision_info()` (same as HF) for image extraction
- ✅ Let the processor handle `<|vision_start|><|image_pad|><|vision_end|>` tokens

**Where:** [utils/inference.py](../utils/inference.py#L232-L256)

```python
# Use the SAME chat template as HuggingFace
prompt = self.processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# Extract images using the same utility as HuggingFace
image_inputs, video_inputs = process_vision_info(messages)
```

---

### ✅ 3. Silent differences in tokenizer behavior

**Status: ADDRESSED**

- ✅ Load `AutoProcessor` to match HF tokenizer
- ✅ Use `trust_remote_code=True` in LLM constructor
- ✅ Use same `apply_chat_template` to ensure identical token sequences

**Where:** [utils/inference.py](../utils/inference.py#L199)

```python
# Load processor for apply_chat_template (to match HF behavior exactly)
self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)

self.llm = LLM(
    ...
    trust_remote_code=True,  # Required for Qwen3-VL tokenizer compatibility
    ...
)
```

---

### ✅ 4. KV cache & context length misunderstandings

**Status: DOCUMENTED**

- ✅ Set `max_model_len=32768` explicitly
- ✅ Added warning about image token costs (~200-400 tokens per image)
- ✅ Document that 15 images can use 3000-6000 tokens

**Where:** [utils/inference.py](../utils/inference.py#L218-L223)

```python
# WARNING: Image token cost
# Each image at 640x480 with these settings uses ~200-400 tokens
# With max 15 images, that's 3000-6000 tokens just for images
# Ensure prompts + images + response fit within max_model_len
```

**Recommendations:**
- Monitor actual token usage per request
- Reduce `max_model_len` if memory pressure is high
- Consider fewer images per prompt for long conversations

---

### ✅ 5. Throughput vs latency assumptions

**Status: DOCUMENTED & WARNED**

- ✅ Documented that single-request latency may be SLOWER than HF
- ✅ Added warnings in `compare_backends.py` about sequential vs batched
- ✅ Documented expected speedup ranges:
  - Sequential: 0.8-2x (may be slower!)
  - Batched (4-8): 2-4x
  - Batched (16+): 4-8x
  - API serving: 5-10x throughput

**Where:** 
- [utils/inference.py](../utils/inference.py#L169-L183) (class docstring)
- [evaluation/compare_backends.py](../evaluation/compare_backends.py#L268-L287)

**Key insight:** vLLM optimizes **throughput**, not **latency**. Use it when:
- Batching many requests
- Serving APIs with concurrent requests
- NOT for offline single-shot inference

---

### ✅ 6. Non-determinism surprises

**Status: ADDRESSED & DOCUMENTED**

- ✅ Use `temperature=0` for deterministic generation
- ✅ Documented that outputs may have minor differences due to:
  - Different CUDA kernels
  - KV cache management differences
  - Floating-point precision differences
- ✅ Implemented output comparison in `compare_backends.py`

**Where:** [evaluation/compare_backends.py](../evaluation/compare_backends.py#L48-L110)

**Expectation:** Outputs should be **semantically identical** but may not match character-for-character.

---

### ✅ 7. Assuming model supports features just because HF version did

**Status: OK (not using advanced features)**

We don't use:
- ❌ Logits processors
- ❌ Custom stopping criteria
- ❌ Streaming token-level probabilities
- ❌ Constrained generation

If needed in future, implement post-processing outside vLLM.

---

### ✅ 8. GPU memory assumptions (Qwen3-VL specific)

**Status: CONFIGURED**

- ✅ Set `gpu_memory_utilization=0.85` (conservative)
- ✅ Set `max_model_len=32768` (not full 262K context)
- ✅ Set `limit_mm_per_prompt={"image": 15, "video": 0}`
- ✅ Added `mm_processor_kwargs` for consistent preprocessing
- ✅ Set `OMP_NUM_THREADS=1` to reduce CPU contention

**Where:** [utils/inference.py](../utils/inference.py#L171-L226)

**Tuning recommendations:**
- If OOM: Reduce `gpu_memory_utilization` to 0.7-0.8
- If OOM: Reduce `max_model_len` to 16384
- If OOM: Reduce `limit_mm_per_prompt["image"]` to 10 or fewer
- If multiple instances: Ensure `OMP_NUM_THREADS=1` is set

---

### ⚠️ 9. Forgetting vLLM is async-by-design

**Status: PARTIALLY ADDRESSED**

- ✅ Implemented `generate_batch()` method for batching
- ⚠️  Current pipeline (sequential.py) runs steps sequentially
- ⚠️  Not using async APIs

**Current usage:**
```python
# Sequential - doesn't leverage vLLM's strength
for step in range(num_steps):
    output = backend.generate(messages, max_new_tokens=256)
```

**Better usage (future improvement):**
```python
# Batch multiple questions together
batch_messages = [messages_q1, messages_q2, ..., messages_q8]
outputs = backend.generate_batch(batch_messages, max_new_tokens=256)
```

**Recommendation:** Consider batching questions or using async API for production.

---

### ✅ 10. Testing only text-only paths

**Status: ADDRESSED**

- ✅ All tests include multimodal (image) inputs
- ✅ Test with multiple images per prompt (up to 15)
- ✅ Test with long prompts (exploration history)
- ✅ Compare outputs between HF and vLLM

**Where:** [evaluation/compare_backends.py](../evaluation/compare_backends.py)

**Test coverage:**
- ✅ Single image + text
- ✅ Multiple images + text
- ✅ Long conversation history
- ⚠️  Mixed batch (image + text-only) - not tested yet

---

## Summary

| Pitfall | Status | Priority |
|---------|--------|----------|
| 1. HF-style generate() | ✅ Fixed | High |
| 2. Multimodal formatting | ✅ Fixed (critical) | Critical |
| 3. Tokenizer differences | ✅ Fixed | High |
| 4. Context length | ✅ Documented | Medium |
| 5. Throughput vs latency | ✅ Documented | High |
| 6. Non-determinism | ✅ Addressed | Medium |
| 7. Unsupported features | ✅ OK | Low |
| 8. GPU memory | ✅ Configured | Medium |
| 9. Async/batching | ⚠️  Partial | Low |
| 10. Multimodal testing | ✅ Done | High |

**Critical fixes applied:**
1. ✅ Use `apply_chat_template` (not manual prompts)
2. ✅ Use `process_vision_info` (not manual image loading)
3. ✅ Set `trust_remote_code=True`
4. ✅ Configure `mm_processor_kwargs` to match HF
5. ✅ Use `SamplingParams` correctly
6. ✅ Document throughput vs latency expectations

**Remaining improvements:**
- Consider batching for production use
- Monitor token usage with long conversations
- Test mixed batches (multimodal + text-only)

---

## Quick Reference: Key Configuration

```python
# vLLM Backend Configuration
VLLMBackend(
    model_id="Qwen/Qwen3-VL-8B-Instruct",
    max_model_len=32768,           # Context budget (not full 262K)
    max_num_seqs=8,                # Batch size for throughput
    gpu_memory_utilization=0.85,   # Conservative allocation
    tensor_parallel_size=1,        # Single GPU
    dtype="float16",               # FP16 for efficiency
)

# Key settings in LLM():
limit_mm_per_prompt={"image": 15, "video": 0}  # Disable video
mm_processor_kwargs={
    "min_pixels": 28 * 28,
    "max_pixels": 1280 * 28 * 28,
}
trust_remote_code=True  # Required for Qwen3-VL

# Environment:
OMP_NUM_THREADS=1  # Reduce CPU contention
```

---

## Testing Recommendations

1. **Output consistency test:**
   ```bash
   python evaluation/compare_backends.py --num-questions 5 --num-steps 3
   ```
   Expected: 80-100% output similarity

2. **Memory profiling:**
   - Monitor GPU memory with `nvidia-smi`
   - Ensure enough headroom for KV cache growth

3. **Throughput test:**
   - Implement batched evaluation
   - Compare tokens/second between HF and vLLM

4. **Long context test:**
   - Test with 10+ images
   - Monitor token counts
   - Ensure no truncation

---

**Last updated:** January 21, 2026
**Version:** 1.0
