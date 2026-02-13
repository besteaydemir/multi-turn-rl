#!/usr/bin/env python3
"""
vLLM Backend with KV Cache Optimization for Sequential Multi-Turn Inference.

This module implements a vLLM backend that maintains the KV cache across
multiple inference calls within a session. This is designed for sequential
pipelines where we iteratively add images to the context.

Key optimization: Instead of discarding KV cache between steps, we keep
the LLM instance warm and rely on vLLM's prefix caching to reuse KV states
for shared prefixes.

Approach B Implementation:
- Enable prefix_caching in vLLM
- Maintain consistent prompt formatting to maximize cache hits
- Keep model loaded between steps (no recreation)
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time


class VLLMKVCacheBackend:
    """
    vLLM backend optimized for sequential multi-turn inference with KV cache reuse.
    
    Key features:
    - enable_prefix_caching=True for automatic KV cache sharing
    - Persistent LLM instance across multiple generate() calls
    - Timing metrics to measure cache effectiveness
    """
    
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        cache_dir: str = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir",
        max_model_len: int = 32768,
        max_num_seqs: int = 4,  # Lower for sequential, single question at a time
        gpu_memory_utilization: float = 0.85,
        tensor_parallel_size: int = 1,
        dtype: str = "float16",
        max_images: int = 48,  # Support up to 48 images (more than 32)
    ):
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor
        import os
        
        self.model_id = model_id
        self.cache_dir = cache_dir
        
        print(f"[vLLM KV-Cache Backend] Loading {model_id}...")
        print(f"[vLLM KV-Cache Backend] Settings: max_num_seqs={max_num_seqs}, dtype={dtype}")
        print(f"[vLLM KV-Cache Backend] ENABLED: prefix_caching for KV reuse")
        
        # Set environment variables
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
        os.environ["OMP_NUM_THREADS"] = "1"
        
        # Load processor for chat template
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        
        self.llm = LLM(
            model=model_id,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            trust_remote_code=True,
            download_dir=cache_dir,
            # Support many images
            limit_mm_per_prompt={"image": max_images, "video": 0},
            # Match HF processor settings
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
            },
            # CRITICAL: Enable prefix caching for KV reuse
            enable_prefix_caching=True,
        )
        
        # Default sampling params (deterministic)
        self.default_sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1024,
        )
        
        # Timing stats
        self.inference_times = []
        self.cache_hit_ratio = []
        
        print("[vLLM KV-Cache Backend] Model loaded with prefix caching enabled.")
    
    def _prepare_prompt(self, messages: List[Dict]) -> Tuple[str, List]:
        """Convert chat messages to vLLM format."""
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        
        # Use same chat template as HF
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Extract images
        image_inputs, video_inputs = process_vision_info(messages)
        
        images = []
        if image_inputs:
            for img in image_inputs:
                if isinstance(img, Image.Image):
                    images.append(img)
                elif isinstance(img, str):
                    images.append(Image.open(img).convert("RGB"))
                else:
                    images.append(img)
        
        return prompt, images
    
    def generate(
        self,
        messages: List[Dict],
        max_new_tokens: int = 1024,
    ) -> str:
        """
        Generate with KV cache optimization.
        
        The key insight: vLLM's prefix caching automatically detects
        when prompts share a common prefix and reuses the KV cache.
        Since our sequential prompts grow by appending new images,
        the prefix (earlier images) stays the same and gets cached.
        """
        from vllm import SamplingParams
        
        start_time = time.time()
        
        prompt, images = self._prepare_prompt(messages)
        
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=max_new_tokens,
        )
        
        # Create input with multi-modal data
        if images:
            inputs = [{
                "prompt": prompt,
                "multi_modal_data": {"image": images},
            }]
        else:
            inputs = [{"prompt": prompt}]
        
        outputs = self.llm.generate(inputs, sampling_params)
        
        elapsed = time.time() - start_time
        self.inference_times.append(elapsed)
        
        result = outputs[0].outputs[0].text
        
        # Log timing info (helpful for debugging cache effectiveness)
        num_images = len(images) if images else 0
        print(f"  [KV-Cache] Generated in {elapsed:.2f}s with {num_images} images")
        
        return result
    
    def generate_batch(
        self,
        batch_messages: List[List[Dict]],
        max_new_tokens: int = 1024,
    ) -> List[str]:
        """Generate for a batch of messages."""
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=max_new_tokens,
        )
        
        inputs = []
        for messages in batch_messages:
            prompt, images = self._prepare_prompt(messages)
            if images:
                inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": {"image": images},
                })
            else:
                inputs.append({"prompt": prompt})
        
        outputs = self.llm.generate(inputs, sampling_params)
        return [output.outputs[0].text for output in outputs]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get timing statistics."""
        if not self.inference_times:
            return {"total_inferences": 0}
        
        import numpy as np
        times = np.array(self.inference_times)
        return {
            "total_inferences": len(times),
            "total_time": float(times.sum()),
            "mean_time": float(times.mean()),
            "std_time": float(times.std()),
            "min_time": float(times.min()),
            "max_time": float(times.max()),
        }
    
    def reset_stats(self):
        """Reset timing statistics."""
        self.inference_times = []
        self.cache_hit_ratio = []
    
    def cleanup(self):
        """Clean up resources."""
        print(f"[vLLM KV-Cache Backend] Final stats: {self.get_stats()}")
        del self.llm
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_kv_cache_backend(
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
    cache_dir: str = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir",
) -> VLLMKVCacheBackend:
    """Factory function to create KV-cache enabled backend."""
    return VLLMKVCacheBackend(
        model_id=model_id,
        cache_dir=cache_dir,
    )
