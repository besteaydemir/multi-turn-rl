#!/usr/bin/env python3
"""
Inference backend abstraction for both HuggingFace and vLLM.

Supports:
- HuggingFace transformers (sequential, exact same behavior as original)
- vLLM (batched inference for significant speedup)

Both backends support Qwen3-VL with multi-image inputs.
"""

import abc
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np


class InferenceBackend(abc.ABC):
    """Abstract base class for inference backends."""
    
    @abc.abstractmethod
    def generate(
        self,
        messages: List[Dict],
        max_new_tokens: int = 1024,
    ) -> str:
        """
        Generate text from a single message (for sequential inference).
        
        Args:
            messages: Chat messages with text and image content
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        pass
    
    @abc.abstractmethod
    def generate_batch(
        self,
        batch_messages: List[List[Dict]],
        max_new_tokens: int = 1024,
    ) -> List[str]:
        """
        Generate text for a batch of messages (for batched inference).
        
        Args:
            batch_messages: List of chat messages, each with text and image content
            max_new_tokens: Maximum tokens to generate per sample
            
        Returns:
            List of generated text responses
        """
        pass
    
    @abc.abstractmethod
    def cleanup(self):
        """Clean up resources."""
        pass


class HuggingFaceBackend(InferenceBackend):
    """HuggingFace transformers backend - sequential inference."""
    
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        cache_dir: str = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir",
        device: str = "cuda",
    ):
        import torch
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        
        self.device = device
        self.model_id = model_id
        self.cache_dir = cache_dir
        
        print(f"[HF Backend] Loading {model_id} on {device}...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype="auto",
            device_map="auto",
            cache_dir=cache_dir
        )
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.processor.tokenizer.padding_side = 'left'
        self.model.to(device)
        print("[HF Backend] Model loaded.")
    
    def generate(
        self,
        messages: List[Dict],
        max_new_tokens: int = 1024,
    ) -> str:
        """Generate for a single sample (sequential)."""
        import torch
        from qwen_vl_utils import process_vision_info
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]
        
        return output_text
    
    def generate_batch(
        self,
        batch_messages: List[List[Dict]],
        max_new_tokens: int = 1024,
    ) -> List[str]:
        """
        Generate for a batch - HF backend does this sequentially.
        For true batching, use vLLM backend.
        """
        results = []
        for messages in batch_messages:
            result = self.generate(messages, max_new_tokens)
            results.append(result)
        return results
    
    def cleanup(self):
        """Clean up GPU memory."""
        import torch
        del self.model
        del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class VLLMBackend(InferenceBackend):
    """
    vLLM backend - batched inference with continuous batching.
    
    Key features:
    - Automatic batching of multiple requests
    - Efficient KV cache management
    - ~4-8x speedup over sequential HF inference (when batching multiple requests)
    
    Configuration tips applied:
    - limit_mm_per_prompt: {"image": 15, "video": 0} to save memory
    - OMP_NUM_THREADS=1 to avoid CPU contention
    - mm_processor_kwargs for consistent image preprocessing with HF
    
    IMPORTANT MIGRATION NOTES:
    -------------------------
    1. Uses apply_chat_template (same as HF) to ensure identical prompts
    2. Uses process_vision_info (same as HF) for consistent image handling
    3. Deterministic with temperature=0, but small numerical differences may occur
       due to different CUDA kernels and KV cache management
    4. Image tokens consume significant context length - each image can be 
       several hundred tokens depending on resolution and model settings
    5. Single-request latency may be SLOWER than HF - vLLM optimizes for throughput
    6. For true speedup, use generate_batch() with multiple requests
    7. trust_remote_code=True ensures tokenizer compatibility with HF
    
    Performance expectations:
    - Sequential (no batching): 1-2x speedup or slower
    - Batching (8+ requests): 4-8x speedup
    - API serving with mixed requests: 5-10x higher throughput
    """
    
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        cache_dir: str = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir",
        max_model_len: int = 32768,
        max_num_seqs: int = 8,  # Max batch size
        gpu_memory_utilization: float = 0.85,
        tensor_parallel_size: int = 1,
        dtype: str = "float16",  # FP16 as requested
    ):
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor
        
        self.model_id = model_id
        self.cache_dir = cache_dir
        
        print(f"[vLLM Backend] Loading {model_id}...")
        print(f"[vLLM Backend] Settings: max_num_seqs={max_num_seqs}, dtype={dtype}")
        
        # Set environment variables for optimization
        import os
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
        # Reduce CPU contention when running multiple vLLM instances
        os.environ["OMP_NUM_THREADS"] = "1"
        
        # Load processor for apply_chat_template (to match HF behavior exactly)
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        
        self.llm = LLM(
            model=model_id,
            max_model_len=max_model_len,  # Context length budget
            max_num_seqs=max_num_seqs,  # Max parallel requests (for batching)
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            trust_remote_code=True,  # Required for Qwen3-VL tokenizer compatibility
            download_dir=cache_dir,
            # Pitfall #2: Disable video to save memory (we only use images)
            # Setting video=0 prevents allocating memory for video embeddings
            limit_mm_per_prompt={"image": 15, "video": 0},
            # Pitfall #2 & #3: Match HF processor settings for consistent preprocessing
            # These pixel settings must match what HF uses to get identical outputs
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
            },
        )
        
        # WARNING: Image token cost
        # Each image at 640x480 with these settings uses ~200-400 tokens
        # With max 15 images, that's 3000-6000 tokens just for images
        # Ensure prompts + images + response fit within max_model_len
        
        # Deterministic sampling (no do_sample)
        self.default_sampling_params = SamplingParams(
            temperature=0,  # Deterministic
            max_tokens=1024,
        )
        
        print("[vLLM Backend] Model loaded.")
    
    def _prepare_prompt(self, messages: List[Dict]) -> Tuple[str, List]:
        """
        Convert chat messages to vLLM format with multi-modal data.
        
        CRITICAL PITFALL AVOIDANCE:
        ---------------------------
        Pitfall #2: We use the SAME apply_chat_template as HuggingFace to avoid
                    incorrect multimodal input formatting. Do NOT manually construct
                    prompts with hardcoded tokens like <|image|> or <|vision_start|>.
        
        Pitfall #3: We use the SAME process_vision_info as HuggingFace to ensure
                    identical tokenizer behavior and image handling.
        
        This ensures:
        - Correct <|vision_start|><|image_pad|><|vision_end|> token placement
        - Proper system message formatting (<|im_start|>system...)
        - Identical prompt structure between HF and vLLM
        - Same image ordering and preprocessing
        """
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        
        # Use the SAME chat template as HuggingFace for consistent prompts
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Extract images using the same utility as HuggingFace
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Convert to list of PIL images for vLLM
        images = []
        if image_inputs:
            for img in image_inputs:
                if isinstance(img, Image.Image):
                    images.append(img)
                elif isinstance(img, str):
                    # If it's a path, load it
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
        Generate for a single sample.
        
        WARNING: Single-request inference does NOT benefit much from vLLM!
        For true speedup, use generate_batch() with multiple requests.
        """
        from vllm import SamplingParams
        
        prompt, images = self._prepare_prompt(messages)
        
        # Pitfall #1: vLLM uses SamplingParams, not HF's generate() kwargs
        # - max_new_tokens -> max_tokens
        # - do_sample is implicit (temperature > 0 enables sampling)
        # - no support for output_scores, return_dict_in_generate, etc.
        sampling_params = SamplingParams(
            temperature=0,  # Deterministic (greedy decoding)
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
        
        return outputs[0].outputs[0].text
    
    def generate_batch(
        self,
        batch_messages: List[List[Dict]],
        max_new_tokens: int = 1024,
    ) -> List[str]:
        """
        Generate for a batch of messages efficiently.
        
        This is where vLLM really shines - it batches all requests
        and uses continuous batching for maximum throughput.
        """
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=0,  # Deterministic
            max_tokens=max_new_tokens,
        )
        
        # Prepare all inputs
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
        
        # Batch generate
        outputs = self.llm.generate(inputs, sampling_params)
        
        # Extract text from outputs
        results = [output.outputs[0].text for output in outputs]
        return results
    
    def cleanup(self):
        """Clean up resources."""
        del self.llm
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_inference_backend(
    backend: str = "hf",
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
    cache_dir: str = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir",
    **kwargs
) -> InferenceBackend:
    """
    Factory function to create the appropriate inference backend.
    
    Args:
        backend: "hf" for HuggingFace, "vllm" for vLLM
        model_id: Model identifier
        cache_dir: Cache directory for model weights
        **kwargs: Additional arguments passed to the backend
        
    Returns:
        InferenceBackend instance
    """
    if backend.lower() == "hf":
        return HuggingFaceBackend(model_id=model_id, cache_dir=cache_dir, **kwargs)
    elif backend.lower() == "vllm":
        return VLLMBackend(model_id=model_id, cache_dir=cache_dir, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose 'hf' or 'vllm'.")
