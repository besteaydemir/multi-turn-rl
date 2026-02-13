#!/usr/bin/env python3
"""
vLLM-backed actor for fast rollout generation **with per-token log-probs**.

Responsibilities
----------------
1. ``generate(messages)`` → text only  (drop-in for eval / dummy rollout).
2. ``generate_with_logprobs(messages)`` → ``GenerateOutput`` containing
   *text*, *token_ids*, *token_logprobs* — everything PPO needs from the
   behaviour policy π_old.
3. ``generate_batch_with_logprobs(batch_messages)`` → list of the above.
4. ``load_weights(state_dict)`` — hot-swap weights from the HF actor after
   each PPO update (online weight sync).

Design notes
~~~~~~~~~~~~
* vLLM natively returns ``logprobs`` per output token when
  ``SamplingParams(logprobs=1)`` is set.  We extract the *chosen* token
  log-prob from the ``Logprob`` objects attached to each ``CompletionOutput``.
* The returned ``token_ids`` tensor is the sequence of *generated* token IDs
  (prompt tokens are **excluded**) so it aligns 1-1 with ``token_logprobs``.
* For PPO we also need π_ref log-probs.  Two strategies are supported:
  (a) *Fast*: compute them on the vLLM ref actor (another ``ActorVLLM``
      instance with frozen weights) — same API.
  (b) *Fallback*: retokenise with HF tokenizer and forward through
      ``ActorHF.forward_logprobs`` / ``ReferenceModel.forward_logprobs``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------
@dataclass
class GenerateOutput:
    """Everything the rollout collector needs from one generation call."""

    text: str = ""
    token_ids: Optional[torch.Tensor] = None          # (gen_len,)  int64
    token_logprobs: Optional[torch.Tensor] = None      # (gen_len,)  float32
    prompt_token_ids: Optional[torch.Tensor] = None    # (prompt_len,) int64
    finish_reason: Optional[str] = None                 # "stop" | "length"

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable snapshot (tensors → lists)."""
        return {
            "text": self.text,
            "token_ids": self.token_ids.tolist() if self.token_ids is not None else None,
            "token_logprobs": self.token_logprobs.tolist() if self.token_logprobs is not None else None,
            "prompt_token_ids": self.prompt_token_ids.tolist() if self.prompt_token_ids is not None else None,
            "finish_reason": self.finish_reason,
        }


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------
class ActorVLLM:
    """
    Thin wrapper around vLLM ``LLM`` for rollout generation with log-probs.

    Designed for Qwen3-VL multimodal models with multi-image inputs.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        cache_dir: str = "/dss/mcmlscratch/06/di38riq",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 32768,
        max_images: int = 32,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 1024,
        dtype: str = "float16",
        enforce_eager: bool = False,
    ):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

        self._tp = tensor_parallel_size
        self._gpu_mem = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._max_images = max_images
        self._dtype = dtype
        self._enforce_eager = enforce_eager

        self._llm = None
        self._processor = None
        self._tokenizer = None

    # ─────────────────────────── lazy init ───────────────────────────
    def _init(self) -> None:
        if self._llm is not None:
            return
        from vllm import LLM
        from transformers import AutoProcessor

        os.environ.setdefault("OMP_NUM_THREADS", "1")

        print(f"[ActorVLLM] Loading {self.model_id} (tp={self._tp}, "
              f"gpu_mem={self._gpu_mem}, dtype={self._dtype}) …")
        self._llm = LLM(
            model=self.model_id,
            tensor_parallel_size=self._tp,
            gpu_memory_utilization=self._gpu_mem,
            trust_remote_code=True,
            download_dir=self.cache_dir,
            dtype=self._dtype,
            limit_mm_per_prompt={"image": self._max_images, "video": 0},
            max_model_len=self._max_model_len,
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
            },
            enforce_eager=self._enforce_eager,
        )
        self._processor = AutoProcessor.from_pretrained(
            self.model_id, cache_dir=self.cache_dir
        )
        self._tokenizer = self._processor.tokenizer
        print("[ActorVLLM] Ready.")

    @property
    def tokenizer(self):
        self._init()
        return self._tokenizer

    # ─────────────────── prompt / image preparation ──────────────────
    def _prepare_inputs(
        self, messages: List[Dict[str, Any]]
    ) -> tuple:
        """
        Convert chat messages → (prompt_str, list_of_PIL_images).

        Uses the same ``apply_chat_template`` and ``process_vision_info``
        pipeline as the existing ``VLLMBackend`` in ``utils/inference.py``
        to guarantee identical tokenisation.
        """
        from PIL import Image as PILImage
        from qwen_vl_utils import process_vision_info

        # Disable Qwen3's "thinking" mode to get direct JSON output
        # Without this, model outputs <think>...</think> tags first
        prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,  # CRITICAL for structured output
        )
        image_inputs, _ = process_vision_info(messages)

        images: List[Any] = []
        if image_inputs:
            for img in image_inputs:
                if isinstance(img, PILImage.Image):
                    images.append(img)
                elif isinstance(img, str):
                    images.append(PILImage.open(img).convert("RGB"))
                else:
                    images.append(img)

        return prompt, images

    # ─────────────────────── text-only generate ──────────────────────
    def generate(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Generate text only (backward-compatible, same as eval backend)."""
        out = self.generate_with_logprobs(
            messages, temperature=temperature, max_new_tokens=max_new_tokens
        )
        return out.text

    # ────────────────── generate **with** logprobs ───────────────────
    def generate_with_logprobs(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> GenerateOutput:
        """
        Generate text and return per-token IDs + log-probs.

        Returns
        -------
        GenerateOutput
            ``.text``           — decoded string
            ``.token_ids``      — ``(gen_len,)`` int64 tensor of generated token IDs
            ``.token_logprobs`` — ``(gen_len,)`` float32 tensor of log π(token)
            ``.prompt_token_ids`` — ``(prompt_len,)`` int64 tensor of prompt token IDs
            ``.finish_reason``  — ``"stop"`` or ``"length"``
        """
        self._init()
        from vllm import SamplingParams

        prompt, images = self._prepare_inputs(messages)

        params = SamplingParams(
            temperature=temperature if temperature is not None else self.temperature,
            top_p=self.top_p,
            max_tokens=max_new_tokens or self.max_new_tokens,
            logprobs=1,       # ← return top-1 log-prob per token
        )

        mm_data = {"image": images} if images else {}
        outputs = self._llm.generate(
            [{"prompt": prompt, "multi_modal_data": mm_data}],
            sampling_params=params,
        )

        return self._parse_vllm_output(outputs[0])

    # ────────────────── batched generate with logprobs ────────────────
    def generate_batch_with_logprobs(
        self,
        batch_messages: List[List[Dict[str, Any]]],
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> List[GenerateOutput]:
        """
        Batched generation — vLLM continuous-batching for throughput.

        Returns one ``GenerateOutput`` per input message list.
        """
        self._init()
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=temperature if temperature is not None else self.temperature,
            top_p=self.top_p,
            max_tokens=max_new_tokens or self.max_new_tokens,
            logprobs=1,
        )

        inputs = []
        for messages in batch_messages:
            prompt, images = self._prepare_inputs(messages)
            mm_data = {"image": images} if images else {}
            inputs.append({"prompt": prompt, "multi_modal_data": mm_data})

        outputs = self._llm.generate(inputs, sampling_params=params)
        return [self._parse_vllm_output(o) for o in outputs]

    # ────────────────── vLLM output → GenerateOutput ─────────────────
    @staticmethod
    def _parse_vllm_output(request_output) -> GenerateOutput:
        """
        Extract token IDs and per-token log-probs from a single
        ``vllm.RequestOutput``.

        vLLM attaches a ``List[Dict[int, Logprob]]`` to each
        ``CompletionOutput`` when ``logprobs >= 1``.  Each element is a
        dict mapping token-id → ``Logprob(logprob=..., ...)``.  We take
        the *sampled* token's log-prob (the one whose id matches the
        generated token).
        """
        completion = request_output.outputs[0]
        text = completion.text
        finish_reason = completion.finish_reason

        # -- token ids --
        gen_ids = list(completion.token_ids)              # tuple → list

        # -- per-token logprobs --
        lp_list: List[float] = []
        if completion.logprobs is not None:
            for step_idx, lp_dict in enumerate(completion.logprobs):
                # lp_dict: Dict[int, Logprob]
                # The sampled token id for this step:
                sampled_id = gen_ids[step_idx]
                if sampled_id in lp_dict:
                    lp_list.append(lp_dict[sampled_id].logprob)
                else:
                    # Shouldn't happen, but be defensive
                    lp_list.append(0.0)
        else:
            # No logprobs returned (shouldn't happen with logprobs=1)
            lp_list = [0.0] * len(gen_ids)

        # -- prompt token ids --
        prompt_ids = list(request_output.prompt_token_ids) if request_output.prompt_token_ids else []

        return GenerateOutput(
            text=text,
            token_ids=torch.tensor(gen_ids, dtype=torch.long),
            token_logprobs=torch.tensor(lp_list, dtype=torch.float32),
            prompt_token_ids=torch.tensor(prompt_ids, dtype=torch.long),
            finish_reason=finish_reason,
        )

    # ─────────────────── weight sync from HF actor ───────────────────
    def load_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """
        Hot-load weights from an HF ``state_dict`` into the running vLLM
        engine, keeping the KV cache warm.
        
        Compatible with vLLM 0.4.x through 0.15.x (API changed across versions).
        """
        self._init()
        
        # Try multiple API paths that work across vLLM versions
        model = None
        
        # === Method 1: vLLM 0.8+ collective_rpc (recommended for TP > 1) ===
        if hasattr(self._llm, 'collective_rpc'):
            try:
                # This works for tensor parallel setups in newer vLLM
                self._llm.collective_rpc("load_weights", args=(list(state_dict.items()),))
                print("[ActorVLLM] Weights synchronised from HF (via collective_rpc).")
                return
            except Exception as e:
                print(f"[ActorVLLM] collective_rpc failed: {e}, trying other methods...")
        
        # === Method 2: vLLM 0.15.x+ get_model() ===
        if hasattr(self._llm, 'get_model'):
            try:
                model = self._llm.get_model()
            except Exception:
                pass
        
        # === Method 3: Try llm_engine paths ===
        if model is None and hasattr(self._llm, 'llm_engine'):
            engine = self._llm.llm_engine
            
            # vLLM 0.6.x+ path via model_executor
            if hasattr(engine, 'model_executor'):
                executor = engine.model_executor
                # Try different executor structures
                if hasattr(executor, 'driver_worker'):
                    try:
                        model = executor.driver_worker.model_runner.model
                    except Exception:
                        pass
                elif hasattr(executor, 'model'):
                    model = executor.model
            
            # vLLM 0.5.x uses _model_executor on engine
            if model is None and hasattr(engine, '_model_executor'):
                try:
                    executor = engine._model_executor
                    if hasattr(executor, 'driver_worker'):
                        model = executor.driver_worker.model_runner.model
                    elif hasattr(executor, 'model'):
                        model = executor.model
                except Exception:
                    pass
        
        # === Method 4: Direct model reference on LLM (some versions) ===
        if model is None and hasattr(self._llm, 'model'):
            model = self._llm.model
        
        if model is None:
            # Debug: print available attributes to help diagnose
            llm_attrs = [a for a in dir(self._llm) if not a.startswith('__')]
            engine_attrs = []
            if hasattr(self._llm, 'llm_engine'):
                engine_attrs = [a for a in dir(self._llm.llm_engine) if not a.startswith('__')]
            print(f"[ActorVLLM] Weight sync failed: Could not locate model in vLLM engine.")
            print(f"[ActorVLLM] LLM attrs (sample): {llm_attrs[:15]}")
            print(f"[ActorVLLM] Engine attrs (sample): {engine_attrs[:15]}")
            print("[ActorVLLM] Continuing without weight sync (rollouts will use original policy).")
            return
        
        try:
            model.load_weights(state_dict.items())
            print("[ActorVLLM] Weights synchronised from HF.")
        except Exception as e:
            print(f"[ActorVLLM] Weight sync failed: {e}")
            print("[ActorVLLM] Continuing without weight sync.")

    # ──────────────────── cleanup ────────────────────────────────────
    def cleanup(self) -> None:
        """Release GPU memory."""
        if self._llm is not None:
            del self._llm
            self._llm = None
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
