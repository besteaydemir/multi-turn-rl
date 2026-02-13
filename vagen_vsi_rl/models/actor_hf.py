#!/usr/bin/env python3
"""
HuggingFace-backed actor for gradient computation during PPO updates.

This model is **not** used during rollout (too slow for multi-turn
generation); instead the vLLM actor collects trajectories and then the
HF actor re-computes log-probs on the collected tokens so we can compute
the PPO loss.

When ``--no-vllm`` is used, this actor is also used for inference (slower
but fits on smaller GPUs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Output container (same as ActorVLLM for compatibility)
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


class ActorHF(nn.Module):
    """Wraps a HuggingFace ``AutoModelForCausalLM`` for PPO training.
    
    Supports optional LoRA for memory-efficient fine-tuning.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        cache_dir: str = "/dss/mcmlscratch/06/di38riq",
        torch_dtype: str = "bfloat16",
        device: str = "cuda",
        # LoRA config
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
    ):
        super().__init__()
        self.model_id = model_id
        self.cache_dir = cache_dir
        self._device = device
        
        # LoRA config
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        self.torch_dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        self.model = None
        self.tokenizer = None
        self.processor = None
        self._is_peft_model = False

    def load(self) -> None:
        """Load the model (call once before training)."""
        from transformers import AutoProcessor

        print(f"[ActorHF] Loading {self.model_id} (LoRA={self.use_lora})…")
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, cache_dir=self.cache_dir, trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer

        # Qwen3-VL requires Qwen3VLForConditionalGeneration, not AutoModelForCausalLM
        if "qwen3" in self.model_id.lower() and "vl" in self.model_id.lower():
            from transformers import Qwen3VLForConditionalGeneration
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
                device_map=self._device,
            )
        else:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
                device_map=self._device,
            )
        
        # Apply LoRA if enabled
        if self.use_lora:
            self._apply_lora()
        
        # Enable gradient checkpointing to reduce memory (trades compute for memory)
        # Skip for LoRA - it's already memory-efficient and has compatibility issues
        # with gradient checkpointing
        if not self.use_lora:
            self.model.gradient_checkpointing_enable()
            ckpt_info = ", gradient checkpointing enabled"
        else:
            ckpt_info = ""
        
        self.model.train()
        lora_info = f", LoRA r={self.lora_r}, α={self.lora_alpha}" if self.use_lora else ""
        print(f"[ActorHF] Ready (training mode{ckpt_info}{lora_info}).")

    def _apply_lora(self) -> None:
        """Apply LoRA adapters to the model."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError(
                "PEFT is required for LoRA. Install with: pip install peft"
            )
        
        print(f"[ActorHF] Applying LoRA: r={self.lora_r}, α={self.lora_alpha}, "
              f"dropout={self.lora_dropout}, targets={self.lora_target_modules}")
        
        # Create LoRA config
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self._is_peft_model = True
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[ActorHF] LoRA applied: {trainable_params:,} trainable / {total_params:,} total "
              f"({100 * trainable_params / total_params:.2f}%)")

    # ── forward: compute log-probs on existing token sequences ──
    def forward_logprobs(
        self,
        input_ids: torch.Tensor,          # (B, seq_len)
        attention_mask: torch.Tensor,      # (B, seq_len)
        labels: torch.Tensor,             # (B, seq_len)  shifted inside
    ) -> torch.Tensor:
        """
        Return per-token log-probs for the *label* positions.

        Shape of returned tensor: ``(B, seq_len)`` (log-prob at each position
        for the corresponding label token; positions outside ``labels`` are 0).
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # (B, seq_len, vocab)
        log_probs = torch.log_softmax(logits, dim=-1)

        # gather log-prob of the actual next token
        # shift: logits[t] predicts token[t+1]
        shifted_logprobs = log_probs[:, :-1, :]      # (B, seq-1, V)
        shifted_labels = labels[:, 1:]                 # (B, seq-1)

        per_token_lp = shifted_logprobs.gather(
            dim=-1, index=shifted_labels.unsqueeze(-1)
        ).squeeze(-1)  # (B, seq-1)

        # pad back to original length
        pad = torch.zeros(per_token_lp.shape[0], 1, device=per_token_lp.device, dtype=per_token_lp.dtype)
        per_token_lp = torch.cat([pad, per_token_lp], dim=1)  # (B, seq_len)
        return per_token_lp

    # ── convenience: export / import state dict ──
    def state_dict_for_sync(self) -> Dict[str, torch.Tensor]:
        """Return model weights suitable for ``ActorVLLM.load_weights``.
        
        For LoRA models, returns merged weights (full model with adapters applied).
        Note: vLLM sync with LoRA is expensive - consider using vLLM's native LoRA support.
        """
        if self._is_peft_model:
            # Merge LoRA weights without destroying the model
            print("[ActorHF] Merging LoRA weights for vLLM sync (this is slow)...")
            self.model.merge_adapter()
            base_model = self.model.get_base_model()
            state_dict = {k: v.cpu() for k, v in base_model.state_dict().items()}
            self.model.unmerge_adapter()
            return state_dict
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def save_lora(self, path: str) -> None:
        """Save LoRA adapter weights to disk."""
        if not self._is_peft_model:
            raise ValueError("Cannot save LoRA - model was not loaded with LoRA")
        from pathlib import Path
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        print(f"[ActorHF] Saved LoRA adapter to {save_path}")

    def load_lora(self, path: str) -> None:
        """Load LoRA adapter weights from disk."""
        if not self._is_peft_model:
            raise ValueError("Cannot load LoRA - model was not loaded with LoRA")
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(
            self.model.get_base_model(), 
            path,
            is_trainable=True,
        )
        print(f"[ActorHF] Loaded LoRA adapter from {path}")

    # ── inference: generate with logprobs (for --no-vllm mode) ──
    def generate_with_logprobs(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_new_tokens: int = 1024,
    ) -> GenerateOutput:
        """
        Generate text with per-token log-probs (for rollout collection).
        
        This is slower than vLLM but allows single-GPU operation.
        """
        from PIL import Image as PILImage
        from qwen_vl_utils import process_vision_info
        
        # Disable gradient checkpointing temporarily for generation (faster)
        # Handle both regular and PEFT models
        base_model = self.model.get_base_model() if self._is_peft_model else self.model
        checkpointing_was_enabled = getattr(base_model, 'is_gradient_checkpointing', False)
        if checkpointing_was_enabled:
            base_model.gradient_checkpointing_disable()
        
        self.model.eval()
        
        try:
            # Prepare inputs using Qwen's chat template
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,  # CRITICAL for structured output
            )
            
            # Process images
            image_inputs, _ = process_vision_info(messages)
            images = []
            if image_inputs:
                for img in image_inputs:
                    if isinstance(img, PILImage.Image):
                        images.append(img)
                    elif isinstance(img, str):
                        images.append(PILImage.open(img).convert("RGB"))
                    else:
                        images.append(img)
            
            # Tokenize with processor (handles images + text)
            if images:
                inputs = self.processor(
                    text=[prompt],
                    images=images,
                    return_tensors="pt",
                    padding=True,
                ).to(self._device)
            else:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                ).to(self._device)
            
            prompt_length = inputs["input_ids"].shape[1]
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else 1.0,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            
            # Extract generated tokens
            full_ids = outputs.sequences[0]  # (full_len,)
            gen_ids = full_ids[prompt_length:]  # (gen_len,)
            
            # Compute log-probs from scores
            scores = outputs.scores  # tuple of (1, vocab_size) for each generated token
            token_logprobs = []
            for i, score in enumerate(scores):
                log_probs = torch.log_softmax(score[0], dim=-1)  # (vocab_size,)
                token_id = gen_ids[i].item()
                token_logprobs.append(log_probs[token_id].item())
            
            # Decode text
            generated_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            
            # Determine finish reason
            finish_reason = "length"
            if gen_ids[-1].item() in [self.tokenizer.eos_token_id]:
                finish_reason = "stop"
            
            return GenerateOutput(
                text=generated_text,
                token_ids=gen_ids.cpu(),
                token_logprobs=torch.tensor(token_logprobs, dtype=torch.float32),
                prompt_token_ids=inputs["input_ids"][0, :prompt_length].cpu(),
                finish_reason=finish_reason,
            )
        finally:
            # Re-enable gradient checkpointing and training mode
            self.model.train()
            if checkpointing_was_enabled:
                base_model = self.model.get_base_model() if self._is_peft_model else self.model
                base_model.gradient_checkpointing_enable()
