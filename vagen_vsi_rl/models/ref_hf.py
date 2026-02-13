#!/usr/bin/env python3
"""
Frozen reference model for KL penalty in PPO.

Loaded once at the start of training; never updated.  Used to compute
``log π_ref(a|s)`` for the KL term ``β · (log π_θ − log π_ref)``.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class ReferenceModel:
    """Read-only copy of the actor used for KL regularisation."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        cache_dir: str = "/dss/mcmlscratch/06/di38riq",
        torch_dtype: str = "bfloat16",
        device: str = "cuda",
    ):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self._device = device

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        self.torch_dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        self.model = None

    def load(self) -> None:
        print(f"[RefModel] Loading frozen reference {self.model_id} …")
        
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
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        print("[RefModel] Ready (frozen).")

    @torch.no_grad()
    def forward_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Same interface as ``ActorHF.forward_logprobs`` but no grad."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        shifted_logprobs = log_probs[:, :-1, :]
        shifted_labels = labels[:, 1:]

        per_token_lp = shifted_logprobs.gather(
            dim=-1, index=shifted_labels.unsqueeze(-1)
        ).squeeze(-1)

        pad = torch.zeros(per_token_lp.shape[0], 1, device=per_token_lp.device, dtype=per_token_lp.dtype)
        return torch.cat([pad, per_token_lp], dim=1)
