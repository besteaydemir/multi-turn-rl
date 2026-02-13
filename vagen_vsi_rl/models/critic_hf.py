#!/usr/bin/env python3
"""
Value-function critic for GAE advantage estimation.

Uses Qwen3-VL-4B as the backbone with a learned scalar value head on top
of the last hidden state.  Supports full multimodal input (images + text)
just like the actor, so the critic sees the same observation the policy saw.

Usage
-----
::

    critic = CriticHF(model_id="Qwen/Qwen3-VL-4B")
    critic.load()

    # per-token values
    values = critic(input_ids, attention_mask,
                    pixel_values=pv, image_grid_thw=grid)   # (B, seq)

    # end-of-turn scalar values
    v_t = critic.forward_at_indices(input_ids, attention_mask,
                                    end_indices=idx,
                                    pixel_values=pv,
                                    image_grid_thw=grid)    # (B,)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class CriticHF(nn.Module):
    """Qwen3-VL backbone + scalar value head for RL value estimation."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
        cache_dir: str = "/dss/mcmlscratch/06/di38riq",
        torch_dtype: str = "bfloat16",
        device: str = "cuda",
        freeze_backbone: bool = False,
        value_head_hidden: int = 256,
    ):
        super().__init__()
        self.model_id = model_id
        self.cache_dir = cache_dir
        self._device = device
        self.freeze_backbone = freeze_backbone
        self._value_head_hidden = value_head_hidden

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.torch_dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        # Populated by load()
        self.backbone: Optional[nn.Module] = None
        self.processor = None
        self.tokenizer = None
        self.value_head: Optional[nn.Module] = None

    # ── loading ──────────────────────────────────────────────────────

    def load(self, share_backbone: Optional[nn.Module] = None) -> None:
        """
        Load the backbone and build the value head.

        Parameters
        ----------
        share_backbone : nn.Module, optional
            If supplied (e.g. the actor's ``model``), the critic reuses it
            instead of loading its own copy — saves GPU memory when both
            models are the same architecture.  When sharing, the backbone
            hidden size is inferred from the shared module.
        """
        from transformers import AutoProcessor

        # ── processor / tokenizer (always needed for input prep) ──
        print(f"[CriticHF] Loading processor for {self.model_id} …")
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )
        self.tokenizer = self.processor.tokenizer

        # ── backbone ──
        if share_backbone is not None:
            self.backbone = share_backbone
            print("[CriticHF] Sharing backbone with actor.")
            self._sharing_backbone = True
        else:
            from transformers import Qwen3VLForConditionalGeneration
            print(f"[CriticHF] Loading backbone {self.model_id} …")
            self.backbone = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                dtype=self.torch_dtype,
                trust_remote_code=True,
                device_map=self._device,
            )
            self._sharing_backbone = False

        # ── optionally freeze backbone ──
        # IMPORTANT: Do NOT freeze when sharing backbone, as actor manages its own grad settings
        # (freezing would disable LoRA adapters)
        if self.freeze_backbone and not self._sharing_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("[CriticHF] Backbone parameters frozen.")
        elif self._sharing_backbone:
            print("[CriticHF] Skipping freeze (backbone owned by actor).")
        else:
            self.backbone.train()

        # ── discover hidden dimension from config ──
        # Qwen3VLConfig nests text params under text_config
        cfg = self.backbone.config
        hidden_dim = (
            cfg.text_config.hidden_size
            if hasattr(cfg, "text_config")
            else cfg.hidden_size
        )  # 2560 for 4B, 4096 for 8B
        print(f"[CriticHF] Backbone hidden_size = {hidden_dim}")

        # ── infer dtype from backbone (important when sharing) ──
        # Get dtype from first parameter of backbone
        backbone_dtype = next(self.backbone.parameters()).dtype
        if backbone_dtype != self.torch_dtype:
            print(f"[CriticHF] Adjusting value head dtype to match backbone: {backbone_dtype}")
            self.torch_dtype = backbone_dtype

        # ── build value head ──
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, self._value_head_hidden),
            nn.Tanh(),
            nn.Linear(self._value_head_hidden, 1),
        ).to(device=self._device, dtype=self.torch_dtype)

        # Initialise value head with small weights so initial values ≈ 0
        with torch.no_grad():
            nn.init.xavier_uniform_(self.value_head[0].weight)
            nn.init.zeros_(self.value_head[0].bias)
            nn.init.zeros_(self.value_head[2].weight)
            nn.init.zeros_(self.value_head[2].bias)

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[CriticHF] Ready. {trainable_params:,} / {total_params:,} params trainable."
        )

    # ── forward: per-token values ────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute a scalar value estimate at every token position.

        Parameters
        ----------
        input_ids : (B, seq_len)
        attention_mask : (B, seq_len)
        pixel_values : optional vision tensor from processor
        image_grid_thw : optional grid info from processor

        Returns
        -------
        values : (B, seq_len)  — scalar value at each position.
        """
        # Build kwargs for the backbone forward pass
        backbone_kwargs: Dict[str, Any] = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        if pixel_values is not None:
            backbone_kwargs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            backbone_kwargs["image_grid_thw"] = image_grid_thw

        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(**backbone_kwargs)
        else:
            outputs = self.backbone(**backbone_kwargs)

        # Use the last hidden state from the model
        # Qwen3VLForConditionalGeneration returns hidden_states tuple
        # when output_hidden_states=True; last element = final layer
        hidden = outputs.hidden_states[-1]  # (B, seq_len, H)

        # Detach if backbone is frozen to be safe
        if self.freeze_backbone:
            hidden = hidden.detach()

        values = self.value_head(hidden).squeeze(-1)  # (B, seq_len)
        return values

    # ── forward at specific indices (end-of-turn) ────────────────────

    def forward_at_indices(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        end_indices: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute value only at specified positions (e.g. end of each turn).

        Parameters
        ----------
        input_ids : (B, seq_len)
        attention_mask : (B, seq_len)
        end_indices : (B,) — index into dim=1 for each sample
        pixel_values : optional vision tensor
        image_grid_thw : optional grid info

        Returns
        -------
        values : (B,) — one scalar value per sample.
        """
        all_values = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )  # (B, seq_len)

        # Gather the value at the specified index for each batch element
        batch_idx = torch.arange(all_values.shape[0], device=all_values.device)
        values = all_values[batch_idx, end_indices]  # (B,)
        return values

    # ── input preparation from chat messages ─────────────────────────

    def prepare_inputs(
        self,
        messages: List[List[Dict[str, Any]]],
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize + process a batch of chat messages (with images) using
        the Qwen3-VL processor.

        Parameters
        ----------
        messages : list of conversations, each a list of
                   ``{"role": ..., "content": [...]}`` dicts (same format
                   used by ``ActorVLLM._prepare_inputs``).

        Returns
        -------
        dict with keys ``input_ids``, ``attention_mask``, and optionally
        ``pixel_values``, ``image_grid_thw`` — all on self._device.
        """
        from qwen_vl_utils import process_vision_info

        # Build text prompts via chat template
        texts = [
            self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=False
            )
            for msg in messages
        ]

        # Extract image/video inputs
        all_images, all_videos = [], []
        for msg in messages:
            images, videos = process_vision_info(msg)
            all_images.append(images)
            all_videos.append(videos)

        # Processor handles tokenization + image preprocessing
        batch = self.processor(
            text=texts,
            images=all_images if any(img for img in all_images) else None,
            videos=all_videos if any(vid for vid in all_videos) else None,
            padding=True,
            return_tensors="pt",
        )

        # Move everything to device
        return {k: v.to(self._device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    # ── state dict helpers ───────────────────────────────────────────

    def value_head_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return only the value head weights (for checkpointing)."""
        return {k: v.cpu() for k, v in self.value_head.state_dict().items()}

    def load_value_head_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load value head weights from a checkpoint."""
        self.value_head.load_state_dict(state_dict)
        self.value_head.to(device=self._device, dtype=self.torch_dtype)

    def trainable_parameters(self):
        """Yield only parameters that require gradients (for optimizer)."""
        return (p for p in self.parameters() if p.requires_grad)
