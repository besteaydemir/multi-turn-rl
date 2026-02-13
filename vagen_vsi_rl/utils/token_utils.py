#!/usr/bin/env python3
"""
Token alignment and masking utilities.

Two critical gotchas this module addresses:

1. **Token alignment between vLLM and HF**
   - vLLM and HuggingFace must use identical tokenizers, special tokens, and
     chat templates. Otherwise log-probs / PPO ratios will be misaligned.
   - Use `validate_tokenizer_alignment()` before training to catch mismatches.

2. **Vision tokens must be masked from loss**
   - Observation tokens (images + prompt) should NOT contribute to the PPO loss.
   - Only generated action tokens should be trained.
   - This is a known failure mode explicitly called out in the VAGEN paper.
   - Use `create_action_mask()` to build proper masks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import torch


class TokenizerMismatchError(Exception):
    """Raised when vLLM and HF tokenizers are not aligned."""
    pass


# ─────────────────────────────────────────────────────────────────────
# Tokenizer alignment validation
# ─────────────────────────────────────────────────────────────────────

def validate_tokenizer_alignment(
    vllm_tokenizer,
    hf_tokenizer,
    test_texts: Optional[List[str]] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Validate that vLLM and HF tokenizers produce identical output.
    
    CRITICAL: Run this before training! Misaligned tokenizers will cause
    PPO ratio computations to be incorrect (different token IDs for same text).
    
    Parameters
    ----------
    vllm_tokenizer
        Tokenizer from ActorVLLM (typically `actor_vllm.tokenizer`)
    hf_tokenizer
        Tokenizer from ActorHF or ReferenceModel (typically `actor_hf.tokenizer`)
    test_texts : List[str], optional
        Sample texts to encode. If None, uses default test cases.
    strict : bool
        If True, raise TokenizerMismatchError on any mismatch.
        If False, return dict with mismatch info but don't raise.
    
    Returns
    -------
    dict
        Alignment report with keys: aligned, mismatches, vocab_size, etc.
    
    Raises
    ------
    TokenizerMismatchError
        If strict=True and any mismatch is found.
    
    Example
    -------
    >>> from vagen_vsi_rl.models import ActorVLLM, ActorHF
    >>> actor_vllm = ActorVLLM(model_id="Qwen/Qwen3-VL-4B-Instruct")
    >>> actor_hf = ActorHF(model_id="Qwen/Qwen3-VL-4B-Instruct")
    >>> actor_hf.load()
    >>> validate_tokenizer_alignment(actor_vllm.tokenizer, actor_hf.tokenizer)
    """
    if test_texts is None:
        test_texts = [
            "Hello, world!",
            '{"rotation_angle_degrees": 45, "forward_meters": 0.5}',
            "The answer is C.",
            "<|im_start|>user\nWhat is this?<|im_end|>",
            "```json\n{}\n```",
        ]
    
    report = {
        "aligned": True,
        "mismatches": [],
        "vocab_size_vllm": len(vllm_tokenizer),
        "vocab_size_hf": len(hf_tokenizer),
    }
    
    # Check vocab size
    if len(vllm_tokenizer) != len(hf_tokenizer):
        report["aligned"] = False
        report["mismatches"].append({
            "type": "vocab_size",
            "vllm": len(vllm_tokenizer),
            "hf": len(hf_tokenizer),
        })
    
    # Check special tokens
    special_tokens = ["bos_token", "eos_token", "pad_token", "unk_token"]
    for tok_name in special_tokens:
        vllm_tok = getattr(vllm_tokenizer, tok_name, None)
        hf_tok = getattr(hf_tokenizer, tok_name, None)
        if vllm_tok != hf_tok:
            report["aligned"] = False
            report["mismatches"].append({
                "type": "special_token",
                "name": tok_name,
                "vllm": vllm_tok,
                "hf": hf_tok,
            })
    
    # Check encoding of test texts
    for text in test_texts:
        vllm_ids = vllm_tokenizer.encode(text, add_special_tokens=False)
        hf_ids = hf_tokenizer.encode(text, add_special_tokens=False)
        
        if vllm_ids != hf_ids:
            report["aligned"] = False
            report["mismatches"].append({
                "type": "encoding",
                "text": text[:50] + ("..." if len(text) > 50 else ""),
                "vllm_ids": vllm_ids[:10],
                "hf_ids": hf_ids[:10],
                "len_vllm": len(vllm_ids),
                "len_hf": len(hf_ids),
            })
    
    if strict and not report["aligned"]:
        msg = (
            f"Tokenizer mismatch between vLLM and HF!\n"
            f"Mismatches: {report['mismatches']}\n\n"
            f"This WILL break PPO: log-probs won't align with token IDs.\n"
            f"Ensure both use the same model_id and cache_dir."
        )
        raise TokenizerMismatchError(msg)
    
    return report


def get_tokenizer_info(tokenizer) -> Dict[str, Any]:
    """Get diagnostic info about a tokenizer."""
    return {
        "vocab_size": len(tokenizer),
        "bos_token": tokenizer.bos_token,
        "eos_token": tokenizer.eos_token,
        "pad_token": tokenizer.pad_token,
        "unk_token": tokenizer.unk_token,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "name_or_path": getattr(tokenizer, "name_or_path", "unknown"),
    }


# ─────────────────────────────────────────────────────────────────────
# Vision token detection
# ─────────────────────────────────────────────────────────────────────

# Qwen3-VL vision token patterns
QWEN3_VL_VISION_TOKENS = {
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
}


def get_vision_token_ids(tokenizer, model_type: str = "qwen3_vl") -> Set[int]:
    """
    Get the set of token IDs that represent vision content.
    
    These tokens encode image/video data and should be EXCLUDED from PPO loss.
    
    Parameters
    ----------
    tokenizer
        The tokenizer to use.
    model_type : str
        Model family. Currently supports: "qwen3_vl"
    
    Returns
    -------
    Set[int]
        Token IDs that represent vision content.
    """
    vision_ids = set()
    
    if model_type == "qwen3_vl":
        for token in QWEN3_VL_VISION_TOKENS:
            try:
                tok_id = tokenizer.convert_tokens_to_ids(token)
                if tok_id != tokenizer.unk_token_id:
                    vision_ids.add(tok_id)
            except Exception:
                pass
        
        # Also add the image_pad token range if it exists
        # Qwen3-VL uses special image tokens that may be in a range
        if hasattr(tokenizer, 'added_tokens_encoder'):
            for token, tok_id in tokenizer.added_tokens_encoder.items():
                if 'image' in token.lower() or 'vision' in token.lower():
                    vision_ids.add(tok_id)
    
    return vision_ids


# ─────────────────────────────────────────────────────────────────────
# Action mask creation
# ─────────────────────────────────────────────────────────────────────

def create_action_mask(
    input_ids: torch.Tensor,
    prompt_length: int,
    tokenizer=None,
    exclude_vision_tokens: bool = True,
) -> torch.Tensor:
    """
    Create a boolean mask where True = action token (train on this).
    
    CRITICAL: Only action tokens should contribute to PPO loss.
    Observation tokens (prompt + images) dominating loss is a known failure mode.
    
    Parameters
    ----------
    input_ids : torch.Tensor
        Full sequence of token IDs (prompt + generated). Shape: (seq_len,) or (B, seq_len)
    prompt_length : int
        Number of tokens in the prompt (before generation started).
        These will be masked OUT (False).
    tokenizer : optional
        If provided and exclude_vision_tokens=True, also masks out vision tokens.
    exclude_vision_tokens : bool
        Whether to additionally mask out vision placeholder tokens in the
        generated portion (rare, but can happen with certain prompts).
    
    Returns
    -------
    torch.Tensor
        Boolean mask, same shape as input_ids. True = train on this token.
    
    Example
    -------
    >>> # Prompt: 100 tokens, Generated: 50 tokens
    >>> input_ids = torch.arange(150)
    >>> mask = create_action_mask(input_ids, prompt_length=100)
    >>> mask[:100].all()  # prompt: all False
    False
    >>> mask[100:].all()  # generated: all True
    True
    """
    if input_ids.dim() == 1:
        seq_len = input_ids.shape[0]
        mask = torch.zeros(seq_len, dtype=torch.bool, device=input_ids.device)
        
        # Action tokens: everything after prompt
        if prompt_length < seq_len:
            mask[prompt_length:] = True
        
        # Optionally mask out vision tokens in generated portion
        if exclude_vision_tokens and tokenizer is not None:
            vision_ids = get_vision_token_ids(tokenizer)
            for i in range(prompt_length, seq_len):
                if input_ids[i].item() in vision_ids:
                    mask[i] = False
        
        return mask
    
    elif input_ids.dim() == 2:
        # Batched
        B, seq_len = input_ids.shape
        masks = []
        for b in range(B):
            masks.append(create_action_mask(
                input_ids[b], 
                prompt_length, 
                tokenizer, 
                exclude_vision_tokens
            ))
        return torch.stack(masks)
    
    else:
        raise ValueError(f"input_ids must be 1D or 2D, got {input_ids.dim()}D")


def create_action_mask_from_ranges(
    seq_len: int,
    action_ranges: List[Tuple[int, int]],
    device: str = "cpu",
) -> torch.Tensor:
    """
    Create action mask from explicit (start, end) ranges.
    
    Useful when you have multiple action segments (multi-turn).
    
    Parameters
    ----------
    seq_len : int
        Total sequence length.
    action_ranges : List[Tuple[int, int]]
        List of (start, end) pairs indicating action token positions.
        End is exclusive: [start, end).
    device : str
        Device for the output tensor.
    
    Returns
    -------
    torch.Tensor
        Boolean mask of shape (seq_len,).
    """
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    for start, end in action_ranges:
        mask[start:end] = True
    return mask


# ─────────────────────────────────────────────────────────────────────
# Sequence construction for PPO
# ─────────────────────────────────────────────────────────────────────

def build_ppo_sequence(
    prompt_ids: torch.Tensor,
    generated_ids: torch.Tensor,
    tokenizer=None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Concatenate prompt + generated tokens and create action mask.
    
    This is the standard way to prepare a sequence for PPO training:
    - Full sequence = prompt_ids || generated_ids
    - Mask = False for prompt, True for generated
    
    Parameters
    ----------
    prompt_ids : torch.Tensor
        Prompt token IDs from vLLM. Shape: (prompt_len,)
    generated_ids : torch.Tensor
        Generated token IDs from vLLM. Shape: (gen_len,)
    tokenizer : optional
        For vision token masking.
    
    Returns
    -------
    input_ids : torch.Tensor
        Concatenated sequence. Shape: (prompt_len + gen_len,)
    action_mask : torch.Tensor
        Boolean mask. Shape: (prompt_len + gen_len,)
    prompt_length : int
        Length of prompt (for reference).
    """
    prompt_len = prompt_ids.shape[0]
    input_ids = torch.cat([prompt_ids, generated_ids], dim=0)
    action_mask = create_action_mask(input_ids, prompt_len, tokenizer)
    return input_ids, action_mask, prompt_len
