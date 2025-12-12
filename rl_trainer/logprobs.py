#!/usr/bin/env python3
"""
Log-probability computation for policy gradient RL.

This module implements teacher-forcing passes to compute:
1. log π(a|s) - log probability of actions under current policy
2. log π_ref(a|s) - log probability under reference (frozen) model
3. Entropy for exploration regularization

Key principles:
- Single forward pass per batch for efficiency
- Proper logit-token alignment for causal LMs
- Action token masking to compute log-probs only on action tokens
- Numerical stability using F.log_softmax
- No gradients for reference model
- Multimodal inputs handled consistently with generation
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List, Any
from dataclasses import dataclass
from PIL import Image

from .batch import EpisodeBatch
from .multimodal import MultimodalInputBuilder, batch_images_for_episodes


@dataclass
class LogProbResult:
    """
    Results from log-probability computation.
    
    All tensors are per-episode scalars (shape: [batch_size]).
    """
    # Policy log-probs (current model)
    logpi_seq: torch.Tensor          # Sum of action token log-probs under policy
    
    # Reference log-probs (frozen model)
    logpref_seq: Optional[torch.Tensor] = None  # Sum of action token log-probs under reference
    
    # KL divergence
    kl_div: Optional[torch.Tensor] = None  # KL(π || π_ref) on action tokens
    
    # Entropy (for exploration)
    entropy_seq: Optional[torch.Tensor] = None  # Sum of entropy on action tokens
    
    # Diagnostics
    num_action_tokens: torch.Tensor = None  # Number of action tokens per episode
    mean_logpi: torch.Tensor = None         # Mean log-prob per action token
    mean_logpref: Optional[torch.Tensor] = None
    mean_entropy: Optional[torch.Tensor] = None


def compute_token_logprobs(
    model,
    batch: EpisodeBatch,
    processor=None,
    images: Optional[List[List[Image.Image]]] = None,
    compute_entropy: bool = False,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute per-token log probabilities using teacher forcing.
    
    This function:
    1. Constructs full input sequences (context + generated)
    2. Handles multimodal inputs (images) if provided
    3. Runs forward pass to get logits
    4. Aligns logits with tokens (accounting for causal shift)
    5. Computes log-probs for generated tokens only
    6. Optionally computes entropy
    
    Args:
        model: Language model (Qwen3VL or similar)
        batch: EpisodeBatch with context and generated tokens
        processor: Model processor (required for multimodal inputs)
        images: List of image lists (one per episode in batch)
        compute_entropy: Whether to compute entropy for regularization
        device: Device for computation
        
    Returns:
        token_logprobs: [batch_size, max_gen_len] - log-probs for each generated token
        token_mask: [batch_size, max_gen_len] - mask for valid (non-padded) tokens
        token_entropy: [batch_size, max_gen_len] - entropy per token (if compute_entropy=True)
        
    Note: Causal LM alignment
        For models like Qwen, logits at position t predict token at position t+1.
        We need to align: logits[t] corresponds to label[t+1].
    
    Note: Multimodal consistency
        Must use same image formatting as during generation to ensure
        log p(action | context, images) is computed with correct conditionals.
    """
    batch_size = batch.batch_size
    max_context_len = batch.max_context_len
    max_gen_len = batch.max_gen_len
    
    # Step 1: Construct full input sequences (context + generated)
    # Shape: [batch_size, max_context_len + max_gen_len]
    full_input_ids = torch.cat([
        batch.context_input_ids,
        batch.generated_ids
    ], dim=1).to(device)
    
    # Construct attention mask
    # Shape: [batch_size, max_context_len + max_gen_len]
    full_attention_mask = batch.attention_masks.to(device)
    
    # Step 2: Prepare multimodal inputs if images provided
    model_inputs = {
        "input_ids": full_input_ids,
        "attention_mask": full_attention_mask,
        "return_dict": True
    }
    
    if images is not None and processor is not None:
        # Build multimodal inputs
        input_builder = MultimodalInputBuilder(processor, device)
        multimodal_inputs = input_builder.prepare_teacher_forcing_inputs(
            context_input_ids=batch.context_input_ids,
            generated_ids=batch.generated_ids,
            attention_mask=full_attention_mask,
            images=images
        )
        # Merge with model_inputs
        model_inputs.update(multimodal_inputs)
    
    # Step 3: Forward pass to get logits
    with torch.set_grad_enabled(model.training):
        outputs = model(**model_inputs)
        logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
    
    # Step 3: Align logits with tokens
    # For causal LM: logits[t] predicts token[t+1]
    # We want log-prob of generated tokens, which start at position max_context_len
    
    # Extract logits for positions that predict generated tokens
    # logits at position [max_context_len-1 : max_context_len+max_gen_len-1]
    # predict tokens at [max_context_len : max_context_len+max_gen_len]
    pred_logits = logits[:, max_context_len-1:-1, :]  # Shape: [batch_size, max_gen_len, vocab_size]
    
    # Extract generated token IDs (labels)
    labels = batch.generated_ids.to(device)  # Shape: [batch_size, max_gen_len]
    
    # Step 4: Compute log-probs
    log_probs = F.log_softmax(pred_logits, dim=-1)  # Shape: [batch_size, max_gen_len, vocab_size]
    
    # Gather log-probs for actual tokens
    # Shape: [batch_size, max_gen_len]
    token_logprobs = log_probs.gather(
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)
    
    # Step 5: Create mask for valid (non-padded) generated tokens
    token_mask = torch.zeros(batch_size, max_gen_len, dtype=torch.bool, device=device)
    for i in range(batch_size):
        gen_len = batch.generation_lengths[i].item()
        token_mask[i, :gen_len] = True
    
    # Zero out log-probs for padded tokens
    token_logprobs = token_logprobs * token_mask.float()
    
    # Step 6: Optionally compute entropy
    token_entropy = None
    if compute_entropy:
        # Entropy = -sum(p * log(p))
        probs = F.softmax(pred_logits, dim=-1)  # Shape: [batch_size, max_gen_len, vocab_size]
        token_entropy = -(probs * log_probs).sum(dim=-1)  # Shape: [batch_size, max_gen_len]
        token_entropy = token_entropy * token_mask.float()
    
    return token_logprobs, token_mask, token_entropy


def compute_sequence_logprobs(
    model,
    batch: EpisodeBatch,
    processor=None,
    images: Optional[List[List[Image.Image]]] = None,
    ref_model: Optional[torch.nn.Module] = None,
    compute_entropy: bool = True,
    compute_kl: bool = True,
    device: str = "cuda"
) -> LogProbResult:
    """
    Compute sequence-level log probabilities for policy gradient.
    
    This is the main function for Step 5. It:
    1. Computes per-token log-probs using teacher forcing
    2. Applies action token masks
    3. Sums over action tokens to get sequence log-probs
    4. Optionally computes reference model log-probs (for KL penalty)
    5. Optionally computes entropy (for exploration)
    
    Args:
        model: Current policy model (with gradients)
        batch: EpisodeBatch with episodes
        processor: Model processor (required for multimodal inputs)
        images: List of image lists (one per episode in batch)
        ref_model: Reference (frozen) model for KL penalty (optional)
        compute_entropy: Whether to compute entropy regularization term
        compute_kl: Whether to compute KL divergence (requires ref_model)
        device: Device for computation
        
    Returns:
        LogProbResult with per-episode log-probs and diagnostics
        
    Mathematical formulation:
        log π(a|s) = sum_{t in action_tokens} log π(a_t | s, a_{<t})
        
    Usage:
        result = compute_sequence_logprobs(policy, batch, processor, images, ref_policy)
        loss = -(result.logpi_seq * advantages).mean()
        loss += kl_coef * result.kl_div.mean()
        loss -= entropy_coef * result.entropy_seq.mean()
    """
    batch_size = batch.batch_size
    
    # Move batch to device
    batch = batch.to(device)
    
    # Ensure model is in correct mode
    is_training = model.training
    
    # Step 1: Compute token-level log-probs with current policy
    token_logprobs, token_mask, token_entropy = compute_token_logprobs(
        model=model,
        batch=batch,
        processor=processor,
        images=images,
        compute_entropy=compute_entropy,
        device=device
    )
    
    # Step 2: Apply action token masks
    action_masks = batch.action_masks.to(device)  # Shape: [batch_size, max_gen_len]
    
    # Action token log-probs (zero out non-action tokens)
    action_logprobs = token_logprobs * action_masks.float()  # Shape: [batch_size, max_gen_len]
    
    # Step 3: Sum over action tokens to get sequence log-probs
    logpi_seq = action_logprobs.sum(dim=1)  # Shape: [batch_size]
    
    # Count action tokens per episode
    num_action_tokens = action_masks.sum(dim=1).float()  # Shape: [batch_size]
    
    # Mean log-prob per action token (for diagnostics)
    mean_logpi = logpi_seq / (num_action_tokens + 1e-8)
    
    # Step 4: Compute entropy if requested
    entropy_seq = None
    mean_entropy = None
    if compute_entropy and token_entropy is not None:
        action_entropy = token_entropy * action_masks.float()
        entropy_seq = action_entropy.sum(dim=1)  # Shape: [batch_size]
        mean_entropy = entropy_seq / (num_action_tokens + 1e-8)
    
    # Step 5: Compute reference model log-probs and KL if requested
    logpref_seq = None
    kl_div = None
    mean_logpref = None
    
    if ref_model is not None and compute_kl:
        # Put reference model in eval mode and disable gradients
        ref_model.eval()
        
        with torch.no_grad():
            ref_token_logprobs, _, _ = compute_token_logprobs(
                model=ref_model,
                batch=batch,
                processor=processor,
                images=images,
                compute_entropy=False,
                device=device
            )
            
            # Sum over action tokens
            ref_action_logprobs = ref_token_logprobs * action_masks.float()
            logpref_seq = ref_action_logprobs.sum(dim=1)  # Shape: [batch_size]
            mean_logpref = logpref_seq / (num_action_tokens + 1e-8)
            
            # KL divergence: E[log π - log π_ref] (on action tokens only)
            # This is forward KL: KL(π || π_ref)
            kl_div = logpi_seq - logpref_seq  # Shape: [batch_size]
    
    # Restore model training mode
    if is_training:
        model.train()
    
    return LogProbResult(
        logpi_seq=logpi_seq,
        logpref_seq=logpref_seq,
        kl_div=kl_div,
        entropy_seq=entropy_seq,
        num_action_tokens=num_action_tokens,
        mean_logpi=mean_logpi,
        mean_logpref=mean_logpref,
        mean_entropy=mean_entropy
    )


def compute_advantages(
    rewards: torch.Tensor,
    baselines: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute advantages from rewards and baselines.
    
    Args:
        rewards: [batch_size] - episode rewards
        baselines: [batch_size] - baseline values (e.g., LOO mean)
        normalize: Whether to normalize advantages (recommended)
        
    Returns:
        advantages: [batch_size] - reward - baseline (optionally normalized)
        
    Note: Normalization helps training stability
    """
    advantages = rewards - baselines
    
    if normalize and len(advantages) > 1:
        # Normalize to zero mean, unit variance
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages


def policy_gradient_loss(
    logprobs_result: LogProbResult,
    advantages: torch.Tensor,
    kl_coef: float = 0.01,
    entropy_coef: float = 0.01
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute policy gradient loss with KL penalty and entropy regularization.
    
    Loss = -E[log π(a|s) * A(s,a)] + β * KL(π || π_ref) - α * H(π)
    
    where:
        - First term: policy gradient (REINFORCE)
        - Second term: KL penalty to prevent divergence from reference
        - Third term: entropy bonus for exploration
    
    Args:
        logprobs_result: LogProbResult from compute_sequence_logprobs
        advantages: [batch_size] - advantages (reward - baseline)
        kl_coef: Coefficient for KL penalty (β)
        entropy_coef: Coefficient for entropy bonus (α)
        
    Returns:
        loss: Scalar loss for optimization
        metrics: Dict of training metrics for logging
    """
    # Policy gradient term: -E[log π * A]
    pg_loss = -(logprobs_result.logpi_seq * advantages).mean()
    
    # KL penalty term
    kl_loss = 0.0
    if logprobs_result.kl_div is not None:
        kl_loss = kl_coef * logprobs_result.kl_div.mean()
    
    # Entropy bonus (negative because we want to maximize entropy)
    entropy_loss = 0.0
    if logprobs_result.entropy_seq is not None:
        entropy_loss = -entropy_coef * logprobs_result.entropy_seq.mean()
    
    # Total loss
    total_loss = pg_loss + kl_loss + entropy_loss
    
    # Metrics for logging
    metrics = {
        "loss/total": total_loss.item(),
        "loss/policy_gradient": pg_loss.item(),
        "loss/kl_penalty": kl_loss if isinstance(kl_loss, float) else kl_loss.item(),
        "loss/entropy_bonus": entropy_loss if isinstance(entropy_loss, float) else entropy_loss.item(),
        "logprobs/mean_logpi": logprobs_result.mean_logpi.mean().item(),
        "advantages/mean": advantages.mean().item(),
        "advantages/std": advantages.std().item(),
    }
    
    if logprobs_result.mean_logpref is not None:
        metrics["logprobs/mean_logpref"] = logprobs_result.mean_logpref.mean().item()
        metrics["kl/mean"] = logprobs_result.kl_div.mean().item()
    
    if logprobs_result.mean_entropy is not None:
        metrics["entropy/mean"] = logprobs_result.mean_entropy.mean().item()
    
    metrics["tokens/num_action_tokens"] = logprobs_result.num_action_tokens.mean().item()
    
    return total_loss, metrics


if __name__ == "__main__":
    print("Log-probability computation module loaded successfully!")
    print("\nKey components:")
    print("- compute_token_logprobs: Per-token log-probs via teacher forcing")
    print("- compute_sequence_logprobs: Sequence-level log-probs with KL and entropy")
    print("- compute_advantages: Advantage estimation (reward - baseline)")
    print("- policy_gradient_loss: Complete PG loss with KL penalty and entropy")
    print("\nFormula: L = -E[log π(a|s) * A] + β*KL(π||π_ref) - α*H(π)")
