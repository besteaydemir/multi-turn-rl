#!/usr/bin/env python3
"""
Enhanced token masking utilities for Step 3.
Robust action token identification with edge case handling.
"""

import torch
import re
from typing import Tuple, Optional, Dict, Any


class ActionTokenMasker:
    """
    Robust action token masking with multiple fallback strategies.
    Implements Step 3 requirements for critical masking logic.
    """
    
    def __init__(
        self,
        processor,
        min_action_tokens: int = 10,
        max_action_tokens: int = 100
    ):
        """
        Initialize masker.
        
        Args:
            processor: Tokenizer processor
            min_action_tokens: Minimum expected action tokens
            max_action_tokens: Maximum expected action tokens
        """
        self.processor = processor
        self.min_action_tokens = min_action_tokens
        self.max_action_tokens = max_action_tokens
        
        # Statistics
        self.stats = {
            "brace_depth": 0,
            "regex_fallback": 0,
            "last_n_fallback": 0,
            "failed": 0
        }
    
    def identify_action_tokens(
        self,
        generated_ids: torch.Tensor,
        generated_text: str
    ) -> Tuple[torch.Tensor, Optional[int], Optional[int], Dict[str, Any]]:
        """
        Identify action tokens with robust fallback strategies (Step 3).
        
        Strategy hierarchy:
        1. Brace depth tracking (primary, most reliable)
        2. Regex pattern matching (fallback 1)
        3. Last-N tokens heuristic (fallback 2, marked as low confidence)
        4. Complete failure (mark for dropout)
        
        Returns:
            (mask, start_index, end_index, diagnostics)
        """
        # Try primary method: brace depth
        mask, start_idx, end_idx, diag = self._try_brace_depth(generated_ids)
        if diag["validation_passed"]:
            return mask, start_idx, end_idx, diag
        
        # Try fallback 1: regex
        mask, start_idx, end_idx, diag = self._try_regex_fallback(generated_ids, generated_text)
        if diag["validation_passed"]:
            return mask, start_idx, end_idx, diag
        
        # Try fallback 2: last-N tokens
        mask, start_idx, end_idx, diag = self._try_last_n_fallback(generated_ids)
        if start_idx is not None:
            return mask, start_idx, end_idx, diag
        
        # Complete failure
        return self._mark_failed(generated_ids)
    
    def _try_brace_depth(
        self,
        generated_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[int], Optional[int], Dict[str, Any]]:
        """Primary method: brace depth tracking."""
        mask = torch.zeros(len(generated_ids), dtype=torch.bool)
        json_start_index = None
        json_end_index = None
        
        in_json = False
        brace_depth = 0
        
        for i, token_id in enumerate(generated_ids):
            token_text = self.processor.tokenizer.decode([token_id.item()], skip_special_tokens=False)
            
            # Detect opening brace
            if '{' in token_text and not in_json:
                in_json = True
                json_start_index = i
                brace_depth = token_text.count('{')
                mask[i] = True
                
            # Track tokens inside JSON
            elif in_json:
                mask[i] = True
                brace_depth += token_text.count('{')
                brace_depth -= token_text.count('}')
                
                # Found closing brace
                if brace_depth <= 0:
                    json_end_index = i + 1
                    break
        
        # Validate
        num_action = mask.sum().item()
        diagnostics = {
            "method": "brace_depth",
            "confidence": 1.0,
            "num_action_tokens": num_action,
            "num_reasoning_tokens": len(generated_ids) - num_action,
            "validation_passed": False,
            "issues": []
        }
        
        if json_start_index is not None and json_end_index is not None:
            if self.min_action_tokens <= num_action <= self.max_action_tokens:
                diagnostics["validation_passed"] = True
                self.stats["brace_depth"] += 1
            else:
                diagnostics["issues"].append(
                    f"Brace depth gave {num_action} tokens, expected {self.min_action_tokens}-{self.max_action_tokens}"
                )
        else:
            if in_json:
                diagnostics["issues"].append("JSON not closed (truncated generation)")
            else:
                diagnostics["issues"].append("No opening brace found")
        
        return mask, json_start_index, json_end_index, diagnostics
    
    def _try_regex_fallback(
        self,
        generated_ids: torch.Tensor,
        generated_text: str
    ) -> Tuple[torch.Tensor, Optional[int], Optional[int], Dict[str, Any]]:
        """Fallback 1: regex pattern matching."""
        mask = torch.zeros(len(generated_ids), dtype=torch.bool)
        json_start_index = None
        json_end_index = None
        
        diagnostics = {
            "method": "regex_fallback",
            "confidence": 0.7,
            "num_action_tokens": 0,
            "num_reasoning_tokens": len(generated_ids),
            "validation_passed": False,
            "issues": ["Used regex fallback instead of brace tracking"]
        }
        
        # Find JSON in text
        json_pattern = re.compile(r'(\{[^}]*"rotation_angle_degrees"[^}]*\})', re.DOTALL)
        match = json_pattern.search(generated_text)
        
        if not match:
            diagnostics["issues"].append("Regex pattern did not match")
            return mask, None, None, diagnostics
        
        json_start_char = match.start()
        json_end_char = match.end()
        
        # Map character positions to token positions
        current_text = ""
        for i, token_id in enumerate(generated_ids):
            token_text = self.processor.tokenizer.decode([token_id.item()], skip_special_tokens=True)
            next_text = current_text + token_text
            
            current_len = len(current_text)
            next_len = len(next_text)
            
            # Token overlaps JSON region
            if not (next_len <= json_start_char or current_len >= json_end_char):
                mask[i] = True
                if json_start_index is None:
                    json_start_index = i
                json_end_index = i + 1
            
            current_text = next_text
        
        num_action = mask.sum().item()
        if num_action > 0:
            diagnostics["num_action_tokens"] = num_action
            diagnostics["num_reasoning_tokens"] = len(generated_ids) - num_action
            diagnostics["validation_passed"] = True
            self.stats["regex_fallback"] += 1
        
        return mask, json_start_index, json_end_index, diagnostics
    
    def _try_last_n_fallback(
        self,
        generated_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[int], Optional[int], Dict[str, Any]]:
        """Fallback 2: Last-N tokens heuristic (last resort)."""
        mask = torch.zeros(len(generated_ids), dtype=torch.bool)
        
        diagnostics = {
            "method": "last_n_fallback",
            "confidence": 0.3,
            "num_action_tokens": 0,
            "num_reasoning_tokens": len(generated_ids),
            "validation_passed": False,
            "issues": []
        }
        
        if len(generated_ids) < 20:
            diagnostics["issues"].append("Sequence too short for last-N fallback")
            return mask, None, None, diagnostics
        
        # Assume last 30% or at least 15 tokens are action
        n_action = min(max(int(len(generated_ids) * 0.3), 15), len(generated_ids))
        json_start_index = len(generated_ids) - n_action
        json_end_index = len(generated_ids)
        mask[json_start_index:json_end_index] = True
        
        diagnostics["num_action_tokens"] = n_action
        diagnostics["num_reasoning_tokens"] = len(generated_ids) - n_action
        diagnostics["issues"].append(f"Used last-N fallback (n={n_action}), episode may be noisy")
        self.stats["last_n_fallback"] += 1
        
        return mask, json_start_index, json_end_index, diagnostics
    
    def _mark_failed(
        self,
        generated_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None, Dict[str, Any]]:
        """Mark complete failure."""
        mask = torch.zeros(len(generated_ids), dtype=torch.bool)
        
        diagnostics = {
            "method": "failed",
            "confidence": 0.0,
            "num_action_tokens": 0,
            "num_reasoning_tokens": len(generated_ids),
            "validation_passed": False,
            "issues": ["All masking strategies failed"]
        }
        
        self.stats["failed"] += 1
        
        return mask, None, None, diagnostics
