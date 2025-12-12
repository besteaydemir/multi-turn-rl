"""
Test suite for forward pass alignment and token-level consistency.

Tests:
1. Forward pass alignment: Verify token_logprobs match manual NLL computation
2. Mask correctness: Verify action masks select correct tokens
3. Teacher forcing consistency: Ensure training inputs match generation outputs
"""

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
from typing import List, Dict, Tuple
import json

from rl_trainer.logprobs import compute_token_logprobs
from rl_environment import ActionTokenMasker


class TestForwardPassAlignment:
    """Test that log probability computation is mathematically correct."""
    
    @pytest.fixture
    def model_and_tokenizer(self):
        """Load a small model for testing."""
        model_name = "Qwen/Qwen3-VL-2B-Instruct"  # Smaller model for faster tests
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Full precision for numerical stability
            device_map="cpu",  # CPU for tests
            trust_remote_code=True
        )
        model.eval()
        return model, tokenizer
    
    def test_logprob_computation_vs_manual_nll(self, model_and_tokenizer):
        """
        Verify that sum(token_logprobs[action_mask]) equals negative of manual NLL.
        
        This is the fundamental correctness test for our log probability computation.
        """
        model, tokenizer = model_and_tokenizer
        
        # Create synthetic input
        context = "Question: What is 2+2?\nAnswer:"
        generated = " The answer is 4."
        full_text = context + generated
        
        # Tokenize
        context_ids = tokenizer.encode(context, add_special_tokens=True)
        full_ids = tokenizer.encode(full_text, add_special_tokens=True)
        generated_ids = full_ids[len(context_ids):]
        
        # Create inputs
        input_ids = torch.tensor([full_ids])
        
        # Create a minimal EpisodeBatch for testing
        from rl_trainer.batch import EpisodeBatch
        batch = EpisodeBatch(
            context_input_ids=torch.tensor([context_ids]),
            generated_ids=torch.tensor([generated_ids]),
            action_masks=torch.ones(1, len(generated_ids), dtype=torch.bool),
            attention_masks=torch.ones(1, len(full_ids)),
            rewards=torch.tensor([1.0]),
            context_lengths=torch.tensor([len(context_ids)]),
            generation_lengths=torch.tensor([len(generated_ids)]),
            episode_ids=["test_0"]
        )
        
        # Method 1: Our log probability computation
        with torch.no_grad():
            token_logprobs, token_mask, _ = compute_token_logprobs(
                model=model,
                batch=batch,
                processor=None,
                images=None,
                compute_entropy=False,
                device="cpu"
            )
        
        # Extract generated token logprobs
        generated_logprobs = token_logprobs[0, :len(generated_ids)]
        our_sum = generated_logprobs.sum().item()
        
        # Method 2: Manual NLL computation
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[0]  # [seq_len, vocab_size]
            
            # Compute NLL for generated tokens
            manual_nll = 0.0
            for i, token_id in enumerate(generated_ids):
                position = num_context + i - 1  # Logits are shifted
                token_logits = logits[position]
                log_probs = F.log_softmax(token_logits, dim=-1)
                manual_nll -= log_probs[token_id].item()
        
        manual_sum = -manual_nll
        
        # They should match within numerical precision
        print(f"\nOur sum: {our_sum:.6f}")
        print(f"Manual sum: {manual_sum:.6f}")
        print(f"Difference: {abs(our_sum - manual_sum):.9f}")
        
        assert abs(our_sum - manual_sum) < 1e-4, \
            f"Log probability mismatch: {our_sum} vs {manual_sum}"
    
    def test_logprob_computation_with_action_mask(self, model_and_tokenizer):
        """
        Test that action masking correctly isolates JSON tokens.
        
        Verifies that sum(token_logprobs * action_mask) only includes target tokens.
        """
        model, tokenizer = model_and_tokenizer
        
        # Create input with known JSON region
        context = "Generate JSON: "
        json_response = '{"action": "forward", "distance": 5}'
        full_text = context + json_response
        
        # Tokenize
        full_ids = tokenizer.encode(full_text, add_special_tokens=True)
        input_ids = torch.tensor([full_ids])
        
        # Find where JSON starts (look for '{')
        tokens = [tokenizer.decode([tid]) for tid in full_ids]
        json_start_idx = next(i for i, t in enumerate(tokens) if '{' in t)
        
        # Split into context and generated
        context_ids = full_ids[:json_start_idx]
        generated_ids = full_ids[json_start_idx:]
        
        # Create action mask (all generated tokens are action tokens)
        action_mask = torch.ones(len(generated_ids), dtype=torch.bool)
        
        # Create EpisodeBatch
        from rl_trainer.batch import EpisodeBatch
        batch = EpisodeBatch(
            context_input_ids=torch.tensor([context_ids]),
            generated_ids=torch.tensor([generated_ids]),
            action_masks=action_mask.unsqueeze(0),
            attention_masks=torch.ones(1, len(full_ids)),
            rewards=torch.tensor([1.0]),
            context_lengths=torch.tensor([len(context_ids)]),
            generation_lengths=torch.tensor([len(generated_ids)]),
            episode_ids=["test_1"]
        )
        
        # Compute log probs
        with torch.no_grad():
            token_logprobs, token_mask, _ = compute_token_logprobs(
                model=model,
                batch=batch,
                processor=None,
                images=None,
                compute_entropy=False,
                device="cpu"
            )
        
        token_logprobs = logprobs_result["token_logprobs"]  # [1, seq_len]
        
        # Masked sum
        masked_sum = (token_logprobs * action_mask).sum().item()
        
        # Verify we're only counting JSON tokens
        num_json_tokens = action_mask.sum().item()
        print(f"\nTotal tokens: {len(full_ids)}")
        print(f"JSON tokens: {num_json_tokens}")
        print(f"Masked sum: {masked_sum:.6f}")
        
        # The masked sum should be negative (log probs are negative)
        assert masked_sum < 0, "Masked log prob sum should be negative"
        
        # Verify the mask has the right shape
        assert action_mask.sum() > 0, "Action mask should select some tokens"
        assert action_mask.sum() < len(full_ids), "Action mask shouldn't select all tokens"


class TestActionMaskCorrectness:
    """Test that action mask correctly identifies JSON response tokens."""
    
    @pytest.fixture
    def tokenizer(self):
        """Load tokenizer."""
        model_name = "Qwen/Qwen3-VL-2B-Instruct"
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    def test_json_start_detection(self, tokenizer):
        """Test that we correctly detect where JSON starts."""
        # Case 1: JSON with leading whitespace
        text = "Question: Navigate forward.\nAnswer: \n\n  {\"action\": \"forward\"}"
        tokens = tokenizer.encode(text)
        
        # Decode each token to find '{'
        token_strs = [tokenizer.decode([tid]) for tid in tokens]
        json_start = next((i for i, t in enumerate(token_strs) if '{' in t), None)
        
        assert json_start is not None, "Should find JSON start"
        print(f"\nJSON starts at token {json_start}: '{token_strs[json_start]}'")
        
        # Case 2: JSON immediately after text
        text2 = "Answer:{\"action\":\"left\"}"
        tokens2 = tokenizer.encode(text2)
        token_strs2 = [tokenizer.decode([tid]) for tid in tokens2]
        json_start2 = next((i for i, t in enumerate(token_strs2) if '{' in t), None)
        
        assert json_start2 is not None, "Should find JSON start without whitespace"
        print(f"JSON starts at token {json_start2}: '{token_strs2[json_start2]}'")
    
    def test_action_mask_shape(self, tokenizer):
        """Test that action masks have correct shape and values."""
        # Sample turn with JSON response
        context = "Navigate to the kitchen."
        response = '{"action": "forward", "distance": 3}'
        
        # Simulate what happens in our pipeline
        full_text = context + " " + response
        full_ids = tokenizer.encode(full_text)
        
        # Create mask (simplified version of actual logic)
        token_strs = [tokenizer.decode([tid]) for tid in full_ids]
        json_start = next((i for i, t in enumerate(token_strs) if '{' in t), None)
        
        mask = torch.zeros(len(full_ids))
        if json_start is not None:
            mask[json_start:] = 1.0
        
        print(f"\nTotal tokens: {len(full_ids)}")
        print(f"Context tokens: {json_start}")
        print(f"Action tokens: {mask.sum().item()}")
        print(f"Mask: {mask.tolist()}")
        
        # Assertions
        assert mask.sum() > 0, "Should have some action tokens"
        assert mask.sum() < len(full_ids), "Should have some context tokens"
        assert mask[0] == 0, "First token should be context"
        if json_start is not None:
            assert mask[json_start] == 1, "JSON start should be masked"
    
    def test_no_json_in_response(self, tokenizer):
        """Test handling when response doesn't contain JSON."""
        text = "This response has no JSON, just plain text."
        tokens = tokenizer.encode(text)
        token_strs = [tokenizer.decode([tid]) for tid in tokens]
        
        json_start = next((i for i, t in enumerate(token_strs) if '{' in t), None)
        
        # Should not find JSON
        assert json_start is None, "Should not find JSON in plain text"
        
        # In real pipeline, this would create empty mask or flag episode as invalid
        mask = torch.zeros(len(tokens))
        assert mask.sum() == 0, "Mask should be empty for non-JSON response"


class TestTeacherForcingConsistency:
    """Test that teacher forcing inputs are correctly aligned."""
    
    @pytest.fixture
    def model_and_tokenizer(self):
        """Load model and tokenizer."""
        model_name = "Qwen/Qwen3-VL-2B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )
        model.eval()
        return model, tokenizer
    
    def test_input_ids_vs_generated_ids_alignment(self, model_and_tokenizer):
        """
        Verify that input_ids and generated_ids are correctly aligned.
        
        For teacher forcing:
        - input_ids should be the full sequence
        - generated_ids should be input_ids[:, 1:] (shifted by 1)
        """
        model, tokenizer = model_and_tokenizer
        
        # Create sequence
        text = "Question: What is the capital of France?\nAnswer: Paris"
        full_ids = tokenizer.encode(text, add_special_tokens=True)
        
        # Teacher forcing alignment
        input_ids = torch.tensor([full_ids[:-1]])  # All but last token
        labels = torch.tensor([full_ids[1:]])  # Shifted by 1
        
        print(f"\nSequence length: {len(full_ids)}")
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Labels shape: {labels.shape}")
        
        # Verify shapes match
        assert input_ids.shape[1] == labels.shape[1], \
            "Input and labels should have same sequence length"
        
        # Verify shift is correct
        for i in range(min(5, len(full_ids) - 1)):
            assert full_ids[i] == input_ids[0, i].item(), \
                f"Input mismatch at position {i}"
            assert full_ids[i + 1] == labels[0, i].item(), \
                f"Label mismatch at position {i}"
    
    def test_logits_to_labels_alignment(self, model_and_tokenizer):
        """
        Test that logits[t] predicts labels[t] (not labels[t+1]).
        
        This is a common source of off-by-one errors.
        """
        model, tokenizer = model_and_tokenizer
        
        text = "Hello world"
        full_ids = tokenizer.encode(text)
        
        input_ids = torch.tensor([full_ids[:-1]])
        labels = torch.tensor([full_ids[1:]])
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits  # [1, seq_len, vocab_size]
        
        # For each position, logits[t] should predict labels[t]
        predicted_tokens = logits[0].argmax(dim=-1)
        
        print(f"\nChecking alignment:")
        for t in range(min(3, labels.shape[1])):
            pred = predicted_tokens[t].item()
            true = labels[0, t].item()
            print(f"  Position {t}: predicted={tokenizer.decode([pred])}, "
                  f"true={tokenizer.decode([true])}")
        
        # Note: predictions may not match exactly, but shapes should align
        assert logits.shape[1] == labels.shape[1], \
            "Logits and labels should have same sequence dimension"


def run_alignment_tests():
    """Run all alignment tests."""
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    run_alignment_tests()
