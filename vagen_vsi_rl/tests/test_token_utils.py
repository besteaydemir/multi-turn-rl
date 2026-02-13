#!/usr/bin/env python3
"""
Tests for token utilities (Critical gotchas: tokenizer alignment & vision masking).

Run: pytest vagen_vsi_rl/tests/test_token_utils.py -v
"""

import pytest
import torch
from vagen_vsi_rl.utils.token_utils import (
    create_action_mask,
    create_action_mask_from_ranges,
    build_ppo_sequence,
    get_vision_token_ids,
    validate_tokenizer_alignment,
    get_tokenizer_info,
    TokenizerMismatchError,
    QWEN3_VL_VISION_TOKENS,
)


class TestCreateActionMask:
    """Tests for create_action_mask() — critical for vision token masking."""

    def test_basic_mask(self):
        """Tokens after prompt_length should be True (action tokens)."""
        input_ids = torch.arange(20)
        mask = create_action_mask(input_ids, prompt_length=10)
        
        # First 10 tokens (prompt) should be False
        assert mask[:10].sum() == 0
        # Last 10 tokens (generated) should be True
        assert mask[10:].sum() == 10

    def test_full_prompt_no_actions(self):
        """If prompt_length = seq_len, all False."""
        input_ids = torch.arange(10)
        mask = create_action_mask(input_ids, prompt_length=10)
        
        assert mask.sum() == 0

    def test_no_prompt_all_actions(self):
        """If prompt_length = 0, all True."""
        input_ids = torch.arange(10)
        mask = create_action_mask(input_ids, prompt_length=0)
        
        assert mask.sum() == 10

    def test_batched_input(self):
        """Should work with 2D batched input."""
        input_ids = torch.arange(20).reshape(2, 10)
        mask = create_action_mask(input_ids, prompt_length=5)
        
        assert mask.shape == (2, 10)
        # Each row: first 5 False, last 5 True
        assert mask[:, :5].sum() == 0
        assert mask[:, 5:].sum() == 10

    def test_3d_raises(self):
        """3D input should raise."""
        input_ids = torch.arange(24).reshape(2, 3, 4)
        with pytest.raises(ValueError, match="1D or 2D"):
            create_action_mask(input_ids, prompt_length=2)


class TestCreateActionMaskFromRanges:
    """Tests for create_action_mask_from_ranges()."""

    def test_single_range(self):
        """Single action range."""
        mask = create_action_mask_from_ranges(10, [(3, 7)])
        
        expected = torch.tensor([False, False, False, True, True, True, True, False, False, False])
        assert mask.tolist() == expected.tolist()

    def test_multiple_ranges(self):
        """Multiple separate action ranges."""
        mask = create_action_mask_from_ranges(10, [(1, 3), (7, 9)])
        
        # Positions 1,2,7,8 should be True
        assert mask[1] and mask[2] and mask[7] and mask[8]
        assert not mask[0] and not mask[5] and not mask[9]

    def test_empty_ranges(self):
        """No ranges = all False."""
        mask = create_action_mask_from_ranges(10, [])
        assert mask.sum() == 0


class TestBuildPPOSequence:
    """Tests for build_ppo_sequence()."""

    def test_concatenation(self):
        """Should concat prompt + generated."""
        prompt_ids = torch.tensor([1, 2, 3])
        generated_ids = torch.tensor([4, 5])
        
        input_ids, action_mask, prompt_len = build_ppo_sequence(prompt_ids, generated_ids)
        
        assert input_ids.tolist() == [1, 2, 3, 4, 5]
        assert prompt_len == 3
        assert action_mask[:3].sum() == 0  # Prompt masked out
        assert action_mask[3:].sum() == 2  # Generated included


class TestGetVisionTokenIds:
    """Tests for get_vision_token_ids()."""

    def test_returns_set(self):
        """Should return a set of token IDs."""
        # Mock tokenizer with vision tokens
        class MockTokenizer:
            unk_token_id = 0
            added_tokens_encoder = {}
            
            def convert_tokens_to_ids(self, token):
                mapping = {
                    "<|vision_start|>": 151643,
                    "<|vision_end|>": 151644,
                    "<|image_pad|>": 151655,
                }
                return mapping.get(token, self.unk_token_id)
        
        tok = MockTokenizer()
        vision_ids = get_vision_token_ids(tok, model_type="qwen3_vl")
        
        assert isinstance(vision_ids, set)
        assert 151643 in vision_ids
        assert 151644 in vision_ids


class TestValidateTokenizerAlignment:
    """Tests for validate_tokenizer_alignment()."""

    def test_identical_tokenizers_pass(self):
        """Identical tokenizers should pass."""
        class MockTokenizer:
            bos_token = "<s>"
            eos_token = "</s>"
            pad_token = "<pad>"
            unk_token = "<unk>"
            
            def __len__(self):
                return 32000
            
            def encode(self, text, add_special_tokens=False):
                return [ord(c) for c in text[:10]]  # Simple mock
        
        tok1 = MockTokenizer()
        tok2 = MockTokenizer()
        
        report = validate_tokenizer_alignment(tok1, tok2, strict=False)
        assert report["aligned"] is True
        assert len(report["mismatches"]) == 0

    def test_different_vocab_size_fails(self):
        """Different vocab sizes should be flagged."""
        class MockTokenizer1:
            bos_token = "<s>"
            eos_token = "</s>"
            pad_token = "<pad>"
            unk_token = "<unk>"
            def __len__(self): return 32000
            def encode(self, text, add_special_tokens=False): return []
        
        class MockTokenizer2(MockTokenizer1):
            def __len__(self): return 50000
        
        tok1 = MockTokenizer1()
        tok2 = MockTokenizer2()
        
        report = validate_tokenizer_alignment(tok1, tok2, strict=False)
        assert report["aligned"] is False
        assert any(m["type"] == "vocab_size" for m in report["mismatches"])

    def test_different_special_tokens_fails(self):
        """Different special tokens should be flagged."""
        class MockTokenizer1:
            bos_token = "<s>"
            eos_token = "</s>"
            pad_token = "<pad>"
            unk_token = "<unk>"
            def __len__(self): return 32000
            def encode(self, text, add_special_tokens=False): return []
        
        class MockTokenizer2(MockTokenizer1):
            eos_token = "[EOS]"  # Different!
        
        tok1 = MockTokenizer1()
        tok2 = MockTokenizer2()
        
        report = validate_tokenizer_alignment(tok1, tok2, strict=False)
        assert report["aligned"] is False
        assert any(m["type"] == "special_token" for m in report["mismatches"])

    def test_strict_mode_raises(self):
        """strict=True should raise on mismatch."""
        class MockTokenizer1:
            bos_token = "<s>"
            eos_token = "</s>"
            pad_token = "<pad>"
            unk_token = "<unk>"
            def __len__(self): return 32000
            def encode(self, text, add_special_tokens=False): return []
        
        class MockTokenizer2(MockTokenizer1):
            def __len__(self): return 50000
        
        tok1 = MockTokenizer1()
        tok2 = MockTokenizer2()
        
        with pytest.raises(TokenizerMismatchError):
            validate_tokenizer_alignment(tok1, tok2, strict=True)


class TestGetTokenizerInfo:
    """Tests for get_tokenizer_info()."""

    def test_returns_expected_keys(self):
        """Should return dict with standard keys."""
        class MockTokenizer:
            bos_token = "<s>"
            eos_token = "</s>"
            pad_token = "<pad>"
            unk_token = "<unk>"
            bos_token_id = 1
            eos_token_id = 2
            pad_token_id = 0
            name_or_path = "mock/tokenizer"
            def __len__(self): return 32000
        
        tok = MockTokenizer()
        info = get_tokenizer_info(tok)
        
        assert "vocab_size" in info
        assert "bos_token" in info
        assert "eos_token" in info
        assert info["vocab_size"] == 32000


class TestQwen3VLVisionTokens:
    """Tests for Qwen3-VL specific vision tokens."""

    def test_vision_tokens_defined(self):
        """QWEN3_VL_VISION_TOKENS should be defined."""
        assert "<|vision_start|>" in QWEN3_VL_VISION_TOKENS
        assert "<|vision_end|>" in QWEN3_VL_VISION_TOKENS
        assert "<|image_pad|>" in QWEN3_VL_VISION_TOKENS
