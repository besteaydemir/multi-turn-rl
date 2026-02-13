#!/usr/bin/env python3
"""
GPU-dependent tests for HabitatEnv and ActorVLLM.

These tests require a GPU and the proper environment setup:
    conda activate habitat_source
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    export __EGL_VENDOR_LIBRARY_FILENAMES=/dss/dsshome1/06/di38riq/habitat-sim/10_nvidia.json
    unset DISPLAY

Run: pytest vagen_vsi_rl/tests/test_gpu.py -v -s
"""

# CRITICAL: Must set these BEFORE importing vLLM/torch
import os
import multiprocessing

# vLLM multiprocessing settings for V1 engine compatibility
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Set spawn method for general multiprocessing (avoids CUDA fork issues)
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

import pytest
import json
import tempfile
from pathlib import Path

# Skip entire module if no GPU
import torch
if not torch.cuda.is_available():
    pytest.skip("GPU required", allow_module_level=True)


class TestHabitatEnv:
    """Tests for HabitatEnv (requires habitat-sim + GPU)."""

    @pytest.fixture
    def env(self):
        """Create a HabitatEnv for testing."""
        from vagen_vsi_rl.env.habitat_env import HabitatEnv, HabitatEnvConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = HabitatEnvConfig(max_steps=3)
            env = HabitatEnv(output_dir=tmpdir, config=cfg)
            yield env
            env.close()

    def test_reset_returns_observation(self, env):
        """Reset should return valid Observation."""
        obs = env.reset({
            "scene_name": "skokloster-castle",
            "question": "What color is the ceiling?",
            "choices": ["Red", "Blue", "White", "Brown"],
            "answer_id": "C",
            "question_type": "color",
            "is_numerical": False,
            "dataset": "habitat",
        })
        
        assert obs is not None
        assert len(obs.image_paths) == 1
        assert obs.step == 0
        assert "ceiling" in obs.prompt_text.lower()

    def test_step_returns_tuple(self, env):
        """Step should return (obs, reward, done, info)."""
        env.reset({
            "scene_name": "skokloster-castle",
            "question": "Test?",
            "choices": ["A", "B"],
            "answer_id": "A",
        })
        
        obs, reward, done, info = env.step({
            "rotation_angle_degrees": 30,
            "forward_meters": 0.5,
            "left_meters": 0,
            "z_delta_meters": 0,
            "answer": None,
            "done": False,
        })
        
        assert obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_correct_answer_reward(self, env):
        """Correct answer should give positive reward."""
        env.reset({
            "scene_name": "skokloster-castle",
            "question": "Test?",
            "choices": ["A", "B"],
            "answer_id": "A",
        })
        
        _, reward, done, info = env.step({
            "rotation_angle_degrees": 0,
            "forward_meters": 0,
            "left_meters": 0,
            "z_delta_meters": 0,
            "answer": "A",
            "done": True,
        })
        
        assert done is True
        assert info.get("is_correct") is True
        assert reward > 0

    def test_wrong_answer_negative_reward(self, env):
        """Wrong answer should give negative reward."""
        env.reset({
            "scene_name": "skokloster-castle",
            "question": "Test?",
            "choices": ["A", "B"],
            "answer_id": "A",
        })
        
        _, reward, done, info = env.step({
            "answer": "B",
            "done": True,
        })
        
        assert done is True
        assert info.get("is_correct") is False
        assert reward < 0

    def test_trajectory_saved(self, env):
        """Trajectory JSON should be saved after episode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from vagen_vsi_rl.env.habitat_env import HabitatEnv, HabitatEnvConfig
            
            cfg = HabitatEnvConfig(max_steps=2)
            test_env = HabitatEnv(output_dir=tmpdir, config=cfg)
            
            test_env.reset({
                "scene_name": "skokloster-castle",
                "question": "Test?",
                "choices": ["A"],
                "answer_id": "A",
            })
            test_env.step({"answer": "A", "done": True})
            
            # Check trajectory.json exists
            traj_path = Path(tmpdir) / "q000" / "trajectory.json"
            assert traj_path.exists(), f"trajectory.json not found at {traj_path}"
            
            with open(traj_path) as f:
                data = json.load(f)
            
            assert "question_id" in data
            assert "is_correct" in data
            assert data["env_type"] == "habitat"
            
            test_env.close()


class TestActorVLLM:
    """Tests for ActorVLLM (requires vLLM + GPU)."""

    @pytest.fixture(scope="class")
    def actor(self):
        """Create ActorVLLM (expensive, reuse across tests)."""
        from vagen_vsi_rl.models.actor_vllm import ActorVLLM
        
        actor = ActorVLLM(
            model_id="Qwen/Qwen3-VL-4B-Instruct",
            gpu_memory_utilization=0.7,
            max_model_len=4096,
            enforce_eager=True,  # Required for pytest compatibility with vLLM V1 engine
        )
        yield actor
        # No explicit cleanup needed

    def test_generate_text_only(self, actor):
        """generate() should return text."""
        messages = [{"role": "user", "content": "What is 2+2?"}]
        text = actor.generate(messages, max_new_tokens=32)
        
        assert isinstance(text, str)
        assert len(text) > 0

    def test_generate_with_logprobs(self, actor):
        """generate_with_logprobs() should return GenerateOutput."""
        messages = [{"role": "user", "content": "What is 2+2?"}]
        out = actor.generate_with_logprobs(messages, max_new_tokens=32)
        
        assert out.text is not None
        assert out.token_ids is not None
        assert out.token_logprobs is not None
        
        # Should align: same number of token IDs and logprobs
        assert len(out.token_ids) == len(out.token_logprobs)

    def test_logprobs_are_negative(self, actor):
        """Log-probabilities should be ≤ 0."""
        messages = [{"role": "user", "content": "Say hello"}]
        out = actor.generate_with_logprobs(messages, max_new_tokens=32)
        
        assert (out.token_logprobs <= 0).all()

    def test_prompt_token_ids_returned(self, actor):
        """Prompt token IDs should be returned."""
        messages = [{"role": "user", "content": "Hello"}]
        out = actor.generate_with_logprobs(messages, max_new_tokens=16)
        
        assert out.prompt_token_ids is not None
        assert len(out.prompt_token_ids) > 0

    def test_to_dict_serializable(self, actor):
        """GenerateOutput.to_dict() should be JSON-serializable."""
        messages = [{"role": "user", "content": "Test"}]
        out = actor.generate_with_logprobs(messages, max_new_tokens=16)
        
        d = out.to_dict()
        json_str = json.dumps(d)  # Should not raise
        assert len(json_str) > 0


class TestTokenizerAlignment:
    """Tests for tokenizer alignment between vLLM and HF."""

    def test_vllm_hf_tokenizers_align(self):
        """ActorVLLM and ActorHF should have aligned tokenizers."""
        from vagen_vsi_rl.models.actor_vllm import ActorVLLM
        from vagen_vsi_rl.models.actor_hf import ActorHF
        from vagen_vsi_rl.utils.token_utils import validate_tokenizer_alignment
        
        model_id = "Qwen/Qwen3-VL-4B-Instruct"
        
        actor_vllm = ActorVLLM(model_id=model_id, gpu_memory_utilization=0.4)
        actor_hf = ActorHF(model_id=model_id)
        actor_hf.load()
        
        report = validate_tokenizer_alignment(
            actor_vllm.tokenizer,
            actor_hf.tokenizer,
            strict=False,
        )
        
        assert report["aligned"], f"Tokenizers not aligned: {report['mismatches']}"
