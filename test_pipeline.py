#!/usr/bin/env python3
"""
End-to-end pipeline test for RL multi-turn project.

Tests:
  1. HabitatEnv trajectory saving (GPU required)
  2. ActorVLLM generate_with_logprobs (GPU required, loads Qwen3-VL-4B)
  
Run: python test_pipeline.py [--habitat-only | --vllm-only | --all]
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ──────────────────────────────────────────────────────────────────
# Test 1: HabitatEnv
# ──────────────────────────────────────────────────────────────────
def test_habitat_env():
    """Test HabitatEnv: reset → 3 steps → answer → check trajectory."""
    print("\n" + "=" * 60)
    print("TEST 1: HabitatEnv trajectory saving")
    print("=" * 60)

    from vagen_vsi_rl.env.habitat_env import HabitatEnv, HabitatEnvConfig

    output_dir = "/tmp/pipeline_test_habitat"
    cfg = HabitatEnvConfig(max_steps=5)
    env = HabitatEnv(output_dir=output_dir, config=cfg)

    # Reset with a demo scene
    question_data = {
        "scene_name": "skokloster-castle",
        "question": "What is the primary color of the walls in this room?",
        "choices": ["Red", "Blue", "White", "Brown"],
        "answer_id": "C",
        "question_type": "color",
        "is_numerical": False,
        "dataset": "habitat",
    }

    print("[1a] Resetting environment...")
    obs = env.reset(question_data)
    assert len(obs.image_paths) == 1, f"Expected 1 image after reset, got {len(obs.image_paths)}"
    assert obs.step == 0
    assert not obs.is_final_step
    assert obs.prompt_text, "Prompt text should not be empty"
    print(f"  ✓ Reset OK: step={obs.step}, images={len(obs.image_paths)}")
    print(f"  ✓ Prompt length: {len(obs.prompt_text)} chars")
    print(f"  ✓ First image: {obs.image_paths[0]}")

    # Step 1: Move forward
    print("[1b] Step 1: move forward 0.5m, rotate 30°...")
    obs, r, done, info = env.step({
        "rotation_angle_degrees": 30,
        "forward_meters": 0.5,
        "left_meters": 0,
        "z_delta_meters": 0,
        "answer": None,
        "done": False,
    })
    assert not done, "Should not be done after step 1"
    assert obs.step == 1
    assert len(obs.image_paths) == 2
    print(f"  ✓ Step 1: step={obs.step}, images={len(obs.image_paths)}, reward={r}")

    # Step 2: Move with lateral
    print("[1c] Step 2: strafe left 0.3m, forward 0.25m...")
    obs, r, done, info = env.step({
        "rotation_angle_degrees": -15,
        "forward_meters": 0.25,
        "left_meters": 0.3,
        "z_delta_meters": 0,
        "answer": None,
        "done": False,
    })
    assert not done
    assert obs.step == 2
    assert len(obs.image_paths) == 3
    print(f"  ✓ Step 2: step={obs.step}, images={len(obs.image_paths)}, reward={r}")

    # Step 3: Answer correctly
    print("[1d] Step 3: submit answer 'C'...")
    obs, r, done, info = env.step({
        "rotation_angle_degrees": 0,
        "forward_meters": 0,
        "left_meters": 0,
        "z_delta_meters": 0,
        "answer": "C",
        "done": True,
    })
    assert done, "Should be done after submitting answer"
    assert info["is_correct"], f"Answer should be correct, got info={info}"
    assert r == 1.0, f"Reward should be 1.0 for correct MCQ, got {r}"
    print(f"  ✓ Step 3: done={done}, reward={r}, correct={info['is_correct']}")

    # Check trajectory file
    traj_path = Path(output_dir) / "q000" / "trajectory.json"
    assert traj_path.exists(), f"trajectory.json not found at {traj_path}"

    with open(traj_path) as f:
        traj = json.load(f)

    print(f"\n[1e] Trajectory validation:")
    assert traj["question_id"] == 0
    assert traj["scene_id"] == "skokloster-castle"
    assert traj["is_correct"] is True
    assert traj["reward"] == 1.0
    assert traj["env_type"] == "habitat"
    assert traj["num_images"] >= 3
    assert len(traj["movement_history"]) == 2  # 2 movement steps before answer
    assert len(traj["image_paths"]) == traj["num_images"]

    # Check all images exist
    for ip in traj["image_paths"]:
        assert Path(ip).exists(), f"Image not found: {ip}"

    print(f"  ✓ question_id: {traj['question_id']}")
    print(f"  ✓ scene_id: {traj['scene_id']}")
    print(f"  ✓ env_type: {traj['env_type']}")
    print(f"  ✓ is_correct: {traj['is_correct']}")
    print(f"  ✓ reward: {traj['reward']}")
    print(f"  ✓ num_steps: {traj['num_steps']}")
    print(f"  ✓ num_images: {traj['num_images']}")
    print(f"  ✓ movement_history entries: {len(traj['movement_history'])}")
    print(f"  ✓ All {traj['num_images']} image files exist on disk")

    # Step 4: wrong answer episode
    print("\n[1f] Testing wrong answer episode...")
    env.close()  # Must close first sim before creating second (GL context)
    env2 = HabitatEnv(output_dir=output_dir, config=cfg, question_id=1)
    obs2 = env2.reset(question_data)
    obs2, r2, done2, info2 = env2.step({
        "rotation_angle_degrees": 0, "forward_meters": 0,
        "left_meters": 0, "z_delta_meters": 0,
        "answer": "A", "done": True,
    })
    assert done2
    assert not info2["is_correct"]
    assert r2 == -0.5
    print(f"  ✓ Wrong answer: reward={r2}, correct={info2['is_correct']}")

    env2.close()
    print("\n✅ TEST 1 PASSED: HabitatEnv trajectory saving works correctly!")
    return True


# ──────────────────────────────────────────────────────────────────
# Test 2: ActorVLLM
# ──────────────────────────────────────────────────────────────────
def test_actor_vllm():
    """Test ActorVLLM: load model, generate with logprobs, verify output."""
    print("\n" + "=" * 60)
    print("TEST 2: ActorVLLM generate_with_logprobs")
    print("=" * 60)

    from vagen_vsi_rl.models.actor_vllm import ActorVLLM, GenerateOutput

    MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
    CACHE_DIR = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir"

    # ── 2a: Instantiate (lazy — no GPU used yet) ──
    print("[2a] Creating ActorVLLM instance (lazy)...")
    actor = ActorVLLM(
        model_id=MODEL_ID,
        cache_dir=CACHE_DIR,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.80,
        max_model_len=4096,
        max_images=8,
        temperature=0.7,
        max_new_tokens=512,
        dtype="float16",
    )
    print("  ✓ ActorVLLM instance created (model not loaded yet)")

    # ── 2b: Test with text-only message ──
    print("\n[2b] Testing text-only generation with logprobs...")
    text_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": "What is 2+2? Reply with just the number."}
        ]},
    ]

    t0 = time.time()
    out = actor.generate_with_logprobs(text_messages, max_new_tokens=64)
    t1 = time.time()

    assert isinstance(out, GenerateOutput), f"Expected GenerateOutput, got {type(out)}"
    assert isinstance(out.text, str) and len(out.text) > 0, "text should be non-empty"
    assert out.token_ids is not None, "token_ids should not be None"
    assert out.token_logprobs is not None, "token_logprobs should not be None"
    assert len(out.token_ids) == len(out.token_logprobs), \
        f"token_ids ({len(out.token_ids)}) and token_logprobs ({len(out.token_logprobs)}) must align"
    assert out.token_ids.dtype.is_floating_point == False, "token_ids should be integer"
    assert out.token_logprobs.dtype.is_floating_point, "token_logprobs should be float"
    assert (out.token_logprobs <= 0).all(), "log-probs should be ≤ 0"
    assert out.prompt_token_ids is not None, "prompt_token_ids should not be None"
    assert len(out.prompt_token_ids) > 0, "prompt_token_ids should not be empty"
    assert out.finish_reason in ("stop", "length"), f"Unexpected finish_reason: {out.finish_reason}"

    print(f"  ✓ Text: {out.text[:100]!r}...")
    print(f"  ✓ token_ids: shape={out.token_ids.shape}, dtype={out.token_ids.dtype}")
    print(f"  ✓ token_logprobs: shape={out.token_logprobs.shape}, dtype={out.token_logprobs.dtype}")
    print(f"  ✓ prompt_token_ids: shape={out.prompt_token_ids.shape}")
    print(f"  ✓ finish_reason: {out.finish_reason}")
    print(f"  ✓ logprob range: [{out.token_logprobs.min():.4f}, {out.token_logprobs.max():.4f}]")
    print(f"  ✓ Time: {t1-t0:.2f}s")

    # ── 2c: Test to_dict() serialization ──
    print("\n[2c] Testing GenerateOutput.to_dict() serialization...")
    d = out.to_dict()
    assert isinstance(d, dict)
    assert "text" in d and "token_ids" in d and "token_logprobs" in d
    assert isinstance(d["token_ids"], list)
    assert isinstance(d["token_logprobs"], list)
    assert len(d["token_ids"]) == len(d["token_logprobs"])
    # Verify JSON-serializable
    json_str = json.dumps(d)
    assert len(json_str) > 0
    print(f"  ✓ to_dict() keys: {list(d.keys())}")
    print(f"  ✓ JSON-serializable: {len(json_str)} chars")

    # ── 2d: Test with image input (using a test image from habitat) ──
    print("\n[2d] Testing generation with image input...")

    # First create a dummy image
    from PIL import Image as PILImage
    import numpy as np
    dummy_img = PILImage.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    dummy_img_path = "/tmp/pipeline_test_dummy.png"
    dummy_img.save(dummy_img_path)

    img_messages = [
        {"role": "system", "content": "You are a spatial reasoning assistant exploring a 3D scene."},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{dummy_img_path}"},
            {"type": "text", "text": (
                "You see this image of a room. "
                "Describe what you see and provide a movement action as JSON:\n"
                '{"rotation_angle_degrees": <degrees>, "forward_meters": <m>, '
                '"left_meters": <m>, "z_delta_meters": <m>, "answer": null, "done": false}'
            )},
        ]},
    ]

    t0 = time.time()
    out_img = actor.generate_with_logprobs(img_messages, max_new_tokens=256)
    t1 = time.time()

    assert isinstance(out_img, GenerateOutput)
    assert len(out_img.text) > 0
    assert out_img.token_ids is not None
    assert out_img.token_logprobs is not None
    assert len(out_img.token_ids) == len(out_img.token_logprobs)
    assert (out_img.token_logprobs <= 0).all()

    print(f"  ✓ Text: {out_img.text[:120]!r}...")
    print(f"  ✓ Generated {len(out_img.token_ids)} tokens with logprobs")
    print(f"  ✓ prompt_token_ids: {len(out_img.prompt_token_ids)} tokens (includes image tokens)")
    print(f"  ✓ Time: {t1-t0:.2f}s")

    # ── 2e: Test text-only generate() (backward compat) ──
    print("\n[2e] Testing text-only generate() method...")
    text_out = actor.generate(text_messages, max_new_tokens=32)
    assert isinstance(text_out, str) and len(text_out) > 0
    print(f"  ✓ generate() returned: {text_out[:80]!r}")

    # ── 2f: Test batch generation ──
    print("\n[2f] Testing batch generation with logprobs...")
    batch = [text_messages, img_messages]
    t0 = time.time()
    batch_out = actor.generate_batch_with_logprobs(batch, max_new_tokens=64)
    t1 = time.time()

    assert isinstance(batch_out, list)
    assert len(batch_out) == 2
    for i, bo in enumerate(batch_out):
        assert isinstance(bo, GenerateOutput), f"batch[{i}]: expected GenerateOutput"
        assert len(bo.token_ids) == len(bo.token_logprobs), f"batch[{i}]: ids/logprobs mismatch"
        assert (bo.token_logprobs <= 0).all(), f"batch[{i}]: logprobs should be ≤ 0"
    print(f"  ✓ Batch of {len(batch_out)} outputs, each with aligned token_ids + logprobs")
    print(f"  ✓ Output[0]: {len(batch_out[0].token_ids)} tokens")
    print(f"  ✓ Output[1]: {len(batch_out[1].token_ids)} tokens")
    print(f"  ✓ Time: {t1-t0:.2f}s")

    # ── 2g: Verify the PPO-critical contract ──
    print("\n[2g] Verifying PPO contract (token_ids, token_logprobs alignment)...")
    # For PPO, we need every generated token to have an aligned logprob
    for bo in batch_out:
        gen_len = len(bo.token_ids)
        lp_len = len(bo.token_logprobs)
        assert gen_len == lp_len, f"MISMATCH: {gen_len} token_ids vs {lp_len} logprobs"
        assert gen_len > 0, "Should have generated at least 1 token"
        # Verify token_ids are valid (non-negative integers)
        assert (bo.token_ids >= 0).all(), "token_ids should be non-negative"
    print(f"  ✓ All outputs have 1:1 token_id ↔ token_logprob alignment")
    print(f"  ✓ All token_ids are non-negative integers")
    print(f"  ✓ All token_logprobs are ≤ 0 (valid log-probabilities)")

    # Cleanup
    actor.cleanup()
    print("\n✅ TEST 2 PASSED: ActorVLLM generate_with_logprobs works correctly!")
    print("   Deliverable verified: generate_with_logprobs(messages) -> {text, token_ids, token_logprobs}")
    return True


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Pipeline test")
    parser.add_argument("--habitat-only", action="store_true")
    parser.add_argument("--vllm-only", action="store_true")
    args = parser.parse_args()

    run_all = not args.habitat_only and not args.vllm_only

    results = {}

    if run_all or args.habitat_only:
        try:
            results["habitat"] = test_habitat_env()
        except Exception as e:
            print(f"\n❌ TEST 1 FAILED: {e}")
            traceback.print_exc()
            results["habitat"] = False

    if run_all or args.vllm_only:
        try:
            results["vllm"] = test_actor_vllm()
        except Exception as e:
            print(f"\n❌ TEST 2 FAILED: {e}")
            traceback.print_exc()
            results["vllm"] = False

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE TEST SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} — {name}")

    if all(results.values()):
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
