#!/usr/bin/env python3
"""
Test script for Video Backend using HuggingFace (MIG-friendly).

This script tests video processing using the HuggingFace backend directly,
which is more compatible with MIG devices than vLLM. Use this to verify
the video format is correct before testing with vLLM.

Usage:
    python test_video_backend_hf.py --num-frames 32
    python test_video_backend_hf.py --num-frames 16 --model 4B
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_hf_video_backend(num_frames: int = 32, model_size: str = "4B"):
    """Test video processing with HuggingFace backend."""
    
    print("=" * 60)
    print(f"Testing HuggingFace Video Processing")
    print(f"Frames: {num_frames}, Model: Qwen3-VL-{model_size}")
    print("=" * 60)
    
    import torch
    print(f"\n[INFO] PyTorch version: {torch.__version__}")
    print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] CUDA device: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[INFO] CUDA memory: {mem_gb:.1f} GB")
    
    # Select model - use 4B for MIG due to memory constraints
    if model_size == "4B":
        model_id = "Qwen/Qwen3-VL-4B-Instruct"
    else:
        model_id = "Qwen/Qwen3-VL-8B-Instruct"
        print("[WARN] 8B model may not fit on MIG partition!")
    
    cache_dir = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir"
    
    print(f"\n[STEP 1] Loading model and processor...")
    print(f"         Model: {model_id}")
    
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=cache_dir,
        )
        print("[STEP 1] ✅ Model loaded!")
        
    except Exception as e:
        print(f"[STEP 1] ❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Find test frames
    print(f"\n[STEP 2] Finding test frames...")
    test_scene_dir = Path("/dss/mcmlscratch/06/di38riq/arkit_vsi/raw/Training/40753679/vga_wide")
    
    if not test_scene_dir.exists():
        print(f"[STEP 2] ❌ Test scene not found: {test_scene_dir}")
        return False
    
    frame_files = sorted(test_scene_dir.glob("*.png"))
    print(f"[STEP 2] Found {len(frame_files)} total frames")
    
    if len(frame_files) < num_frames:
        selected_frames = frame_files
    else:
        import numpy as np
        indices = np.linspace(0, len(frame_files) - 1, num_frames, dtype=int)
        selected_frames = [frame_files[i] for i in indices]
    
    print(f"[STEP 2] ✅ Selected {len(selected_frames)} frames")
    
    # Build message with VIDEO format
    print(f"\n[STEP 3] Building video message...")
    
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": [str(p) for p in selected_frames],
            },
            {
                "type": "text",
                "text": "Describe what you see in this video. Be brief (1-2 sentences)."
            }
        ]
    }]
    
    print("[STEP 3] ✅ Message built")
    
    # Process with qwen_vl_utils
    print(f"\n[STEP 4] Processing video with qwen_vl_utils...")
    
    try:
        from qwen_vl_utils import process_vision_info
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        print(f"         image_inputs: {type(image_inputs)}, len={len(image_inputs) if image_inputs else 0}")
        print(f"         video_inputs: {type(video_inputs)}, len={len(video_inputs) if video_inputs else 0}")
        
        if video_inputs:
            print(f"         video shape: {video_inputs[0].shape if hasattr(video_inputs[0], 'shape') else 'N/A'}")
            print("[STEP 4] ✅ Video processed correctly (video_inputs is populated)")
        else:
            print("[STEP 4] ⚠️ video_inputs is empty! This may indicate a format issue.")
            if image_inputs:
                print(f"         BUT image_inputs has {len(image_inputs)} items")
        
    except Exception as e:
        print(f"[STEP 4] ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Apply chat template and generate
    print(f"\n[STEP 5] Generating response...")
    
    try:
        import time
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(model.device)
        
        start = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=100,
            )
        elapsed = time.time() - start
        
        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        print(f"[STEP 5] ✅ Generated in {elapsed:.1f}s")
        print(f"\n{'='*60}")
        print("MODEL RESPONSE:")
        print("="*60)
        print(response)
        print("="*60)
        
    except Exception as e:
        print(f"[STEP 5] ❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n{'='*60}")
    print(f"SUCCESS! HF video processing works with {num_frames} frames!")
    print("="*60)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Test HF Video Backend")
    parser.add_argument("--num-frames", type=int, default=32, help="Number of frames to test")
    parser.add_argument("--model", type=str, default="4B", choices=["4B", "8B"], help="Model size")
    
    args = parser.parse_args()
    
    success = test_hf_video_backend(num_frames=args.num_frames, model_size=args.model)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
