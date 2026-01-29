#!/usr/bin/env python3
"""
Test script for vLLM Video Backend on MIG devices.

MIG (Multi-Instance GPU) devices require special configuration for vLLM:
- tensor_parallel_size must be 1
- GPU memory utilization should be conservative
- May need to set CUDA_VISIBLE_DEVICES explicitly

Usage:
    python test_video_backend_mig.py --num-frames 32
    python test_video_backend_mig.py --num-frames 16 --model 4B
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_video_backend(num_frames: int = 32, model_size: str = "4B"):
    """Test the vLLM video backend with specified number of frames."""
    
    print("=" * 60)
    print(f"Testing vLLM Video Backend on MIG")
    print(f"Frames: {num_frames}, Model: Qwen3-VL-{model_size}")
    print("=" * 60)
    
    # MIG-specific environment settings
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["OMP_NUM_THREADS"] = "1"
    
    # Check CUDA availability
    import torch
    print(f"\n[INFO] PyTorch version: {torch.__version__}")
    print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Select model
    if model_size == "4B":
        model_id = "Qwen/Qwen3-VL-4B-Instruct"
    else:
        model_id = "Qwen/Qwen3-VL-8B-Instruct"
    
    print(f"\n[STEP 1] Creating vLLM Video Backend...")
    print(f"         Model: {model_id}")
    
    try:
        from utils import create_inference_backend
        
        # MIG-optimized settings:
        # - Lower GPU memory utilization for MIG partitions
        # - Smaller max_model_len to fit in MIG memory
        # - tensor_parallel_size=1 (required for MIG)
        backend = create_inference_backend(
            backend="vllm_video",
            model_id=model_id,
            gpu_memory_utilization=0.80,  # Conservative for MIG
            max_model_len=16384,  # Smaller for MIG memory constraints
            tensor_parallel_size=1,  # Required for MIG
        )
        print("[STEP 1] ✅ Backend created successfully!")
        
    except Exception as e:
        print(f"[STEP 1] ❌ Failed to create backend: {e}")
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
        print(f"[STEP 2] ⚠️ Only {len(frame_files)} frames available, using all")
        selected_frames = frame_files
    else:
        # Sample equally spaced frames
        import numpy as np
        indices = np.linspace(0, len(frame_files) - 1, num_frames, dtype=int)
        selected_frames = [frame_files[i] for i in indices]
    
    print(f"[STEP 2] ✅ Selected {len(selected_frames)} frames")
    
    # Build test message
    print(f"\n[STEP 3] Building video message with {len(selected_frames)} frames...")
    
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": [str(p) for p in selected_frames],
            },
            {
                "type": "text",
                "text": "Describe what you see in this video walkthrough of an indoor space. What type of room is this? Be brief."
            }
        ]
    }]
    
    print("[STEP 3] ✅ Message built")
    
    # Generate response
    print(f"\n[STEP 4] Generating response (this may take a moment)...")
    
    try:
        import time
        start = time.time()
        response = backend.generate(messages, max_new_tokens=200)
        elapsed = time.time() - start
        
        print(f"[STEP 4] ✅ Response generated in {elapsed:.1f}s")
        print(f"\n{'='*60}")
        print("MODEL RESPONSE:")
        print("="*60)
        print(response)
        print("="*60)
        
    except Exception as e:
        print(f"[STEP 4] ❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Check for the specific error we're trying to fix
        error_str = str(e)
        if "16 image" in error_str or "At most 16" in error_str:
            print("\n" + "!"*60)
            print("ERROR: This is the '16 image limit' error!")
            print("The video backend is NOT working correctly.")
            print("vLLM is still treating video frames as images.")
            print("!"*60)
        
        return False
    
    # Cleanup
    print(f"\n[STEP 5] Cleaning up...")
    try:
        backend.cleanup()
        print("[STEP 5] ✅ Cleanup complete")
    except:
        pass
    
    print(f"\n{'='*60}")
    print(f"SUCCESS! Video backend works with {num_frames} frames!")
    print("="*60)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Test vLLM Video Backend on MIG")
    parser.add_argument("--num-frames", type=int, default=32, help="Number of frames to test")
    parser.add_argument("--model", type=str, default="4B", choices=["4B", "8B"], help="Model size")
    
    args = parser.parse_args()
    
    success = test_video_backend(num_frames=args.num_frames, model_size=args.model)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
