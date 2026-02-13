#!/usr/bin/env python3
"""Test gradient updates and memory usage with real model and images."""

import torch
import gc
from PIL import Image
import numpy as np

def print_memory_stats(prefix=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"{prefix}")
        print(f"  Allocated: {allocated:.2f} GB, Peak: {peak:.2f} GB, Available: {(80 - allocated):.2f} GB")
        return allocated
    return 0

def create_dummy_image(size=(384, 384)):
    """Create a dummy RGB image."""
    return Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype=np.uint8))

def main():
    print("=" * 80)
    print("RL Training Memory Test - 16 Frames")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("No GPU available!")
        return
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.reset_peak_memory_stats()
    
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    
    model_path = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
    
    print(f"\nLoading model from: {model_path}")
    print_memory_stats("Before load:")
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    print_memory_stats("\nAfter model load:")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    # Test with 16 frames
    num_frames = 16
    batch_size = 2  # Start conservative with images
    
    print(f"\n--- Testing with {num_frames} frames, batch_size={batch_size} ---")
    
    # Create messages with 16 images
    dummy_images = [create_dummy_image() for _ in range(num_frames)]
    
    # Build message with multiple images
    image_content = [{"type": "image", "image": img} for img in dummy_images]
    messages = [
        {
            "role": "user",
            "content": image_content + [
                {"type": "text", "text": "Based on these 16 views of the room, answer: What color is the chair? Options: A) red B) blue C) green. Think step by step."}
            ]
        }
    ]
    
    # Process for each batch item
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text] * batch_size,
        images=image_inputs * batch_size if image_inputs else None,
        videos=video_inputs * batch_size if video_inputs else None,
        return_tensors="pt",
        padding=True,
    ).to("cuda")
    
    print(f"Input IDs shape: {inputs['input_ids'].shape}")
    if 'pixel_values' in inputs:
        print(f"Pixel values shape: {inputs['pixel_values'].shape}")
    
    model.gradient_checkpointing_enable()
    model.train()
    
    print_memory_stats("\nBefore forward:")
    
    outputs = model(
        **inputs,
        labels=inputs['input_ids'],
    )
    
    print(f"Loss: {outputs.loss.item():.4f}")
    print_memory_stats("After forward:")
    
    outputs.loss.backward()
    print_memory_stats("After backward:")
    
    from torch.optim import AdamW
    optimizer = AdamW(model.parameters(), lr=1e-5)
    optimizer.step()
    optimizer.zero_grad()
    print_memory_stats("After optimizer step:")
    
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n📊 Peak memory for {num_frames} frames, batch_size={batch_size}: {peak:.2f} GB")
    print(f"📊 Memory per sample: ~{peak / batch_size:.2f} GB")
    print(f"📊 Estimated max batch_size for 80GB: ~{int(80 / (peak / batch_size))}")
    print("\n✓ All tests passed!")

if __name__ == "__main__":
    main()
