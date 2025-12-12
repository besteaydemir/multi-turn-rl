"""
Multimodal input handling for Qwen3-VL models.

This module ensures proper handling of images and text for:
1. Episode generation (simulator.py)
2. Teacher-forcing for log-probability computation (logprobs.py)
3. Maintaining consistency between generation and training passes

Qwen3-VL requirements:
- Uses special <|vision_start|> and <|vision_end|> tokens around image regions
- Processor handles image encoding and token insertion automatically
- During training, must use same formatting as generation for correct conditionals
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
from pathlib import Path


class MultimodalInputBuilder:
    """
    Build model inputs with proper multimodal formatting for Qwen3-VL.
    
    This class ensures consistency between:
    - Generation (simulator): Uses processor.apply_chat_template with images
    - Training (logprobs): Uses same formatting for teacher-forcing
    """
    
    def __init__(self, processor, device: str = "cuda"):
        """
        Initialize multimodal input builder.
        
        Args:
            processor: Qwen3VL processor (handles tokenization + image encoding)
            device: Device for tensors
        """
        self.processor = processor
        self.device = device
        
    def prepare_generation_inputs(
        self,
        messages: List[Dict[str, Any]],
        images: Optional[List[Image.Image]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for model.generate().
        
        This uses the processor's apply_chat_template which:
        1. Inserts special vision tokens where images appear
        2. Encodes images to pixel_values or image_embeds
        3. Returns proper input_ids, attention_mask, and image tensors
        
        Args:
            messages: Chat-formatted messages (list of dicts with 'role' and 'content')
            images: List of PIL images (if multimodal)
            
        Returns:
            Dict with keys:
                - input_ids: [1, seq_len] with vision tokens inserted
                - attention_mask: [1, seq_len]
                - pixel_values or image_embeds: Image data (if images provided)
        """
        # Use processor's chat template formatting
        # This handles vision token insertion automatically
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def prepare_teacher_forcing_inputs(
        self,
        context_input_ids: torch.Tensor,
        generated_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: Optional[List[Image.Image]] = None,
        image_grid_thw: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for teacher-forcing forward pass (used in log-prob computation).
        
        CRITICAL: Must use same multimodal formatting as generation to get correct
        conditional probabilities log p(action | context, images).
        
        Args:
            context_input_ids: [batch_size, max_context_len] - Context tokens (includes vision tokens)
            generated_ids: [batch_size, max_gen_len] - Generated action tokens
            attention_mask: [batch_size, seq_len] - Attention mask for full sequence
            images: List of PIL images (batch_size items, each can be single image or list)
            image_grid_thw: [num_images, 3] - Image grid dimensions (if using Qwen3-VL)
            
        Returns:
            Dict with keys:
                - input_ids: [batch_size, context_len + gen_len] - Full sequence
                - attention_mask: [batch_size, context_len + gen_len]
                - pixel_values or image_embeds: Image data (if images provided)
                - image_grid_thw: Grid dimensions (if using Qwen3-VL)
        """
        # Concatenate context + generated
        full_input_ids = torch.cat([context_input_ids, generated_ids], dim=1)
        
        inputs = {
            "input_ids": full_input_ids,
            "attention_mask": attention_mask
        }
        
        # Add image inputs if provided
        if images is not None:
            # Process images through processor
            # For batch processing, we need to handle each episode's images
            if isinstance(images[0], list):
                # Multiple images per episode (multi-turn case)
                # Flatten and process
                all_images = []
                for episode_images in images:
                    all_images.extend(episode_images)
                
                # Encode images
                image_inputs = self.processor.image_processor(
                    images=all_images,
                    return_tensors="pt"
                )
                
                # Move to device
                for k, v in image_inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                
                # Add grid dimensions if provided
                if image_grid_thw is not None:
                    inputs["image_grid_thw"] = image_grid_thw.to(self.device)
            else:
                # Single image per episode
                image_inputs = self.processor.image_processor(
                    images=images,
                    return_tensors="pt"
                )
                
                for k, v in image_inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                
                if image_grid_thw is not None:
                    inputs["image_grid_thw"] = image_grid_thw.to(self.device)
        
        return inputs
    
    def extract_context_from_generation(
        self,
        generation_inputs: Dict[str, torch.Tensor],
        input_length: int
    ) -> Dict[str, Any]:
        """
        Extract context portion from generation inputs for later teacher-forcing.
        
        When we generate actions during episode collection, we need to save:
        1. Context input IDs (including vision tokens)
        2. Image data (pixel_values or embeddings)
        3. Image grid dimensions
        
        This data will be used later during training for teacher-forcing.
        
        Args:
            generation_inputs: Full inputs used for generation
            input_length: Length of context (before generation starts)
            
        Returns:
            Dict with:
                - context_input_ids: [1, input_length] - Context tokens
                - image data (pixel_values/image_embeds if present)
                - image_grid_thw (if present)
        """
        context_data = {
            "context_input_ids": generation_inputs["input_ids"][:, :input_length]
        }
        
        # Copy image-related keys
        image_keys = ["pixel_values", "image_embeds", "image_grid_thw"]
        for key in image_keys:
            if key in generation_inputs:
                context_data[key] = generation_inputs[key]
        
        return context_data
    
    def validate_multimodal_consistency(
        self,
        generation_inputs: Dict[str, torch.Tensor],
        training_inputs: Dict[str, torch.Tensor]
    ) -> Tuple[bool, str]:
        """
        Validate that training inputs match generation inputs in structure.
        
        This ensures we're computing log p(action | context) with the same
        conditional distribution used during generation.
        
        Args:
            generation_inputs: Inputs used during generation
            training_inputs: Inputs used for teacher-forcing
            
        Returns:
            (is_valid, error_message)
        """
        # Check that both have same keys (modulo sequence length difference)
        gen_keys = set(generation_inputs.keys())
        train_keys = set(training_inputs.keys())
        
        # Core required keys
        required = {"input_ids", "attention_mask"}
        if not required.issubset(gen_keys):
            return False, f"Generation inputs missing required keys: {required - gen_keys}"
        if not required.issubset(train_keys):
            return False, f"Training inputs missing required keys: {required - train_keys}"
        
        # Check image keys consistency
        image_keys = {"pixel_values", "image_embeds", "image_grid_thw"}
        gen_image_keys = gen_keys & image_keys
        train_image_keys = train_keys & image_keys
        
        if gen_image_keys != train_image_keys:
            return False, f"Image key mismatch: gen={gen_image_keys}, train={train_image_keys}"
        
        # Check image tensor shapes match (except for batch dimension)
        for key in gen_image_keys:
            gen_shape = generation_inputs[key].shape
            train_shape = training_inputs[key].shape
            
            # Shapes should match except possibly batch size
            if gen_shape[1:] != train_shape[1:]:
                return False, f"{key} shape mismatch: gen={gen_shape}, train={train_shape}"
        
        return True, ""
    
    def get_vision_token_positions(
        self,
        input_ids: torch.Tensor
    ) -> List[Tuple[int, int]]:
        """
        Find positions of vision tokens in input sequence.
        
        Qwen3-VL uses special tokens like <|vision_start|> and <|vision_end|>
        to mark where image embeddings are inserted.
        
        Args:
            input_ids: [batch_size, seq_len] or [seq_len]
            
        Returns:
            List of (start_pos, end_pos) tuples for each vision region
        """
        if input_ids.dim() == 2:
            input_ids = input_ids[0]  # Take first in batch
        
        # Get special vision token IDs from processor
        # This depends on the specific Qwen3-VL tokenizer
        try:
            vision_start_id = self.processor.tokenizer.encode("<|vision_start|>", add_special_tokens=False)[0]
            vision_end_id = self.processor.tokenizer.encode("<|vision_end|>", add_special_tokens=False)[0]
        except:
            # Fallback: no vision tokens found
            return []
        
        positions = []
        start_idx = None
        
        for i, token_id in enumerate(input_ids):
            if token_id == vision_start_id:
                start_idx = i
            elif token_id == vision_end_id and start_idx is not None:
                positions.append((start_idx, i + 1))
                start_idx = None
        
        return positions


def load_images_from_paths(image_paths: List[str]) -> List[Image.Image]:
    """
    Load PIL images from file paths.
    
    Args:
        image_paths: List of paths to image files
        
    Returns:
        List of PIL Image objects
    """
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"Warning: Failed to load image {path}: {e}")
            # Create blank image as fallback
            images.append(Image.new("RGB", (224, 224), color=(0, 0, 0)))
    
    return images


def batch_images_for_episodes(episodes) -> Tuple[List[List[Image.Image]], List[torch.Tensor]]:
    """
    Extract and batch images from episodes for training.
    
    Args:
        episodes: List of Episode objects
        
    Returns:
        images_per_episode: List of image lists (one list per episode)
        image_grid_thw: Stacked grid dimensions if available
    """
    images_per_episode = []
    
    for episode in episodes:
        episode_images = []
        
        # Collect all images from all turns
        for turn in episode.turns:
            if hasattr(turn.observation, 'images') and turn.observation.images:
                # Load images
                images = load_images_from_paths(turn.observation.images)
                episode_images.extend(images)
        
        images_per_episode.append(episode_images)
    
    # For now, return images without grid dimensions
    # Grid dimensions can be computed by processor if needed
    return images_per_episode, None
