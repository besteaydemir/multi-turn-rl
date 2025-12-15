#!/usr/bin/env python3
"""
Batch formation utilities for RL training.

This module handles:
1. Batching episodes with variable lengths
2. Padding and alignment for model inputs
3. Leave-One-Out (LOO) baseline preparation (requires N>=2 episodes)
4. DataLoader abstraction for shuffling and batching

Key concepts:
- context_input_ids: Tokens fed to model during generation (prompt + history)
- generated_ids: Tokens produced by model in this turn
- action_mask: Boolean mask indicating which tokens are action tokens
- full_input_ids: Concatenation of context + generated (for re-running model)
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Iterator
from dataclasses import dataclass, field
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_environment import Episode, Turn


@dataclass
class EpisodeBatch:
    """
    A batch of episodes prepared for RL training.
    
    Context and generated sequences are stored separately and padded to their respective max lengths.
    Supports Leave-One-Out (LOO) baseline computation.
    
    Shapes:
        batch_size: N episodes
        max_context_len: Maximum context length across all episodes
        max_gen_len: Maximum generation length across all episodes
        
    Note: For LOO baseline, N >= 2 is required. N >= 8 recommended for stability.
    """
    
    # Core tensors
    context_input_ids: torch.Tensor      # Shape: [batch_size, max_context_len]
    generated_ids: torch.Tensor          # Shape: [batch_size, max_gen_len]
    action_masks: torch.Tensor           # Shape: [batch_size, max_gen_len] - masks for action tokens
    attention_masks: torch.Tensor        # Shape: [batch_size, max_context_len + max_gen_len]
    
    # Metadata (shape: [batch_size])
    rewards: torch.Tensor                # Scalar rewards per episode
    context_lengths: torch.Tensor        # Length of context (index C)
    generation_lengths: torch.Tensor     # Length of generated sequence
    episode_ids: List[str]               # Episode identifiers
    
    # Additional info
    batch_size: int = field(init=False)
    max_context_len: int = field(init=False)
    max_gen_len: int = field(init=False)
    max_len: int = field(init=False)  # max_context_len + max_gen_len
    device: str = "cpu"
    
    def __post_init__(self):
        self.batch_size = len(self.rewards)
        self.max_context_len = self.context_input_ids.shape[1]
        self.max_gen_len = self.generated_ids.shape[1]
        self.max_len = self.max_context_len + self.max_gen_len
        
        # Validate shapes
        assert self.context_input_ids.shape == (self.batch_size, self.max_context_len)
        assert self.generated_ids.shape == (self.batch_size, self.max_gen_len)
        assert self.action_masks.shape == (self.batch_size, self.max_gen_len)
        assert self.attention_masks.shape == (self.batch_size, self.max_len)
        assert len(self.rewards) == self.batch_size
        assert len(self.context_lengths) == self.batch_size
        assert len(self.generation_lengths) == self.batch_size
        assert len(self.episode_ids) == self.batch_size
    
    def to(self, device: str) -> 'EpisodeBatch':
        """Move all tensors to specified device."""
        return EpisodeBatch(
            context_input_ids=self.context_input_ids.to(device),
            generated_ids=self.generated_ids.to(device),
            action_masks=self.action_masks.to(device),
            attention_masks=self.attention_masks.to(device),
            rewards=self.rewards.to(device),
            context_lengths=self.context_lengths.to(device),
            generation_lengths=self.generation_lengths.to(device),
            episode_ids=self.episode_ids,
            device=device
        )
    
    def __len__(self) -> int:
        """Return the batch size."""
        return self.batch_size
    
    def get_full_input_ids(self, idx: int) -> torch.Tensor:
        """
        Get full input sequence for episode idx: context + generated.
        
        Returns:
            Tensor of shape [total_len] containing concatenated sequence.
        """
        context_len = self.context_lengths[idx].item()
        gen_len = self.generation_lengths[idx].item()
        
        # Extract valid tokens (no padding)
        context = self.context_input_ids[idx, :context_len]
        generated = self.generated_ids[idx, :gen_len]
        
        return torch.cat([context, generated], dim=0)
    
    def get_action_token_indices(self, idx: int) -> torch.Tensor:
        """
        Get indices of action tokens for episode idx (relative to full sequence).
        
        Returns:
            Tensor of indices where action_mask is True.
        """
        gen_len = self.generation_lengths[idx].item()
        action_mask = self.action_masks[idx, :gen_len]
        
        # Get indices (relative to generated sequence)
        action_indices = torch.where(action_mask)[0]
        
        # Offset by context length to get absolute indices
        context_len = self.context_lengths[idx].item()
        return action_indices + context_len


def prepare_batch(
    episodes: List[Episode],
    processor,
    pad_token_id: Optional[int] = None,
    device: str = "cpu"
) -> EpisodeBatch:
    """
    Prepare a batch of episodes for training.
    
    Steps:
    1. Extract context_input_ids and generated_ids from each episode
    2. Concatenate all turns in multi-turn episodes
    3. Pad to max_len
    4. Create attention masks
    5. Align action masks
    
    Args:
        episodes: List of Episode objects (N >= 1)
        processor: Qwen processor (for tokenization if needed)
        pad_token_id: Token ID for padding (default: processor.tokenizer.pad_token_id)
        device: Device to place tensors on
        
    Returns:
        EpisodeBatch ready for training
        
    Note: For LOO baseline, ensure len(episodes) >= 2
    """
    if pad_token_id is None:
        if processor is None:
            raise ValueError("Either processor or pad_token_id must be provided")
        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = processor.tokenizer.eos_token_id
    
    batch_size = len(episodes)
    
    # Collect data from all episodes
    all_context_ids = []
    all_generated_ids = []
    all_action_masks = []
    all_rewards = []
    all_episode_ids = []
    context_lengths = []
    generation_lengths = []
    
    for episode in episodes:
        # For multi-turn episodes, concatenate all turns
        # Context = initial prompt + all previous turn outputs
        # Generated = current turn output (we train on all turns)
        
        # Get the full conversation for this episode
        # For simplicity in Step 4, we'll process each turn as a separate training example
        # In practice, you might want to train on entire multi-turn sequences
        
        for turn in episode.turns:
            # Context: everything before this turn's generation
            context_ids = turn.context_input_ids
            
            # Generated: tokens produced in this turn
            generated_ids = turn.generated_ids
            
            # Action mask: marks action tokens within generated sequence
            action_mask = turn.action_token_mask
            
            # Store
            all_context_ids.append(context_ids)
            all_generated_ids.append(generated_ids)
            all_action_masks.append(action_mask)
            all_rewards.append(episode.final_reward)  # All turns get same episode reward
            all_episode_ids.append(f"{episode.episode_id}_turn{turn.turn_index}")
            
            context_lengths.append(len(context_ids))
            generation_lengths.append(len(generated_ids))
    
    # Find max length for padding
    max_context_len = max(len(ids) for ids in all_context_ids)
    max_gen_len = max(len(ids) for ids in all_generated_ids)
    max_len = max_context_len + max_gen_len
    
    # Pad all sequences
    padded_contexts = []
    padded_generated = []
    padded_action_masks = []
    attention_masks = []
    
    for context_ids, gen_ids, action_mask in zip(all_context_ids, all_generated_ids, all_action_masks):
        context_len = len(context_ids)
        gen_len = len(gen_ids)
        
        # Get device from existing tensors
        device = context_ids.device
        
        # Ensure all tensors are on the same device
        gen_ids = gen_ids.to(device)
        action_mask = action_mask.to(device)
        
        # Pad context to max_context_len
        context_padding = max_context_len - context_len
        padded_context = torch.cat([
            context_ids,
            torch.full((context_padding,), pad_token_id, dtype=torch.long, device=device)
        ])
        
        # Pad generated to max_gen_len
        gen_padding = max_gen_len - gen_len
        padded_gen = torch.cat([
            gen_ids,
            torch.full((gen_padding,), pad_token_id, dtype=torch.long, device=device)
        ])
        
        # Pad action mask to max_gen_len
        mask_padding = max_gen_len - len(action_mask)
        padded_mask = torch.cat([
            action_mask,
            torch.zeros(mask_padding, dtype=torch.bool, device=device)
        ])
        
        # Attention mask (1 for real tokens, 0 for padding)
        # Shape: [max_len] = [max_context_len + max_gen_len]
        attention_mask = torch.zeros(max_len, dtype=torch.long, device=device)
        attention_mask[:context_len] = 1  # Real context tokens
        attention_mask[max_context_len:max_context_len + gen_len] = 1  # Real generated tokens
        
        padded_contexts.append(padded_context)
        padded_generated.append(padded_gen)
        padded_action_masks.append(padded_mask)
        attention_masks.append(attention_mask)
    
    # Stack into tensors
    context_input_ids = torch.stack(padded_contexts)
    generated_ids = torch.stack(padded_generated)
    action_masks = torch.stack(padded_action_masks)
    attention_masks = torch.stack(attention_masks)
    rewards = torch.tensor(all_rewards, dtype=torch.float32)
    context_lengths = torch.tensor(context_lengths, dtype=torch.long)
    generation_lengths = torch.tensor(generation_lengths, dtype=torch.long)
    
    # Create batch
    batch = EpisodeBatch(
        context_input_ids=context_input_ids,
        generated_ids=generated_ids,
        action_masks=action_masks,
        attention_masks=attention_masks,
        rewards=rewards,
        context_lengths=context_lengths,
        generation_lengths=generation_lengths,
        episode_ids=all_episode_ids,
        device=device
    )
    
    return batch.to(device)


def prepare_loo_batches(
    batch: EpisodeBatch,
) -> List[EpisodeBatch]:
    """
    Prepare Leave-One-Out (LOO) batches for baseline computation.
    
    For a batch of N episodes, creates N sub-batches, each with N-1 episodes
    (leaving out one episode at a time). This is used to compute the LOO baseline:
    
        baseline_i = mean(rewards of all episodes except i)
    
    Args:
        batch: Original batch with N episodes (N >= 2 required)
        
    Returns:
        List of N batches, each with N-1 episodes
        
    Raises:
        ValueError: If batch_size < 2
        
    Example:
        Original batch: [ep0, ep1, ep2, ep3]  (N=4)
        LOO batches:
            - [ep1, ep2, ep3]  (leave out ep0)
            - [ep0, ep2, ep3]  (leave out ep1)
            - [ep0, ep1, ep3]  (leave out ep2)
            - [ep0, ep1, ep2]  (leave out ep3)
    """
    N = batch.batch_size
    
    if N < 2:
        raise ValueError(f"LOO baseline requires N >= 2 episodes, got {N}")
    
    loo_batches = []
    
    for i in range(N):
        # Create mask: all True except index i
        mask = torch.ones(N, dtype=torch.bool)
        mask[i] = False
        
        # Extract all episodes except i
        loo_batch = EpisodeBatch(
            context_input_ids=batch.context_input_ids[mask],
            generated_ids=batch.generated_ids[mask],
            action_masks=batch.action_masks[mask],
            attention_masks=batch.attention_masks[mask],
            rewards=batch.rewards[mask],
            context_lengths=batch.context_lengths[mask],
            generation_lengths=batch.generation_lengths[mask],
            episode_ids=[batch.episode_ids[j] for j in range(N) if j != i],
            device=batch.device
        )
        
        loo_batches.append(loo_batch)
    
    return loo_batches


def compute_loo_baseline(batch) -> torch.Tensor:
    """
    Compute Leave-One-Out baseline for each episode in the batch.
    
    For episode i, the baseline is the mean reward of all other episodes:
        baseline_i = mean(rewards[j] for j != i)
    
    Args:
        batch: Either EpisodeBatch or tensor of rewards with shape [N]
        
    Returns:
        Tensor of shape [N] with baseline for each episode
        
    Raises:
        ValueError: If batch_size < 2
    """
    # Handle both EpisodeBatch and plain tensor inputs
    if isinstance(batch, torch.Tensor):
        # Plain tensor of rewards
        rewards = batch
        N = len(rewards)
        device = rewards.device
    else:
        # EpisodeBatch object
        N = batch.batch_size
        rewards = batch.rewards
        device = batch.device
    
    if N < 2:
        # Fallback: use zero baseline (no variance reduction)
        print(f"Warning: LOO baseline requires N >= 2 episodes, got {N}. Using zero baseline.")
        return torch.zeros(N, dtype=torch.float32, device=device)
    
    baselines = torch.zeros(N, dtype=torch.float32, device=device)
    
    for i in range(N):
        # Mean of all rewards except i
        mask = torch.ones(N, dtype=torch.bool, device=device)
        mask[i] = False
        baselines[i] = rewards[mask].mean()
    
    return baselines


class EpisodeDataLoader:
    """
    DataLoader-like abstraction for batching and shuffling episodes.
    
    Features:
    - Batching with configurable batch_size (N >= 8 recommended)
    - Shuffling for each epoch
    - Filtering invalid episodes
    - Support for multi-epoch training
    
    Usage:
        loader = EpisodeDataLoader(episodes, batch_size=8, shuffle=True)
        
        for epoch in range(num_epochs):
            for batch in loader:
                # Train on batch
                loss = compute_loss(batch)
                loss.backward()
    """
    
    def __init__(
        self,
        episodes: List[Episode],
        batch_size: int = 8,
        shuffle: bool = True,
        filter_invalid: bool = True,
        processor = None,
        pad_token_id: Optional[int] = None,
        device: str = "cpu",
        drop_last: bool = False
    ):
        """
        Initialize episode data loader.
        
        Args:
            episodes: List of Episode objects
            batch_size: Number of episodes per batch (N >= 8 recommended)
            shuffle: Whether to shuffle episodes each epoch
            filter_invalid: Whether to filter out invalid episodes
            processor: Qwen processor for tokenization
            pad_token_id: Token ID for padding
            device: Device to place batches on
            drop_last: Whether to drop incomplete batches
        """
        self.episodes = episodes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.filter_invalid = filter_invalid
        self.processor = processor
        self.pad_token_id = pad_token_id
        self.device = device
        self.drop_last = drop_last
        
        # Filter invalid episodes if requested
        if self.filter_invalid:
            self.episodes = [ep for ep in self.episodes if ep.is_valid]
        
        # Validate batch size for LOO
        if self.batch_size < 2:
            print(f"Warning: batch_size={self.batch_size} < 2. LOO baseline will not work.")
            print("Consider using batch_size >= 8 for stability.")
        elif self.batch_size < 8:
            print(f"Warning: batch_size={self.batch_size} < 8. Recommend >= 8 for stability.")
        
        self.num_episodes = len(self.episodes)
        self.num_batches = self._compute_num_batches()
    
    def _compute_num_batches(self) -> int:
        """Compute number of batches per epoch."""
        if self.drop_last:
            return self.num_episodes // self.batch_size
        else:
            return (self.num_episodes + self.batch_size - 1) // self.batch_size
    
    def __len__(self) -> int:
        """Number of batches per epoch."""
        return self.num_batches
    
    def __iter__(self) -> Iterator[EpisodeBatch]:
        """Iterate over batches."""
        # Shuffle if requested
        if self.shuffle:
            indices = np.random.permutation(self.num_episodes)
        else:
            indices = np.arange(self.num_episodes)
        
        # Yield batches
        for i in range(0, self.num_episodes, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            # Skip incomplete batches if drop_last
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue
            
            # Get episodes for this batch
            batch_episodes = [self.episodes[idx] for idx in batch_indices]
            
            # Prepare batch
            batch = prepare_batch(
                episodes=batch_episodes,
                processor=self.processor,
                pad_token_id=self.pad_token_id,
                device=self.device
            )
            
            yield batch
    
    def get_statistics(self) -> Dict[str, float]:
        """Get statistics about the dataset."""
        total = len(self.episodes)
        
        if self.filter_invalid:
            # Already filtered
            valid = total
            invalid = 0
        else:
            valid = sum(1 for ep in self.episodes if ep.is_valid)
            invalid = total - valid
        
        avg_turns = np.mean([len(ep.turns) for ep in self.episodes])
        avg_reward = np.mean([ep.final_reward for ep in self.episodes])
        
        return {
            "total_episodes": total,
            "valid_episodes": valid,
            "invalid_episodes": invalid,
            "avg_turns_per_episode": avg_turns,
            "avg_reward": avg_reward,
            "num_batches": self.num_batches,
            "batch_size": self.batch_size
        }
    
    def log_statistics(self):
        """Print statistics about the dataset."""
        stats = self.get_statistics()
        
        print("=" * 80)
        print("EPISODE DATALOADER STATISTICS")
        print("=" * 80)
        print(f"Total episodes: {stats['total_episodes']}")
        print(f"Valid episodes: {stats['valid_episodes']}")
        print(f"Invalid episodes: {stats['invalid_episodes']}")
        print(f"Average turns per episode: {stats['avg_turns_per_episode']:.2f}")
        print(f"Average reward: {stats['avg_reward']:.3f}")
        print(f"Batch size: {stats['batch_size']}")
        print(f"Number of batches per epoch: {stats['num_batches']}")
        
        # Additional debugging for zero rewards
        if stats['avg_reward'] == 0.0:
            print("\n⚠️  WARNING: Average reward is 0.0!")
            print("Checking individual episode rewards:")
            for i, ep in enumerate(self.episodes[:5]):  # Show first 5
                print(f"  Episode {i}: {len(ep.turns)} turns, reward={ep.final_reward}")
                if len(ep.turns) > 0:
                    print(f"    Turn 0 has {len(ep.turns[0].action_tokens) if ep.turns[0].action_tokens else 0} action tokens")
        print("=" * 80)


if __name__ == "__main__":
    # Quick test
    print("Batch formation module loaded successfully!")
    print("\nKey components:")
    print("- EpisodeBatch: Padded batch of episodes ready for training")
    print("- prepare_batch: Convert list of episodes to EpisodeBatch")
    print("- prepare_loo_batches: Create LOO batches for baseline")
    print("- compute_loo_baseline: Compute LOO baseline values")
    print("- EpisodeDataLoader: DataLoader abstraction for batching/shuffling")
    print("\nRecommended batch_size >= 8 for LOO baseline stability")
