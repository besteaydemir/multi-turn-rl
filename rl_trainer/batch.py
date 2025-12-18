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
    
    IMPORTANT: This batch keeps multi-turn episodes together!
    - Each episode can have multiple turns (1-N turns)
    - RLOO baseline is computed at EPISODE level (using episode rewards)
    - Loss is computed by aggregating logprobs across ALL turns in each episode
    
    Structure:
        episodes: List of Episode objects (length = batch_size)
        rewards: Tensor of episode-level rewards [batch_size]
        
    For episode i with T_i turns:
        - All T_i turns contribute to the same loss term
        - Loss_i = -advantage_i * sum_{t=1}^{T_i} sum_{action_tokens} logprob_t
        
    Note: For LOO baseline, batch_size >= 2 required. Batch size of 4-8 recommended.
    """
    
    # Episodes (kept as structured objects)
    episodes: List[Episode]              # List of Episode objects
    
    # Episode-level metadata
    rewards: torch.Tensor                # Shape: [batch_size] - episode rewards
    episode_ids: List[str]               # Episode identifiers
    
    # Additional info
    batch_size: int = field(init=False)
    device: str = "cpu"
    
    def __post_init__(self):
        """Initialize batch size and validate structure."""
        self.batch_size = len(self.rewards)
        
        # Validate shapes
        assert len(self.episodes) == self.batch_size, f"Mismatch: {len(self.episodes)} episodes vs {self.batch_size} rewards"
        assert len(self.episode_ids) == self.batch_size, f"Mismatch: {len(self.episode_ids)} IDs vs {self.batch_size} rewards"
    
    def to(self, device: str) -> 'EpisodeBatch':
        """Move all tensors to specified device."""
        return EpisodeBatch(
            episodes=self.episodes,  # Episode objects don't need device movement
            rewards=self.rewards.to(device),
            episode_ids=self.episode_ids,
            device=device
        )
    
    def __len__(self) -> int:
        """Return the batch size (number of episodes)."""
        return self.batch_size
    
    def get_num_turns(self, idx: int) -> int:
        """Get number of turns in episode idx."""
        return len(self.episodes[idx].turns)
    
    def get_episode(self, idx: int) -> Episode:
        """Get episode by index."""
        return self.episodes[idx]


def prepare_batch(
    episodes: List[Episode],
    processor,
    pad_token_id: Optional[int] = None,
    device: str = "cpu"
) -> EpisodeBatch:
    """
    Prepare a batch of episodes for training.
    
    NEW BEHAVIOR: Keeps episodes as structured objects instead of unpacking turns!
    - Each episode retains its multi-turn structure
    - RLOO baseline computed at episode level
    - Loss aggregates across all turns within each episode
    
    Args:
        episodes: List of Episode objects (N >= 1)
        processor: Not used in new implementation (kept for compatibility)
        pad_token_id: Not used in new implementation (kept for compatibility)
        device: Device to place tensors on
        
    Returns:
        EpisodeBatch with episodes kept as structured objects
        
    Note: For LOO baseline, ensure len(episodes) >= 2
    """
    batch_size = len(episodes)
    
    # Extract episode-level rewards
    rewards = torch.tensor([ep.final_reward for ep in episodes], dtype=torch.float32, device=device)
    episode_ids = [ep.episode_id for ep in episodes]
    
    # Return batch with episodes as structured objects
    return EpisodeBatch(
        episodes=episodes,
        rewards=rewards,
        episode_ids=episode_ids,
        device=device
    )


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
                    num_action_tokens = ep.turns[0].action_token_mask.sum().item() if ep.turns[0].action_token_mask is not None else 0
                    print(f"    Turn 0 has {num_action_tokens} action tokens")
                    if hasattr(ep.turns[0], 'action') and ep.turns[0].action:
                        print(f"    Turn 0 action: forward={ep.turns[0].action.forward_meters}, done={ep.turns[0].action.done}")
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
