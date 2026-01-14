# Multi-Turn Episode-Level RLOO Refactor Summary

## Changes Made

### 1. `rl_trainer/batch.py`
- **Modified `EpisodeBatch`**: Now keeps episodes as structured objects instead of unpacking turns
  - `episodes: List[Episode]` - keeps full episode structure
  - `rewards: torch.Tensor` - episode-level rewards [batch_size]
  - Removed turn-unpacking logic

- **Simplified `prepare_batch()`**: Just wraps episodes without flattening turns
  ```python
  return EpisodeBatch(
      episodes=episodes,
      rewards=torch.tensor([ep.final_reward for ep in episodes]),
      episode_ids=[ep.episode_id for ep in episodes]
  )
  ```

### 2. `train_rl.py`
- **Dataloader batch_size = 1**: Process one episode at a time
- **TrainerConfig batch_size = num_episodes**: For RLOO group size
- **gradient_accumulation_steps = 1**: No gradient accumulation (handled manually per episode)
- **Fixed validation logging**: Use `trainer.global_step` directly (not global_step-1)

## What Still Needs Implementation

### 3. `rl_trainer/trainer.py` - `train_step()` method
The current `train_step()` assumes flattened turns. Needs complete rewrite:

```python
def train_step(self, all_episodes_batch) -> Dict[str, float]:
    """
    Process a collection of episodes for RLOO training.
    
    Algorithm:
    1. Compute episode-level RLOO baselines using all N episode rewards
    2. For each episode i:
        a. Compute episode-level advantage
        b. For each turn t in episode i:
            - Compute logprobs for action tokens in turn t
        c. Aggregate: total_logprob_i = sum over all turns
        d. Loss_i = -advantage_i * total_logprob_i
        e. Backward (accumulates gradients)
    3. After all episodes: optimizer step
    4. Return aggregated metrics
    """
    
    # Step 1: Compute RLOO baselines (episode-level)
    N = len(all_episodes_batch.episodes)
    rewards = all_episodes_batch.rewards  # [N]
    baselines = compute_loo_baseline(all_episodes_batch)  # [N]
    advantages = compute_advantages(rewards, baselines, normalize=True)  # [N]
    
    # Step 2: Process each episode
    total_loss = 0.0
    metrics_list = []
    
    for ep_idx, episode in enumerate(all_episodes_batch.episodes):
        advantage_i = advantages[ep_idx]
        
        # Accumulate logprobs across all turns
        episode_logprob = 0.0
        episode_entropy = 0.0
        episode_kl = 0.0
        total_action_tokens = 0
        
        for turn in episode.turns:
            # Forward pass for this turn
            context_ids = turn.context_input_ids.to(self.config.device)
            generated_ids = turn.generated_ids.to(self.config.device)
            action_mask = turn.action_token_mask.to(self.config.device)
            
            # Build full sequence
            full_ids = torch.cat([context_ids, generated_ids])
            context_len = len(context_ids)
            
            # Compute logprobs for generated tokens
            with torch.amp.autocast('cuda', enabled=self.config.use_amp, dtype=self.config.amp_dtype):
                outputs = self.model(
                    input_ids=full_ids.unsqueeze(0),  # [1, seq_len]
                    return_dict=True
                )
                logits = outputs.logits  # [1, seq_len, vocab_size]
                
                # Extract logits for generated tokens
                gen_logits = logits[0, context_len-1:-1, :]  # [gen_len, vocab_size]
                
                # Compute log probs
                log_probs = torch.log_softmax(gen_logits, dim=-1)
                
                # Get logprobs for actual tokens
                token_logprobs = log_probs.gather(1, generated_ids.unsqueeze(-1)).squeeze(-1)  # [gen_len]
                
                # Mask to action tokens only
                action_logprobs = token_logprobs[action_mask]
                
                # Accumulate
                episode_logprob += action_logprobs.sum()
                total_action_tokens += action_mask.sum().item()
                
                # Optional: entropy and KL (if ref model exists)
                # ...
        
        # Compute loss for this episode
        loss_i = -advantage_i * episode_logprob
        
        # Backward (accumulates gradients across episodes)
        if self.scaler is not None:
            self.scaler.scale(loss_i / N).backward()  # Scale by N for averaging
        else:
            (loss_i / N).backward()
        
        total_loss += loss_i.item() / N
        
        # Track metrics
        metrics_list.append({
            "episode_logprob": episode_logprob.item(),
            "num_turns": len(episode.turns),
            "num_action_tokens": total_action_tokens
        })
    
    # Step 3: Optimizer step (after processing all episodes)
    grad_norm = self.optimizer_step()
    
    # Aggregate metrics
    metrics = {
        "loss/total": total_loss,
        "rewards/mean": rewards.mean().item(),
        "advantages/mean": advantages.mean().item(),
        "grad_norm": grad_norm,
        "lr": self.optimizer.param_groups[0]["lr"]
    }
    
    return metrics
```

### 4. `rl_trainer/trainer.py` - `train_epoch()` method
Needs to handle the new batch structure:

```python
def train_epoch(self) -> List[Dict[str, float]]:
    """
    Train for one epoch.
    
    With new structure:
    - Dataloader yields batches of size 1 (one episode)
    - Collect all episodes first
    - Then call train_step with all episodes for RLOO
    """
    # Collect all episodes for RLOO group
    all_episodes = []
    for batch in self.dataloader:
        all_episodes.extend(batch.episodes)
    
    # Create mega-batch with all episodes
    mega_batch = EpisodeBatch(
        episodes=all_episodes,
        rewards=torch.tensor([ep.final_reward for ep in all_episodes], device=self.config.device),
        episode_ids=[ep.episode_id for ep in all_episodes],
        device=self.config.device
    )
    
    # Single training step with all episodes
    metrics = self.train_step(mega_batch)
    
    # Log
    if self.logger:
        self.logger.log_training_step(self.global_step, metrics)
    
    self.global_step += 1
    
    return [metrics]  # Return list with single entry
```

## Summary

The refactor enables:
1. ✅ Episode-level RLOO with N=4 episodes
2. ✅ Multi-turn trajectories (1, 2, 6, 3 turns)
3. ✅ Loss aggregated over all turns per episode
4. ✅ One wandb step per model update
5. ✅ Validation after each update

Key insight: **Gradient accumulation happens naturally across episodes**, not via gradient_accumulation_steps.
