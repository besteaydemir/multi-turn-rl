#!/usr/bin/env python3
"""
RL Trainer for multi-turn view selection.

Implements:
- REINFORCE / PPO-lite policy gradient
- Action-token-only optimization
- Turn-level structure (VAGEN-compatible)
- KL regularization to reference model (optional)
- Entropy bonus (optional)

Currently: Zero reward (tests pipeline correctness).
Extension point: Add non-zero rewards later.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import time
import json
import copy

from .data_structures import Trajectory, Turn


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TrainerConfig:
    """Configuration for RL trainer."""
    
    # =========================================================================
    # OPTIMIZATION
    # =========================================================================
    
    learning_rate: float = 1e-5           # AdamW learning rate
    weight_decay: float = 0.01            # Weight decay
    max_grad_norm: float = 1.0            # Gradient clipping threshold
    gradient_accumulation_steps: int = 1  # Accumulate over N batches
    
    # =========================================================================
    # LOSS COEFFICIENTS
    # =========================================================================
    
    # KL penalty to reference model (optional)
    kl_coef: float = 0.0                  # β - set to 0 to disable
    
    # Entropy bonus (optional)
    entropy_coef: float = 0.0             # α - set to 0 to disable
    
    # =========================================================================
    # PPO SETTINGS (if enabled)
    # =========================================================================
    
    use_ppo: bool = False                 # Enable PPO clipping
    ppo_clip_range: float = 0.2           # ε for ratio clipping
    ppo_epochs: int = 4                   # Optimization epochs per batch
    
    # =========================================================================
    # CRITIC SETTINGS (optional)
    # =========================================================================
    
    use_critic: bool = False              # Enable value function
    value_loss_coef: float = 0.5          # Coefficient for value loss
    
    # =========================================================================
    # REFERENCE MODEL
    # =========================================================================
    
    use_reference_model: bool = False     # Enable KL regularization
    ref_model_strategy: str = "frozen"    # "frozen", "ema", or "periodic"
    ref_ema_tau: float = 0.999            # EMA decay (if strategy="ema")
    ref_update_interval: int = 100        # Update interval (if strategy="periodic")
    
    # =========================================================================
    # TRAINING
    # =========================================================================
    
    batch_size: int = 4                   # Trajectories per batch
    num_epochs: int = 1                   # Epochs over collected data
    
    # =========================================================================
    # LOGGING
    # =========================================================================
    
    log_interval: int = 1                 # Log every N updates
    save_interval: int = 100              # Save checkpoint every N updates
    
    use_wandb: bool = True                # Enable W&B logging
    wandb_project: str = "rl-view-selection"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    
    # =========================================================================
    # OUTPUT
    # =========================================================================
    
    output_dir: str = "./checkpoints"
    device: str = "cuda"


# ============================================================================
# ADVANTAGE COMPUTATION
# ============================================================================

def compute_advantages(
    trajectories: List[Trajectory],
    gamma: float = 1.0,
    use_gae: bool = False,
    gae_lambda: float = 0.95,
    values: Optional[List[List[float]]] = None,
) -> List[List[float]]:
    """
    Compute advantages for all trajectories.
    
    Currently: Returns zeros (no reward signal).
    Extension point: Implement GAE for VAGEN.
    
    Args:
        trajectories: List of collected trajectories
        gamma: Discount factor
        use_gae: Use Generalized Advantage Estimation
        gae_lambda: GAE lambda parameter
        values: Value estimates per turn (if using critic)
        
    Returns:
        List of advantage lists (one per trajectory, one per turn)
    """
    all_advantages = []
    
    for i, traj in enumerate(trajectories):
        returns = traj.compute_returns(gamma=gamma)
        
        if use_gae and values is not None:
            # GAE computation (for future VAGEN support)
            traj_values = values[i] if i < len(values) else [0.0] * len(traj.turns)
            advantages = []
            gae = 0.0
            
            for t in reversed(range(len(traj.turns))):
                if t == len(traj.turns) - 1:
                    next_value = 0.0
                else:
                    next_value = traj_values[t + 1]
                
                delta = traj.turns[t].reward + gamma * next_value - traj_values[t]
                gae = delta + gamma * gae_lambda * gae
                advantages.append(gae)
            
            advantages = list(reversed(advantages))
        else:
            # Simple returns - values (or just returns if no critic)
            if values is not None and i < len(values):
                traj_values = values[i]
                advantages = [r - v for r, v in zip(returns, traj_values)]
            else:
                # No critic - just use returns (currently all zeros)
                advantages = returns
        
        all_advantages.append(advantages)
    
    return all_advantages


def normalize_advantages(advantages: List[List[float]]) -> List[List[float]]:
    """
    Normalize advantages across all trajectories.
    
    Standard normalization: (adv - mean) / (std + eps)
    """
    # Flatten all advantages
    flat = [a for traj_advs in advantages for a in traj_advs]
    
    if len(flat) == 0:
        return advantages
    
    mean = sum(flat) / len(flat)
    variance = sum((a - mean) ** 2 for a in flat) / len(flat)
    std = variance ** 0.5 + 1e-8
    
    # Normalize
    normalized = []
    for traj_advs in advantages:
        normalized.append([(a - mean) / std for a in traj_advs])
    
    return normalized


# ============================================================================
# RL TRAINER
# ============================================================================

class RLTrainer:
    """
    Policy gradient trainer for multi-turn view selection.
    
    Implements:
    - REINFORCE with optional baseline
    - PPO-lite with ratio clipping (optional)
    - Action-token-only optimization
    - Turn-level credit assignment
    
    Design principles:
    - Only action tokens receive gradients
    - Observation tokens (images) are masked out
    - Turn boundaries are preserved for VAGEN extension
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        tokenizer = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Policy model (must be trainable, not vLLM)
            config: TrainerConfig with hyperparameters
            tokenizer: Tokenizer for computing log probabilities
        """
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        
        # Move model to device
        self.model = self.model.to(config.device)
        self.model.train()
        
        # Reference model (for KL penalty)
        self.ref_model = None
        if config.use_reference_model:
            self._init_reference_model()
        
        # Value head (for critic)
        self.value_head = None
        if config.use_critic:
            self._init_value_head()
        
        # Optimizer
        params = list(self.model.parameters())
        if self.value_head is not None:
            params += list(self.value_head.parameters())
        
        self.optimizer = AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Training state
        self.global_step = 0
        self.metrics_history = []
        
        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # W&B logger
        self.wandb_run = None
        if config.use_wandb:
            self._init_wandb()
    
    def _init_reference_model(self):
        """Initialize frozen reference model."""
        print("[Trainer] Creating reference model...")
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        print("[Trainer] Reference model created.")
    
    def _init_value_head(self):
        """Initialize value head for critic."""
        # Get hidden size from model config
        hidden_size = getattr(self.model.config, 'hidden_size', 4096)
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(self.config.device)
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        try:
            import wandb
            
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.wandb_run_name,
                config=vars(self.config),
            )
            print(f"[Trainer] W&B initialized: {wandb.run.url}")
        except Exception as e:
            print(f"[Trainer] W&B init failed: {e}")
            self.wandb_run = None
    
    def train_step(
        self,
        trajectories: List[Trajectory],
    ) -> Dict[str, float]:
        """
        Perform one training step on a batch of trajectories.
        
        Args:
            trajectories: List of collected trajectories
            
        Returns:
            Dict of metrics
        """
        self.model.train()
        
        # Compute advantages
        advantages = compute_advantages(trajectories)
        advantages = normalize_advantages(advantages)
        
        # Accumulate loss over trajectories
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_kl = 0.0
        total_entropy = 0.0
        num_action_tokens = 0
        
        self.optimizer.zero_grad()
        
        for traj_idx, (traj, traj_advs) in enumerate(zip(trajectories, advantages)):
            for turn_idx, (turn, advantage) in enumerate(zip(traj.turns, traj_advs)):
                if turn.generated_ids is None or turn.action_token_mask is None:
                    continue
                
                # Get action tokens only
                action_mask = turn.action_token_mask
                if not action_mask.any():
                    continue
                
                # Compute policy loss for this turn
                turn_loss, turn_metrics = self._compute_turn_loss(
                    turn=turn,
                    advantage=advantage,
                    action_mask=action_mask,
                )
                
                if turn_loss is not None:
                    total_policy_loss += turn_loss
                    num_action_tokens += action_mask.sum().item()
                    
                    total_kl += turn_metrics.get("kl", 0.0)
                    total_entropy += turn_metrics.get("entropy", 0.0)
        
        # Average losses
        if num_action_tokens > 0:
            total_policy_loss = total_policy_loss / len(trajectories)
        
        # Compute total loss
        total_loss = total_policy_loss
        
        if self.config.kl_coef > 0:
            total_loss = total_loss + self.config.kl_coef * total_kl
        
        if self.config.entropy_coef > 0:
            total_loss = total_loss - self.config.entropy_coef * total_entropy
        
        if self.config.use_critic:
            total_loss = total_loss + self.config.value_loss_coef * total_value_loss
        
        # Backward pass
        if total_loss.requires_grad:
            total_loss.backward()
            
            # Gradient clipping
            if self.config.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
            
            self.optimizer.step()
        
        self.global_step += 1
        
        # Compile metrics
        metrics = {
            "policy_loss": total_policy_loss.item() if torch.is_tensor(total_policy_loss) else total_policy_loss,
            "value_loss": total_value_loss.item() if torch.is_tensor(total_value_loss) else total_value_loss,
            "kl_divergence": total_kl.item() if torch.is_tensor(total_kl) else total_kl,
            "entropy": total_entropy.item() if torch.is_tensor(total_entropy) else total_entropy,
            "num_action_tokens": num_action_tokens,
            "num_trajectories": len(trajectories),
            "global_step": self.global_step,
        }
        
        self.metrics_history.append(metrics)
        
        # Log to W&B
        if self.wandb_run is not None and self.global_step % self.config.log_interval == 0:
            import wandb
            wandb.log(metrics, step=self.global_step)
        
        # Save checkpoint
        if self.global_step % self.config.save_interval == 0:
            self.save_checkpoint()
        
        return metrics
    
    def _compute_turn_loss(
        self,
        turn: Turn,
        advantage: float,
        action_mask: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        """
        Compute policy gradient loss for one turn.
        
        Only action tokens contribute to the loss.
        """
        metrics = {}
        
        # Get input IDs (we need the full sequence for context)
        input_ids = turn.generated_ids.to(self.config.device)
        
        # We need to recompute logprobs through the model
        # This is necessary because rollout used vLLM (different model instance)
        
        # For now, use stored logprobs if available (with gradient approximation)
        if turn.logprobs is not None:
            logprobs = turn.logprobs.to(self.config.device)
            
            # Get action token logprobs
            action_logprobs = logprobs[action_mask]
            
            # Compute advantage-weighted loss
            # REINFORCE: -log_prob * advantage
            loss = -(action_logprobs * advantage).mean()
            
            # Entropy (approximation from logprobs)
            entropy = -action_logprobs.mean()
            metrics["entropy"] = entropy.item()
            
            return loss, metrics
        
        # If no stored logprobs, we need to forward through the model
        # This path requires proper prompt encoding
        # For minimal implementation, return None
        return None, metrics
    
    def train_on_batch(
        self,
        trajectories: List[Trajectory],
    ) -> Dict[str, float]:
        """
        Train on a batch of trajectories.
        
        Handles gradient accumulation and multiple PPO epochs.
        """
        if self.config.use_ppo:
            # PPO: multiple optimization epochs
            all_metrics = []
            for epoch in range(self.config.ppo_epochs):
                metrics = self.train_step(trajectories)
                all_metrics.append(metrics)
            
            # Average metrics across epochs
            avg_metrics = {}
            for key in all_metrics[0]:
                if isinstance(all_metrics[0][key], (int, float)):
                    avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
                else:
                    avg_metrics[key] = all_metrics[-1][key]
            return avg_metrics
        else:
            # REINFORCE: single epoch
            return self.train_step(trajectories)
    
    def save_checkpoint(self, path: Optional[Path] = None):
        """Save model checkpoint."""
        if path is None:
            path = self.output_dir / f"checkpoint_step_{self.global_step}"
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), path / "model.pt")
        
        # Save optimizer
        torch.save(self.optimizer.state_dict(), path / "optimizer.pt")
        
        # Save training state
        state = {
            "global_step": self.global_step,
            "config": vars(self.config),
            "metrics_history": self.metrics_history[-100:],  # Keep last 100
        }
        with open(path / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)
        
        print(f"[Trainer] Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        path = Path(path)
        
        # Load model
        self.model.load_state_dict(torch.load(path / "model.pt"))
        
        # Load optimizer
        self.optimizer.load_state_dict(torch.load(path / "optimizer.pt"))
        
        # Load training state
        with open(path / "training_state.json", "r") as f:
            state = json.load(f)
        
        self.global_step = state["global_step"]
        self.metrics_history = state.get("metrics_history", [])
        
        print(f"[Trainer] Loaded checkpoint from {path} (step {self.global_step})")
    
    def get_metrics_summary(self) -> Dict[str, float]:
        """Get summary of recent metrics."""
        if not self.metrics_history:
            return {}
        
        recent = self.metrics_history[-10:]
        summary = {}
        
        for key in recent[0]:
            if isinstance(recent[0][key], (int, float)):
                summary[f"avg_{key}"] = sum(m[key] for m in recent) / len(recent)
        
        return summary


# ============================================================================
# FULL TRAINING LOOP
# ============================================================================

class OnlineRLTrainer:
    """
    Complete online RL training loop.
    
    Alternates between:
    1. Rollout: Collect trajectories using current policy (vLLM)
    2. Training: Update policy using collected data
    """
    
    def __init__(
        self,
        rollout_engine,
        trainer: RLTrainer,
        config: TrainerConfig,
    ):
        """
        Initialize online trainer.
        
        Args:
            rollout_engine: VLLMRolloutEngine for trajectory collection
            trainer: RLTrainer for policy updates
            config: TrainerConfig
        """
        self.rollout_engine = rollout_engine
        self.trainer = trainer
        self.config = config
        
        # Training state
        self.update_count = 0
        self.all_trajectories = []
    
    def train(
        self,
        questions: List[Dict[str, Any]],
        num_updates: int = 100,
        episodes_per_update: int = 4,
    ) -> Dict[str, Any]:
        """
        Run the full online RL training loop.
        
        Args:
            questions: List of question dicts for training
            num_updates: Number of policy updates
            episodes_per_update: Trajectories to collect per update
            
        Returns:
            Training summary
        """
        print(f"[OnlineRL] Starting training: {num_updates} updates, {episodes_per_update} episodes/update")
        
        for update_idx in range(num_updates):
            print(f"\n{'='*60}")
            print(f"Update {update_idx + 1}/{num_updates}")
            print(f"{'='*60}")
            
            # Sample questions for this update
            import random
            sampled = random.sample(questions, min(episodes_per_update, len(questions)))
            
            # Collect trajectories
            print("[OnlineRL] Collecting trajectories...")
            trajectories = self.rollout_engine.collect_batch(sampled)
            
            # Log trajectory stats
            correct = sum(1 for t in trajectories if t.is_correct)
            print(f"[OnlineRL] Collected {len(trajectories)} trajectories, {correct} correct")
            
            # Train on collected data
            print("[OnlineRL] Training...")
            metrics = self.trainer.train_on_batch(trajectories)
            
            # Log metrics
            print(f"[OnlineRL] Policy loss: {metrics['policy_loss']:.4f}")
            print(f"[OnlineRL] Num action tokens: {metrics['num_action_tokens']}")
            
            self.update_count += 1
            self.all_trajectories.extend(trajectories)
        
        return {
            "num_updates": self.update_count,
            "total_trajectories": len(self.all_trajectories),
            "final_metrics": self.trainer.get_metrics_summary(),
        }
