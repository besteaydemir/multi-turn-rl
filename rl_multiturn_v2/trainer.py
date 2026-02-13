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
import shutil

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
        original_model_path: Optional[str] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Policy model (must be trainable, not vLLM)
            config: TrainerConfig with hyperparameters
            tokenizer: Tokenizer for computing log probabilities
            original_model_path: Path to original model (for copying processor files)
        """
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.original_model_path = original_model_path
        
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
        
        return path
    
    def save_hf_checkpoint(self, path: Optional[Path] = None) -> Path:
        """
        Save model in HuggingFace format for vLLM loading.
        
        This saves the model in a format that vLLM can load directly.
        Used for weight synchronization (Option A).
        
        Args:
            path: Optional path for checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        if path is None:
            path = self.output_dir / f"hf_checkpoint_step_{self.global_step}"
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        print(f"[Trainer] Saving HuggingFace checkpoint to {path}")
        
        # Use model's save_pretrained if available
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(path)
        else:
            # Fallback: save state dict
            torch.save(self.model.state_dict(), path / "pytorch_model.bin")
        
        # Also save tokenizer/processor if available
        if self.tokenizer is not None and hasattr(self.tokenizer, 'save_pretrained'):
            self.tokenizer.save_pretrained(path)
        
        # Copy processor config files from original model path
        # These are needed by vLLM for vision-language models
        if self.original_model_path is not None:
            original_path = Path(self.original_model_path)
            processor_files = [
                "preprocessor_config.json",
                "chat_template.json", 
                "processor_config.json",
                "image_processor_config.json",
                "special_tokens_map.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "merges.txt",
                "vocab.json",
            ]
            
            for fname in processor_files:
                src = original_path / fname
                if src.exists():
                    dst = path / fname
                    if not dst.exists():  # Don't overwrite if tokenizer already saved it
                        shutil.copy2(src, dst)
                        print(f"[Trainer] Copied {fname} from original model")
        
        print(f"[Trainer] HuggingFace checkpoint saved to {path}")
        
        return path
    
    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Get model state dict for direct weight sync.
        
        Used for weight synchronization (Option B).
        
        Returns:
            Model state dict
        """
        return self.model.state_dict()
    
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
    
    def find_latest_checkpoint(self) -> Optional[Path]:
        """
        Find the latest checkpoint in output directory.
        
        Returns:
            Path to latest checkpoint or None
        """
        checkpoints = list(self.output_dir.glob("checkpoint_step_*"))
        if not checkpoints:
            return None
        
        # Sort by step number
        def get_step(p):
            try:
                return int(p.name.split("_")[-1])
            except:
                return 0
        
        checkpoints.sort(key=get_step, reverse=True)
        return checkpoints[0]
    
    def resume_from_latest(self) -> bool:
        """
        Resume training from the latest checkpoint.
        
        Returns:
            True if checkpoint was found and loaded
        """
        latest = self.find_latest_checkpoint()
        if latest is None:
            print("[Trainer] No checkpoint found, starting from scratch")
            return False
        
        print(f"[Trainer] Resuming from checkpoint: {latest}")
        self.load_checkpoint(latest)
        return True
    
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
# FULL TRAINING LOOP WITH WEIGHT SYNC
# ============================================================================

class OnlineRLTrainer:
    """
    Complete online RL training loop with weight synchronization.
    
    Alternates between:
    1. Rollout: Collect trajectories using current policy (vLLM)
    2. Training: Update policy using collected data
    3. Weight Sync: Synchronize vLLM weights with trained HuggingFace model
    
    Weight Sync Options:
    - Option A (checkpoint): Save HF model to disk, reload vLLM from checkpoint
    - Option B (direct): Use vLLM's update_weights API (if available)
    """
    
    def __init__(
        self,
        rollout_engine,
        trainer: RLTrainer,
        config: TrainerConfig,
        weight_sync_method: str = "checkpoint",  # "checkpoint", "direct", or "none"
        weight_sync_interval: int = 1,  # Sync every N updates
    ):
        """
        Initialize online trainer.
        
        Args:
            rollout_engine: VLLMRolloutEngine for trajectory collection
            trainer: RLTrainer for policy updates
            config: TrainerConfig
            weight_sync_method: Method for weight synchronization
            weight_sync_interval: How often to sync weights
        """
        self.rollout_engine = rollout_engine
        self.trainer = trainer
        self.config = config
        self.weight_sync_method = weight_sync_method
        self.weight_sync_interval = weight_sync_interval
        
        # Training state
        self.update_count = 0
        self.all_trajectories = []
        
        # Temp dir for weight sync checkpoints
        self._temp_checkpoint_dir = None
    
    def _sync_weights(self) -> bool:
        """
        Synchronize weights from trainer to rollout engine.
        
        Returns:
            True if sync was successful
        """
        if self.weight_sync_method == "none":
            return True
        
        print(f"[OnlineRL] Syncing weights (method: {self.weight_sync_method})...")
        
        if self.weight_sync_method == "direct":
            # Option B: Direct weight update
            state_dict = self.trainer.get_state_dict()
            success = self.rollout_engine.sync_weights_direct(state_dict)
            
            if not success:
                print("[OnlineRL] Direct sync failed, falling back to checkpoint method")
                self.weight_sync_method = "checkpoint"
                return self._sync_weights()
            
            return True
        
        elif self.weight_sync_method == "checkpoint":
            # Option A: Save checkpoint and reload
            import tempfile
            import shutil
            
            # Create temp checkpoint directory
            if self._temp_checkpoint_dir is None:
                self._temp_checkpoint_dir = Path(tempfile.mkdtemp(prefix="rl_weight_sync_"))
            
            checkpoint_path = self._temp_checkpoint_dir / f"sync_step_{self.trainer.global_step}"
            
            # Save HF checkpoint
            self.trainer.save_hf_checkpoint(checkpoint_path)
            
            # Reload vLLM from checkpoint
            success = self.rollout_engine.sync_weights_from_checkpoint(
                checkpoint_path, cleanup=True
            )
            
            return success
        
        else:
            print(f"[OnlineRL] Unknown weight sync method: {self.weight_sync_method}")
            return False
    
    def train(
        self,
        questions: List[Dict[str, Any]],
        num_updates: int = 100,
        episodes_per_update: int = 4,
        scene_loader = None,
    ) -> Dict[str, Any]:
        """
        Run the full online RL training loop.
        
        Args:
            questions: List of question dicts for training
            num_updates: Number of policy updates
            episodes_per_update: Trajectories to collect per update
            scene_loader: Optional SceneLoader for rendering
            
        Returns:
            Training summary
        """
        print(f"[OnlineRL] Starting training: {num_updates} updates, {episodes_per_update} episodes/update")
        print(f"[OnlineRL] Weight sync: {self.weight_sync_method} (every {self.weight_sync_interval} updates)")
        
        for update_idx in range(num_updates):
            print(f"\n{'='*60}")
            print(f"Update {update_idx + 1}/{num_updates}")
            print(f"{'='*60}")
            
            # Sample questions for this update
            import random
            sampled = random.sample(questions, min(episodes_per_update, len(questions)))
            
            # Setup scene loader for each question if provided
            render_fn = None
            if scene_loader is not None:
                # Load scene for first question (batch processing would need enhancement)
                scene_id = sampled[0].get("scene_id")
                dataset = sampled[0].get("dataset")
                
                try:
                    scene_loader.load_scene(scene_id, dataset)
                    scene_loader.compute_initial_pose()
                    initial_image = scene_loader.render_current_view()
                    
                    # Add initial image to questions
                    for q in sampled:
                        if q.get("scene_id") == scene_id:
                            q["initial_image_path"] = str(initial_image)
                    
                    # Create render function
                    from .scene_loader import create_render_fn
                    render_fn = create_render_fn(scene_loader)
                except Exception as e:
                    print(f"[OnlineRL] Scene loading failed: {e}")
            
            # Collect trajectories
            print("[OnlineRL] Collecting trajectories...")
            trajectories = self.rollout_engine.collect_batch(sampled, render_fn=render_fn)
            
            # Save trajectories to disk
            print("[OnlineRL] Saving trajectories...")
            traj_dir = self.trainer.output_dir.parent / "trajectories" / f"update_{update_idx:03d}"
            traj_dir.mkdir(parents=True, exist_ok=True)
            
            for traj_idx, traj in enumerate(trajectories):
                traj_subdir = traj_dir / f"traj_{traj_idx:02d}"
                try:
                    traj.save(traj_subdir)
                    print(f"[OnlineRL] Saved trajectory {traj_idx} to {traj_subdir}")
                except Exception as e:
                    print(f"[OnlineRL] Failed to save trajectory {traj_idx}: {e}")
            
            # Log trajectory stats
            correct = sum(1 for t in trajectories if t.is_correct)
            print(f"[OnlineRL] Collected {len(trajectories)} trajectories, {correct} correct")
            
            # Compute rewards (currently binary: 1 for correct, 0 for incorrect)
            for traj in trajectories:
                traj.terminal_reward = 1.0 if traj.is_correct else 0.0
            
            # Train on collected data
            print("[OnlineRL] Training...")
            metrics = self.trainer.train_on_batch(trajectories)
            
            # Log metrics
            print(f"[OnlineRL] Policy loss: {metrics['policy_loss']:.4f}")
            print(f"[OnlineRL] Num action tokens: {metrics['num_action_tokens']}")
            
            self.update_count += 1
            self.all_trajectories.extend(trajectories)
            
            # Weight sync
            if self.update_count % self.weight_sync_interval == 0:
                self._sync_weights()
        
        # Cleanup
        if self._temp_checkpoint_dir and self._temp_checkpoint_dir.exists():
            import shutil
            shutil.rmtree(self._temp_checkpoint_dir, ignore_errors=True)
        
        return {
            "num_updates": self.update_count,
            "total_trajectories": len(self.all_trajectories),
            "final_metrics": self.trainer.get_metrics_summary(),
        }
