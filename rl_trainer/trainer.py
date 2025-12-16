#!/usr/bin/env python3
"""
Trainer module for policy gradient RL with VLMs.

This module implements the full training loop with:
- AdamW optimizer with configurable learning rate
- Gradient clipping for stability
- Gradient accumulation for large batches
- Logging and checkpointing
- Learning rate scheduling
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from typing import Optional, Dict, List, Callable
from pathlib import Path
from dataclasses import dataclass, field
import json
import time
from tqdm import tqdm

from .batch import EpisodeDataLoader, compute_loo_baseline
from .logprobs import compute_sequence_logprobs, compute_advantages, policy_gradient_loss
from .reference import ReferenceModelManager, KLScheduler
from .logging_utils import WandBLogger, format_training_metrics
from .checkpoint import CheckpointManager, should_save_checkpoint


@dataclass
class TrainerConfig:
    """Configuration for RL trainer."""
    
    # Optimization
    learning_rate: float = 1e-5           # AdamW learning rate (1e-5 to 5e-6 for large models)
    weight_decay: float = 0.01            # Weight decay for AdamW
    max_grad_norm: float = 1.0            # Gradient clipping threshold
    gradient_accumulation_steps: int = 1  # Accumulate gradients over N microbatches
    
    # Loss coefficients
    kl_coef: float = 0.01                 # β - KL penalty coefficient (initial)
    entropy_coef: float = 0.01            # α - Entropy bonus coefficient
    
    # Reference model management
    ref_model_strategy: str = "ema"       # "frozen", "periodic", or "ema"
    ref_update_interval: int = 100        # For periodic strategy
    ref_ema_tau: float = 0.999            # For EMA strategy (τ near 0.999)
    
    # KL scheduling
    use_kl_scheduler: bool = True         # Enable adaptive β scheduling
    target_kl: float = 0.01               # Target KL divergence
    kl_tolerance: float = 0.005           # Tolerance around target
    kl_adaptation_rate: float = 1.5       # Multiplier for β adaptation
    
    # Training
    num_epochs: int = 3
    batch_size: int = 8                   # Episodes per batch (N >= 8 recommended)
    
    # Learning rate schedule
    warmup_steps: int = 100               # Linear warmup steps
    use_cosine_schedule: bool = True      # Cosine annealing after warmup
    
    # Logging
    log_interval: int = 10                # Log every N steps
    save_interval: int = 100              # Save checkpoint every N steps
    log_examples: bool = True             # Log example episodes
    num_log_examples: int = 5             # Number of examples to log
    
    # WandB
    use_wandb: bool = True                # Enable WandB logging
    wandb_project: str = "vlm-navigation-rl"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    
    # Checkpointing
    output_dir: str = "./checkpoints"
    save_total_limit: int = 3             # Keep only N latest checkpoints
    save_optimizer: bool = True           # Save optimizer state
    save_every_n_epochs: int = 5          # Save checkpoint every N epochs (0 = use step-based saving)
    resume_from_checkpoint: Optional[str] = None  # Path to checkpoint to resume from
    
    # Device
    device: str = "cuda"
    
    # Mixed precision (optional)
    use_amp: bool = False                 # Automatic Mixed Precision
    amp_dtype: torch.dtype = torch.bfloat16  # Dtype for autocast (bfloat16 or float16)


class RLTrainer:
    """
    Policy gradient trainer for VLM navigation.
    
    Implements REINFORCE with:
    - Leave-One-Out baseline
    - KL penalty relative to reference model
    - Entropy regularization
    - Gradient clipping
    - Gradient accumulation
    """
    
    def __init__(
        self,
        model: nn.Module,
        dataloader: EpisodeDataLoader,
        config: TrainerConfig
    ):
        """
        Initialize trainer.
        
        Args:
            model: Policy model to train (will have gradients)
            dataloader: EpisodeDataLoader with training episodes
            config: TrainerConfig with hyperparameters
        
        Note:
            Reference model is created automatically by ReferenceModelManager
            based on config.ref_model_strategy.
        """
        self.model = model
        self.dataloader = dataloader
        self.config = config
        
        # Move models to device
        self.model = self.model.to(config.device)
        
        # Setup reference model manager (if enabled)
        if config.ref_model_strategy is not None:
            self.ref_manager = ReferenceModelManager(
                policy_model=self.model,
                strategy=config.ref_model_strategy,
                update_interval=config.ref_update_interval,
                ema_tau=config.ref_ema_tau,
                device=config.device
            )
            self.ref_model = self.ref_manager.get_reference_model()
        else:
            self.ref_manager = None
            self.ref_model = None
        
        # Setup KL scheduler
        if config.use_kl_scheduler:
            self.kl_scheduler = KLScheduler(
                initial_kl_coef=config.kl_coef,
                target_kl=config.target_kl,
                kl_tolerance=config.kl_tolerance,
                adaptation_rate=config.kl_adaptation_rate,
                warmup_steps=config.warmup_steps
            )
        else:
            self.kl_scheduler = None
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Setup learning rate scheduler
        # For online RL with None dataloader, skip schedulers or use dummy values
        if dataloader is not None:
            total_steps = len(dataloader) * config.num_epochs
        else:
            total_steps = 1000  # Dummy value for online RL
        
        if config.warmup_steps > 0:
            self.warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=config.warmup_steps
            )
        else:
            self.warmup_scheduler = None
        
        if config.use_cosine_schedule and dataloader is not None:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - config.warmup_steps,
                eta_min=config.learning_rate * 0.1
            )
        else:
            self.scheduler = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.metrics_history = []
        self.last_save_time = time.time()
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config (convert non-serializable objects to strings)
        config_dict = vars(config).copy()
        if 'amp_dtype' in config_dict:
            config_dict['amp_dtype'] = str(config_dict['amp_dtype'])
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Setup logging
        if config.use_wandb:
            self.logger = WandBLogger(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=vars(config),
                entity=config.wandb_entity,
                tags=config.wandb_tags,
                enabled=True
            )
            # DISABLED: watch_model causes massive slowdown during generation
            # It hooks into EVERY forward pass (100+ times per generation)
            # self.logger.watch_model(self.model, log_freq=config.log_interval)
        else:
            self.logger = None
        
        # Setup checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(self.output_dir),
            save_total_limit=config.save_total_limit,
            save_optimizer=config.save_optimizer,
            save_reference=True
        )
        
        # Resume from checkpoint if specified
        if config.resume_from_checkpoint:
            self._resume_from_checkpoint(config.resume_from_checkpoint)
        
        # Mixed precision scaler (only for float16, not bfloat16)
        # Note: autocast can be used with bfloat16, but GradScaler is only for float16
        self.scaler = None
        if config.use_amp and config.amp_dtype == torch.float16:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def _resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        
        training_state = self.checkpoint_manager.load_checkpoint(
            Path(checkpoint_path),
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            kl_scheduler=self.kl_scheduler,
            ref_manager=self.ref_manager
        )
        
        # Load reference model
        self.checkpoint_manager.load_reference_model(
            Path(checkpoint_path),
            self.ref_model
        )
        
        # Restore training state
        self.global_step = training_state.get("step", 0)
        self.epoch = training_state.get("epoch", 0)
        
        print(f"Resumed from step {self.global_step}, epoch {self.epoch}")
    
    def train_step(self, batch) -> Dict[str, float]:
        """
        Single training step on a batch.
        
        Args:
            batch: EpisodeBatch
            
        Returns:
            metrics: Dict of training metrics
        """
        # Ensure model is in training mode
        self.model.train()
        
        # DEBUG: Check if batch has valid data (once per epoch)
        if self.global_step == 0 or (hasattr(self, '_last_debug_epoch') and self._last_debug_epoch != self.epoch):
            print(f"\n[DEBUG] Epoch {self.epoch + 1} - First batch stats:")
            print(f"  Batch size: {batch.batch_size}")
            print(f"  Rewards shape: {batch.rewards.shape}")
            print(f"  Rewards: {batch.rewards.tolist()}")
            print(f"  Context input_ids shape: {batch.context_input_ids.shape}")
            print(f"  Generated_ids shape: {batch.generated_ids.shape}")
            print(f"  Action masks shape: {batch.action_masks.shape}")
            print(f"  Action masks sum: {batch.action_masks.sum(dim=1).tolist()}")
            self._last_debug_epoch = self.epoch
        
        # Step 1: Compute LOO baseline
        baselines = compute_loo_baseline(batch)
        
        # Step 2: Compute advantages (normalized)
        advantages = compute_advantages(
            rewards=batch.rewards,
            baselines=baselines,
            normalize=True
        )
        
        # DEBUG: Check advantages (only first time)
        if self.global_step == 0:
            print(f"  Baselines: {baselines.tolist()}")
            print(f"  Advantages (before norm): {(batch.rewards - baselines).tolist()}")
            print(f"  Advantages (after norm): {advantages.tolist()}")
        
        # Step 3: Compute sequence log-probs
        with torch.amp.autocast('cuda', enabled=self.config.use_amp, dtype=self.config.amp_dtype):
            logprobs_result = compute_sequence_logprobs(
                model=self.model,
                batch=batch,
                ref_model=self.ref_model,
                compute_entropy=True,
                compute_kl=True,
                device=self.config.device
            )
            
            # Step 4: Compute loss with current KL coefficient
            current_kl_coef = self.kl_scheduler.get_kl_coef() if self.kl_scheduler else self.config.kl_coef
            
            loss, metrics = policy_gradient_loss(
                logprobs_result=logprobs_result,
                advantages=advantages,
                kl_coef=current_kl_coef,
                entropy_coef=self.config.entropy_coef
            )
            
            # DEBUG: Check loss components
            if self.global_step == 0:
                print(f"  LogProbs (logpi_seq): {logprobs_result.logpi_seq.tolist()}")
                print(f"  Num action tokens: {logprobs_result.num_action_tokens.tolist()}")
                print(f"  Mean logpi: {logprobs_result.mean_logpi.tolist()}")
                pg_term = -(logprobs_result.logpi_seq * advantages)
                print(f"  PG loss term (before mean): {pg_term.tolist()}")
                print(f"  Loss components: PG={metrics['loss/policy_gradient']:.6f}, KL={metrics['loss/kl_penalty']:.6f}, Entropy={metrics['loss/entropy_bonus']:.6f}")
                print(f"  Total loss: {loss.item():.6f}")
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
        
        # Step 5: Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Add additional metrics
        metrics["rewards/mean"] = batch.rewards.mean().item()
        metrics["rewards/std"] = batch.rewards.std().item()
        metrics["baselines/mean"] = baselines.mean().item()
        metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        
        return metrics
    
    def optimizer_step(self):
        """
        Perform optimizer step with gradient clipping.
        """
        # Gradient clipping
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )
        
        # Optimizer step
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Learning rate scheduling
        if self.warmup_scheduler is not None and self.global_step < self.config.warmup_steps:
            self.warmup_scheduler.step()
        elif self.scheduler is not None:
            self.scheduler.step()
        
        # Update reference model
        if self.ref_manager is not None:
            ref_updated = self.ref_manager.maybe_update(self.model, self.global_step)
            if ref_updated and self.config.ref_model_strategy == "periodic":
                print(f"  Reference model updated at step {self.global_step}")
        
        return grad_norm.item()
    
    def train_epoch(self) -> List[Dict[str, float]]:
        """
        Train for one epoch.
        
        Returns:
            List of metrics for each step
        """
        epoch_metrics = []
        accumulated_metrics = {}
        
        pbar = tqdm(self.dataloader, desc=f"Epoch {self.epoch + 1}/{self.config.num_epochs}")
        
        for step, batch in enumerate(pbar):
            # Move batch to device
            batch = batch.to(self.config.device)
            
            # Training step
            metrics = self.train_step(batch)
            
            # Accumulate metrics
            for key, value in metrics.items():
                if key not in accumulated_metrics:
                    accumulated_metrics[key] = []
                accumulated_metrics[key].append(value)
            
            # Optimizer step (every N accumulation steps OR at end of epoch)
            is_last_step = (step + 1) == len(self.dataloader)
            should_update = ((step + 1) % self.config.gradient_accumulation_steps == 0) or is_last_step
            
            if should_update:
                grad_norm = self.optimizer_step()
                
                # Average accumulated metrics
                step_metrics = {
                    key: sum(values) / len(values)
                    for key, values in accumulated_metrics.items()
                }
                step_metrics["grad_norm"] = grad_norm
                step_metrics["step"] = self.global_step
                step_metrics["epoch"] = self.epoch
                
                # Update KL scheduler if enabled
                if self.kl_scheduler is not None and "kl/mean" in step_metrics:
                    current_kl = step_metrics["kl/mean"]
                    new_kl_coef = self.kl_scheduler.step(current_kl, self.global_step)
                    step_metrics["kl_coef"] = new_kl_coef
                    
                    # Add KL scheduler stats
                    kl_stats = self.kl_scheduler.get_stats()
                    step_metrics.update({f"kl_scheduler/{k}": v for k, v in kl_stats.items()})
                
                epoch_metrics.append(step_metrics)
                accumulated_metrics = {}
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{step_metrics['loss/total']:.4f}",
                    "reward": f"{step_metrics['rewards/mean']:.3f}",
                    "lr": f"{step_metrics['lr']:.2e}"
                })
                
                # Log metrics
                if self.global_step % self.config.log_interval == 0:
                    self._log_step_metrics(step_metrics, batch)
                
                # Save checkpoint (only if using step-based saving)
                if self.config.save_every_n_epochs == 0:
                    if should_save_checkpoint(
                        step=self.global_step,
                        save_interval=self.config.save_interval,
                        last_save_time=self.last_save_time
                    ):
                        self._save_training_checkpoint()
                        self.last_save_time = time.time()
                
                self.global_step += 1
        
        return epoch_metrics
    
    def train(self):
        """
        Full training loop with WandB logging and checkpointing.
        """
        print("=" * 80)
        print("TRAINING STARTED")
        print("=" * 80)
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Dataset: {len(self.dataloader)} batches per epoch")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"KL coefficient: {self.config.kl_coef}")
        print(f"Entropy coefficient: {self.config.entropy_coef}")
        print(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        print(f"Max grad norm: {self.config.max_grad_norm}")
        print(f"Device: {self.config.device}")
        print(f"WandB logging: {self.config.use_wandb}")
        print("=" * 80)
        
        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            
            # Train epoch
            epoch_start = time.time()
            epoch_metrics = self.train_epoch()
            epoch_time = time.time() - epoch_start
            
            # Compute epoch statistics
            epoch_stats = self._compute_epoch_stats(epoch_metrics)
            
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs} completed in {epoch_time:.1f}s")
            print(f"  Loss: {epoch_stats.get('loss/total', 0):.4f}")
            print(f"  Mean reward: {epoch_stats.get('rewards/mean', 0):.3f}")
            print(f"  Mean advantage: {epoch_stats.get('advantages/mean', 0):.3f}")
            if 'kl/mean' in epoch_stats:
                print(f"  Mean KL: {epoch_stats['kl/mean']:.4f}")
            if 'entropy/mean' in epoch_stats:
                print(f"  Mean entropy: {epoch_stats['entropy/mean']:.4f}")
            
            # Log epoch stats to WandB
            if self.logger:
                epoch_log = {f"epoch/{k}": v for k, v in epoch_stats.items()}
                self.logger.log_training_step(self.global_step, epoch_log)
            
            # Save epoch checkpoint (only if using epoch-based saving)
            if self.config.save_every_n_epochs > 0:
                if (epoch + 1) % self.config.save_every_n_epochs == 0:
                    print(f"[Checkpoint] Saving at epoch {epoch + 1}")
                    self._save_training_checkpoint(is_epoch_end=True)
                else:
                    print(f"[Checkpoint] Skipping save (will save every {self.config.save_every_n_epochs} epochs)")
            else:
                # Step-based saving (save every epoch for backward compatibility)
                self._save_training_checkpoint(is_epoch_end=True)
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED")
        print("=" * 80)
        
        # Save final model
        final_path = self._save_training_checkpoint(is_final=True)
        
        # Log final checkpoint to WandB
        if self.logger:
            self.logger.log_checkpoint(
                final_path,
                step=self.global_step,
                is_best=False,
                metadata={"type": "final", "epoch": self.epoch}
            )
            self.logger.finish()
    
    def _log_step_metrics(self, metrics: Dict[str, float], batch):
        """Log metrics for a single training step."""
        # Log to file
        log_file = self.output_dir / "metrics.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        
        self.metrics_history.append(metrics)
        
        # Log to WandB
        if self.logger:
            self.logger.log_training_step(self.global_step, metrics, commit=False)
            
            # Log action histograms periodically
            if self.global_step % (self.config.log_interval * 10) == 0:
                episodes = batch.episodes if hasattr(batch, 'episodes') else []
                if episodes:
                    self.logger.log_action_histograms(self.global_step, episodes, commit=False)
            
            # Log example episodes periodically
            if self.config.log_examples and self.global_step % (self.config.log_interval * 20) == 0:
                episodes = batch.episodes if hasattr(batch, 'episodes') else []
                if episodes:
                    self.logger.log_example_episodes(
                        self.global_step,
                        episodes,
                        num_examples=self.config.num_log_examples,
                        commit=True
                    )
            else:
                # Commit the training step metrics
                self.logger.log_training_step(self.global_step, {}, commit=True)
    
    def _save_training_checkpoint(
        self,
        is_epoch_end: bool = False,
        is_final: bool = False
    ) -> Path:
        """Save training checkpoint."""
        # Get current metrics
        current_metrics = self.metrics_history[-1] if self.metrics_history else {}
        
        # Determine if this is the best checkpoint
        is_best = False
        if current_metrics and 'reward/mean' in current_metrics:
            current_reward = current_metrics['reward/mean']
            if self.checkpoint_manager.best_metric is None or \
               current_reward > self.checkpoint_manager.best_metric.get('reward/mean', float('-inf')):
                is_best = True
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            step=self.global_step,
            epoch=self.epoch,
            config=vars(self.config),
            ref_model=self.ref_model,
            scheduler=self.scheduler,
            kl_scheduler=self.kl_scheduler,
            ref_manager=self.ref_manager,
            metrics=current_metrics,
            is_best=is_best
        )
        
        # Log to WandB if this is the best checkpoint
        if self.logger and (is_best or is_final):
            metadata = {
                "step": self.global_step,
                "epoch": self.epoch,
                "metrics": current_metrics
            }
            if is_final:
                metadata["type"] = "final"
            
            self.logger.log_checkpoint(
                checkpoint_path,
                step=self.global_step,
                is_best=is_best,
                metadata=metadata
            )
        
        return checkpoint_path
    
    def _compute_epoch_stats(self, epoch_metrics: List[Dict]) -> Dict[str, float]:
        """Compute statistics over an epoch."""
        stats = {}
        
        if not epoch_metrics:
            return stats
        
        # Average each metric
        for key in epoch_metrics[0].keys():
            if key not in ["step", "epoch"]:
                values = [m[key] for m in epoch_metrics if key in m]
                if values:
                    stats[key] = sum(values) / len(values)
        
        return stats
        
        # Remove oldest checkpoints
        while len(checkpoints) > self.config.save_total_limit:
            oldest = checkpoints.pop(0)
            print(f"Removing old checkpoint: {oldest}")
            import shutil
            shutil.rmtree(oldest)


if __name__ == "__main__":
    print("RL Trainer module loaded successfully!")
    print("\nKey components:")
    print("- TrainerConfig: Configuration for training")
    print("- RLTrainer: Full training loop with optimizer, gradient clipping, logging")
    print("\nLoss: L = -E[log π(a|s) * A] + β*KL(π||π_ref) - α*H(π)")
    print("Optimizer: AdamW with lr=1e-5, gradient clipping=1.0")
