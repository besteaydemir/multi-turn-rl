"""
Checkpoint management for RL training.

Handles:
- Saving model, optimizer, and training state
- Loading checkpoints for resuming training
- Managing checkpoint rotation (keeping only N latest)
- Saving best model based on metrics
"""

import torch
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
from datetime import datetime
import os


class CheckpointManager:
    """
    Manages model checkpoints during training.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        save_total_limit: int = 3,
        save_optimizer: bool = True,
        save_reference: bool = True
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_total_limit: Maximum number of checkpoints to keep (0 = unlimited)
            save_optimizer: Whether to save optimizer state
            save_reference: Whether to save reference model
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_total_limit = save_total_limit
        self.save_optimizer = save_optimizer
        self.save_reference = save_reference
        
        # Track checkpoints
        self.checkpoints = []
        self.best_checkpoint = None
        self.best_metric = None
        
        print(f"CheckpointManager initialized:")
        print(f"  Directory: {self.checkpoint_dir}")
        print(f"  Save limit: {self.save_total_limit}")
    
    def save_checkpoint(
        self,
        model,
        optimizer,
        step: int,
        epoch: int,
        config: Dict[str, Any],
        ref_model: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        kl_scheduler: Optional[Any] = None,
        ref_manager: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False
    ) -> Path:
        """
        Save a training checkpoint.
        
        Args:
            model: Policy model to save
            optimizer: Optimizer to save
            step: Global training step
            epoch: Current epoch
            config: Training configuration
            ref_model: Reference model (optional)
            scheduler: Learning rate scheduler (optional)
            kl_scheduler: KL coefficient scheduler (optional)
            ref_manager: Reference model manager (optional)
            metrics: Current metrics (optional)
            is_best: Whether this is the best checkpoint
            
        Returns:
            Path to saved checkpoint directory
        """
        # Create checkpoint directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_step_{step}_{timestamp}"
        if is_best:
            checkpoint_name = "checkpoint_best"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving checkpoint: {checkpoint_name}")
        
        # 1. Save policy model (HuggingFace format if possible)
        model_path = checkpoint_path / "policy_model"
        model_path.mkdir(exist_ok=True)
        
        try:
            # Try HuggingFace save_pretrained
            model.save_pretrained(model_path)
            print(f"  ✓ Policy model saved (HF format)")
        except AttributeError:
            # Fallback: save state dict
            torch.save(model.state_dict(), model_path / "pytorch_model.bin")
            print(f"  ✓ Policy model saved (state dict)")
        
        # 2. Save reference model (if enabled and provided)
        if self.save_reference and ref_model is not None:
            ref_path = checkpoint_path / "reference_model"
            ref_path.mkdir(exist_ok=True)
            
            try:
                ref_model.save_pretrained(ref_path)
                print(f"  ✓ Reference model saved (HF format)")
            except AttributeError:
                torch.save(ref_model.state_dict(), ref_path / "pytorch_model.bin")
                print(f"  ✓ Reference model saved (state dict)")
        
        # 3. Save optimizer state (if enabled)
        if self.save_optimizer and optimizer is not None:
            optimizer_path = checkpoint_path / "optimizer.pt"
            torch.save(optimizer.state_dict(), optimizer_path)
            print(f"  ✓ Optimizer state saved")
        
        # 4. Save scheduler states
        if scheduler is not None:
            scheduler_path = checkpoint_path / "scheduler.pt"
            torch.save(scheduler.state_dict(), scheduler_path)
            print(f"  ✓ LR scheduler saved")
        
        if kl_scheduler is not None:
            kl_scheduler_path = checkpoint_path / "kl_scheduler.pt"
            torch.save({
                "kl_coef": kl_scheduler.kl_coef,
                "kl_history": kl_scheduler.kl_history,
                "kl_coef_history": kl_scheduler.kl_coef_history,
                "total_steps": kl_scheduler.total_steps
            }, kl_scheduler_path)
            print(f"  ✓ KL scheduler saved")
        
        # 5. Save reference manager state
        if ref_manager is not None:
            ref_manager_path = checkpoint_path / "ref_manager.pt"
            torch.save({
                "strategy": ref_manager.strategy,
                "total_updates": ref_manager.total_updates,
                "update_interval": ref_manager.update_interval,
                "ema_tau": ref_manager.ema_tau
            }, ref_manager_path)
            print(f"  ✓ Reference manager state saved")
        
        # 6. Save RNG states
        rng_path = checkpoint_path / "rng_state.pt"
        torch.save({
            "python_rng_state": torch.get_rng_state(),
            "numpy_rng_state": torch.get_rng_state(),  # Can add numpy if needed
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }, rng_path)
        print(f"  ✓ RNG states saved")
        
        # 7. Save training state
        training_state = {
            "step": step,
            "epoch": epoch,
            "config": config,
            "metrics": metrics or {},
            "timestamp": timestamp,
            "is_best": is_best
        }
        
        state_path = checkpoint_path / "training_state.json"
        with open(state_path, "w") as f:
            json.dump(training_state, f, indent=2)
        print(f"  ✓ Training state saved")
        
        # 8. Track checkpoint
        if not is_best:
            self.checkpoints.append(checkpoint_path)
            self._rotate_checkpoints()
        else:
            self.best_checkpoint = checkpoint_path
            if metrics:
                self.best_metric = metrics
        
        print(f"Checkpoint saved to: {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: Path,
        model,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        kl_scheduler: Optional[Any] = None,
        ref_manager: Optional[Any] = None,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Load a checkpoint and restore training state.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            model: Policy model to load into
            optimizer: Optimizer to load into (optional)
            scheduler: LR scheduler to load into (optional)
            kl_scheduler: KL scheduler to load into (optional)
            ref_manager: Reference manager to load into (optional)
            strict: Whether to strictly enforce state dict matching
            
        Returns:
            Dictionary with training state (step, epoch, config, metrics)
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"\nLoading checkpoint from: {checkpoint_path}")
        
        # 1. Load policy model
        model_path = checkpoint_path / "policy_model"
        if (model_path / "config.json").exists():
            # HuggingFace format
            model.from_pretrained(model_path)
            print(f"  ✓ Policy model loaded (HF format)")
        elif (model_path / "pytorch_model.bin").exists():
            # State dict format
            state_dict = torch.load(model_path / "pytorch_model.bin")
            model.load_state_dict(state_dict, strict=strict)
            print(f"  ✓ Policy model loaded (state dict)")
        else:
            raise ValueError(f"Model not found in {model_path}")
        
        # 2. Load optimizer state
        optimizer_path = checkpoint_path / "optimizer.pt"
        if optimizer is not None and optimizer_path.exists():
            optimizer.load_state_dict(torch.load(optimizer_path))
            print(f"  ✓ Optimizer state loaded")
        
        # 3. Load scheduler states
        scheduler_path = checkpoint_path / "scheduler.pt"
        if scheduler is not None and scheduler_path.exists():
            scheduler.load_state_dict(torch.load(scheduler_path))
            print(f"  ✓ LR scheduler loaded")
        
        kl_scheduler_path = checkpoint_path / "kl_scheduler.pt"
        if kl_scheduler is not None and kl_scheduler_path.exists():
            kl_state = torch.load(kl_scheduler_path)
            kl_scheduler.kl_coef = kl_state["kl_coef"]
            kl_scheduler.kl_history = kl_state["kl_history"]
            kl_scheduler.kl_coef_history = kl_state["kl_coef_history"]
            kl_scheduler.total_steps = kl_state["total_steps"]
            print(f"  ✓ KL scheduler loaded")
        
        # 4. Load reference manager state
        ref_manager_path = checkpoint_path / "ref_manager.pt"
        if ref_manager is not None and ref_manager_path.exists():
            ref_state = torch.load(ref_manager_path)
            ref_manager.strategy = ref_state["strategy"]
            ref_manager.total_updates = ref_state["total_updates"]
            print(f"  ✓ Reference manager state loaded")
        
        # 5. Load RNG states
        rng_path = checkpoint_path / "rng_state.pt"
        if rng_path.exists():
            rng_state = torch.load(rng_path)
            torch.set_rng_state(rng_state["python_rng_state"])
            if torch.cuda.is_available() and rng_state["cuda_rng_state"] is not None:
                torch.cuda.set_rng_state_all(rng_state["cuda_rng_state"])
            print(f"  ✓ RNG states loaded")
        
        # 6. Load training state
        state_path = checkpoint_path / "training_state.json"
        if state_path.exists():
            with open(state_path, "r") as f:
                training_state = json.load(f)
            print(f"  ✓ Training state loaded")
            print(f"    Step: {training_state['step']}")
            print(f"    Epoch: {training_state['epoch']}")
        else:
            training_state = {}
        
        print(f"Checkpoint loaded successfully!")
        return training_state
    
    def load_reference_model(
        self,
        checkpoint_path: Path,
        ref_model,
        strict: bool = True
    ):
        """
        Load reference model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            ref_model: Reference model to load into
            strict: Whether to strictly enforce state dict matching
        """
        checkpoint_path = Path(checkpoint_path)
        ref_path = checkpoint_path / "reference_model"
        
        if not ref_path.exists():
            print("Warning: Reference model not found in checkpoint")
            return
        
        if (ref_path / "config.json").exists():
            ref_model.from_pretrained(ref_path)
            print(f"  ✓ Reference model loaded (HF format)")
        elif (ref_path / "pytorch_model.bin").exists():
            state_dict = torch.load(ref_path / "pytorch_model.bin")
            ref_model.load_state_dict(state_dict, strict=strict)
            print(f"  ✓ Reference model loaded (state dict)")
    
    def _rotate_checkpoints(self):
        """Remove old checkpoints to maintain save_total_limit."""
        if self.save_total_limit <= 0:
            return
        
        # Sort by modification time
        self.checkpoints.sort(key=lambda p: p.stat().st_mtime)
        
        # Remove oldest checkpoints
        while len(self.checkpoints) > self.save_total_limit:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                shutil.rmtree(old_checkpoint)
                print(f"  Removed old checkpoint: {old_checkpoint.name}")
    
    def find_latest_checkpoint(self) -> Optional[Path]:
        """
        Find the most recent checkpoint in checkpoint_dir.
        
        Returns:
            Path to latest checkpoint, or None if no checkpoints found
        """
        # Look for checkpoint directories
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*"))
        
        if not checkpoints:
            return None
        
        # Sort by modification time
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest = checkpoints[0]
        
        print(f"Found latest checkpoint: {latest}")
        return latest
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """
        Get path to best checkpoint.
        
        Returns:
            Path to best checkpoint, or None if not saved
        """
        best_path = self.checkpoint_dir / "checkpoint_best"
        if best_path.exists():
            return best_path
        return None


def should_save_checkpoint(
    step: int,
    save_interval: int,
    last_save_time: Optional[float] = None,
    wall_time_interval: Optional[float] = None
) -> bool:
    """
    Determine if checkpoint should be saved.
    
    Args:
        step: Current training step
        save_interval: Save every N steps
        last_save_time: Timestamp of last save (optional)
        wall_time_interval: Save every N seconds (optional)
        
    Returns:
        True if should save checkpoint
    """
    import time
    
    # Check step interval
    if step % save_interval == 0:
        return True
    
    # Check wall time interval
    if wall_time_interval is not None and last_save_time is not None:
        elapsed = time.time() - last_save_time
        if elapsed >= wall_time_interval:
            return True
    
    return False
