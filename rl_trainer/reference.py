#!/usr/bin/env python3
"""
Reference model management and KL coefficient scheduling.

This module implements different strategies for updating the reference model:
1. Frozen snapshot: Never update (simple but may diverge)
2. Periodic snapshot: Copy policy every K steps
3. EMA (recommended): Exponential moving average update

Also implements adaptive KL coefficient scheduling to maintain target KL range.
"""

import torch
import torch.nn as nn
from typing import Optional, Literal
from copy import deepcopy


class ReferenceModelManager:
    """
    Manages reference model updates for KL penalty computation.
    
    Three strategies:
    1. frozen: Never update reference (simple)
    2. periodic: Copy policy weights every K steps
    3. ema: Exponential moving average (recommended)
    """
    
    def __init__(
        self,
        policy_model: nn.Module,
        strategy: Literal["frozen", "periodic", "ema"] = "ema",
        update_interval: int = 100,  # For periodic strategy
        ema_tau: float = 0.999,      # For EMA strategy (τ near 0.999)
        device: str = "cuda"
    ):
        """
        Initialize reference model manager.
        
        Args:
            policy_model: Current policy model
            strategy: Update strategy ("frozen", "periodic", or "ema")
            update_interval: Steps between updates (for periodic)
            ema_tau: EMA coefficient (for ema), τ in [0, 1]
                    ref = τ * ref + (1-τ) * policy
                    Higher τ = slower updates (0.999 recommended)
            device: Device for reference model
        """
        self.strategy = strategy
        self.update_interval = update_interval
        self.ema_tau = ema_tau
        self.device = device
        
        # Create reference model as a copy of policy
        print(f"Creating reference model (strategy: {strategy})")
        self.ref_model = self._create_reference_copy(policy_model)
        
        # Freeze reference model
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Track update count
        self.total_updates = 0
        
        print(f"Reference model initialized:")
        print(f"  Strategy: {strategy}")
        if strategy == "periodic":
            print(f"  Update interval: {update_interval} steps")
        elif strategy == "ema":
            print(f"  EMA tau: {ema_tau}")
    
    def _create_reference_copy(self, policy_model: nn.Module) -> nn.Module:
        """Create a deep copy of the policy model."""
        # For HuggingFace models, use from_pretrained or deepcopy
        # This is a simplified version - may need model-specific logic
        
        try:
            # Try to use model's own cloning if available
            ref_model = deepcopy(policy_model)
        except Exception as e:
            print(f"Warning: Could not deepcopy model: {e}")
            print("Using state_dict copy instead")
            # Fallback: create new instance and copy state_dict
            ref_model = type(policy_model)(policy_model.config)
            ref_model.load_state_dict(policy_model.state_dict())
        
        return ref_model.to(self.device)
    
    def maybe_update(self, policy_model: nn.Module, step: int) -> bool:
        """
        Maybe update reference model based on strategy.
        
        Args:
            policy_model: Current policy model
            step: Current training step (global step count)
            
        Returns:
            True if reference was updated, False otherwise
        """
        if self.strategy == "frozen":
            # Never update
            return False
        
        elif self.strategy == "periodic":
            # Update every K steps based on step count
            if step > 0 and step % self.update_interval == 0:
                self._periodic_update(policy_model)
                self.total_updates += 1
                return True
            
            return False
        
        elif self.strategy == "ema":
            # Update every step with EMA
            self._ema_update(policy_model)
            self.total_updates += 1
            return True
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _periodic_update(self, policy_model: nn.Module):
        """Copy policy weights to reference (periodic snapshot)."""
        print(f"Updating reference model (periodic snapshot, update #{self.total_updates + 1})")
        
        # Copy all parameters
        with torch.no_grad():
            for ref_param, policy_param in zip(
                self.ref_model.parameters(),
                policy_model.parameters()
            ):
                ref_param.copy_(policy_param.data)
    
    def _ema_update(self, policy_model: nn.Module):
        """Update reference with EMA: ref = τ * ref + (1-τ) * policy."""
        with torch.no_grad():
            for ref_param, policy_param in zip(
                self.ref_model.parameters(),
                policy_model.parameters()
            ):
                # ref = tau * ref + (1 - tau) * policy
                ref_param.mul_(self.ema_tau).add_(
                    policy_param.data,
                    alpha=1.0 - self.ema_tau
                )
    
    def get_reference_model(self) -> nn.Module:
        """Get the reference model."""
        return self.ref_model


class KLScheduler:
    """
    Adaptive KL coefficient (β) scheduler.
    
    Adjusts β to keep KL divergence in a target range:
    - If KL too high: increase β (penalize more)
    - If KL too low: decrease β (allow more exploration)
    
    This prevents policy from diverging too far from reference
    while still allowing learning.
    """
    
    def __init__(
        self,
        initial_kl_coef: float = 0.01,
        target_kl: float = 0.01,          # Target KL divergence
        kl_tolerance: float = 0.005,       # Tolerance around target
        adaptation_rate: float = 1.5,      # Multiplier/divisor for β
        min_kl_coef: float = 0.001,        # Minimum β
        max_kl_coef: float = 0.1,          # Maximum β
        warmup_steps: int = 100,           # Don't adapt during warmup
        update_interval: int = 10          # Check KL every N steps
    ):
        """
        Initialize KL scheduler.
        
        Args:
            initial_kl_coef: Starting β value
            target_kl: Target KL divergence (e.g., 0.01)
            kl_tolerance: ±tolerance around target before adapting
            adaptation_rate: How aggressively to adapt β
            min_kl_coef: Minimum allowed β
            max_kl_coef: Maximum allowed β
            warmup_steps: Don't adapt during warmup
            update_interval: Steps between adaptations
        """
        self.kl_coef = initial_kl_coef
        self.target_kl = target_kl
        self.kl_tolerance = kl_tolerance
        self.adaptation_rate = adaptation_rate
        self.min_kl_coef = min_kl_coef
        self.max_kl_coef = max_kl_coef
        self.warmup_steps = warmup_steps
        self.update_interval = update_interval
        
        # Track history
        self.kl_history = []
        self.kl_coef_history = [initial_kl_coef]
        self.steps_since_update = 0
        self.total_steps = 0
        
        print(f"KL scheduler initialized:")
        print(f"  Initial β: {initial_kl_coef}")
        print(f"  Target KL: {target_kl} ± {kl_tolerance}")
        print(f"  β range: [{min_kl_coef}, {max_kl_coef}]")
    
    def step(self, current_kl: float, step: int) -> float:
        """
        Update KL coefficient based on observed KL.
        
        Args:
            current_kl: Current KL divergence (mean over batch)
            step: Current training step
            
        Returns:
            Updated kl_coef (β)
        """
        self.total_steps = step
        self.kl_history.append(current_kl)
        self.steps_since_update += 1
        
        # Don't adapt during warmup
        if step < self.warmup_steps:
            return self.kl_coef
        
        # Only adapt every N steps
        if self.steps_since_update < self.update_interval:
            return self.kl_coef
        
        # Compute mean KL over recent steps
        recent_kl = sum(self.kl_history[-self.update_interval:]) / min(
            self.update_interval,
            len(self.kl_history)
        )
        
        # Adapt β based on KL
        old_kl_coef = self.kl_coef
        
        if recent_kl > self.target_kl + self.kl_tolerance:
            # KL too high: increase β (penalize more)
            self.kl_coef = min(
                self.kl_coef * self.adaptation_rate,
                self.max_kl_coef
            )
            if self.kl_coef != old_kl_coef:
                print(f"KL too high ({recent_kl:.4f} > {self.target_kl + self.kl_tolerance:.4f}): "
                      f"β {old_kl_coef:.4f} → {self.kl_coef:.4f}")
        
        elif recent_kl < self.target_kl - self.kl_tolerance:
            # KL too low: decrease β (allow more exploration)
            self.kl_coef = max(
                self.kl_coef / self.adaptation_rate,
                self.min_kl_coef
            )
            if self.kl_coef != old_kl_coef:
                print(f"KL too low ({recent_kl:.4f} < {self.target_kl - self.kl_tolerance:.4f}): "
                      f"β {old_kl_coef:.4f} → {self.kl_coef:.4f}")
        
        # Record update
        self.kl_coef_history.append(self.kl_coef)
        self.steps_since_update = 0
        
        return self.kl_coef
    
    def get_kl_coef(self) -> float:
        """Get current KL coefficient."""
        return self.kl_coef
    
    def get_stats(self) -> dict:
        """Get statistics about KL and β."""
        if not self.kl_history:
            return {}
        
        recent_kl = self.kl_history[-min(10, len(self.kl_history)):]
        
        return {
            "kl_coef": self.kl_coef,
            "current_kl": self.kl_history[-1] if self.kl_history else 0.0,
            "mean_kl_recent": sum(recent_kl) / len(recent_kl),
            "target_kl": self.target_kl,
            "num_adaptations": len(self.kl_coef_history) - 1
        }


if __name__ == "__main__":
    print("Reference model management module loaded successfully!")
    print("\nKey components:")
    print("- ReferenceModelManager: Manages reference model updates")
    print("  - frozen: Never update (simple)")
    print("  - periodic: Copy every K steps")
    print("  - ema: Exponential moving average (recommended, τ=0.999)")
    print("\n- KLScheduler: Adaptive β scheduling")
    print("  - Increases β when KL too high")
    print("  - Decreases β when KL too low")
    print("  - Keeps KL in target range")
