"""
WandB logging utilities for RL training.

Provides comprehensive logging of:
- Training metrics (loss, KL, entropy, rewards, advantages)
- Episode statistics (success rate, action distributions)
- Model checkpoints
- Example episodes for visualization
"""

import wandb
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
from datetime import datetime


class WandBLogger:
    """
    WandB logger for RL training metrics and visualizations.
    """
    
    def __init__(
        self,
        project: str = "vlm-navigation-rl",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
        resume: Optional[str] = None,
        id: Optional[str] = None,
        enabled: bool = True
    ):
        """
        Initialize WandB logger.
        
        Args:
            project: WandB project name
            name: Run name (auto-generated if None)
            config: Hyperparameters to log
            entity: WandB entity/username
            tags: Tags for this run
            resume: Resume mode ("allow", "must", "never")
            id: Run ID for resuming
            enabled: Whether to actually log to WandB (False for debugging)
        """
        self.enabled = enabled
        
        if not enabled:
            print("WandB logging disabled")
            return
        
        # Generate run name if not provided
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"rl_training_{timestamp}"
        
        # Initialize WandB
        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            entity=entity,
            tags=tags,
            resume=resume,
            id=id
        )
        
        print(f"WandB run initialized: {self.run.name}")
        print(f"  Project: {project}")
        print(f"  URL: {self.run.url}")
    
    def log_training_step(
        self,
        step: int,
        metrics: Dict[str, float],
        commit: bool = True
    ):
        """
        Log training metrics for a single update step.
        
        Expected metrics:
            - loss/total: Total loss
            - loss/policy: Policy gradient loss (-A * logpi)
            - loss/kl: KL penalty term
            - loss/entropy: Entropy regularization
            - reward/mean: Average reward in batch
            - reward/max: Maximum reward
            - reward/min: Minimum reward
            - advantage/mean: Average advantage
            - advantage/std: Advantage standard deviation
            - logprob/policy_mean: Average log π(a|s)
            - logprob/reference_mean: Average log π_ref(a|s)
            - kl/mean: Average KL divergence
            - entropy/mean: Average entropy
            - training/grad_norm: Gradient norm
            - training/learning_rate: Current learning rate
            - kl_scheduler/beta: Current KL coefficient (if using scheduler)
        
        Args:
            step: Global training step
            metrics: Dictionary of metric name -> value
            commit: Whether to commit this log (set False to accumulate)
        """
        if not self.enabled:
            return
        
        wandb.log(metrics, step=step, commit=commit)
    
    def log_episode_statistics(
        self,
        step: int,
        episodes,
        commit: bool = True
    ):
        """
        Log episode-level statistics.
        
        Args:
            step: Global training step
            episodes: List of Episode objects
            commit: Whether to commit this log
        """
        if not self.enabled:
            return
        
        # Basic statistics
        total_episodes = len(episodes)
        valid_episodes = [ep for ep in episodes if ep.is_valid]
        num_valid = len(valid_episodes)
        
        # Success rate
        success_rate = sum(ep.is_correct for ep in valid_episodes) / max(num_valid, 1)
        
        # Dropout statistics
        dropout_rate = (total_episodes - num_valid) / max(total_episodes, 1)
        dropout_reasons = {}
        for ep in episodes:
            if not ep.is_valid and ep.dropout_reason:
                dropout_reasons[ep.dropout_reason] = dropout_reasons.get(ep.dropout_reason, 0) + 1
        
        # Action distribution statistics
        action_stats = self._compute_action_statistics(valid_episodes)
        
        # Log metrics
        metrics = {
            "episode/total": total_episodes,
            "episode/valid": num_valid,
            "episode/success_rate": success_rate,
            "episode/dropout_rate": dropout_rate,
            **action_stats
        }
        
        # Add dropout reasons
        for reason, count in dropout_reasons.items():
            metrics[f"dropout/{reason}"] = count
        
        wandb.log(metrics, step=step, commit=commit)
    
    def _compute_action_statistics(self, episodes) -> Dict[str, float]:
        """Compute statistics about action distributions."""
        all_actions = []
        
        for episode in episodes:
            for turn in episode.turns:
                if turn.action and turn.action_valid:
                    all_actions.append(turn.action)
        
        if not all_actions:
            return {}
        
        # Extract movement parameters
        forward_values = [a.forward_meters for a in all_actions]
        left_values = [a.left_meters for a in all_actions]
        rotation_values = [a.rotation_angle_degrees for a in all_actions]
        z_values = [a.z_delta_meters for a in all_actions]
        
        # Compute statistics
        stats = {
            "action/forward_mean": np.mean(forward_values),
            "action/forward_std": np.std(forward_values),
            "action/left_mean": np.mean(left_values),
            "action/left_std": np.std(left_values),
            "action/rotation_mean": np.mean(rotation_values),
            "action/rotation_std": np.std(rotation_values),
            "action/z_mean": np.mean(z_values),
            "action/z_std": np.std(z_values),
            "action/done_rate": sum(a.done for a in all_actions) / len(all_actions)
        }
        
        return stats
    
    def log_action_histograms(
        self,
        step: int,
        episodes,
        commit: bool = True
    ):
        """
        Log histograms of action distributions.
        
        Args:
            step: Global training step
            episodes: List of Episode objects
            commit: Whether to commit this log
        """
        if not self.enabled:
            return
        
        valid_episodes = [ep for ep in episodes if ep.is_valid]
        all_actions = []
        
        for episode in valid_episodes:
            for turn in episode.turns:
                if turn.action and turn.action_valid:
                    all_actions.append(turn.action)
        
        if not all_actions:
            return
        
        # Create histograms
        forward_values = [a.forward_meters for a in all_actions]
        rotation_values = [a.rotation_angle_degrees for a in all_actions]
        
        wandb.log({
            "action_dist/forward": wandb.Histogram(forward_values),
            "action_dist/rotation": wandb.Histogram(rotation_values)
        }, step=step, commit=commit)
    
    def log_example_episodes(
        self,
        step: int,
        episodes,
        num_examples: int = 5,
        commit: bool = True
    ):
        """
        Log example episodes as WandB tables for visualization.
        
        Args:
            step: Global training step
            episodes: List of Episode objects
            num_examples: Number of example episodes to log
            commit: Whether to commit this log
        """
        if not self.enabled:
            return
        
        # Select diverse examples (some success, some failure)
        valid_episodes = [ep for ep in episodes if ep.is_valid]
        if not valid_episodes:
            return
        
        success_episodes = [ep for ep in valid_episodes if ep.is_correct]
        failure_episodes = [ep for ep in valid_episodes if not ep.is_correct]
        
        examples = []
        examples.extend(success_episodes[:num_examples//2])
        examples.extend(failure_episodes[:num_examples//2])
        examples = examples[:num_examples]
        
        # Create table
        columns = ["episode_id", "question", "num_turns", "final_answer", 
                   "ground_truth", "reward", "success"]
        data = []
        
        for ep in examples:
            data.append([
                ep.episode_id,
                ep.question,
                len(ep.turns),
                ep.final_answer or "N/A",
                ep.ground_truth,
                ep.final_reward,
                ep.is_correct
            ])
        
        table = wandb.Table(columns=columns, data=data)
        wandb.log({"examples/episodes": table}, step=step, commit=commit)
    
    def log_reference_update(
        self,
        step: int,
        update_type: str,
        commit: bool = True
    ):
        """
        Log when reference model is updated.
        
        Args:
            step: Global training step
            update_type: Type of update ("periodic", "ema", "frozen")
            commit: Whether to commit this log
        """
        if not self.enabled:
            return
        
        wandb.log({
            "reference/update_type": update_type,
            "reference/update_step": step
        }, step=step, commit=commit)
    
    def log_checkpoint(
        self,
        checkpoint_path: Path,
        step: int,
        is_best: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log model checkpoint as WandB artifact.
        
        Args:
            checkpoint_path: Path to checkpoint directory or file
            step: Global training step
            is_best: Whether this is the best checkpoint so far
            metadata: Additional metadata to attach to artifact
        """
        if not self.enabled:
            return
        
        # Create artifact
        artifact_name = f"checkpoint_step_{step}"
        if is_best:
            artifact_name = "checkpoint_best"
        
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            metadata=metadata or {}
        )
        
        # Add checkpoint files
        if checkpoint_path.is_dir():
            artifact.add_dir(str(checkpoint_path))
        else:
            artifact.add_file(str(checkpoint_path))
        
        # Log artifact
        self.run.log_artifact(artifact)
        print(f"Logged checkpoint artifact: {artifact_name}")
    
    def watch_model(self, model, log_freq: int = 100):
        """
        Watch model parameters and gradients.
        
        Args:
            model: PyTorch model to watch
            log_freq: How often to log histograms (every N steps)
        """
        if not self.enabled:
            return
        
        wandb.watch(model, log="all", log_freq=log_freq)
    
    def finish(self):
        """Finish WandB run."""
        if not self.enabled:
            return
        
        wandb.finish()
        print("WandB run finished")


def format_training_metrics(
    loss: float,
    policy_loss: float,
    kl_loss: float,
    entropy_loss: float,
    logprobs_result,
    advantages: torch.Tensor,
    rewards: torch.Tensor,
    grad_norm: float,
    learning_rate: float,
    kl_coef: Optional[float] = None
) -> Dict[str, float]:
    """
    Format training metrics for logging.
    
    Args:
        loss: Total loss value
        policy_loss: Policy gradient loss component
        kl_loss: KL penalty component
        entropy_loss: Entropy regularization component
        logprobs_result: LogProbResult object
        advantages: Advantage values [batch_size]
        rewards: Reward values [batch_size]
        grad_norm: Gradient norm
        learning_rate: Current learning rate
        kl_coef: Current KL coefficient (if using scheduler)
        
    Returns:
        Dictionary of metrics ready for logging
    """
    metrics = {
        # Loss components
        "loss/total": loss,
        "loss/policy": policy_loss,
        "loss/kl": kl_loss,
        "loss/entropy": entropy_loss,
        
        # Rewards
        "reward/mean": rewards.mean().item(),
        "reward/max": rewards.max().item(),
        "reward/min": rewards.min().item(),
        "reward/std": rewards.std().item(),
        
        # Advantages
        "advantage/mean": advantages.mean().item(),
        "advantage/std": advantages.std().item(),
        "advantage/max": advantages.max().item(),
        "advantage/min": advantages.min().item(),
        
        # Log probabilities
        "logprob/policy_mean": logprobs_result.mean_logpi.mean().item(),
        "logprob/policy_std": logprobs_result.mean_logpi.std().item(),
        
        # Training stats
        "training/grad_norm": grad_norm,
        "training/learning_rate": learning_rate,
    }
    
    # Add reference model metrics if available
    if logprobs_result.mean_logpref is not None:
        metrics["logprob/reference_mean"] = logprobs_result.mean_logpref.mean().item()
    
    # Add KL metrics if available
    if logprobs_result.kl_div is not None:
        metrics["kl/mean"] = logprobs_result.kl_div.mean().item()
        metrics["kl/std"] = logprobs_result.kl_div.std().item()
        metrics["kl/max"] = logprobs_result.kl_div.max().item()
    
    # Add entropy metrics if available
    if logprobs_result.mean_entropy is not None:
        metrics["entropy/mean"] = logprobs_result.mean_entropy.mean().item()
        metrics["entropy/std"] = logprobs_result.mean_entropy.std().item()
    
    # Add KL coefficient if using scheduler
    if kl_coef is not None:
        metrics["kl_scheduler/beta"] = kl_coef
    
    return metrics
