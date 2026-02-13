from .rewards import compute_rewards, compute_turn_reward, compute_terminal_reward, RewardConfig
from .advantage import (
    compute_gae, 
    compute_monte_carlo_returns, 
    compute_bilevel_gae,
    compute_advantage,
)
from .ppo import ppo_step, PPOConfig
from .sync import sync_weights
from .vagen_gae import (
    VAGENConfig,
    TokenData,
    TurnData,
    compute_kl_rewards,
    compute_token_level_gae,
    compute_turn_level_gae,
    compute_bilevel_gae_full,
    compute_critic_loss,
    ppo_critic_step,
    PPOCriticConfig,
)

__all__ = [
    # Rewards (Step 3)
    "compute_rewards",
    "compute_turn_reward",
    "compute_terminal_reward",
    "RewardConfig",
    # Advantage (Step 7B, 9)
    "compute_gae",
    "compute_monte_carlo_returns",
    "compute_bilevel_gae",
    "compute_advantage",
    # VAGEN GAE (full implementation)
    "VAGENConfig",
    "TokenData",
    "TurnData",
    "compute_kl_rewards",
    "compute_token_level_gae",
    "compute_turn_level_gae",
    "compute_bilevel_gae_full",
    "compute_critic_loss",
    "ppo_critic_step",
    "PPOCriticConfig",
    # PPO (Step 7C)
    "ppo_step",
    "PPOConfig",
    # Sync (Step 8)
    "sync_weights",
]
