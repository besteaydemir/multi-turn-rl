"""
RL Multi-Turn View Selection v2

A minimal, stable implementation for multi-turn RL with VLMs.
Designed to be extended toward the full VAGEN framework.

Components:
- data_structures: CameraPose, Action, Turn, Trajectory
- output_parser: OutputParser for bracketed marker format
- rollout: vLLM-based trajectory collection
- trainer: REINFORCE/PPO policy gradient training
- logging_utils: W&B and file logging at three levels
"""

from .data_structures import (
    CameraPose,
    Action,
    FinalAnswer,
    Turn,
    Trajectory,
)

from .output_parser import (
    OutputParser,
    ParseResult,
    ActionTokenMasker,
    create_system_prompt,
    create_turn_prompt,
)

from .rollout import (
    RolloutConfig,
    VLLMRolloutEngine,
    MockRolloutEngine,
)

from .trainer import (
    TrainerConfig,
    RLTrainer,
    OnlineRLTrainer,
    compute_advantages,
    normalize_advantages,
)

from .logging_utils import (
    TurnLog,
    TrajectoryLog,
    UpdateLog,
    WandBLogger,
    FileLogger,
    Logger,
    format_update_summary,
    format_trajectory_summary,
)

__version__ = "2.0.0"

__all__ = [
    # Data structures
    "CameraPose",
    "Action",
    "FinalAnswer",
    "Turn",
    "Trajectory",
    # Parser
    "OutputParser",
    "ParseResult",
    "ActionTokenMasker",
    "create_system_prompt",
    "create_turn_prompt",
    # Rollout
    "RolloutConfig",
    "VLLMRolloutEngine",
    "MockRolloutEngine",
    # Trainer
    "TrainerConfig",
    "RLTrainer",
    "OnlineRLTrainer",
    "compute_advantages",
    "normalize_advantages",
    # Logging
    "TurnLog",
    "TrajectoryLog",
    "UpdateLog",
    "WandBLogger",
    "FileLogger",
    "Logger",
    "format_update_summary",
    "format_trajectory_summary",
]
