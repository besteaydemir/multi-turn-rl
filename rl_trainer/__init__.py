"""
RL Trainer package for policy gradient training with VLMs.
"""

from .batch import (
    EpisodeBatch,
    EpisodeDataLoader,
    prepare_batch,
    prepare_loo_batches,
    compute_loo_baseline
)

from .logprobs import (
    LogProbResult,
    compute_token_logprobs,
    compute_sequence_logprobs,
    compute_advantages,
    policy_gradient_loss
)

from .reference import (
    ReferenceModelManager,
    KLScheduler
)

from .multimodal import (
    MultimodalInputBuilder,
    load_images_from_paths,
    batch_images_for_episodes
)

from .logging_utils import (
    WandBLogger,
    format_training_metrics
)

from .checkpoint import (
    CheckpointManager,
    should_save_checkpoint
)

from .trainer import (
    TrainerConfig,
    RLTrainer
)

__all__ = [
    # Batching
    'EpisodeBatch',
    'EpisodeDataLoader',
    'prepare_batch',
    'prepare_loo_batches',
    'compute_loo_baseline',
    # Log-probs
    'LogProbResult',
    'compute_token_logprobs',
    'compute_sequence_logprobs',
    'compute_advantages',
    'policy_gradient_loss',
    # Reference model
    'ReferenceModelManager',
    'KLScheduler',
    # Multimodal
    'MultimodalInputBuilder',
    'load_images_from_paths',
    'batch_images_for_episodes',
    # Logging
    'WandBLogger',
    'format_training_metrics',
    # Checkpointing
    'CheckpointManager',
    'should_save_checkpoint',
    # Trainer
    'TrainerConfig',
    'RLTrainer'
]
