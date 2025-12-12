"""
Comprehensive configuration system for VLM navigation RL training.

This module provides a structured configuration system with:
- Hyperparameter management
- Config validation
- Serialization/deserialization
- Config templates for different scenarios
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Literal
from pathlib import Path
import json
import yaml


@dataclass
class ModelConfig:
    """Model and architecture settings."""
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    cache_dir: Optional[str] = None
    torch_dtype: str = "bfloat16"  # "float32", "float16", "bfloat16"
    device_map: str = "auto"  # "auto", "cuda", "cpu", or custom mapping
    trust_remote_code: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False


@dataclass
class DataConfig:
    """Dataset and data loading settings."""
    dataset_path: str = "/path/to/dataset"
    dataset_name: str = "arkitscenes"  # "arkitscenes", "scannet", etc.
    split: str = "train"  # "train", "val", "test"
    
    # Episode collection
    num_episodes: int = 100
    episodes_per_scene: int = 5
    max_episode_len: int = 10  # Maximum turns per episode
    
    # Filtering
    filter_invalid_episodes: bool = True
    min_turns_per_episode: int = 1
    
    # Data loading
    num_workers: int = 4
    prefetch_factor: int = 2


@dataclass
class GenerationConfig:
    """Decoding/generation settings for episode collection."""
    # Sampling strategy
    do_sample: bool = True
    temperature: float = 1.0  # 0.7-1.2 typical range
    top_p: float = 0.9  # Nucleus sampling
    top_k: int = 50  # Top-k sampling
    
    # Generation limits
    max_new_tokens: int = 512
    min_new_tokens: int = 10
    
    # Special tokens
    pad_token_id: Optional[int] = None  # Set from tokenizer if None
    eos_token_id: Optional[int] = None  # Set from tokenizer if None
    
    # Beam search (if not sampling)
    num_beams: int = 1
    
    # Repetition control
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    
    # Other
    use_cache: bool = True


@dataclass
class OptimizationConfig:
    """Optimizer and learning rate settings."""
    # Optimizer
    optimizer: str = "adamw"  # "adamw", "adam", "sgd"
    learning_rate: float = 1e-5  # 5e-6 to 2e-5 for large models
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Gradient
    max_grad_norm: float = 1.0  # Gradient clipping
    gradient_accumulation_steps: int = 1
    
    # Learning rate schedule
    lr_scheduler_type: str = "cosine"  # "cosine", "linear", "constant", "constant_with_warmup"
    warmup_steps: int = 100
    warmup_ratio: float = 0.0  # Alternative to warmup_steps
    
    # Mixed precision
    use_amp: bool = False  # Automatic Mixed Precision
    fp16: bool = False
    bf16: bool = False


@dataclass
class RLConfig:
    """Reinforcement learning specific settings."""
    # Batch size
    batch_size: int = 8  # Episodes per update (8-32 recommended)
    
    # Loss coefficients
    kl_coef: float = 0.01  # β - KL penalty coefficient (0.005, 0.01, 0.02)
    entropy_coef: float = 0.01  # α - Entropy bonus (0.0-0.01)
    
    # Advantage computation
    advantage_normalize: bool = True  # Normalize advantages
    advantage_eps: float = 1e-8  # Small constant for stability
    
    # Reference model strategy
    ref_model_strategy: Literal["frozen", "periodic", "ema"] = "ema"
    ref_update_interval: int = 100  # For periodic strategy (steps)
    ref_ema_tau: float = 0.999  # For EMA strategy (0.99-0.999)
    
    # KL divergence scheduling
    use_kl_scheduler: bool = True  # Adaptive β scheduling
    target_kl: float = 0.01  # Target KL divergence
    kl_tolerance: float = 0.005  # Tolerance around target
    kl_adaptation_rate: float = 1.5  # Multiplier for β adaptation
    kl_coef_min: float = 0.001  # Minimum β
    kl_coef_max: float = 0.1  # Maximum β
    
    # Reward settings
    reward_success: float = 1.0  # Reward for correct answer
    reward_failure: float = 0.0  # Reward for incorrect answer
    reward_invalid: float = 0.0  # Reward for invalid episodes
    
    # Episode dropout (quality control)
    dropout_invalid_json: bool = True
    dropout_low_confidence_masking: bool = False
    min_action_tokens: int = 10
    max_action_tokens: int = 100


@dataclass
class TrainingConfig:
    """Training loop settings."""
    # Epochs and steps
    num_epochs: int = 3
    max_steps: Optional[int] = None  # Override num_epochs if set
    
    # Evaluation
    eval_strategy: str = "epoch"  # "no", "steps", "epoch"
    eval_steps: int = 100
    eval_episodes: int = 50
    
    # Logging
    logging_strategy: str = "steps"  # "steps", "epoch"
    logging_steps: int = 10
    logging_first_step: bool = True
    
    # Checkpointing
    save_strategy: str = "steps"  # "no", "steps", "epoch"
    save_steps: int = 100
    save_total_limit: int = 3  # Keep only N checkpoints
    save_optimizer: bool = True
    save_reference_model: bool = True
    
    # Resume
    resume_from_checkpoint: Optional[str] = None
    
    # Output
    output_dir: str = "./checkpoints"
    overwrite_output_dir: bool = False
    
    # Device
    device: str = "cuda"
    local_rank: int = -1  # For distributed training
    
    # Reproducibility
    seed: int = 42
    
    # Performance
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True


@dataclass
class WandbConfig:
    """Weights & Biases logging configuration."""
    use_wandb: bool = True
    project: str = "vlm-navigation-rl"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    group: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    
    # What to log
    log_model: bool = True  # Log model checkpoints as artifacts
    log_examples: bool = True  # Log example episodes
    num_log_examples: int = 5
    log_histograms: bool = True  # Log action distributions
    log_gradients: bool = False  # Log gradient histograms (can be expensive)
    
    # Logging frequency
    log_interval: int = 10  # Log every N steps
    log_examples_interval: int = 100
    log_histograms_interval: int = 50


@dataclass
class RLTrainingConfig:
    """
    Master configuration for RL training.
    
    Combines all sub-configurations into a single object.
    """
    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    
    # Experiment metadata
    experiment_name: Optional[str] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self):
        """Validate configuration values."""
        # Learning rate validation
        if self.optimization.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.optimization.learning_rate}")
        
        # Batch size validation
        if self.rl.batch_size < 2:
            raise ValueError(f"batch_size must be >= 2 for LOO baseline, got {self.rl.batch_size}")
        
        # KL coefficient validation
        if self.rl.kl_coef < 0:
            raise ValueError(f"kl_coef must be >= 0, got {self.rl.kl_coef}")
        
        # Temperature validation
        if self.generation.do_sample and self.generation.temperature <= 0:
            raise ValueError(f"temperature must be > 0 when sampling, got {self.generation.temperature}")
        
        # Reference model strategy validation
        if self.rl.ref_model_strategy not in ["frozen", "periodic", "ema"]:
            raise ValueError(f"Invalid ref_model_strategy: {self.rl.ref_model_strategy}")
        
        # EMA tau validation
        if self.rl.ref_model_strategy == "ema" and not (0 < self.rl.ref_ema_tau < 1):
            raise ValueError(f"ref_ema_tau must be in (0, 1), got {self.rl.ref_ema_tau}")
        
        # Output directory
        if not self.training.output_dir:
            raise ValueError("output_dir cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model": asdict(self.model),
            "data": asdict(self.data),
            "generation": asdict(self.generation),
            "optimization": asdict(self.optimization),
            "rl": asdict(self.rl),
            "training": asdict(self.training),
            "wandb": asdict(self.wandb),
            "experiment_name": self.experiment_name,
            "description": self.description
        }
    
    def to_json(self, path: str):
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Config saved to {path}")
    
    def to_yaml(self, path: str):
        """Save config to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
        print(f"Config saved to {path}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RLTrainingConfig":
        """Load config from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            data=DataConfig(**config_dict.get("data", {})),
            generation=GenerationConfig(**config_dict.get("generation", {})),
            optimization=OptimizationConfig(**config_dict.get("optimization", {})),
            rl=RLConfig(**config_dict.get("rl", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            wandb=WandbConfig(**config_dict.get("wandb", {})),
            experiment_name=config_dict.get("experiment_name"),
            description=config_dict.get("description")
        )
    
    @classmethod
    def from_json(cls, path: str) -> "RLTrainingConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_yaml(cls, path: str) -> "RLTrainingConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def print_summary(self):
        """Print a human-readable summary of the configuration."""
        print("=" * 80)
        print("RL TRAINING CONFIGURATION")
        print("=" * 80)
        
        if self.experiment_name:
            print(f"\nExperiment: {self.experiment_name}")
        if self.description:
            print(f"Description: {self.description}")
        
        print(f"\n--- MODEL ---")
        print(f"  Model ID: {self.model.model_id}")
        print(f"  Device: {self.training.device}")
        print(f"  Dtype: {self.model.torch_dtype}")
        
        print(f"\n--- DATA ---")
        print(f"  Dataset: {self.data.dataset_name}")
        print(f"  Episodes: {self.data.num_episodes}")
        print(f"  Max episode length: {self.data.max_episode_len}")
        
        print(f"\n--- GENERATION ---")
        print(f"  Sampling: {self.generation.do_sample}")
        print(f"  Temperature: {self.generation.temperature}")
        print(f"  Top-p: {self.generation.top_p}")
        print(f"  Top-k: {self.generation.top_k}")
        
        print(f"\n--- OPTIMIZATION ---")
        print(f"  Learning rate: {self.optimization.learning_rate}")
        print(f"  Batch size: {self.rl.batch_size}")
        print(f"  Gradient accumulation: {self.optimization.gradient_accumulation_steps}")
        print(f"  Max grad norm: {self.optimization.max_grad_norm}")
        print(f"  Warmup steps: {self.optimization.warmup_steps}")
        
        print(f"\n--- REINFORCEMENT LEARNING ---")
        print(f"  KL coefficient: {self.rl.kl_coef}")
        print(f"  Entropy coefficient: {self.rl.entropy_coef}")
        print(f"  Reference strategy: {self.rl.ref_model_strategy}")
        if self.rl.ref_model_strategy == "ema":
            print(f"  EMA tau: {self.rl.ref_ema_tau}")
        elif self.rl.ref_model_strategy == "periodic":
            print(f"  Update interval: {self.rl.ref_update_interval}")
        print(f"  Advantage normalize: {self.rl.advantage_normalize}")
        print(f"  KL scheduler: {self.rl.use_kl_scheduler}")
        
        print(f"\n--- TRAINING ---")
        print(f"  Epochs: {self.training.num_epochs}")
        print(f"  Output dir: {self.training.output_dir}")
        print(f"  Save strategy: {self.training.save_strategy} (every {self.training.save_steps} steps)")
        print(f"  Logging: every {self.training.logging_steps} steps")
        
        print(f"\n--- WANDB ---")
        print(f"  Enabled: {self.wandb.use_wandb}")
        if self.wandb.use_wandb:
            print(f"  Project: {self.wandb.project}")
            print(f"  Run name: {self.wandb.run_name or 'auto-generated'}")
            print(f"  Log examples: {self.wandb.log_examples}")
        
        print("=" * 80)


# ============================================================================
# Configuration Templates
# ============================================================================

def get_debug_config() -> RLTrainingConfig:
    """Small config for quick debugging."""
    return RLTrainingConfig(
        experiment_name="debug",
        data=DataConfig(
            num_episodes=10,
            max_episode_len=3
        ),
        rl=RLConfig(
            batch_size=2
        ),
        training=TrainingConfig(
            num_epochs=1,
            logging_steps=1,
            save_steps=10,
            save_total_limit=1
        ),
        wandb=WandbConfig(
            use_wandb=False
        )
    )


def get_small_scale_config() -> RLTrainingConfig:
    """Config for small-scale experiments (quick iteration)."""
    return RLTrainingConfig(
        experiment_name="small_scale",
        data=DataConfig(
            num_episodes=100,
            max_episode_len=10
        ),
        optimization=OptimizationConfig(
            learning_rate=1e-5,
            warmup_steps=50
        ),
        rl=RLConfig(
            batch_size=8,
            kl_coef=0.01,
            entropy_coef=0.01
        ),
        training=TrainingConfig(
            num_epochs=3,
            save_steps=100
        )
    )


def get_full_scale_config() -> RLTrainingConfig:
    """Config for full-scale training."""
    return RLTrainingConfig(
        experiment_name="full_scale",
        data=DataConfig(
            num_episodes=1000,
            max_episode_len=10
        ),
        optimization=OptimizationConfig(
            learning_rate=5e-6,
            warmup_steps=200,
            gradient_accumulation_steps=2
        ),
        rl=RLConfig(
            batch_size=16,
            kl_coef=0.01,
            entropy_coef=0.005,
            ref_model_strategy="ema",
            ref_ema_tau=0.999
        ),
        training=TrainingConfig(
            num_epochs=5,
            save_steps=200,
            save_total_limit=5
        ),
        wandb=WandbConfig(
            use_wandb=True,
            log_examples=True,
            log_histograms=True
        )
    )


def get_hyperparameter_sweep_configs() -> List[RLTrainingConfig]:
    """
    Generate configs for hyperparameter sweep.
    
    Grid search over:
    - batch_size: [8, 16, 32]
    - learning_rate: [5e-6, 1e-5, 2e-5]
    - kl_coef: [0.005, 0.01, 0.02]
    - entropy_coef: [0.0, 0.005, 0.01]
    """
    configs = []
    
    batch_sizes = [8, 16, 32]
    learning_rates = [5e-6, 1e-5, 2e-5]
    kl_coefs = [0.005, 0.01, 0.02]
    entropy_coefs = [0.0, 0.005, 0.01]
    
    for bs in batch_sizes:
        for lr in learning_rates:
            for kl in kl_coefs:
                for ent in entropy_coefs:
                    config = get_small_scale_config()
                    config.experiment_name = f"sweep_bs{bs}_lr{lr:.0e}_kl{kl}_ent{ent}"
                    config.rl.batch_size = bs
                    config.optimization.learning_rate = lr
                    config.rl.kl_coef = kl
                    config.rl.entropy_coef = ent
                    configs.append(config)
    
    return configs


# ============================================================================
# CLI Integration
# ============================================================================

def parse_config_from_cli(args) -> RLTrainingConfig:
    """
    Parse configuration from command-line arguments.
    
    Supports loading from file and overriding specific values.
    """
    # Load base config
    if hasattr(args, 'config') and args.config:
        if args.config.endswith('.json'):
            config = RLTrainingConfig.from_json(args.config)
        elif args.config.endswith('.yaml') or args.config.endswith('.yml'):
            config = RLTrainingConfig.from_yaml(args.config)
        else:
            raise ValueError(f"Unknown config format: {args.config}")
    else:
        config = RLTrainingConfig()
    
    # Override with CLI arguments
    if hasattr(args, 'learning_rate') and args.learning_rate is not None:
        config.optimization.learning_rate = args.learning_rate
    
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        config.rl.batch_size = args.batch_size
    
    if hasattr(args, 'kl_coef') and args.kl_coef is not None:
        config.rl.kl_coef = args.kl_coef
    
    if hasattr(args, 'entropy_coef') and args.entropy_coef is not None:
        config.rl.entropy_coef = args.entropy_coef
    
    if hasattr(args, 'output_dir') and args.output_dir is not None:
        config.training.output_dir = args.output_dir
    
    if hasattr(args, 'wandb_project') and args.wandb_project is not None:
        config.wandb.project = args.wandb_project
    
    if hasattr(args, 'wandb_name') and args.wandb_name is not None:
        config.wandb.run_name = args.wandb_name
    
    return config
