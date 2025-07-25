from typing import Optional
import torch
from dataclasses import dataclass


@dataclass
class ModelParams:
    d_model: int
    num_heads: int
    d_ff: int
    rope_theta: float
    context_length: int
    num_layers: int
    device: str | torch.device
    dtype: torch.dtype


@dataclass
class OptimizerParams:
    learning_rate: float
    betas: tuple[float, float]
    weight_decay: float
    learning_rate_schedule_max_iterations: int
    gradient_clip_norm: float


@dataclass
class TrainingParams:
    run_name: str
    data_path: str
    batch_size: int
    total_steps: int
    checkpoint_interval: int
    checkpoint_dir: str
    compile_backend: Optional[str]


@dataclass
class RandomSeeds:
    python: int
    numpy: int
    pytorch: int


@dataclass
class WandbParams:
    """Parameters for wandb integration."""

    project: str
    entity: Optional[str] = None
    tags: Optional[list[str]] = None
    notes: Optional[str] = None
    gradient_log_frequency: int = 10
    log_artifacts: bool = False


@dataclass
class ValidationParams:
    """Parameters for validation and early stopping."""

    validation_data_path: Optional[str] = None
    validation_interval: int = 50
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001


@dataclass
class TrainingRunParams:
    model_params: ModelParams
    optimimizer_params: OptimizerParams
    training_params: TrainingParams
    random_seeds: RandomSeeds
    wandb_params: WandbParams
    validation_params: ValidationParams
