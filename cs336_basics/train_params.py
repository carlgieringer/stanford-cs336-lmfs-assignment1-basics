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
    batches_per_epoch: int
    epochs: int
    checkpoint_interval: int
    checkpoint_path: str


@dataclass
class RandomSeeds:
    python: int
    numpy: int
    pytorch: int
