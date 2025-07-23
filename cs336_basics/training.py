import argparse
from dataclasses import dataclass
import logging
import random

import numpy as np
import torch
from tqdm import tqdm
import wandb

from torch.optim.lr_scheduler import CosineAnnealingLR

from cs336_basics.checkpointing import save_checkpoint
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.adamw import Adamw
from cs336_basics.data_loading import load_data
from cs336_basics.transformer import TransformerLm

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

arg_parser = argparse.ArgumentParser()

# Model params
_DEFAULT_D_MODEL = 768
arg_parser.add_argument("--d-model", type=int, default=768)
arg_parser.add_argument("--num-heads", type=int, default=4)
arg_parser.add_argument("--d-ff", type=int, default=_DEFAULT_D_MODEL * 4)
arg_parser.add_argument("--rope-theta", type=int, default=1e-8)
arg_parser.add_argument("--context-length", type=int, default=256)
arg_parser.add_argument("--num-layers", type=int, default=8)
arg_parser.add_argument("--device")
arg_parser.add_argument("--dtype", default="float32")

# Optimizer params
arg_parser.add_argument("--learning-rate", type=float, default=1e-3)
arg_parser.add_argument(
    "--learning-rate-schedule-max-iterations", type=int, default=100
)
arg_parser.add_argument(
    "--betas", type=lambda v: tuple(map(float, v.split(","))), default=(0.9, 0.95)
)
arg_parser.add_argument("--weight-decay", type=float, default=1e-4)

# Training params
arg_parser.add_argument("--run-name", required=True)
arg_parser.add_argument("--data-path")
arg_parser.add_argument("--batch-size", type=int, default=128)
arg_parser.add_argument("--epochs", type=int, default=10)
arg_parser.add_argument("--checkpoint-interval", type=int, default=100)
arg_parser.add_argument("--checkpoint-path", default="data/checkpoints")

arg_parser.add_argument("--python-random-seed", type=int, default=42)
arg_parser.add_argument("--numpy-random-seed", type=int, default=42)
arg_parser.add_argument("--pytorch-random-seed", type=int, default=42)


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


@dataclass
class TrainingParams:
    run_name: str
    data_path: str
    batch_size: int
    epochs: int
    checkpoint_interval: int
    checkpoint_path: str


@dataclass
class RandomSeeds:
    python: int
    numpy: int
    pytorch: int


def train_model(
    model_params: ModelParams,
    optimizer_params: OptimizerParams,
    training_params: TrainingParams,
    random_seeds: RandomSeeds,
):
    random.seed(random_seeds.python)
    np.random.seed(random_seeds.numpy)
    torch.manual_seed(random_seeds.pytorch)

    training_data = np.load(training_params.data_path, mmap_mode="r")
    # +1 for 0th indexed vocab items
    vocab_size = training_data.max() + 1

    model = TransformerLm(
        model_params.d_model,
        model_params.num_heads,
        model_params.d_ff,
        model_params.rope_theta,
        vocab_size,
        model_params.context_length,
        model_params.num_layers,
        model_params.device,
        model_params.dtype,
    )
    optimizer = Adamw(
        model.parameters(),
        optimizer_params.learning_rate,
        optimizer_params.betas,
        optimizer_params.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer, optimizer_params.learning_rate_schedule_max_iterations
    )

    for epoch in tqdm(range(training_params.epochs), desc="Epochs", unit="epoch"):

        inputs, targets = load_data(
            training_data,
            training_params.batch_size,
            model_params.context_length,
            model_params.device,
        )

        logits = model(inputs)

        loss = cross_entropy(logits, targets)
        loss.backward()

        optimizer.step()
        scheduler.step()

        optimizer.zero_grad()

        print(f"epoch {epoch}: loss: {loss}")
        if epoch % training_params.checkpoint_interval == 0:
            iteration = epoch
            save_checkpoint(
                model,
                optimizer,
                iteration,
                f"{training_params.checkpoint_path}/{training_params.run_name}-epoch-{epoch}.pt",
            )

    save_checkpoint(
        model,
        optimizer,
        iteration,
        f"{training_params.checkpoint_path}/{training_params.run_name}-final.pt",
    )


if __name__ == "__main__":
    # Hyperparams
    # https://docs.ray.io/en/latest/tune/index.html

    # run = wandb.init(
    #     entity="carl-gieringer",
    #     project="stanford-cs336-language-model",
    #     # Track hyperparameters and run metadata.
    #     config={
    #         "learning_rate": 0.02,
    #         "architecture": "CNN",
    #         "dataset": "CIFAR-100",
    #         "epochs": 10,
    #     },
    # )
    args = arg_parser.parse_args()

    logger.info(f"Running {arg_parser.prog} with args: {args}")

    if args.device:
        device = args.device
    elif torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
    else:
        device = "cpu"

    dtype = getattr(torch, args.dtype)
    if not isinstance(dtype, torch.dtype):
        raise Exception(f"Invalid dtype: {args.dtype}")

    train_model(
        ModelParams(
            args.d_model,
            args.num_heads,
            args.d_ff,
            args.rope_theta,
            args.context_length,
            args.num_layers,
            device,
            dtype,
        ),
        OptimizerParams(
            args.learning_rate,
            args.betas,
            args.weight_decay,
            args.learning_rate_schedule_max_iterations,
        ),
        TrainingParams(
            args.run_name,
            args.data_path,
            args.batch_size,
            args.epochs,
            args.checkpoint_interval,
            args.checkpoint_path,
        ),
        RandomSeeds(
            args.python_random_seed,
            args.numpy_random_seed,
            args.pytorch_random_seed,
        ),
    )
