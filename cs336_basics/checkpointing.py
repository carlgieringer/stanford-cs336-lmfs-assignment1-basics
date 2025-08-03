"""
Test:
    uv run pytest -k test_checkpointing
"""

from dataclasses import dataclass
import os
from typing import IO, BinaryIO, Optional

import torch

from cs336_basics.train_params import TrainingRunParams
from cs336_basics.training_objects import make_training_objects


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """Should also save random seeds. Is iteration enough to resume the training data?"""
    checkpoint_data = dict(
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        iteration=iteration,
    )
    torch.save(checkpoint_data, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint_data = torch.load(src)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
    return checkpoint_data["iteration"]


def save_train_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: Optional[int],
    training_run_params: TrainingRunParams,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    train_state = dict(
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        iteration=iteration,
        training_run_params=training_run_params,
    )
    torch.save(train_state, out)


def load_train_state(
    vocab_size: int,
    src: str | os.PathLike | BinaryIO | IO[bytes],
    device: Optional[str | torch.device],
):
    train_state = torch.load(src, weights_only=False, map_location=device)
    if "training_run_params" in train_state:
        training_run_params = train_state["training_run_params"]
        model_params = training_run_params.model_params
        # I accidentally serialized some checkpoints using this misspelling
        if hasattr(training_run_params, "optimimizer_params"):
            optimizer_params = getattr(training_run_params, "optimimizer_params")
        else:
            optimizer_params = training_run_params.optimizer_params
    else:
        # Older checkpoints stored params directly on a 'params' dict.
        model_params = train_state["params"]["model_params"]
        optimizer_params = train_state["params"]["optimizer_params"]

    if device:
        model_params.device = device

    model, optimizer, scheduler = make_training_objects(
        vocab_size, model_params, optimizer_params
    )

    model.load_state_dict(train_state["model_state_dict"])
    optimizer.load_state_dict(train_state["optimizer_state_dict"])

    return TrainState(
        model,
        optimizer,
        scheduler,
        training_run_params,
        train_state["iteration"],
    )


@dataclass
class TrainState:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    training_run_params: TrainingRunParams
    iteration: int
