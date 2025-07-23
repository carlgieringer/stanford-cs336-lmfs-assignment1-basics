"""
Test:
    uv run pytest -k test_checkpointing
"""

import os
import typing

import torch

from cs336_basics.training_objects import make_training_objects


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    """Should also save random seeds. Is iteration enough to resume the training data?"""
    checkpoint_data = dict(
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        iteration=iteration,
    )
    torch.save(checkpoint_data, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
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
    iteration: int,
    params: dict,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    train_state = dict(
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        iteration=iteration,
        params=params,
    )
    torch.save(train_state, out)


def load_train_state(
    vocab_size: int,
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    train_state = torch.load(src, weights_only=False)
    params = train_state["params"]
    model_params = params["model_params"]
    optimizer_params = params["optimizer_params"]
    training_params = params["training_params"]
    random_seeds = params["random_seeds"]

    # if torch.accelerator.is_available():
    #     model_params.device = torch.accelerator.current_accelerator()

    model, optimizer, scheduler = make_training_objects(
        vocab_size, model_params, optimizer_params
    )
    # if torch.accelerator.is_available():
    #     model.to(torch.accelerator.current_accelerator())

    model.load_state_dict(train_state["model_state_dict"])
    optimizer.load_state_dict(train_state["optimizer_state_dict"])

    return dict(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_params=training_params,
        random_seeds=random_seeds,
        iteration=train_state["iteration"],
    )
