import argparse
import logging
import random

import numpy as np
import torch
from tqdm import tqdm

from cs336_basics.checkpointing import save_train_state
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.data_loading import load_data
from cs336_basics.gradient_clipping import clip_gradients
from cs336_basics.train_params import (
    ModelParams,
    OptimizerParams,
    RandomSeeds,
    TrainingParams,
)
from cs336_basics.training_objects import make_training_objects

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

arg_parser = argparse.ArgumentParser()

# Model params
_DEFAULT_D_MODEL = 768
arg_parser.add_argument("--d-model", type=int, default=768)
arg_parser.add_argument("--num-heads", type=int, default=12)
arg_parser.add_argument("--d-ff", type=int, default=_DEFAULT_D_MODEL * 4)
arg_parser.add_argument("--rope-theta", type=int, default=1e-6)
arg_parser.add_argument("--context-length", type=int, default=256)
arg_parser.add_argument("--num-layers", type=int, default=12)
arg_parser.add_argument("--device")
arg_parser.add_argument("--dtype", default="float32")

# Optimizer params
arg_parser.add_argument("--learning-rate", type=float, default=1e-4)
arg_parser.add_argument(
    "--learning-rate-schedule-max-iterations", type=int, default=1000
)
arg_parser.add_argument(
    "--betas", type=lambda v: tuple(map(float, v.split(","))), default=(0.9, 0.95)
)
arg_parser.add_argument("--weight-decay", type=float, default=1e-4)
arg_parser.add_argument("--gradient-clip-norm", type=float, default=1.0)

# Training params
arg_parser.add_argument("--run-name", required=True)
arg_parser.add_argument("--data-path")
arg_parser.add_argument("--batch-size", type=int, default=64)
arg_parser.add_argument("--epochs", type=int, default=10)
arg_parser.add_argument("--checkpoint-interval", type=int, default=100)
arg_parser.add_argument("--checkpoint-path", default="data/checkpoints")

arg_parser.add_argument("--python-random-seed", type=int, default=42)
arg_parser.add_argument("--numpy-random-seed", type=int, default=42)
arg_parser.add_argument("--pytorch-random-seed", type=int, default=42)


def train_model(
    model_params: ModelParams,
    optimizer_params: OptimizerParams,
    training_params: TrainingParams,
    random_seeds: RandomSeeds,
):
    """
    TODO: test/validation set
    """
    random.seed(random_seeds.python)
    np.random.seed(random_seeds.numpy)
    torch.manual_seed(random_seeds.pytorch)

    training_data = np.load(training_params.data_path, mmap_mode="r")
    # +1 for 0th indexed vocab items
    vocab_size = training_data.max() + 1

    model, optimizer, scheduler = make_training_objects(
        vocab_size, model_params, optimizer_params
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

        # Apply gradient clipping
        clip_gradients(model.parameters(), optimizer_params.gradient_clip_norm)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        print(f"epoch {epoch}: loss: {loss:.4f}")

        if epoch % training_params.checkpoint_interval == 0:
            save_train_state(
                model,
                optimizer,
                epoch,
                dict(
                    model_params=model_params,
                    optimizer_params=optimizer_params,
                    training_params=training_params,
                    random_seeds=random_seeds,
                ),
                f"{training_params.checkpoint_path}/{training_params.run_name}-epoch-{epoch}.pt",
            )

    save_train_state(
        model,
        optimizer,
        epoch,
        dict(
            model_params=model_params,
            optimizer_params=optimizer_params,
            training_params=training_params,
            random_seeds=random_seeds,
        ),
        f"{training_params.checkpoint_path}/{training_params.run_name}-final.pt",
    )


if __name__ == "__main__":
    # Hyperparams
    # https://docs.ray.io/en/latest/tune/index.html

    # run = wandb.init(
    #     entity="carl-gieringer-self",
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
            args.gradient_clip_norm,
        ),
        TrainingParams(
            args.run_name,
            args.data_path,
            args.batch_size,
            args.batches_per_epoch,
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
