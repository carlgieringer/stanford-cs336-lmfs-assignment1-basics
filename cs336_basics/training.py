"""

```
uv run python cs336_basics/training.py\
 --action=RunHoptStudy\
 --data-path=data/tokens-TinyStoriesV2-GPT4-train.npy\
 --run-name=TinyStories-hopt\
 --hopt-trials=20\
 --total-steps=1000
```

"""

import argparse
from dataclasses import dataclass
import enum
import logging
import random
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm
import optuna
from optuna import Trial

from cs336_basics.checkpointing import save_train_state
from cs336_basics.cross_entropy import batched_cross_entropy
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


class Action(enum.Enum):
    RunSingleTraining = "RunSingleTraining"
    RunHoptStudy = "RunHoptStudy"


arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("--action", type=Action, choices=list(Action))

# Model params
_DEFAULT_D_MODEL = 768
arg_parser.add_argument("--d-model", type=int, default=512)
arg_parser.add_argument("--num-heads", type=int, default=16)
arg_parser.add_argument("--d-ff", type=int, default=1344)
arg_parser.add_argument("--rope-theta", type=int, default=1e4)
arg_parser.add_argument("--context-length", type=int, default=256)
arg_parser.add_argument("--num-layers", type=int, default=4)
arg_parser.add_argument("--device")
arg_parser.add_argument("--dtype", default="float32")

# Optimizer params
arg_parser.add_argument("--learning-rate", type=float, default=1e-4)
arg_parser.add_argument("--learning-rate-schedule-max-iterations", type=int)
arg_parser.add_argument(
    "--betas", type=lambda v: tuple(map(float, v.split(","))), default=(0.9, 0.95)
)
arg_parser.add_argument("--weight-decay", type=float, default=1e-4)
arg_parser.add_argument("--gradient-clip-norm", type=float, default=1.0)

# Training params
arg_parser.add_argument("--run-name", required=True)
arg_parser.add_argument("--data-path")
arg_parser.add_argument("--batch-size", type=int, default=64)
arg_parser.add_argument("--total-steps", type=int, default=10)
arg_parser.add_argument("--checkpoint-interval", type=int, default=100)
arg_parser.add_argument("--checkpoint-path", default="data/checkpoints")
arg_parser.add_argument("--save-intermediate-checkpoints", action="store_true")

# Random seeds
arg_parser.add_argument("--python-random-seed", type=int, default=42)
arg_parser.add_argument("--numpy-random-seed", type=int, default=42)
arg_parser.add_argument("--pytorch-random-seed", type=int, default=42)

# HOpt
arg_parser.add_argument("--hopt-trials", type=int)
arg_parser.add_argument("--pruner-patience", type=int, default=100)


@dataclass
class TrainingRunParams:
    model_params: ModelParams
    optimimizer_params: OptimizerParams
    training_params: TrainingParams
    random_seeds: RandomSeeds


def train_model(training_run_params: TrainingRunParams, trial: Optional[Trial] = None):
    """
    TODO: test/validation set
    """
    model_params = training_run_params.model_params
    optimizer_params = training_run_params.optimimizer_params
    training_params = training_run_params.training_params
    random_seeds = training_run_params.random_seeds

    random.seed(random_seeds.python)
    np.random.seed(random_seeds.numpy)
    torch.manual_seed(random_seeds.pytorch)

    training_data = np.load(training_params.data_path, mmap_mode="r")
    # +1 for 0th indexed vocab items
    vocab_size = training_data.max() + 1

    model, optimizer, scheduler = make_training_objects(
        vocab_size, model_params, optimizer_params
    )
    model.compile(backend="aot_eager")

    for step in tqdm(range(training_params.total_steps), desc="Steps", unit="step"):

        inputs, targets = load_data(
            training_data,
            training_params.batch_size,
            model_params.context_length,
            model_params.device,
        )

        logits = model(inputs)
        loss = batched_cross_entropy(logits, targets)
        loss.backward()

        # Apply gradient clipping
        clip_gradients(model.parameters(), optimizer_params.gradient_clip_norm)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if trial:
            trial.report(loss.item(), step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if step % 10 == 0:
            logger.info(f"Step {step}: loss: {loss:.4f}")

        if (
            training_params.save_intermediate_checkpoints
            and step % training_params.checkpoint_interval == 0
        ):
            save_train_state(
                model,
                optimizer,
                step,
                dict(
                    model_params=model_params,
                    optimizer_params=optimizer_params,
                    training_params=training_params,
                    random_seeds=random_seeds,
                ),
                f"{training_params.checkpoint_path}/{training_params.run_name}-step-{step}.pt",
            )

    save_train_state(
        model,
        optimizer,
        None,
        dict(
            model_params=model_params,
            optimizer_params=optimizer_params,
            training_params=training_params,
            random_seeds=random_seeds,
        ),
        f"{training_params.checkpoint_path}/{training_params.run_name}-final.pt",
    )

    return loss.item()


def make_params(args: argparse.Namespace) -> TrainingRunParams:
    if args.device:
        device = args.device
    elif torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
    else:
        device = "cpu"

    dtype = getattr(torch, args.dtype)
    if not isinstance(dtype, torch.dtype):
        raise Exception(f"Invalid dtype: {args.dtype}")

    if args.learning_rate_schedule_max_iterations is not None:
        learning_rate_schedule_max_iterations = (
            args.learning_rate_schedule_max_iterations
        )
    else:
        learning_rate_schedule_max_iterations = args.total_steps

    return TrainingRunParams(
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
            learning_rate_schedule_max_iterations,
            args.gradient_clip_norm,
        ),
        TrainingParams(
            args.run_name,
            args.data_path,
            args.batch_size,
            args.total_steps,
            args.checkpoint_interval,
            args.checkpoint_path,
            args.save_intermediate_checkpoints,
        ),
        RandomSeeds(
            args.python_random_seed,
            args.numpy_random_seed,
            args.pytorch_random_seed,
        ),
    )


def run_hopt_study(args: argparse.Namespace):
    study = optuna.create_study(
        study_name=args.run_name,
        pruner=optuna.pruners.PatientPruner(
            optuna.pruners.MedianPruner(), patience=args.pruner_patience
        ),
    )

    def objective(trial: Trial):

        training_run_params = make_params(args)
        training_run_params.optimimizer_params.learning_rate = trial.suggest_float(
            "lr", 1e-5, 1e-1, log=True
        )
        training_run_params.training_params.run_name = (
            f"{training_run_params.training_params.run_name}-{trial.number}"
        )

        logger.info(f"Running objective with training run params {training_run_params}")

        return train_model(training_run_params, trial)

    study.optimize(objective, n_trials=args.hopt_trials)
    logging.info(f"Best params {study.best_params} (best value {study.best_value})")


def run_single_training(args: argparse.Namespace):
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

    training_run_params = make_params(args)
    logger.info(f"Running single training with params {training_run_params}")
    train_model(training_run_params)


if __name__ == "__main__":
    args = arg_parser.parse_args()
    logger.info(f"Running {arg_parser.prog} with args: {args}")
    if args.action == Action.RunSingleTraining:
        run_single_training(args)
    elif args.action == Action.RunHoptStudy:
        run_hopt_study(args)
