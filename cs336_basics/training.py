"""
Training script with wandb integration for experiment tracking and hyperparameter optimization.

Examples:

Login with WandB API key first

```
uv run wandb login
```

Single training run:

```
uv run python cs336_basics/training.py\
 --action=RunSingleTraining\
 --data-path=data/tokens-TinyStoriesV2-GPT4-train.npy\
 --validation-data-path=data/tokens-TinyStoriesV2-GPT4-valid.npy\
 --run-name=TinyStories-single-with-validation\
 --total-steps=1000\
 --validation-interval=50\
 --early-stopping-patience=5\
 --early-stopping-min-delta=0.001\
 --wandb-project=stanford-cs336-language-model\
 --wandb-entity=carl-gieringer-self
```

Wandb sweep for hyperparameter optimization:

```
uv run python cs336_basics/training.py\
 --action=RunWandbSweep\
 --data-path=data/tokens-TinyStoriesV2-GPT4-train.npy\
 --validation-data-path=data/tokens-TinyStoriesV2-GPT4-valid.npy\
 --run-name=TinyStories-sweep\
 --sweep-count=20\
 --total-steps=1000\
 --validation-interval=50\
 --early-stopping-patience=5\
 --early-stopping-min-delta=0.001\
 --wandb-project=stanford-cs336-language-model\
 --wandb-entity=carl-gieringer-self
```

"""

import argparse
import dataclasses
from dataclasses import dataclass
import enum
import logging
import random
from typing import Optional, List

import numpy as np
import torch
from tqdm import tqdm
import wandb
import wandb.wandb_run

from cs336_basics import git
from cs336_basics.checkpointing import save_train_state
from cs336_basics.cross_entropy import batched_cross_entropy
from cs336_basics.data_loading import load_data
from cs336_basics.gradient_clipping import clip_gradients
from cs336_basics.train_params import (
    ModelParams,
    OptimizerParams,
    RandomSeeds,
    TrainingParams,
    TrainingRunParams,
    ValidationParams,
    WandbParams,
)
from cs336_basics.training_objects import make_training_objects

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Action(enum.Enum):
    RunSingleTraining = "RunSingleTraining"
    RunWandbSweep = "RunWandbSweep"


arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("--action", type=Action, choices=list(Action))

# Model params
arg_parser.add_argument("--d-model", type=int, default=512)
arg_parser.add_argument("--num-heads", type=int, default=16)
arg_parser.add_argument("--d-ff", type=int, default=1344)
arg_parser.add_argument("--rope-theta", type=int, default=1e4)
arg_parser.add_argument("--context-length", type=int, default=256)
arg_parser.add_argument("--num-layers", type=int, default=4)
arg_parser.add_argument("--device")
arg_parser.add_argument("--dtype", default="float32")

# Optimizer params
arg_parser.add_argument("--learning-rate", type=float, default=4e-4)
arg_parser.add_argument("--learning-rate-schedule-max-iterations", type=int)
arg_parser.add_argument(
    "--betas", type=lambda v: tuple(map(float, v.split(","))), default=(0.9, 0.95)
)
arg_parser.add_argument("--weight-decay", type=float, default=1e-5)
arg_parser.add_argument("--gradient-clip-norm", type=float, default=1.0)

# Training params
arg_parser.add_argument("--run-name", required=True)
arg_parser.add_argument("--data-path")
arg_parser.add_argument("--batch-size", type=int, default=64)
arg_parser.add_argument("--total-steps", type=int, default=10)
arg_parser.add_argument(
    "--checkpoint-interval", type=int, help="If missing, no intermediate checkpointing."
)
arg_parser.add_argument("--checkpoint-dir", default="data/checkpoints")
arg_parser.add_argument("--compile-model", action="store_true")

# Random seeds
arg_parser.add_argument("--python-random-seed", type=int, default=42)
arg_parser.add_argument("--numpy-random-seed", type=int, default=42)
arg_parser.add_argument("--pytorch-random-seed", type=int, default=42)

# Wandb
arg_parser.add_argument("--wandb-project", default="stanford-cs336-language-model")
arg_parser.add_argument("--wandb-entity")
arg_parser.add_argument("--wandb-tags", type=lambda v: v.split(",") if v else [])
arg_parser.add_argument("--wandb-notes")
arg_parser.add_argument("--sweep-count", type=int, default=20)
arg_parser.add_argument("--gradient-log-frequency", type=int, default=10)
arg_parser.add_argument(
    "--log-artifacts",
    action="store_true",
    help="Upload model checkpoints to wandb (uses storage quota)",
)

# Validation and early stopping
arg_parser.add_argument("--validation-data-path")
arg_parser.add_argument("--validation-interval", type=int, default=50)
arg_parser.add_argument("--early-stopping-patience", type=int, default=5)
arg_parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)


@dataclass
class EarlyStoppingInfo:
    best_validation_loss: float = float("inf")
    patience_counter: int = 0
    do_stop: bool = False


def make_wandb_config(training_run_params: TrainingRunParams):
    model_params = training_run_params.model_params
    optimizer_params = training_run_params.optimizer_params
    training_params = training_run_params.training_params
    random_seeds = training_run_params.random_seeds
    validation_params = training_run_params.validation_params
    return {
        # Model params
        "d_model": model_params.d_model,
        "num_heads": model_params.num_heads,
        "d_ff": model_params.d_ff,
        "rope_theta": model_params.rope_theta,
        "context_length": model_params.context_length,
        "num_layers": model_params.num_layers,
        "device": str(model_params.device),
        "dtype": str(model_params.dtype),
        # Optimizer params
        "learning_rate": optimizer_params.learning_rate,
        "betas": optimizer_params.betas,
        "weight_decay": optimizer_params.weight_decay,
        "gradient_clip_norm": optimizer_params.gradient_clip_norm,
        # Training params
        "batch_size": training_params.batch_size,
        "total_steps": training_params.total_steps,
        # Random seeds
        "python_random_seed": random_seeds.python,
        "numpy_random_seed": random_seeds.numpy,
        "pytorch_random_seed": random_seeds.pytorch,
        # Validation params
        "validation_data_path": validation_params.validation_data_path,
        "validation_interval": validation_params.validation_interval,
        "early_stopping_patience": validation_params.early_stopping_patience,
        "early_stopping_min_delta": validation_params.early_stopping_min_delta,
        # Reproducibility
        "git_commit": git.get_git_commit_hash(),
    }


def init_seeds(random_seeds: RandomSeeds):
    random.seed(random_seeds.python)
    np.random.seed(random_seeds.numpy)
    torch.manual_seed(random_seeds.pytorch)


def train_model(training_run_params: TrainingRunParams):
    model_params = training_run_params.model_params
    optimizer_params = training_run_params.optimizer_params
    training_params = training_run_params.training_params
    random_seeds = training_run_params.random_seeds
    wandb_params = training_run_params.wandb_params
    validation_params = training_run_params.validation_params

    run = wandb.init(
        project=wandb_params.project,
        entity=wandb_params.entity,
        name=training_params.run_name,
        config=make_wandb_config(training_run_params),
        tags=wandb_params.tags,
        notes=wandb_params.notes,
    )

    init_seeds(random_seeds)

    training_data = np.load(training_params.data_path, mmap_mode="r")
    # +1 for 0th indexed vocab items
    vocab_size = training_data.max() + 1

    model, optimizer, scheduler = make_training_objects(
        vocab_size, model_params, optimizer_params
    )
    if training_params.compile_backend:
        model.compile(backend=training_params.compile_backend)

    validation_data = None
    if validation_params.validation_data_path:
        validation_data = np.load(validation_params.validation_data_path, mmap_mode="r")

    if wandb_params.gradient_log_frequency:
        wandb.watch(model, log="all", log_freq=wandb_params.gradient_log_frequency)

    early_stopping_info = EarlyStoppingInfo()

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

        clip_gradients(model.parameters(), optimizer_params.gradient_clip_norm)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        early_stopping_info = log_and_validate(
            training_run_params,
            model,
            scheduler,
            loss.item(),
            step,
            validation_data,
            early_stopping_info,
        )
        if early_stopping_info.do_stop:
            break

        if (
            training_params.checkpoint_interval is not None
            and step % training_params.checkpoint_interval == 0
        ):
            checkpoint_and_log_artifact(
                training_run_params, model, optimizer, run, step
            )

    checkpoint_and_log_artifact(training_run_params, model, optimizer, run, step=None)

    final_loss = loss.item()
    wandb.log(
        {
            "final_loss": final_loss,
            "early_stopped": early_stopping_info.do_stop,
            "final_step": step,
        }
    )

    wandb.finish()
    return final_loss


def checkpoint_and_log_artifact(
    training_run_params: TrainingRunParams,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    run: wandb.wandb_run.Run,
    step: Optional[int],
):
    step_description = "step-{step}" if step is not None else "final"
    checkpoint_path = f"{training_run_params.training_params.checkpoint_dir}/{training_run_params.training_params.run_name}-{step_description}.pt"
    save_train_state(
        model,
        optimizer,
        step,
        training_run_params,
        checkpoint_path,
    )

    if training_run_params.wandb_params.log_artifacts:
        artifact = wandb.Artifact(f"model-checkpoint-{step_description}", type="model")
        artifact.add_file(checkpoint_path)
        run.log_artifact(artifact)


def log_and_validate(
    training_run_params: TrainingRunParams,
    model: torch.nn.Module,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss: float,
    step: int,
    validation_data: Optional[np.ndarray],
    early_stopping_info: EarlyStoppingInfo,
):
    model_params = training_run_params.model_params
    optimizer_params = training_run_params.optimizer_params
    training_params = training_run_params.training_params
    validation_params = training_run_params.validation_params

    # Log metrics to wandb
    current_lr = (
        scheduler.get_last_lr()[0]
        if hasattr(scheduler, "get_last_lr")
        else optimizer_params.learning_rate
    )

    log_dict = {
        "loss": loss,
        "learning_rate": current_lr,
        "step": step,
    }

    # Validation and early stopping
    do_stop = early_stopping_info.do_stop
    patience_counter = early_stopping_info.patience_counter
    if (
        validation_data is not None
        and step % validation_params.validation_interval == 0
        and step > 0
    ):
        model.eval()
        with torch.no_grad():
            val_inputs, val_targets = load_data(
                validation_data,
                training_params.batch_size,
                model_params.context_length,
                model_params.device,
            )
            val_logits = model(val_inputs)
            validation_loss = batched_cross_entropy(val_logits, val_targets).item()
        model.train()
        log_dict["validation_loss"] = validation_loss

        # Early stopping logic
        if (
            validation_loss
            < early_stopping_info.best_validation_loss
            - validation_params.early_stopping_min_delta
        ):
            best_validation_loss = validation_loss
            patience_counter = 0
            log_dict["best_validation_loss"] = best_validation_loss
        else:
            patience_counter += 1

        if patience_counter >= validation_params.early_stopping_patience:
            logger.info(
                f"Early stopping triggered at step {step}. Best validation loss: {best_validation_loss}"
            )
            do_stop = True

    wandb.log(log_dict)

    return dataclasses.replace(
        early_stopping_info, patience_counter=patience_counter, do_stop=do_stop
    )


def make_params(args: argparse.Namespace) -> TrainingRunParams:
    compile_backend = "inductor"
    if args.device:
        device = args.device
    elif torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
    else:
        device = "cpu"
    if device == "mps" or isinstance(device, torch.device) and device.type == "mps":
        compile_backend = "aot_eager"
    if not args.compile_model:
        compile_backend = None

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
            args.checkpoint_dir,
            compile_backend,
        ),
        RandomSeeds(
            args.python_random_seed,
            args.numpy_random_seed,
            args.pytorch_random_seed,
        ),
        WandbParams(
            project=args.wandb_project,
            entity=args.wandb_entity,
            tags=args.wandb_tags or [],
            notes=args.wandb_notes,
            gradient_log_frequency=args.gradient_log_frequency,
            log_artifacts=args.log_artifacts,
        ),
        ValidationParams(
            validation_data_path=args.validation_data_path,
            validation_interval=args.validation_interval,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
        ),
    )


def create_wandb_sweep_config(args: argparse.Namespace):
    """Create wandb sweep configuration with early termination support."""
    # Use validation_loss if validation data is provided, otherwise use final_loss
    metric_name = "validation_loss" if args.validation_data_path else "final_loss"

    sweep_config = {
        "method": "bayes",  # or 'grid', 'random'
        "metric": {"name": metric_name, "goal": "minimize"},
        "parameters": {
            # "learning_rate": {
            #     "distribution": "log_uniform_values",
            #     "min": 1e-5,
            #     "max": 1e-1,
            # },
            # Add more hyperparameters to sweep over
            # "weight_decay": {"values": [1e-5, 1e-4, 1e-3]},
            "batch_size": {"values": [1, 2, 4, 8, 16]},
        },
        # Early termination configuration
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 100,  # Minimum steps before considering termination
            "eta": 3,  # Proportion of runs to keep at each iteration
        },
    }
    return sweep_config


def run_wandb_sweep(args: argparse.Namespace):
    """Run wandb sweep for hyperparameter optimization."""

    def train_with_sweep():
        """Training function to be called by wandb agent."""
        # Initialize wandb run (this will be done by the sweep agent)
        run = wandb.init()

        # Get hyperparameters from wandb config
        config = wandb.config

        # Create base training params from args (no mutation)
        training_run_params = make_params(args)

        # Update specific fields from sweep config
        if hasattr(config, "learning_rate"):
            training_run_params.optimizer_params.learning_rate = config.learning_rate
        if hasattr(config, "weight_decay"):
            training_run_params.optimizer_params.weight_decay = config.weight_decay
        if hasattr(config, "batch_size"):
            training_run_params.training_params.batch_size = config.batch_size

        # Update run name to include sweep info
        training_run_params.training_params.run_name = f"{args.run_name}-{run.id}"

        logger.info(f"Running sweep with config: {config}")
        logger.info(f"Training run params: {training_run_params}")

        # Train the model
        final_loss = train_model(training_run_params)

        return final_loss

    # Create sweep configuration
    sweep_config = create_wandb_sweep_config(args)

    # Initialize the sweep
    sweep_id = wandb.sweep(
        sweep_config, project=args.wandb_project, entity=args.wandb_entity
    )

    logger.info(f"Created wandb sweep with ID: {sweep_id}")
    logger.info(f"Starting wandb agent with {args.sweep_count} runs")

    # Run the sweep agent
    wandb.agent(sweep_id, train_with_sweep, count=args.sweep_count)

    logger.info("Wandb sweep completed")


def run_single_training(args: argparse.Namespace):
    """Run a single training run with wandb logging."""
    training_run_params = make_params(args)
    logger.info(f"Running single training with params {training_run_params}")
    train_model(training_run_params)


if __name__ == "__main__":
    args = arg_parser.parse_args()
    logger.info(f"Running {arg_parser.prog} with args: {args}")
    if args.action == Action.RunSingleTraining:
        run_single_training(args)
    elif args.action == Action.RunWandbSweep:
        run_wandb_sweep(args)
