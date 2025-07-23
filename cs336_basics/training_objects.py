from torch.optim.lr_scheduler import CosineAnnealingLR

from cs336_basics.adamw import Adamw
from cs336_basics.train_params import ModelParams, OptimizerParams
from cs336_basics.transformer import TransformerLm


def make_training_objects(
    vocab_size: int,
    model_params: ModelParams,
    optimizer_params: OptimizerParams,
):

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
    # model.compile()
    optimizer = Adamw(
        model.parameters(),
        optimizer_params.learning_rate,
        optimizer_params.betas,
        optimizer_params.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer, optimizer_params.learning_rate_schedule_max_iterations
    )
    return model, optimizer, scheduler
