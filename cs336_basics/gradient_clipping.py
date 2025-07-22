from collections.abc import Iterable
import math

import torch


def clip_gradients(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps=1e-6
):
    norms = []
    for parameter in parameters:
        if parameter.grad is None:
            continue
        norms.append(math.sqrt(parameter.grad.data.pow(2).sum().item()))
    total_norm = math.sqrt(sum(n**2 for n in norms))
    if total_norm > max_l2_norm:
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.data.mul_(max_l2_norm / (total_norm + eps))
