"""
Tests:
    uv run pytest -k test_cross_entropy
"""

import torch
from torch import Tensor

from jaxtyping import Float, Int


def batched_cross_entropy(
    logits: Float[Tensor, "batch_size seq_len vocab_size"],
    targets: Int[Tensor, "batch_size seq_len"],
) -> Float[Tensor, ""]:
    """During training our inputs are batched, but we want to calculate cross entropy across"""
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)
    return cross_entropy(logits_flat, targets_flat)


def cross_entropy(
    logits: Float[Tensor, "batch_size*seq_len vocab_size"],
    targets: Int[Tensor, "batch_size*seq_len"],
) -> Float[Tensor, ""]:
    """compute the cross entropy loss, which takes in predicted logits (oi) and targets (xi+1) and
    computes the cross entropy"""

    [maxes, _indices] = logits.max(dim=-1, keepdim=True)
    shifted = logits - maxes
    # use shifted directly in numerator because we have log(exp(x)).
    # Use log(a/b) = log(a) - log(b)
    log_softmax = shifted - shifted.exp().sum(dim=-1, keepdim=True).log()
    selected_log_probs = log_softmax[range(logits.size(0)), targets]
    return -selected_log_probs.mean()
