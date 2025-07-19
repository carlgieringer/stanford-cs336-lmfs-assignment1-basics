"""
Tests:

    uv run pytest -k test_softmax_matches_pytorch
"""

import torch


def softmax(tensor: torch.Tensor, *, dim: int):
    shifted = tensor - tensor.max(dim=dim, keepdim=True)[0]
    exped = shifted.exp()
    return exped / exped.sum(dim=dim, keepdim=True)
