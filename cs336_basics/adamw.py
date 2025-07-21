"""
Test:
    uv run pytest -k test_adamw
"""

from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class Adamw(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01, eps=1e-8):
        if lr <= 0:
            raise ValueError(f"lr must be positive: {lr}")
        if betas[0] < 0 or betas[0] >= 1:
            raise ValueError(f"beta_1 must be in [0,1): {betas[0]}")
        if betas[1] < 0 or betas[1] >= 1:
            raise ValueError(f"beta_2 must be in [0,1): {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta_1, beta_2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m")
                v = state.get("v")
                if m is None:
                    m = torch.zeros_like(p)
                if v is None:
                    v = torch.zeros_like(p)

                grad = p.grad.data
                m = beta_1 * m + (1 - beta_1) * grad
                v = beta_2 * v + (1 - beta_2) * grad**2
                adam_lr = lr * math.sqrt(1 - beta_2**t) / (1 - beta_1**t)

                p.data -= adam_lr * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss
