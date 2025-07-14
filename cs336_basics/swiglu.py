import math

import torch
from jaxtyping import Float
from torch import Tensor


class Swiglu(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        std = 2 / (d_model + d_ff)
        sigma = math.sqrt(std)

        self.w1 = torch.nn.Parameter(
            torch.empty(d_ff, d_model, device=device, dtype=dtype)
        )
        self.w2 = torch.nn.Parameter(
            torch.empty(d_model, d_ff, device=device, dtype=dtype)
        )
        self.w3 = torch.nn.Parameter(
            torch.empty(d_ff, d_model, device=device, dtype=dtype)
        )
        torch.nn.init.trunc_normal_(self.w1, 0, std, -3 * sigma, 3 * sigma)
        torch.nn.init.trunc_normal_(self.w2, 0, std, -3 * sigma, 3 * sigma)
        torch.nn.init.trunc_normal_(self.w3, 0, std, -3 * sigma, 3 * sigma)

    def forward(
        self, x: Float[Tensor, " ... d_model"]
    ) -> Float[Tensor, " ... d_model"]:
        ff = x @ self.w1.T
        silu = ff * torch.sigmoid(ff)
        return (silu * (x @ self.w3.T)) @ self.w2.T
