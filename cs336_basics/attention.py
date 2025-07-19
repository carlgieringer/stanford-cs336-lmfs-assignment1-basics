"""
Tests:

    uv run pytest -k test_softmax_matches_pytorch

    uv run pytest -k test_scaled_dot_product_attention
    uv run pytest -k test_4d_scaled_dot_product_attention

    uv run pytest -k test_multihead_self_attention
    uv run pytest -k test_multihead_self_attention_with_rope
"""

import math
from typing import Optional

import torch
from torch import Tensor
from jaxtyping import Float, Bool, Int

from cs336_basics import rope as rope_lib


def softmax(tensor: torch.Tensor, *, dim: int):
    shifted = tensor - tensor.max(dim=dim, keepdim=True)[0]
    exped = shifted.exp()
    return exped / exped.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Float[Tensor, "batch_size ... seq_len d_k"],
    K: Float[Tensor, "batch_size ... seq_len d_k"],
    V: Float[Tensor, "batch_size ... seq_len d_v"],
    mask: Optional[Bool[Tensor, "seq_len seq_len"]] = None,
):
    pre_attention = torch.einsum("...ij,...kj", Q, K)
    if mask is not None:
        pre_attention = torch.where(mask, pre_attention, -math.inf)
    d_k = Q.shape[-1]
    attention = softmax(pre_attention / math.sqrt(d_k), dim=-1)
    return torch.einsum("...ij,...jk", attention, V)


class CausalMultiheadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.d_k = self.d_v = d_model // n_heads

        self.weights_q = torch.nn.Parameter(
            torch.empty(d_model, d_model, device=device, dtype=dtype)
        )
        self.weights_k = torch.nn.Parameter(
            torch.empty(d_model, d_model, device=device, dtype=dtype)
        )
        self.weights_v = torch.nn.Parameter(
            torch.empty(d_model, d_model, device=device, dtype=dtype)
        )
        self.weights_o = torch.nn.Parameter(
            torch.empty(d_model, d_model, device=device, dtype=dtype)
        )

        std = 1 / d_model  # = 2 / (d_model + d_model)
        sigma = math.sqrt(std)
        torch.nn.init.trunc_normal_(self.weights_q, 0, std, -3 * sigma, 3 * sigma)
        torch.nn.init.trunc_normal_(self.weights_k, 0, std, -3 * sigma, 3 * sigma)
        torch.nn.init.trunc_normal_(self.weights_v, 0, std, -3 * sigma, 3 * sigma)
        torch.nn.init.trunc_normal_(self.weights_o, 0, std, -3 * sigma, 3 * sigma)

        if theta > 0 and max_seq_len > 0:
            self.rope = rope_lib.RotaryPositionalEmbedding(theta, self.d_k, max_seq_len)
        else:
            if theta > 0 or max_seq_len > 0:
                raise Exception(
                    "If either theta or max_seq_len are gte zero, both must be."
                )
            self.rope = None

    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_in"],
        token_positions: Optional[Int[Tensor, " ... sequence_length"]] = None,
    ) -> Float[Tensor, "... d_model"]:
        q = x @ self.weights_q.T
        k = x @ self.weights_k.T
        v = x @ self.weights_v.T

        q_heads = q.unflatten(-1, (self.n_heads, self.d_k)).transpose(-3, -2)
        k_heads = k.unflatten(-1, (self.n_heads, self.d_k)).transpose(-3, -2)
        v_heads = v.unflatten(-1, (self.n_heads, self.d_v)).transpose(-3, -2)

        if token_positions is not None:
            if not self.rope:
                raise Exception("rope must have been initialized for token_positions")
            q_heads = self.rope(q_heads, token_positions)
            k_heads = self.rope(k_heads, token_positions)

        seq_len = x.shape[-2]
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        )

        attention_heads = scaled_dot_product_attention(q_heads, k_heads, v_heads, mask)

        attention = attention_heads.transpose(-2, -3).flatten(-2, -1)

        o = attention @ self.weights_o.T
        return o
