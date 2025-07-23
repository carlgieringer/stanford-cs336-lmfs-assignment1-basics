"""
Tests:

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

from cs336_basics import linear, rope as rope_lib
from cs336_basics.softmax import softmax


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

        self.q_proj = linear.Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = linear.Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = linear.Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = linear.Linear(d_model, d_model, device=device, dtype=dtype)

        if theta > 0 and max_seq_len > 0:
            self.rope = rope_lib.RotaryPositionalEmbedding(
                theta, self.d_k, max_seq_len, device=device
            )
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
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # unflatten: split d_model across the heads
        # transpose: move the head dim into a batch position
        q_heads = q.unflatten(-1, (self.n_heads, self.d_k)).transpose(-3, -2)
        k_heads = k.unflatten(-1, (self.n_heads, self.d_k)).transpose(-3, -2)
        v_heads = v.unflatten(-1, (self.n_heads, self.d_v)).transpose(-3, -2)

        if token_positions is not None:
            if not self.rope:
                raise Exception("rope must have been initialized for token_positions")
            # Insert a singleton dimension where the head dim is so that RoPE ops are compatible.
            head_token_positions = token_positions.unsqueeze(-2)
            q_heads = self.rope(q_heads, head_token_positions)
            k_heads = self.rope(k_heads, head_token_positions)

        seq_len = x.shape[-2]
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        )

        attention_heads = scaled_dot_product_attention(q_heads, k_heads, v_heads, mask)

        attention = attention_heads.transpose(-2, -3).flatten(-2, -1)

        o = self.output_proj(attention)
        return o
