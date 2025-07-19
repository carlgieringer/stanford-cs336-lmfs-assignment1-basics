"""
Tests:
    uv run pytest -k test_transformer_block
    uv run pytest -k test_transformer_lm
"""

from typing import Optional

import torch

from cs336_basics import attention, embedding, linear, rmsnorm, swiglu
from cs336_basics.softmax import softmax


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.ln1 = rmsnorm.RmsNorm(d_model, device=device, dtype=dtype)
        self.attn = attention.CausalMultiheadSelfAttention(
            d_model, num_heads, max_seq_len, theta, device, dtype
        )
        self.ln2 = rmsnorm.RmsNorm(d_model, device=device, dtype=dtype)
        self.ffn = swiglu.Swiglu(d_model, d_ff, device, dtype)

    def forward(self, x):
        seq_len = x.shape[-2]
        token_positions = (
            torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.shape[:-1])
        )

        out = self.attn(self.ln1(x), token_positions) + x
        out = self.ffn(self.ln2(out)) + out
        return out


class TransformerLm(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.token_embeddings = embedding.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype
        )
        self.layers = torch.nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    max_seq_len=context_length,
                    theta=theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = rmsnorm.RmsNorm(d_model, device=device, dtype=dtype)
        self.lm_head = linear.Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x):
        out = self.token_embeddings(x)
        for layer in self.layers:
            out = layer(out)
        out = self.lm_head(self.ln_final(out))
        return out
