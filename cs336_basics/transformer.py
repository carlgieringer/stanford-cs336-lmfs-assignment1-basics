"""
Test:
    uv run pytest -k test_transformer_block
"""

from typing import Optional

import torch

from cs336_basics import attention, rmsnorm, swiglu


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
        token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.shape[:-1])

        out = self.attn(self.ln1(x), token_positions) + x
        out = self.ffn(self.ln2(out)) + out
        return out
