"""
Test:
    uv run pytest -k test_rope
"""

from jaxtyping import Float, Int
import torch
from torch import Tensor


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """Construct the RoPE module and create buffers if needed.

        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()

        positions = torch.arange(max_seq_len, device=device).float()
        dim_indices = torch.arange(0, d_k, 2, device=device).float()

        freqs = 1.0 / (theta ** (dim_indices / d_k))
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0)

        cos_vals = torch.cos(angles)
        sin_vals = torch.sin(angles)

        self.register_buffer("cos_cached", cos_vals, persistent=False)
        self.register_buffer("sin_cached", sin_vals, persistent=False)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Int[Tensor, "... seq_len"],
    ) -> torch.Tensor:
        """Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. Note
        that you should tolerate x with an arbitrary number of batch dimensions. You should assume
        that the token positions are a tensor of shape (..., seq_len) specifying the token positions of
        x along the sequence dimension.

        You should use the token positions to slice your (possibly precomputed) cos and sin tensors
        along the sequence dimension.
        """

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        assert isinstance(self.cos_cached, torch.Tensor)
        assert isinstance(self.sin_cached, torch.Tensor)
        cos_vals = self.cos_cached[token_positions]
        sin_vals = self.sin_cached[token_positions]

        x_even_rotated = x_even * cos_vals - x_odd * sin_vals
        x_odd_rotated = x_even * sin_vals + x_odd * cos_vals

        result = torch.zeros_like(x)
        result[..., 0::2] = x_even_rotated
        result[..., 1::2] = x_odd_rotated
        return result
