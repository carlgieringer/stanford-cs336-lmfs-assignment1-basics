import torch


class RmsNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """Construct the RMSNorm module. This function should accept the following parameters:
        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()

        self.eps = eps
        self.weight = torch.nn.Parameter(
            torch.ones((d_model,), device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor
        of the same shape."""
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.sum(x * x, 2, keepdim=True) / x.shape[2] + self.eps)
        rms_norm = torch.div(x * self.weight, rms)

        rms_norm.to(in_dtype)
        return rms_norm
