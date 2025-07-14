import torch


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        Construct a linear transformation module. This function should accept the following parameters:
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.weights = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        sigma = 2 / (in_features + out_features)
        torch.nn.init.trunc_normal_(self.weights, 0, sigma, -3 * sigma, 3 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation to the input."""
        return torch.einsum("...ij,kj", x, self.weights)
