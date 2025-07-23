import torch
from torch import Tensor
from jaxtyping import Int, Float


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """Construct an embedding module.

        This function should accept the following parameters:
            num_embeddings: int Size of the vocabulary
            embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )

    def forward(
        self, token_ids: Int[Tensor, "batch context_length"]
    ) -> Float[Tensor, "batch d_model"]:
        """Lookup the embedding vectors for the given token IDs."""
        one_hot = torch.nn.functional.one_hot(
            token_ids, num_classes=self.weight.shape[0]
        ).float()
        return torch.einsum("...v,ve", one_hot, self.weight)
