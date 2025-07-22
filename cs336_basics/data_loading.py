"""
Test:
    uv run pytest -k test_get_batch
"""

import numpy.typing as npt
import torch


def load_data(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """returns a pair of tensors: the sampled input sequences and the corresponding next-token
    targets. Both tensors should have shape (batch_size, context_length) containing token IDs, and
    both should be placed on the requested device.
    """
    # Get batch_size contexts with random offsets in dataset
    input_offsets = torch.randint(0, len(dataset) - context_length, (batch_size,))
    target_offsets = input_offsets + 1
    inputs = [dataset[offset : (offset + context_length)] for offset in input_offsets]
    targets = [dataset[offset : (offset + context_length)] for offset in target_offsets]
    tensor_inputs = torch.as_tensor(inputs, device=device)
    tensor_targets = torch.as_tensor(targets, device=device)
    return (tensor_inputs, tensor_targets)
