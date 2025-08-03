from typing import Optional
import torch


def get_device(device: Optional[str | torch.device]):
    if device:
        return device
    elif torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
        if not device:
            raise RuntimeError("No current_accelerator despite is_available being True")
        return device
    else:
        return "cpu"
