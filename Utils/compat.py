"""
PyTorch compatibility helpers for SelfGNN: seed and device.
"""
import random
import numpy as np
import torch


def set_seed(seed=42):
    """Set global seeds for reproducibility (random, numpy, PyTorch, CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda=True, device_override=None):
    """Return torch device. Uses device_override if set (e.g. 'cuda', 'cuda:0', 'cpu'), else cuda if available."""
    if device_override is not None:
        override = str(device_override).strip().lower()
        if override in ("cuda", "gpu", ""):
            if torch.cuda.is_available():
                return torch.device("cuda")
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False. "
                "Install PyTorch with CUDA: https://pytorch.org/get-started/locally/"
            )
        if override.startswith("cuda"):
            if torch.cuda.is_available():
                return torch.device(override)
            raise RuntimeError(
                "CUDA requested but not available. Install PyTorch with CUDA support."
            )
        if override == "cpu":
            return torch.device("cpu")
        return torch.device(override)
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
