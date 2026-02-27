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


def get_device(prefer_cuda=True):
    """Return torch device: cuda if available and prefer_cuda else cpu."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
