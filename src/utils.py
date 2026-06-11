# Description: Small shared helpers (device selection, reproducibility).
# -*- coding: utf-8 -*-

import random

import numpy as np
import torch


def get_device() -> torch.device:
    """Pick the best available accelerator: CUDA, then Apple Silicon (MPS), then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy and Torch so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
