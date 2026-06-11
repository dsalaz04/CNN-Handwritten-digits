# Description: MNIST loading, normalization and data augmentation.
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Dataset-wide statistics used to normalize inputs to roughly zero mean / unit std.
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


class AddGaussianNoise:
    """Add zero-mean Gaussian noise to a tensor. Makes the model robust to speckle."""

    def __init__(self, std: float = 0.08):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn_like(tensor) * self.std

    def __repr__(self) -> str:
        return f"{type(self).__name__}(std={self.std})"


def build_transforms(train: bool, augment: bool = True, noise_std: float = 0.08):
    """Compose the preprocessing pipeline.

    Training (with augmentation) applies random affine transforms — small rotations,
    translations, scaling and shear — followed by normalization and additive noise.
    Evaluation only normalizes, so we measure accuracy on clean digits.
    """
    ops = []
    if train and augment:
        ops.append(
            transforms.RandomAffine(
                degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10
            )
        )
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(MNIST_MEAN, MNIST_STD))
    if train and augment and noise_std > 0:
        ops.append(AddGaussianNoise(noise_std))
    return transforms.Compose(ops)


def load_datasets(root: str = "data", augment: bool = True, noise_std: float = 0.08):
    """Return (train, test) MNIST datasets, downloading them on first run."""
    train = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=build_transforms(train=True, augment=augment, noise_std=noise_std),
    )
    test = datasets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=build_transforms(train=False),
    )
    return train, test


def make_loaders(
    root: str = "data",
    batch_size: int = 128,
    val_split: float = 0.1,
    augment: bool = True,
    noise_std: float = 0.08,
    num_workers: int = 2,
    seed: int = 42,
):
    """Build train / validation / test dataloaders.

    A slice of the training set is held out for validation so we can pick the best
    checkpoint without touching the test set.
    """
    train_full, test = load_datasets(root, augment=augment, noise_std=noise_std)

    val_size = int(len(train_full) * val_split)
    train_size = len(train_full) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(train_full, [train_size, val_size], generator=generator)

    # The validation split should be evaluated on clean digits, not augmented ones.
    val_ds.dataset = load_datasets(root, augment=False)[0]

    # pin_memory only helps (and is only supported) for CUDA transfers.
    common = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader
