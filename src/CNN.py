# Author: Daniel Salazar
# Created: 01.26.2022
# Description: CNN with MNIST Dataset
# -*- coding: utf-8 -*-
# Copyright 2021 Daniel Salazar

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def prepare_data():
    train_data = datasets.MNIST(
        root='data',
        train=True,
        transform=ToTensor(),
        download=True,
    )

    test_data = datasets.MNIST(
        root='data',
        train=False,
        transform=ToTensor()
    )

    plt.imshow(train_data.data[0], cmap='gray')
    plt.title('%i' % train_data.targets[0])
    plt.show()

    figure = plt.figure(figsize=(10, 8))
    cols, rows = 5, 5
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_data), size=(1,)).item()
        img, label = train_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

    print(train_data)
    print(test_data)


if __name__ == '__main__':
    prepare_data()
