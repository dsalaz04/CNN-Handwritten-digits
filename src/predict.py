# Description: Run a trained model on test samples or your own digit image.
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from data import MNIST_MEAN, MNIST_STD, build_transforms, load_datasets
from model import DigitCNN
from utils import get_device


def load_model(checkpoint_path: str, device: torch.device) -> DigitCNN:
    model = DigitCNN().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def predict_image(model, image_path: str, device: torch.device):
    """Classify an external 28x28-ish grayscale digit image (white digit on black)."""
    from PIL import Image

    img = Image.open(image_path).convert("L").resize((28, 28))
    transform = build_transforms(train=False)
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]
    pred = int(probs.argmax())
    print(f"{image_path} -> predicted {pred} (confidence {probs[pred]:.3f})")
    return pred


def show_test_grid(model, device, n: int = 25, out: str = "predictions.png"):
    """Sample n test digits, predict them, and save a labelled grid."""
    import matplotlib.pyplot as plt

    test = load_datasets(augment=False)[1]
    mean, std = MNIST_MEAN[0], MNIST_STD[0]

    cols = 5
    rows = (n + cols - 1) // cols
    figure = plt.figure(figsize=(cols * 1.6, rows * 1.8))
    correct = 0
    for i in range(1, n + 1):
        idx = torch.randint(len(test), size=(1,)).item()
        image, label = test[idx]
        with torch.no_grad():
            pred = int(model(image.unsqueeze(0).to(device)).argmax())
        correct += pred == label

        ax = figure.add_subplot(rows, cols, i)
        ax.axis("off")
        ax.set_title(f"pred {pred} / true {label}", color="green" if pred == label else "red", fontsize=8)
        ax.imshow(image.squeeze() * std + mean, cmap="gray")  # un-normalize for display

    figure.tight_layout()
    figure.savefig(out, dpi=120)
    print(f"Saved {out} — {correct}/{n} correct on this sample.")


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained MNIST CNN.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/digitcnn.pt")
    parser.add_argument("--image", type=str, default=None, help="Path to a digit image to classify.")
    parser.add_argument("--grid", type=int, default=25, help="How many test digits to show in the grid.")
    parser.add_argument("--out", type=str, default="predictions.png")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        raise SystemExit(f"No checkpoint at {args.checkpoint}. Train one first with: python src/train.py")

    device = get_device()
    model = load_model(args.checkpoint, device)

    if args.image:
        predict_image(model, args.image, device)
    else:
        show_test_grid(model, device, n=args.grid, out=args.out)


if __name__ == "__main__":
    main()
