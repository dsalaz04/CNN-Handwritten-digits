# Description: Train and evaluate the MNIST CNN.
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from data import make_loaders
from model import DigitCNN
from utils import get_device, seed_everything


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None, desc=""):
    """Run one pass over `loader`. Trains when `optimizer` is given, else evaluates."""
    train = optimizer is not None
    model.train(train)

    total_loss, correct, seen = 0.0, 0, 0
    progress = tqdm(loader, desc=desc, leave=False)
    with torch.set_grad_enabled(train):
        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += images.size(0)
            progress.set_postfix(loss=total_loss / seen, acc=correct / seen)

    return total_loss / seen, correct / seen


def main():
    parser = argparse.ArgumentParser(description="Train a CNN on MNIST.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--noise-std", type=float, default=0.08)
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation.")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--out", type=str, default="checkpoints/digitcnn.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = make_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        augment=not args.no_augment,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    model = DigitCNN(dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader)
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, scheduler,
            desc=f"Epoch {epoch}/{args.epochs} [train]",
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device, desc=f"Epoch {epoch}/{args.epochs} [val]"
        )
        print(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(), "val_acc": val_acc}, out_path)
            print(f"  ↳ saved new best model to {out_path} (val acc {val_acc:.4f})")

    # Final evaluation on the untouched test set, using the best checkpoint.
    checkpoint = torch.load(out_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_acc = run_epoch(model, test_loader, criterion, device, desc="Test")
    print(f"\nBest val acc {best_val_acc:.4f} | Test acc {test_acc:.4f} ({test_acc * 100:.2f}%)")


if __name__ == "__main__":
    main()
