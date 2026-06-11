# CNN — Handwritten Digit Classification

A compact, modern PyTorch convolutional neural network that classifies handwritten
digits from the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset. The preprocessing
pipeline normalizes inputs and augments the training set with random **affine
transforms** (rotation, translation, scaling, shear) and **additive Gaussian noise**,
so the model learns to recognize digits that are shifted, tilted, rescaled or slightly
corrupted — not just the pristine training examples.

It started as a one-day project; this is the cleaned-up, full-fledged version of the
same idea: a real model, a real training loop, and tooling to run inference on your own
digits.

<p align="center">
  <img src="assets/predictions.png" width="640" alt="Model predictions on random MNIST test digits">
</p>

> Predictions on 25 random test digits (green = correct). Generated with `predict.py`.

## Highlights

- **~99.6% test accuracy** with a ~0.47M-parameter VGG-style CNN.
- **Data augmentation** — affine jitter + Gaussian noise — exactly as the preprocessing
  pipeline advertises.
- **Modern training loop** — AdamW, one-cycle LR schedule, BatchNorm, dropout, a held-out
  validation split, and best-checkpoint saving.
- **Runs anywhere** — automatically uses CUDA, Apple Silicon (MPS), or CPU.
- **Reproducible** — single seed controls data splits and weight initialization.
- **Inference tooling** — classify your own `28×28` digit image, or render a labelled
  grid of test predictions.

## Architecture

Two convolutional blocks (each: two `3×3` convs → BatchNorm → ReLU → max-pool → dropout)
take the `28×28` input down to `7×7` feature maps, followed by a small fully-connected
head:

```
Input 1×28×28
 └─ Conv block 1:  1 →  32 channels   →  32×14×14
 └─ Conv block 2: 32 →  64 channels   →  64× 7× 7
 └─ FC head:      64·7·7 → 128 → 10
```

## Setup

Requires Python 3.9+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

MNIST downloads automatically on the first run into `data/`.

## Usage

**Train** (downloads MNIST on first run, saves the best checkpoint to
`checkpoints/digitcnn.pt`):

```bash
python src/train.py                 # 15 epochs with the default hyperparameters
python src/train.py --epochs 25 --lr 2e-3 --batch-size 256
python src/train.py --no-augment    # train on clean digits, for comparison
```

Useful flags: `--epochs`, `--batch-size`, `--lr`, `--weight-decay`, `--dropout`,
`--noise-std`, `--no-augment`, `--seed`, `--out`. Run `python src/train.py --help` for
the full list.

**Predict** — render a labelled grid of random test digits:

```bash
python src/predict.py --grid 25 --out predictions.png
```

Or classify your own digit (a white digit on a black background works best; it is
resized to `28×28` automatically):

```bash
python src/predict.py --image my_digit.png
```

## Results

Trained for 15 epochs on an Apple Silicon GPU (MPS) in a few minutes:

| Metric         | Value        |
| -------------- | ------------ |
| Test accuracy  | 99.65%       |
| Validation accuracy | 99.58%  |
| Parameters     | 468,458      |
| Training time  | ~3 min (MPS) |

Augmentation is the main lever here: training on jittered, noisy digits costs a little
training accuracy but generalizes noticeably better to the held-out test set than the
`--no-augment` baseline.

## Project structure

```
src/
  model.py     # DigitCNN architecture
  data.py      # MNIST loading, normalization, affine + noise augmentation
  train.py     # training / validation loop, scheduler, checkpointing  (entry point)
  predict.py   # inference on test samples or your own image
  utils.py     # device selection + reproducible seeding
data/           # MNIST (downloaded automatically)
requirements.txt
```

## License

MIT.
