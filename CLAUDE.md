# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Coding Guidelines

- Max 120 characters per line
- 2 spaces for indentation

## Overview

This repo contains ML experiments and projects built with PyTorch. Python 3.10+ is required.

## Projects

### Root-level scripts
Standalone PyTorch exploration scripts — no package structure, run directly:

```bash
python main.py          # tensor basics and Perceptron stub
python neural_network.py  # FashionMNIST-style feedforward net (training loop commented out)
python test-pytorch.py  # DataLoader demo (requires FashionMNIST data downloaded)
```

### image_classifier/
A CNN-based image classifier trained on FashionMNIST. **Always run from inside the `image_classifier/` directory** so relative paths (`data/`, `checkpoints/`) resolve correctly.

```bash
cd image_classifier
pip install -r requirements.txt
python main.py   # downloads FashionMNIST, trains, saves checkpoint, runs inference demo
```

## Architecture: image_classifier

Data flows through four modules:

1. **`src/dataset.py`** — `get_dataloaders` downloads FashionMNIST via torchvision into `data/`, applies
   random horizontal flip and random crop (train only), and normalises with FashionMNIST channel statistics.
   Returns train / val / test `DataLoader`s. Val is split from the training set with `random_split`.

2. **`src/model.py`** — `ImageClassifier`: three `ConvBlock`s (Conv2d → BatchNorm → ReLU → optional MaxPool)
   followed by a fully-connected head. Feature map shrinks 28×28 → 14×14 → 7×7; channels grow 1 → 32 → 64 → 128.
   The flattened 128×7×7 = 6272 features pass through `Linear(6272, 256) → Dropout → Linear(256, 10)`.

3. **`src/train.py`** — `train()` drives Adam with weight decay and a `CosineAnnealingLR` scheduler.
   `_run_epoch` handles both train and eval modes (pass `optimizer=None` for eval).

4. **`src/inference.py`** — `load_model` reconstructs the model from a checkpoint. `predict` accepts a
   single image tensor and returns the top label + per-class probability dict. `predict_from_loader`
   prints results for the first N images in any DataLoader.

Hyperparameters and paths live at the top of `image_classifier/main.py` (`BATCH_SIZE`, `EPOCHS`, `LR`, etc.).
Checkpoints are saved to `image_classifier/checkpoints/model.pt` (created at runtime).
FashionMNIST data is auto-downloaded to `image_classifier/data/`.

### text_classifier/
A fully self-contained text classification project. **Always run from inside the `text_classifier/` directory** so relative paths (`data/`, `checkpoints/`) resolve correctly.

```bash
cd text_classifier
pip install -r requirements.txt
python main.py   # trains, saves checkpoint, runs inference demo
```

## Architecture: text_classifier

Data flows through four modules:

1. **`src/dataset.py`** — `load_csv` reads `data/sample_data.csv` (columns: `text`, `label`). `Vocabulary` builds word→index and label→index mappings from the training split. `make_dataloader` pads sequences to the batch's max length via `_collate_fn`.

2. **`src/model.py`** — `TextClassifier`: `Embedding(vocab_size, embed_dim)` → `Bi-LSTM(hidden_dim, num_layers)` → `Linear(hidden_dim*2, num_classes)`. The forward and backward final hidden states are concatenated before the linear layer.

3. **`src/train.py`** — `train()` drives the Adam optimizer with `CrossEntropyLoss`. A single `_run_epoch` helper handles both train and eval modes (pass `optimizer=None` for eval).

4. **`src/inference.py`** — `load_model_and_vocab` reconstructs the model from a checkpoint; `predict` returns the top label and a per-class probability dict.

Hyperparameters and paths live at the top of `text_classifier/main.py` as module-level constants (`EMBED_DIM`, `HIDDEN_DIM`, `EPOCHS`, etc.).

Checkpoints are saved to `text_classifier/checkpoints/` (created at runtime): `model.pt` (state dict) and `vocab.json` (word/label mappings).

## Data format

CSVs must have `text` and `label` columns. The tokenizer lowercases and strips non-alphanumeric characters (`re.sub(r"[^a-z0-9\s]", "", text.lower())`). Replace `data/sample_data.csv` to train on new data.
