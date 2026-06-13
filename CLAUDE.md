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
