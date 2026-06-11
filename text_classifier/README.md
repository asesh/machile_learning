# Text Classifier

A PyTorch text classification model built from scratch using an Embedding → Bidirectional LSTM → Linear architecture.

## Project Structure

```
text_classifier/
├── data/
│   └── sample_data.csv       # CSV with "text" and "label" columns
├── src/
│   ├── dataset.py            # Vocabulary, tokenizer, Dataset, DataLoader
│   ├── model.py              # TextClassifier neural network
│   ├── train.py              # Training and evaluation loop
│   └── inference.py          # predict() and load_model_and_vocab()
├── checkpoints/              # Saved model weights and vocabulary (created at runtime)
├── main.py                   # Entry point: trains, saves, and demos inference
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- PyTorch 2.0+

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running

From inside the `text_classifier/` directory:

```bash
python main.py
```

This will:
1. Load `data/sample_data.csv`
2. Build a word-level vocabulary from the training split
3. Train for 15 epochs, printing loss and accuracy each epoch
4. Save the model weights to `checkpoints/model.pt` and vocabulary to `checkpoints/vocab.json`
5. Run an inference demo on five example sentences

## Using Your Own Data

Replace `data/sample_data.csv` with any CSV that has `text` and `label` columns:

```csv
text,label
"I loved the movie",positive
"Boring and slow",negative
```

## Architecture

| Layer | Details |
|-------|---------|
| Embedding | vocab_size × 64, padding-aware |
| Bi-LSTM | 128 hidden units, 2 layers, dropout 0.3 |
| Linear | (128×2) → num_classes |

The forward and backward final hidden states are concatenated before the linear layer, giving the model bidirectional context.

## Customising Hyperparameters

Edit the constants near the top of `main.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `EMBED_DIM` | 64 | Embedding dimension |
| `HIDDEN_DIM` | 128 | LSTM hidden units |
| `NUM_LAYERS` | 2 | LSTM depth |
| `DROPOUT` | 0.3 | Dropout probability |
| `EPOCHS` | 15 | Training epochs |
| `LR` | 1e-3 | Adam learning rate |
| `BATCH_SIZE` | 16 | Mini-batch size |
| `VAL_SPLIT` | 0.2 | Fraction held out for validation |

## Inference API

```python
import torch
from src.inference import load_model_and_vocab, predict
from src.model import TextClassifier

device = torch.device("cpu")
model, vocab = load_model_and_vocab(
    "checkpoints/model.pt",
    "checkpoints/vocab.json",
    TextClassifier,
    dict(vocab_size=..., embed_dim=64, hidden_dim=128,
         num_classes=..., num_layers=2, dropout=0.3),
    device,
)

label, confidence = predict("This product is fantastic!", model, vocab, device)
print(label, confidence)
# positive  {'negative': 0.02, 'neutral': 0.05, 'positive': 0.93}
```
