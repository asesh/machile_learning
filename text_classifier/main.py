import os
import random

import torch

from src.dataset import Vocabulary, load_csv, make_dataloader
from src.inference import load_model_and_vocab, predict
from src.model import TextClassifier
from src.train import train

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH = "data/sample_data.csv"
MODEL_PATH = "checkpoints/model.pt"
VOCAB_PATH = "checkpoints/vocab.json"

# ── Hyperparameters ───────────────────────────────────────────────────────────
EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-3
VAL_SPLIT = 0.2
SEED = 42


def make_model(vocab: Vocabulary) -> TextClassifier:
    return TextClassifier(
        vocab_size=vocab.vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=vocab.num_classes,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"Device: {device}\n")

    os.makedirs("checkpoints", exist_ok=True)

    # ── Load & split data ─────────────────────────────────────────────────────
    texts, labels = load_csv(DATA_PATH)
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)

    split = int(len(texts) * (1 - VAL_SPLIT))
    train_texts, val_texts = list(texts[:split]), list(texts[split:])
    train_labels, val_labels = list(labels[:split]), list(labels[split:])

    # ── Build vocabulary ──────────────────────────────────────────────────────
    vocab = Vocabulary(min_freq=1)
    vocab.build(train_texts, train_labels)
    print(f"Vocab size : {vocab.vocab_size}")
    print(f"Classes    : {list(vocab.label2idx.keys())}")
    print(f"Train / Val: {len(train_texts)} / {len(val_texts)}\n")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = make_dataloader(train_texts, train_labels, vocab, BATCH_SIZE, shuffle=True)
    val_loader = make_dataloader(val_texts, val_labels, vocab, BATCH_SIZE, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = make_model(vocab)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters : {n_params:,}\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    model = train(model, train_loader, val_loader, EPOCHS, LR, device)

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), MODEL_PATH)
    vocab.save(VOCAB_PATH)
    print(f"\nSaved model → {MODEL_PATH}")
    print(f"Saved vocab → {VOCAB_PATH}")

    # ── Inference demo ────────────────────────────────────────────────────────
    print("\n── Inference demo ──────────────────────────────────────────────────")
    model_kwargs = dict(
        vocab_size=vocab.vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=vocab.num_classes,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )
    loaded_model, loaded_vocab = load_model_and_vocab(
        MODEL_PATH, VOCAB_PATH, TextClassifier, model_kwargs, device
    )

    demo_sentences = [
        "I absolutely love this product, it's amazing!",
        "This was a terrible experience, very disappointed.",
        "The item arrived on time and works as expected.",
        "Not bad, does the job but nothing spectacular.",
        "Completely broken on arrival, total waste of money.",
    ]
    for sentence in demo_sentences:
        label, conf = predict(sentence, loaded_model, loaded_vocab, device)
        top = ", ".join(f"{k}: {v:.2%}" for k, v in sorted(conf.items(), key=lambda x: -x[1]))
        print(f"  [{label:8s}]  \"{sentence[:55]}\"")
        print(f"            {top}")


if __name__ == "__main__":
    main()
