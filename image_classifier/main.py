import os
import torch

from src.dataset import FASHION_MNIST_CLASSES, get_dataloaders
from src.inference import load_model, predict_from_loader
from src.model import ImageClassifier
from src.train import train

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = "data"
MODEL_PATH = "checkpoints/model.pt"

# ── Hyperparameters ───────────────────────────────────────────────────────────
NUM_CLASSES = 10
DROPOUT = 0.4
BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-3
VAL_SPLIT = 0.1
SEED = 42


def main() -> None:
  torch.manual_seed(SEED)
  device = torch.device("mps" if torch.mps.is_available() else "cpu")
  print(f"Device: {device}\n")

  os.makedirs("checkpoints", exist_ok=True)

  # ── Data ───────────────────────────────────────────────────────────────────
  train_loader, val_loader, test_loader = get_dataloaders(
    DATA_DIR, batch_size=BATCH_SIZE, val_split=VAL_SPLIT, seed=SEED
  )
  print(f"Train batches : {len(train_loader)}")
  print(f"Val   batches : {len(val_loader)}")
  print(f"Test  batches : {len(test_loader)}")
  print(f"Classes       : {FASHION_MNIST_CLASSES}\n")

  # ── Model ──────────────────────────────────────────────────────────────────
  model = ImageClassifier(num_classes=NUM_CLASSES, dropout=DROPOUT)
  n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f"Parameters : {n_params:,}\n")

  # ── Train ──────────────────────────────────────────────────────────────────
  model = train(model, train_loader, val_loader, EPOCHS, LR, device)

  # ── Evaluate on test set ───────────────────────────────────────────────────
  model.eval()
  correct, total = 0, 0
  with torch.no_grad():
    for images, labels in test_loader:
      images, labels = images.to(device), labels.to(device)
      preds = model(images).argmax(dim=1)
      correct += (preds == labels).sum().item()
      total += len(labels)
  print(f"\nTest Accuracy: {correct / total:.4f}  ({correct}/{total})")

  # ── Save ───────────────────────────────────────────────────────────────────
  torch.save(model.state_dict(), MODEL_PATH)
  print(f"Saved model → {MODEL_PATH}")

  # ── Inference demo ─────────────────────────────────────────────────────────
  print("\n── Inference demo (first 5 test images) ───────────────────────────────")
  model_kwargs = dict(num_classes=NUM_CLASSES, dropout=DROPOUT)
  loaded_model = load_model(MODEL_PATH, ImageClassifier, model_kwargs, device)
  predict_from_loader(loaded_model, test_loader, device)


if __name__ == "__main__":
  main()
