import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


def _run_epoch(
  model: nn.Module,
  loader: DataLoader,
  criterion: nn.Module,
  optimizer: Adam | None,
  device: torch.device,
) -> tuple[float, float]:
  training = optimizer is not None
  model.train(training)
  total_loss, correct, total = 0.0, 0, 0

  ctx = torch.enable_grad() if training else torch.no_grad()
  with ctx:
    for images, labels in loader:
      images, labels = images.to(device), labels.to(device)
      if training:
        optimizer.zero_grad()
      logits = model(images)
      loss = criterion(logits, labels)
      if training:
        loss.backward()
        optimizer.step()
      total_loss += loss.item() * len(labels)
      correct += (logits.argmax(dim=1) == labels).sum().item()
      total += len(labels)

  return total_loss / total, correct / total


def train(
  model: nn.Module,
  train_loader: DataLoader,
  val_loader: DataLoader,
  epochs: int,
  lr: float,
  device: torch.device,
) -> nn.Module:
  model.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
  scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

  best_val_acc = 0.0
  for epoch in range(1, epochs + 1):
    train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = _run_epoch(model, val_loader, criterion, None, device)
    scheduler.step()

    marker = " *" if val_acc > best_val_acc else ""
    best_val_acc = max(best_val_acc, val_acc)

    print(
      f"Epoch {epoch:3d}/{epochs} | "
      f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
      f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}{marker}"
    )

  return model
