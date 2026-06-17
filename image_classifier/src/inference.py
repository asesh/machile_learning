import torch
import torch.nn as nn
from torchvision import transforms

from .dataset import FASHION_MNIST_CLASSES, make_transforms


def load_model(model_path: str, model_cls, model_kwargs: dict, device: torch.device) -> nn.Module:
  model = model_cls(**model_kwargs)
  model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
  model.to(device)
  model.eval()
  return model


def predict(
  image: torch.Tensor,
  model: nn.Module,
  device: torch.device,
  class_names: list[str] = FASHION_MNIST_CLASSES,
) -> tuple[str, dict[str, float]]:
  """
  Predict the class of a single image tensor (C, H, W) or (1, C, H, W).
  Returns the top label and a per-class probability dict.
  """
  model.eval()
  if image.dim() == 3:
    image = image.unsqueeze(0)
  image = image.to(device)

  with torch.no_grad():
    logits = model(image)
    probs = torch.softmax(logits, dim=1).squeeze(0)

  pred_idx = int(probs.argmax().item())
  label = class_names[pred_idx]
  confidence = {class_names[i]: round(p.item(), 4) for i, p in enumerate(probs)}
  return label, confidence


def predict_from_loader(
  model: nn.Module,
  loader,
  device: torch.device,
  class_names: list[str] = FASHION_MNIST_CLASSES,
  n: int = 5,
) -> None:
  """Print predictions for the first n images in a DataLoader."""
  model.eval()
  images, labels = next(iter(loader))
  images, labels = images[:n].to(device), labels[:n]

  with torch.no_grad():
    logits = model(images)
    probs = torch.softmax(logits, dim=1)

  for i in range(n):
    pred = int(probs[i].argmax().item())
    true = int(labels[i].item())
    conf = probs[i][pred].item()
    pred_name = class_names[pred]
    true_name = class_names[true]
    match = "OK" if pred == true else "WRONG"
    print(f"  [{match}] Predicted: {pred_name:12s} ({conf:.2%})  True: {true_name}")
