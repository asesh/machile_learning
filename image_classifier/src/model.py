import torch
import torch.nn as nn


class ConvBlock(nn.Module):
  """Conv2d -> BatchNorm -> ReLU -> optional MaxPool."""

  def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
    super().__init__()
    layers: list[nn.Module] = [
      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
    ]
    if pool:
      layers.append(nn.MaxPool2d(2))
    self.block = nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.block(x)


class ImageClassifier(nn.Module):
  """
  CNN for 1-channel 28x28 images (FashionMNIST).

  Architecture:
    ConvBlock(1  -> 32, pool)  -> 32x14x14
    ConvBlock(32 -> 64, pool)  -> 64x7x7
    ConvBlock(64 -> 128)       -> 128x7x7
    Flatten -> FC(6272, 256) -> Dropout -> FC(256, num_classes)
  """

  def __init__(self, num_classes: int = 10, dropout: float = 0.4):
    super().__init__()
    self.features = nn.Sequential(
      ConvBlock(1, 32, pool=True),
      ConvBlock(32, 64, pool=True),
      ConvBlock(64, 128, pool=False),
    )
    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(128 * 7 * 7, 256),
      nn.ReLU(inplace=True),
      nn.Dropout(dropout),
      nn.Linear(256, num_classes),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.classifier(self.features(x))
