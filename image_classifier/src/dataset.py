import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

FASHION_MNIST_CLASSES = [
  "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def make_transforms(augment: bool = False) -> transforms.Compose:
  base = [
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),  # FashionMNIST channel mean/std
  ]
  if augment:
    base = [
      transforms.RandomHorizontalFlip(),
      transforms.RandomCrop(28, padding=4),
    ] + base
  return transforms.Compose(base)


def get_dataloaders(
  data_dir: str,
  batch_size: int = 64,
  val_split: float = 0.1,
  seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
  train_dataset = datasets.FashionMNIST(
    root=data_dir, train=True, download=True, transform=make_transforms(augment=True)
  )
  test_dataset = datasets.FashionMNIST(
    root=data_dir, train=False, download=True, transform=make_transforms(augment=False)
  )

  val_size = int(len(train_dataset) * val_split)
  train_size = len(train_dataset) - val_size
  generator = torch.Generator().manual_seed(seed)
  train_subset, val_subset = random_split(train_dataset, [train_size, val_size], generator=generator)

  # Val uses no-augment transforms
  val_subset.dataset = datasets.FashionMNIST(
    root=data_dir, train=True, download=False, transform=make_transforms(augment=False)
  )

  train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
  val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

  return train_loader, val_loader, test_loader
