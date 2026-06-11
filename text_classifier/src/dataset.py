import csv
import json
import re
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def load_csv(path: str) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"])
            labels.append(row["label"])
    return texts, labels


def tokenize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).split()


class Vocabulary:
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.word2idx: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.idx2word: dict[int, str] = {0: PAD_TOKEN, 1: UNK_TOKEN}
        self.label2idx: dict[str, int] = {}
        self.idx2label: dict[int, str] = {}

    def build(self, texts: list[str], labels: list[str]) -> None:
        counter: Counter = Counter()
        for text in texts:
            counter.update(tokenize(text))
        for word, freq in counter.items():
            if freq >= self.min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        for i, label in enumerate(sorted(set(labels))):
            self.label2idx[label] = i
            self.idx2label[i] = label

    def encode_text(self, text: str) -> list[int]:
        return [self.word2idx.get(w, 1) for w in tokenize(text)]

    def encode_label(self, label: str) -> int:
        return self.label2idx[label]

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

    @property
    def num_classes(self) -> int:
        return len(self.label2idx)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(
                {
                    "word2idx": self.word2idx,
                    "label2idx": self.label2idx,
                    "min_freq": self.min_freq,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path) as f:
            data = json.load(f)
        vocab = cls(min_freq=data["min_freq"])
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(v): k for k, v in data["word2idx"].items()}
        vocab.label2idx = data["label2idx"]
        vocab.idx2label = {int(v): k for k, v in data["label2idx"].items()}
        return vocab


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[str], vocab: Vocabulary):
        self.sequences = [vocab.encode_text(t) for t in texts]
        self.labels = [vocab.encode_label(l) for l in labels]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        return self.sequences[idx], self.labels[idx]


def _collate_fn(batch: list[tuple[list[int], int]]) -> tuple[torch.Tensor, torch.Tensor]:
    sequences, labels = zip(*batch)
    max_len = max(len(s) for s in sequences)
    padded = [s + [0] * (max_len - len(s)) for s in sequences]
    return torch.tensor(padded, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def make_dataloader(
    texts: list[str],
    labels: list[str],
    vocab: Vocabulary,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    dataset = TextDataset(texts, labels, vocab)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_fn)
