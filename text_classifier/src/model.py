import torch
import torch.nn as nn


class TextClassifier(nn.Module):
    """Embedding -> Bi-LSTM -> fully-connected classifier."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # bidirectional doubles the hidden state size
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        embedded = self.dropout(self.embedding(x))          # (batch, seq_len, embed_dim)
        _, (hidden, _) = self.lstm(embedded)                # hidden: (num_layers*2, batch, hidden_dim)
        # Concatenate the final forward and backward hidden states
        forward_hidden = hidden[-2]                         # (batch, hidden_dim)
        backward_hidden = hidden[-1]                        # (batch, hidden_dim)
        combined = self.dropout(torch.cat([forward_hidden, backward_hidden], dim=1))
        return self.fc(combined)                            # (batch, num_classes)
