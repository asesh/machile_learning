import torch
import torch.nn as nn

from .dataset import Vocabulary


def predict(
    text: str,
    model: nn.Module,
    vocab: Vocabulary,
    device: torch.device,
) -> tuple[str, dict[str, float]]:
    """Return the predicted label and a confidence dict for every class."""
    model.eval()
    tokens = vocab.encode_text(text)
    if not tokens:
        raise ValueError(f"Text '{text}' produced no tokens after preprocessing.")

    tensor = torch.tensor([tokens], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    pred_idx = int(probs.argmax().item())
    label = vocab.idx2label[pred_idx]
    confidence = {vocab.idx2label[i]: round(p.item(), 4) for i, p in enumerate(probs)}
    return label, confidence


def load_model_and_vocab(
    model_path: str,
    vocab_path: str,
    model_cls,
    model_kwargs: dict,
    device: torch.device,
) -> tuple[nn.Module, Vocabulary]:
    vocab = Vocabulary.load(vocab_path)
    model = model_cls(**model_kwargs)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, vocab
