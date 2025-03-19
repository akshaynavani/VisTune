"""CLIP text-prompt utilities used to build zero-shot classification heads."""

from __future__ import annotations

import torch
from torch import nn

# A small prompt ensemble in the spirit of the CLIP paper. Averaging normalised embeddings over
# several phrasings is a cheap, consistent zero-shot improvement.
DEFAULT_TEMPLATES = [
    "a photo of a {}.",
    "a close-up photo of a {}.",
    "a cropped photo of the {}.",
    "a good photo of a {}.",
    "a bright photo of a {}.",
]


def render_prompts(class_names: list[str], templates: list[str]) -> list[list[str]]:
    return [[t.format(name) for t in templates] for name in class_names]


@torch.no_grad()
def text_classifier_weights(
    clip_model: nn.Module,
    tokenizer,
    class_names: list[str],
    templates: list[str],
    device: torch.device | str = "cpu",
    batch_size: int = 256,
) -> torch.Tensor:
    """Return a [num_classes, embed_dim] matrix of L2-normalised class text embeddings."""
    clip_model = clip_model.to(device).eval()
    weights = []
    for prompts in render_prompts(class_names, templates):
        embs = []
        for start in range(0, len(prompts), batch_size):
            batch = tokenizer(prompts[start:start + batch_size], padding=True,
                              truncation=True, return_tensors="pt").to(device)
            embs.append(clip_model.get_text_features(**batch))
        emb = torch.cat(embs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        mean = emb.mean(dim=0)
        weights.append(mean / mean.norm())
    return torch.stack(weights).cpu()
