"""A from-scratch LoRA implementation for nn.Linear layers.

LoRA (Hu et al., 2021) freezes a pretrained weight W and learns a low-rank update
delta_W = B @ A with A in R^{r x in}, B in R^{out x r}. The forward pass becomes

    y = x W^T + b + (alpha / r) * dropout(x) A^T B^T

B is zero-initialised, so the adapted model is identical to the pretrained one at step 0.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear with a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        factory = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, **factory))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, **factory))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.merged = False

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> torch.Tensor:  # some HF code paths read .weight directly
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.merged:
            return out
        lora = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return out + lora * self.scaling

    @torch.no_grad()
    def merge(self) -> None:
        """Fold the adapter into the base weight for zero-overhead inference."""
        if not self.merged:
            self.base.weight.add_(self.delta_weight().to(self.base.weight.dtype))
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if self.merged:
            self.base.weight.sub_(self.delta_weight().to(self.base.weight.dtype))
            self.merged = False

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, rank={self.rank}, "
                f"alpha={self.alpha}, merged={self.merged}")


# --------------------------------------------------------------------------- injection

_LAYER_INDEX = re.compile(r"\.(?:layer|layers)\.(\d+)\.")


def layer_index(module_name: str) -> int | None:
    """Extract the transformer block index from a module path, e.g. 'encoder.layer.7.x' -> 7."""
    match = _LAYER_INDEX.search(f".{module_name}.")
    return int(match.group(1)) if match else None


def _normalise_layers(layers: Iterable[int] | None, num_layers: int | None) -> set[int] | None:
    if layers is None:
        return None
    resolved = set()
    for idx in layers:
        if idx < 0:
            if num_layers is None:
                raise ValueError("Negative layer indices need the number of encoder layers")
            idx = num_layers + idx
        resolved.add(idx)
    return resolved


def _matches(name: str, targets: Iterable[str]) -> bool:
    return any(name == t or name.endswith("." + t) for t in targets)


def _set_submodule(root: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child, new)


@dataclass
class InjectionReport:
    replaced: list[str]

    def __len__(self) -> int:
        return len(self.replaced)


def inject_lora(
    model: nn.Module,
    targets: Iterable[str],
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    layers: Iterable[int] | None = None,
    num_layers: int | None = None,
    exclude: Iterable[str] = (),
) -> InjectionReport:
    """Replace every nn.Linear whose name ends with one of `targets` by a LoRALinear.

    `targets` are module-name suffixes such as "attention.query" or "q_proj". If `layers` is
    given, only modules inside those transformer blocks are adapted.
    """
    targets = list(targets)
    exclude = list(exclude)
    allowed_layers = _normalise_layers(layers, num_layers)
    to_replace = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or isinstance(module, LoRALinear):
            continue
        if not _matches(name, targets) or any(e in name for e in exclude):
            continue
        if allowed_layers is not None:
            idx = layer_index(name)
            if idx is None or idx not in allowed_layers:
                continue
        to_replace.append((name, module))

    if not to_replace:
        raise ValueError(f"No nn.Linear modules matched targets={targets} layers={layers}")

    for name, module in to_replace:
        _set_submodule(model, name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
    return InjectionReport([name for name, _ in to_replace])


def lora_modules(model: nn.Module) -> list[tuple[str, LoRALinear]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]


def merge_lora(model: nn.Module) -> None:
    for _, m in lora_modules(model):
        m.merge()


def unmerge_lora(model: nn.Module) -> None:
    for _, m in lora_modules(model):
        m.unmerge()


# --------------------------------------------------------------------------- freezing / stats

def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def mark_lora_trainable(model: nn.Module) -> None:
    for _, m in lora_modules(model):
        m.lora_A.requires_grad_(True)
        m.lora_B.requires_grad_(True)


def count_parameters(model: nn.Module) -> dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora = sum(m.lora_A.numel() + m.lora_B.numel() for _, m in lora_modules(model))
    return {
        "total": total,
        "trainable": trainable,
        "lora": lora,
        "trainable_pct": 100.0 * trainable / max(total, 1),
    }


# --------------------------------------------------------------------------- checkpoints

def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if k.endswith("lora_A") or k.endswith("lora_B")}


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Everything with requires_grad (adapters + head), i.e. what a run actually learned."""
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in trainable}


def load_partial_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    missing_in_model = set(state) - set(model.state_dict())
    if missing_in_model:
        raise KeyError(f"Checkpoint keys not present in model: {sorted(missing_in_model)[:5]}...")
    model.load_state_dict(state, strict=False)
