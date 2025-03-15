"""Experiment configuration: dataclasses, YAML loading and dotted-key overrides."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    name: str = "flowers102"  # registry key, or "imagefolder" / "fake"
    root: str = "data"
    train_split: str = "train"
    eval_split: str = "test"
    batch_size: int = 64
    num_workers: int = 4
    # Limit the number of training samples per class (few-shot experiments). None = all.
    shots_per_class: int | None = None
    augment: bool = True


@dataclass
class ModelConfig:
    family: str = "vit"  # "vit" or "clip"
    checkpoint: str = "google/vit-base-patch16-224-in21k"
    # CLIP only: initialise the classification head from text prompt embeddings.
    zero_shot_head: bool = True
    prompt_template: str = "a photo of a {}."


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.1
    # Preset name ("qv", "attn", "mlp", "all") or explicit list of module-name suffixes.
    targets: Any = "qv"
    # Restrict adapters to these encoder layer indices. None = all layers; negative ints
    # count from the end (e.g. [-4, -3, -2, -1] = last four blocks).
    layers: list[int] | None = None


@dataclass
class TrainConfig:
    # "linear_probe" (head only), "lora" (adapters + head), "full" (everything).
    strategy: str = "lora"
    epochs: int = 10
    lr: float = 1e-3
    head_lr: float | None = None  # defaults to lr
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    label_smoothing: float = 0.1
    grad_clip: float | None = 1.0
    amp: bool = True
    seed: int = 42
    device: str = "auto"
    eval_every: int = 1


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    output_dir: str = "runs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls, values: dict[str, Any]):
    """Recursively build a dataclass from a dict, rejecting unknown keys."""
    known = {f.name: f for f in fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise KeyError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    defaults = cls()
    for name, f in known.items():
        if name not in values:
            continue
        default_value = getattr(defaults, name)
        value = values[name]
        if is_dataclass(default_value) and isinstance(value, dict):
            kwargs[name] = _build(type(default_value), value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def set_dotted(d: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = d
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def parse_overrides(overrides: list[str] | None) -> dict[str, Any]:
    """Turn ["train.lr=3e-4", "lora.rank=4"] into a nested dict (values parsed as YAML)."""
    result: dict[str, Any] = {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        key, raw = item.split("=", 1)
        set_dotted(result, key.strip(), yaml.safe_load(raw))
    return result


def config_from_dict(values: dict[str, Any]) -> ExperimentConfig:
    return _build(ExperimentConfig, values)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> ExperimentConfig:
    values: dict[str, Any] = {}
    if path is not None:
        with open(path) as fh:
            values = yaml.safe_load(fh) or {}
    values = deep_merge(values, parse_overrides(overrides))
    return config_from_dict(values)


def save_config(cfg: ExperimentConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False)
