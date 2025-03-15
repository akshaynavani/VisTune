"""Dataset registry, backbone-aware transforms and loaders."""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .config import DataConfig, ModelConfig

log = logging.getLogger(__name__)

# Fallback normalisation stats if the image processor cannot be fetched.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
VIT_MEAN = (0.5, 0.5, 0.5)
VIT_STD = (0.5, 0.5, 0.5)


@dataclass
class PreprocessSpec:
    image_size: int
    mean: tuple[float, ...]
    std: tuple[float, ...]


def preprocess_spec(model_cfg: ModelConfig) -> PreprocessSpec:
    """Read image size and normalisation from the checkpoint's image processor."""
    default = PreprocessSpec(
        224,
        CLIP_MEAN if model_cfg.family == "clip" else VIT_MEAN,
        CLIP_STD if model_cfg.family == "clip" else VIT_STD,
    )
    try:
        from transformers import AutoImageProcessor

        proc = AutoImageProcessor.from_pretrained(model_cfg.checkpoint)
    except Exception as exc:  # offline, unknown checkpoint, etc.
        log.warning("Falling back to default preprocessing for %s (%s)", model_cfg.checkpoint, exc)
        return default

    size = getattr(proc, "crop_size", None) or getattr(proc, "size", None) or {}
    if isinstance(size, dict):
        image_size = size.get("height") or size.get("shortest_edge") or default.image_size
    else:
        image_size = int(size)
    mean = tuple(getattr(proc, "image_mean", None) or default.mean)
    std = tuple(getattr(proc, "image_std", None) or default.std)
    return PreprocessSpec(int(image_size), mean, std)


def build_transforms(spec: PreprocessSpec, train: bool, augment: bool = True) -> Callable:
    normalize = transforms.Normalize(spec.mean, spec.std)
    if train and augment:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(spec.image_size, scale=(0.6, 1.0),
                                             interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(spec.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(spec.image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


# --------------------------------------------------------------------------- registry

def _flowers(root, split, tf):
    split = {"train": "train", "val": "val", "test": "test"}[split]
    return datasets.Flowers102(root, split=split, transform=tf, download=True)


def _pets(root, split, tf):
    split = "trainval" if split in ("train", "trainval") else "test"
    return datasets.OxfordIIITPet(root, split=split, transform=tf, download=True)


def _dtd(root, split, tf):
    return datasets.DTD(root, split=split, transform=tf, download=True)


def _aircraft(root, split, tf):
    split = "trainval" if split in ("train", "trainval") else split
    return datasets.FGVCAircraft(root, split=split, transform=tf, download=True)


def _eurosat(root, split, tf):
    # EuroSAT ships without official splits: use a deterministic 80/20 split.
    full = datasets.EuroSAT(root, transform=tf, download=True)
    indices = list(range(len(full)))
    random.Random(0).shuffle(indices)
    cut = int(0.8 * len(indices))
    chosen = indices[:cut] if split == "train" else indices[cut:]
    subset = Subset(full, chosen)
    subset.classes = full.classes  # type: ignore[attr-defined]
    return subset


def _imagefolder(root, split, tf):
    return datasets.ImageFolder(str(Path(root) / split), transform=tf)


def _fake(root, split, tf):
    size = 256 if split == "train" else 64
    ds = datasets.FakeData(size=size, image_size=(3, 224, 224), num_classes=10,
                           transform=tf, random_offset=0 if split == "train" else 10_000)
    ds.classes = [f"class {i}" for i in range(10)]  # type: ignore[attr-defined]
    return ds


DATASETS: dict[str, Callable] = {
    "flowers102": _flowers,
    "pets": _pets,
    "dtd": _dtd,
    "aircraft": _aircraft,
    "eurosat": _eurosat,
    "imagefolder": _imagefolder,
    "fake": _fake,
}


def build_dataset(name: str, root: str, split: str, tf: Callable) -> Dataset:
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}. Available: {sorted(DATASETS)}")
    return DATASETS[name](root, split, tf)


# --------------------------------------------------------------------------- helpers

def dataset_labels(ds: Dataset) -> list[int]:
    """Best-effort extraction of integer labels without decoding images."""
    if isinstance(ds, Subset):
        parent = dataset_labels(ds.dataset)
        return [parent[i] for i in ds.indices]
    for attr in ("targets", "_labels", "_label_ids"):
        if hasattr(ds, attr):
            return [int(x) for x in getattr(ds, attr)]
    if hasattr(ds, "samples"):
        return [int(t) for _, t in ds.samples]
    return [int(ds[i][1]) for i in range(len(ds))]  # slow fallback


def class_names(ds: Dataset, names_file: str | None = None) -> list[str]:
    if names_file:
        with open(names_file) as fh:
            return [line.strip() for line in fh if line.strip()]
    base = ds.dataset if isinstance(ds, Subset) and not hasattr(ds, "classes") else ds
    names = getattr(base, "classes", None)
    if names:
        return [str(n).replace("_", " ") for n in names]
    n = max(dataset_labels(ds)) + 1
    log.warning("Dataset exposes no class names; using generic names (CLIP prompts will be weak)")
    return [f"class {i}" for i in range(n)]


def few_shot_subset(ds: Dataset, shots: int, seed: int) -> Subset:
    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(dataset_labels(ds)):
        by_class[label].append(idx)
    rng = random.Random(seed)
    chosen: list[int] = []
    for label in sorted(by_class):
        idxs = by_class[label]
        rng.shuffle(idxs)
        chosen.extend(idxs[:shots])
    return Subset(ds, sorted(chosen))


def train_val_split(ds: Dataset, val_fraction: float, seed: int) -> tuple[Subset, Subset]:
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    n_val = max(1, int(len(indices) * val_fraction))
    return Subset(ds, sorted(indices[n_val:])), Subset(ds, sorted(indices[:n_val]))


@dataclass
class DataBundle:
    train: DataLoader
    val: DataLoader | None
    test: DataLoader
    class_names: list[str]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def build_dataloaders(data_cfg: DataConfig, model_cfg: ModelConfig, seed: int = 42) -> DataBundle:
    spec = preprocess_spec(model_cfg)
    train_tf = build_transforms(spec, train=True, augment=data_cfg.augment)
    eval_tf = build_transforms(spec, train=False)

    train_ds = build_dataset(data_cfg.name, data_cfg.root, data_cfg.train_split, train_tf)
    test_ds = build_dataset(data_cfg.name, data_cfg.root, data_cfg.eval_split, eval_tf)
    names = class_names(train_ds, data_cfg.class_names_file)

    val_ds = None
    if data_cfg.val_fraction > 0:
        # Validation images should not be augmented: re-open the split with eval transforms
        # and select the same indices.
        train_sub, val_sub = train_val_split(train_ds, data_cfg.val_fraction, seed)
        eval_view = build_dataset(data_cfg.name, data_cfg.root, data_cfg.train_split, eval_tf)
        val_ds = Subset(eval_view, val_sub.indices)
        train_ds = train_sub
    if data_cfg.shots_per_class:
        train_ds = few_shot_subset(train_ds, data_cfg.shots_per_class, seed)

    pin = torch.cuda.is_available()

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=data_cfg.batch_size, shuffle=shuffle,
                          num_workers=data_cfg.num_workers, pin_memory=pin,
                          drop_last=False, persistent_workers=data_cfg.num_workers > 0)

    return DataBundle(
        train=loader(train_ds, True),
        val=loader(val_ds, False) if val_ds is not None else None,
        test=loader(test_ds, False),
        class_names=names,
    )
