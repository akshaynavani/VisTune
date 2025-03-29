"""Ablation harness: expand a YAML grid of adapter settings into runs and execute them.

Ablation file format::

    name: vit_flowers_rank          # output folder under output_dir
    base: ../base_vit.yaml          # base experiment config (relative to this file)
    output_dir: runs/ablations      # optional
    overrides:                      # applied to every run
      train: {epochs: 5}
    grid:                           # cartesian product over dotted keys
      lora.rank: [2, 4, 8, 16]
      lora.targets: [qv, all]
    variants:                       # optional extra named runs, crossed with the grid
      - {name: lora}                #   (an empty variant keeps the grid as-is)
      - {name: linear_probe, train.strategy: linear_probe}
    seeds: [0, 1, 2]
"""

from __future__ import annotations

import copy
import itertools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import ExperimentConfig, config_from_dict, deep_merge, set_dotted
from .utils import read_json

log = logging.getLogger(__name__)

# Keys that do not affect the result of a strategy and so collapse duplicate runs.
LORA_ONLY_PREFIX = "lora."


@dataclass
class AblationSpec:
    name: str
    base: dict[str, Any]
    output_dir: str = "runs/ablations"
    overrides: dict[str, Any] = field(default_factory=dict)
    grid: dict[str, list[Any]] = field(default_factory=dict)
    variants: list[dict[str, Any]] = field(default_factory=lambda: [{}])
    seeds: list[int] = field(default_factory=lambda: [42])


def load_ablation(path: str | Path) -> AblationSpec:
    path = Path(path)
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    base: dict[str, Any] = {}
    if raw.get("base"):
        base_path = (path.parent / raw["base"]).resolve()
        with open(base_path) as fh:
            base = yaml.safe_load(fh) or {}
    return AblationSpec(
        name=raw.get("name", path.stem),
        base=base,
        output_dir=raw.get("output_dir", "runs/ablations"),
        overrides=raw.get("overrides", {}) or {},
        grid=raw.get("grid", {}) or {},
        variants=raw.get("variants") or [{}],
        seeds=raw.get("seeds") or [base.get("train", {}).get("seed", 42)],
    )


def _slug_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "-".join(_slug_value(v) for v in value) or "none"
    if value is None:
        return "none"
    return str(value).replace("/", "_").replace(" ", "")


def _short_key(dotted: str) -> str:
    return dotted.split(".")[-1]


def expand(spec: AblationSpec) -> list[ExperimentConfig]:
    """Cartesian product of grid x variants x seeds, de-duplicated by final config."""
    keys = list(spec.grid)
    combos = list(itertools.product(*(spec.grid[k] for k in keys))) if keys else [()]
    configs: list[ExperimentConfig] = []
    seen: set[str] = set()

    for variant in spec.variants:
        variant = dict(variant)
        variant_name = variant.pop("name", None)
        for combo in combos:
            for seed in spec.seeds:
                values = deep_merge(spec.base, spec.overrides)
                for key, value in variant.items():
                    set_dotted(values, key, copy.deepcopy(value))
                for key, value in zip(keys, combo):
                    set_dotted(values, key, copy.deepcopy(value))
                set_dotted(values, "train.seed", seed)

                strategy = values.get("train", {}).get("strategy", "lora")
                parts = [variant_name] if variant_name else []
                for key, value in zip(keys, combo):
                    # Adapter hyper-parameters are irrelevant for non-LoRA strategies.
                    if strategy != "lora" and key.startswith(LORA_ONLY_PREFIX):
                        continue
                    parts.append(f"{_short_key(key)}{_slug_value(value)}")
                parts.append(f"seed{seed}")
                run_name = "_".join(parts)
                if run_name in seen:
                    continue
                seen.add(run_name)

                values["name"] = run_name
                values["output_dir"] = str(Path(spec.output_dir) / spec.name)
                configs.append(config_from_dict(values))
    return configs


def is_complete(cfg: ExperimentConfig) -> bool:
    metrics = cfg.run_dir / "metrics.json"
    if not metrics.exists():
        return False
    try:
        return "final_test" in read_json(metrics)
    except Exception:
        return False


def run_ablation(spec: AblationSpec, resume: bool = True, dry_run: bool = False) -> list[dict]:
    configs = expand(spec)
    log.info("Ablation %s: %d runs", spec.name, len(configs))
    results = []
    for i, cfg in enumerate(configs, 1):
        if resume and is_complete(cfg):
            log.info("[%d/%d] %s already complete, skipping", i, len(configs), cfg.name)
            results.append(read_json(cfg.run_dir / "metrics.json"))
            continue
        log.info("[%d/%d] running %s", i, len(configs), cfg.name)
        if dry_run:
            continue
        from .engine import run_experiment  # heavy import only when actually training

        try:
            results.append(run_experiment(cfg))
        except Exception:
            log.exception("Run %s failed; continuing with the rest of the ablation", cfg.name)
    return results
