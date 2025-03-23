"""Training / evaluation loops and the end-to-end `run_experiment` entry point."""

from __future__ import annotations

import contextlib
import logging
import math
import time
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import ExperimentConfig, TrainConfig, save_config
from .data import DataBundle, build_dataloaders
from .lora import load_partial_state_dict, trainable_state_dict
from .metrics import AccuracyMeter
from .models import VisionClassifier, apply_strategy, build_model
from .utils import resolve_device, seed_everything, write_json

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- optimisation

def _is_no_decay(name: str, param: torch.Tensor) -> bool:
    # Biases, norm weights and other 1-D tensors are conventionally excluded from decay.
    return param.ndim <= 1 or name.endswith(".bias")


def build_optimizer(model: VisionClassifier, cfg: TrainConfig) -> torch.optim.Optimizer:
    head_ids = {id(p) for p in model.head_parameters()}
    head_lr = cfg.head_lr if cfg.head_lr is not None else cfg.lr
    groups: dict[tuple[str, bool], list[torch.Tensor]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        key = ("head" if id(p) in head_ids else "body", _is_no_decay(name, p))
        groups.setdefault(key, []).append(p)
    if not groups:
        raise ValueError("Model has no trainable parameters")
    param_groups = [
        {
            "params": params,
            "lr": head_lr if part == "head" else cfg.lr,
            "weight_decay": 0.0 if no_decay else cfg.weight_decay,
            "name": f"{part}{'_nodecay' if no_decay else ''}",
        }
        for (part, no_decay), params in groups.items()
    ]
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.999))


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int, min_ratio: float = 0.0):
    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class Autocast:
    """Mixed precision helper: bf16 where supported, fp16 + GradScaler otherwise, CUDA only."""

    def __init__(self, device: torch.device, enabled: bool):
        self.enabled = enabled and device.type == "cuda"
        self.dtype = torch.bfloat16 if self.enabled and torch.cuda.is_bf16_supported() else torch.float16
        self.device_type = device.type
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.enabled and self.dtype == torch.float16)

    def context(self):
        if not self.enabled:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device_type, dtype=self.dtype)


# --------------------------------------------------------------------------- loops

def train_one_epoch(model, loader: DataLoader, optimizer, scheduler, criterion, amp: Autocast,
                    device, num_classes: int, grad_clip: float | None, epoch: int) -> dict:
    model.train()
    meter = AccuracyMeter(num_classes)
    trainable = [p for p in model.parameters() if p.requires_grad]
    bar = tqdm(loader, desc=f"train {epoch}", leave=False)
    for images, targets in bar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with amp.context():
            logits = model(images)
            loss = criterion(logits.float(), targets)
        optimizer.zero_grad(set_to_none=True)
        amp.scaler.scale(loss).backward()
        if grad_clip:
            amp.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        amp.scaler.step(optimizer)
        amp.scaler.update()
        scheduler.step()
        meter.update(logits, targets, loss.item())
        bar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
    return meter.compute()


@torch.no_grad()
def evaluate(model, loader: DataLoader, device, num_classes: int, amp: Autocast | None = None,
             desc: str = "eval") -> dict:
    model.eval()
    meter = AccuracyMeter(num_classes)
    criterion = nn.CrossEntropyLoss()
    ctx = amp.context if amp is not None else contextlib.nullcontext
    for images, targets in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with ctx():
            logits = model(images)
        meter.update(logits, targets, criterion(logits.float(), targets).item())
    return meter.compute()


# --------------------------------------------------------------------------- orchestration

def prepare(cfg: ExperimentConfig) -> tuple[VisionClassifier, DataBundle, dict, torch.device]:
    seed_everything(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    data = build_dataloaders(cfg.data, cfg.model, seed=cfg.train.seed)
    model = build_model(cfg.model, data.class_names, device=device)
    summary = apply_strategy(model, cfg.train.strategy, cfg.lora)
    return model.to(device), data, summary, device


def run_experiment(cfg: ExperimentConfig) -> dict[str, Any]:
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")

    model, data, summary, device = prepare(cfg)
    amp = Autocast(device, cfg.train.amp)
    tc = cfg.train
    results: dict[str, Any] = {
        "name": cfg.name,
        "config": cfg.to_dict(),
        "params": summary["params"],
        "num_adapted_modules": len(summary["adapted_modules"]),
        "num_classes": data.num_classes,
        "history": [],
    }

    # Accuracy before any training: zero-shot for CLIP with a text head, chance level otherwise.
    results["initial_test"] = evaluate(model, data.test, device, data.num_classes, amp, "initial")
    log.info("initial test top1=%.2f top5=%.2f", results["initial_test"]["top1"],
             results["initial_test"]["top5"])

    if tc.strategy == "zero_shot" or tc.epochs <= 0:
        results["final_test"] = results["initial_test"]
        results["best_test"] = results["initial_test"]
        write_json(results, run_dir / "metrics.json")
        return results

    optimizer = build_optimizer(model, tc)
    criterion = nn.CrossEntropyLoss(label_smoothing=tc.label_smoothing)
    steps_per_epoch = len(data.train)
    total_steps = steps_per_epoch * tc.epochs
    scheduler = cosine_with_warmup(optimizer, int(tc.warmup_epochs * steps_per_epoch), total_steps)

    select_loader = data.val if data.val is not None else data.test
    select_name = "val" if data.val is not None else "test"
    best_score, best_epoch = -1.0, -1
    start = time.time()

    for epoch in range(1, tc.epochs + 1):
        train_stats = train_one_epoch(model, data.train, optimizer, scheduler, criterion, amp,
                                      device, data.num_classes, tc.grad_clip, epoch)
        record = {"epoch": epoch, "train": train_stats}
        if epoch % tc.eval_every == 0 or epoch == tc.epochs:
            sel = evaluate(model, select_loader, device, data.num_classes, amp, select_name)
            record[select_name] = sel
            if sel["top1"] > best_score:
                best_score, best_epoch = sel["top1"], epoch
                torch.save(trainable_state_dict(model), run_dir / "best_trainable.pt")
            log.info("epoch %d train_loss=%.4f %s_top1=%.2f %s_top5=%.2f", epoch,
                     train_stats["loss"], select_name, sel["top1"], select_name, sel["top5"])
        results["history"].append(record)
        write_json(results, run_dir / "metrics.json")

    results["train_seconds"] = time.time() - start
    torch.save(trainable_state_dict(model), run_dir / "final_trainable.pt")
    results["final_test"] = evaluate(model, data.test, device, data.num_classes, amp, "test")

    # Reload the selected checkpoint for the "best" numbers. When no validation split exists,
    # selection happened on the test set, which is optimistic; this is flagged in the output.
    load_partial_state_dict(model, torch.load(run_dir / "best_trainable.pt", map_location=device))
    results["best_epoch"] = best_epoch
    results["selected_on"] = select_name
    results["best_test"] = (results["history"][best_epoch - 1].get("test")
                            if select_name == "test"
                            else evaluate(model, data.test, device, data.num_classes, amp, "test"))
    write_json(results, run_dir / "metrics.json")
    log.info("done: final top1=%.2f top5=%.2f | best(%s, epoch %d) top1=%.2f",
             results["final_test"]["top1"], results["final_test"]["top5"], select_name,
             best_epoch, results["best_test"]["top1"])
    return results


def evaluate_checkpoint(cfg: ExperimentConfig, checkpoint: str | None) -> dict[str, Any]:
    """Rebuild the model for `cfg`, load a trainable-state checkpoint and evaluate on test."""
    model, data, _, device = prepare(cfg)
    if checkpoint:
        load_partial_state_dict(model, torch.load(checkpoint, map_location=device))
    return evaluate(model, data.test, device, data.num_classes, Autocast(device, cfg.train.amp))
