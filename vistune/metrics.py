"""Accuracy metrics (top-k, per-class) with streaming accumulation."""

from __future__ import annotations

import torch


def topk_correct(logits: torch.Tensor, targets: torch.Tensor, ks=(1, 5)) -> dict[int, int]:
    """Number of samples whose target is within the top-k predictions, for each k."""
    max_k = min(max(ks), logits.size(1))
    _, pred = logits.topk(max_k, dim=1)
    hits = pred.eq(targets.view(-1, 1))
    return {k: int(hits[:, : min(k, max_k)].any(dim=1).sum().item()) for k in ks}


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, ks=(1, 5)) -> dict[int, float]:
    correct = topk_correct(logits, targets, ks)
    n = max(targets.numel(), 1)
    return {k: 100.0 * c / n for k, c in correct.items()}


class AccuracyMeter:
    """Accumulates top-k hits, per-class top-1 and mean loss across batches."""

    def __init__(self, num_classes: int, ks=(1, 5)):
        self.num_classes = num_classes
        self.ks = tuple(ks)
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.loss_sum = 0.0
        self.correct = {k: 0 for k in self.ks}
        self.class_total = torch.zeros(self.num_classes, dtype=torch.long)
        self.class_correct = torch.zeros(self.num_classes, dtype=torch.long)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: float | None = None) -> None:
        logits = logits.detach().float().cpu()
        targets = targets.detach().cpu()
        batch = targets.numel()
        self.n += batch
        if loss is not None:
            self.loss_sum += float(loss) * batch
        for k, c in topk_correct(logits, targets, self.ks).items():
            self.correct[k] += c
        pred = logits.argmax(dim=1)
        self.class_total += torch.bincount(targets, minlength=self.num_classes)
        self.class_correct += torch.bincount(targets[pred == targets], minlength=self.num_classes)

    def compute(self) -> dict[str, float]:
        n = max(self.n, 1)
        out = {f"top{k}": 100.0 * self.correct[k] / n for k in self.ks}
        out["loss"] = self.loss_sum / n
        seen = self.class_total > 0
        if seen.any():
            per_class = self.class_correct[seen].float() / self.class_total[seen].float()
            out["mean_per_class"] = 100.0 * per_class.mean().item()
        out["samples"] = self.n
        return out
