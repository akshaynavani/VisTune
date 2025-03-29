"""Aggregate run metrics into tables and an accuracy-vs-parameters plot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .utils import read_json

GROUP_COLUMNS = ["family", "strategy", "targets", "rank", "alpha", "layers", "shots"]


def _fmt_targets(targets: Any) -> str:
    if isinstance(targets, (list, tuple)):
        return "+".join(map(str, targets))
    return str(targets)


def flatten(metrics: dict[str, Any]) -> dict[str, Any]:
    cfg = metrics["config"]
    strategy = cfg["train"]["strategy"]
    is_lora = strategy == "lora"
    best = metrics.get("best_test") or {}
    final = metrics.get("final_test") or {}
    initial = metrics.get("initial_test") or {}
    return {
        "run": metrics["name"],
        "family": cfg["model"]["family"],
        "checkpoint": cfg["model"]["checkpoint"],
        "dataset": cfg["data"]["name"],
        "shots": cfg["data"].get("shots_per_class") or "all",
        "strategy": strategy,
        "targets": _fmt_targets(cfg["lora"]["targets"]) if is_lora else "-",
        "rank": cfg["lora"]["rank"] if is_lora else "-",
        "alpha": cfg["lora"]["alpha"] if is_lora else "-",
        "layers": _fmt_targets(cfg["lora"]["layers"]) if is_lora and cfg["lora"]["layers"] else "all",
        "seed": cfg["train"]["seed"],
        "trainable_params": metrics["params"]["trainable"],
        "trainable_pct": metrics["params"]["trainable_pct"],
        "initial_top1": initial.get("top1"),
        "final_top1": final.get("top1"),
        "final_top5": final.get("top5"),
        "best_top1": best.get("top1"),
        "best_top5": best.get("top5"),
        "selected_on": metrics.get("selected_on", "-"),
        "train_seconds": metrics.get("train_seconds"),
    }


def collect(root: str | Path) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(root).rglob("metrics.json")):
        metrics = read_json(path)
        if "final_test" in metrics:
            rows.append(flatten(metrics))
    if not rows:
        raise FileNotFoundError(f"No completed runs (metrics.json with final_test) under {root}")
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame, metric: str = "final") -> pd.DataFrame:
    """Mean and std over seeds for each configuration, sorted by top-1."""
    df = df.copy()
    df[GROUP_COLUMNS] = df[GROUP_COLUMNS].astype(str)
    agg = df.groupby(GROUP_COLUMNS, dropna=False).agg(
        runs=("seed", "count"),
        trainable_params=("trainable_params", "first"),
        trainable_pct=("trainable_pct", "first"),
        top1_mean=(f"{metric}_top1", "mean"),
        top1_std=(f"{metric}_top1", "std"),
        top5_mean=(f"{metric}_top5", "mean"),
        top5_std=(f"{metric}_top5", "std"),
    ).reset_index()
    agg[["top1_std", "top5_std"]] = agg[["top1_std", "top5_std"]].fillna(0.0)
    return agg.sort_values("top1_mean", ascending=False).reset_index(drop=True)


def to_markdown(agg: pd.DataFrame) -> str:
    header = ("| Family | Strategy | Targets | Rank | Alpha | Layers | Shots | Trainable params "
              "| Trainable % | Top-1 | Top-5 | Runs |")
    sep = "|" + "---|" * 12
    lines = [header, sep]
    for r in agg.itertuples(index=False):
        lines.append(
            f"| {r.family} | {r.strategy} | {r.targets} | {r.rank} | {r.alpha} | {r.layers} "
            f"| {r.shots} | {int(r.trainable_params):,} | {r.trainable_pct:.3f} "
            f"| {r.top1_mean:.2f} ± {r.top1_std:.2f} | {r.top5_mean:.2f} ± {r.top5_std:.2f} "
            f"| {r.runs} |"
        )
    return "\n".join(lines) + "\n"


def plot_accuracy_vs_params(agg: pd.DataFrame, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for strategy, group in agg.groupby("strategy"):
        ax.errorbar(group["trainable_params"], group["top1_mean"], yerr=group["top1_std"],
                    fmt="o", capsize=3, label=strategy)
        if strategy == "lora":
            for r in group.itertuples(index=False):
                ax.annotate(f"r{r.rank}/{r.targets}", (r.trainable_params, r.top1_mean),
                            textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("Accuracy vs. trainable parameters")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(root: str | Path, metric: str = "final", plot: bool = True) -> dict[str, Path]:
    root = Path(root)
    runs = collect(root)
    agg = aggregate(runs, metric)
    outputs = {
        "runs_csv": root / "runs.csv",
        "summary_csv": root / "summary.csv",
        "summary_md": root / "summary.md",
    }
    runs.to_csv(outputs["runs_csv"], index=False)
    agg.to_csv(outputs["summary_csv"], index=False)
    outputs["summary_md"].write_text(
        f"# Ablation summary: {root.name}\n\nMetric: `{metric}` test accuracy, "
        f"mean ± std over seeds.\n\n" + to_markdown(agg)
    )
    if plot:
        outputs["plot"] = root / "accuracy_vs_params.png"
        plot_accuracy_vs_params(agg, outputs["plot"])
    return outputs
