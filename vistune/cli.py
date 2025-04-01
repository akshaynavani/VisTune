"""Command line interface: `vistune train|evaluate|ablate|report` (also `python -m vistune`)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import load_config


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", "-c", help="YAML experiment config")
    p.add_argument("--set", "-s", nargs="*", default=[], metavar="KEY=VALUE",
                   help="Dotted overrides, e.g. -s lora.rank=4 train.epochs=3")


def cmd_train(args) -> None:
    from .engine import run_experiment

    cfg = load_config(args.config, args.set)
    results = run_experiment(cfg)
    summary = {
        "name": results["name"],
        "trainable_params": results["params"]["trainable"],
        "initial_test": results["initial_test"],
        "final_test": results["final_test"],
        "best_test": results["best_test"],
    }
    print(json.dumps(summary, indent=2))


def cmd_evaluate(args) -> None:
    from .engine import evaluate_checkpoint

    cfg = load_config(args.config, args.set)
    print(json.dumps(evaluate_checkpoint(cfg, args.checkpoint), indent=2))


def cmd_ablate(args) -> None:
    from .ablation import expand, load_ablation, run_ablation

    spec = load_ablation(args.spec)
    if args.list:
        for cfg in expand(spec):
            print(cfg.run_dir)
        return
    run_ablation(spec, resume=not args.no_resume, dry_run=args.dry_run)
    if not args.dry_run and not args.no_report:
        from .report import write_report

        outputs = write_report(Path(spec.output_dir) / spec.name, metric=args.metric)
        print((outputs["summary_md"]).read_text())


def cmd_report(args) -> None:
    from .report import write_report

    outputs = write_report(args.root, metric=args.metric, plot=not args.no_plot)
    print(outputs["summary_md"].read_text())
    for key, path in outputs.items():
        print(f"{key}: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vistune", description=__doc__)
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="Fine-tune one configuration")
    _add_config_args(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="Evaluate a config, optionally loading a checkpoint")
    _add_config_args(p)
    p.add_argument("--checkpoint", help="*_trainable.pt saved by `train`")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("ablate", help="Run every configuration in an ablation YAML")
    p.add_argument("spec", help="Ablation YAML, e.g. configs/ablations/vit_rank.yaml")
    p.add_argument("--list", action="store_true", help="Only print the expanded run names")
    p.add_argument("--dry-run", action="store_true", help="Log the plan without training")
    p.add_argument("--no-resume", action="store_true", help="Re-run completed configurations")
    p.add_argument("--no-report", action="store_true")
    p.add_argument("--metric", choices=["final", "best"], default="final")
    p.set_defaults(func=cmd_ablate)

    p = sub.add_parser("report", help="Summarise finished runs under a directory")
    p.add_argument("root", help="Directory containing run folders with metrics.json")
    p.add_argument("--metric", choices=["final", "best"], default="final")
    p.add_argument("--no-plot", action="store_true")
    p.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
