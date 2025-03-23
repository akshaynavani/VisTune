"""Command line interface: `vistune train|evaluate ...` (also `python -m vistune`)."""

from __future__ import annotations

import argparse
import json
import logging

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
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
