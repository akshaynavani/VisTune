from pathlib import Path

import pytest

from vistune.ablation import AblationSpec, expand, is_complete, load_ablation
from vistune.report import aggregate, collect, to_markdown
from vistune.utils import write_json

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "ablations"


def test_grid_cross_variants_and_dedup():
    spec = AblationSpec(
        name="t",
        base={"train": {"strategy": "lora"}},
        grid={"lora.rank": [2, 4], "lora.targets": ["qv", "all"]},
        variants=[{"name": "lora"}, {"name": "probe", "train.strategy": "linear_probe"}],
        seeds=[0, 1],
    )
    cfgs = expand(spec)
    lora = [c for c in cfgs if c.train.strategy == "lora"]
    probe = [c for c in cfgs if c.train.strategy == "linear_probe"]
    assert len(lora) == 2 * 2 * 2
    # LoRA hyper-parameters are irrelevant for a linear probe: one run per seed.
    assert len(probe) == 2
    assert len({c.name for c in cfgs}) == len(cfgs)
    assert {c.train.seed for c in cfgs} == {0, 1}


def test_shipped_ablation_files_expand():
    spec = load_ablation(CONFIGS / "vit_rank_targets.yaml")
    cfgs = expand(spec)
    assert len(cfgs) == 5 * 4 * 3 + 3 + 3
    full = next(c for c in cfgs if c.train.strategy == "full")
    assert full.train.lr == pytest.approx(5e-5)
    assert all(c.model.family == "vit" for c in cfgs)


@pytest.mark.parametrize("name", ["clip_alpha_layers.yaml", "clip_few_shot.yaml", "smoke_fake.yaml"])
def test_all_ablation_files_parse(name):
    assert expand(load_ablation(CONFIGS / name))


def _fake_metrics(name, strategy, rank, seed, top1, trainable):
    return {
        "name": name,
        "config": {
            "model": {"family": "vit", "checkpoint": "x"},
            "data": {"name": "fake", "shots_per_class": None},
            "train": {"strategy": strategy, "seed": seed},
            "lora": {"targets": "qv", "rank": rank, "alpha": 16, "layers": None},
        },
        "params": {"trainable": trainable, "trainable_pct": 0.5},
        "initial_test": {"top1": 10.0, "top5": 50.0},
        "final_test": {"top1": top1, "top5": top1 + 10},
        "best_test": {"top1": top1, "top5": top1 + 10},
    }


def test_report_aggregates_over_seeds(tmp_path):
    write_json(_fake_metrics("a0", "lora", 4, 0, 80.0, 1000), tmp_path / "a0" / "metrics.json")
    write_json(_fake_metrics("a1", "lora", 4, 1, 82.0, 1000), tmp_path / "a1" / "metrics.json")
    write_json(_fake_metrics("p0", "linear_probe", 4, 0, 70.0, 100), tmp_path / "p0" / "metrics.json")
    agg = aggregate(collect(tmp_path))
    assert list(agg["strategy"]) == ["lora", "linear_probe"]
    assert agg.loc[0, "top1_mean"] == pytest.approx(81.0)
    assert agg.loc[0, "runs"] == 2
    assert "lora" in to_markdown(agg)


def test_is_complete(tmp_path):
    spec = AblationSpec(name="t", base={}, output_dir=str(tmp_path), seeds=[0])
    cfg = expand(spec)[0]
    assert not is_complete(cfg)
    write_json({"final_test": {"top1": 1.0}}, cfg.run_dir / "metrics.json")
    assert is_complete(cfg)
