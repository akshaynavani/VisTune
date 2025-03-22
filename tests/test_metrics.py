import torch

from vistune.metrics import AccuracyMeter, topk_accuracy


def test_topk_accuracy_basic():
    logits = torch.tensor([
        [0.9, 0.05, 0.05],   # pred 0, target 0 -> top1 hit
        [0.1, 0.2, 0.7],     # pred 2, target 1 -> top2 hit
        [0.5, 0.3, 0.2],     # pred 0, target 2 -> only top3
    ])
    targets = torch.tensor([0, 1, 2])
    acc = topk_accuracy(logits, targets, ks=(1, 2, 3))
    assert acc[1] == 100.0 / 3
    assert acc[2] == 200.0 / 3
    assert acc[3] == 100.0


def test_topk_clamps_k_to_num_classes():
    logits = torch.randn(4, 3)
    acc = topk_accuracy(logits, torch.tensor([0, 1, 2, 0]), ks=(1, 5))
    assert acc[5] == 100.0


def test_meter_accumulates_batches():
    meter = AccuracyMeter(num_classes=2, ks=(1,))
    meter.update(torch.tensor([[2.0, 0.0], [2.0, 0.0]]), torch.tensor([0, 1]), loss=1.0)
    meter.update(torch.tensor([[0.0, 2.0]]), torch.tensor([1]), loss=4.0)
    out = meter.compute()
    assert out["samples"] == 3
    assert abs(out["top1"] - 200.0 / 3) < 1e-6
    assert abs(out["loss"] - 2.0) < 1e-6
    # class 0: 1/1, class 1: 1/2 -> mean 75%
    assert abs(out["mean_per_class"] - 75.0) < 1e-6
