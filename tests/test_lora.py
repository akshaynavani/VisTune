import pytest
import torch
from torch import nn

from vistune.lora import (
    LoRALinear,
    count_parameters,
    freeze_all,
    inject_lora,
    layer_index,
    lora_state_dict,
    mark_lora_trainable,
    merge_lora,
    unmerge_lora,
)


class Block(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.query = nn.Linear(d, d)
        self.value = nn.Linear(d, d)
        self.fc = nn.Linear(d, d)

    def forward(self, x):
        return self.fc(self.query(x) + self.value(x))


class Tiny(nn.Module):
    def __init__(self, n=4, d=16):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([Block(d) for _ in range(n)])
        self.head = nn.Linear(d, 3)

    def forward(self, x):
        for blk in self.encoder.layer:
            x = blk(x)
        return self.head(x)


def test_lora_starts_as_identity_update():
    torch.manual_seed(0)
    base = nn.Linear(8, 5)
    x = torch.randn(3, 8)
    expected = base(x)
    wrapped = LoRALinear(base, rank=2, alpha=4)
    assert torch.allclose(wrapped(x), expected)


def test_merge_matches_unmerged_forward():
    torch.manual_seed(0)
    layer = LoRALinear(nn.Linear(8, 5), rank=4, alpha=8)
    nn.init.normal_(layer.lora_B)
    layer.eval()
    x = torch.randn(3, 8)
    before = layer(x)
    layer.merge()
    assert torch.allclose(layer(x), before, atol=1e-5)
    layer.unmerge()
    assert torch.allclose(layer(x), before, atol=1e-5)


def test_inject_targets_and_layers():
    model = Tiny()
    report = inject_lora(model, ["query", "value"], rank=2, layers=[-1, -2], num_layers=4)
    assert len(report) == 4
    assert all(layer_index(n) in (2, 3) for n in report.replaced)
    assert isinstance(model.encoder.layer[3].query, LoRALinear)
    assert not isinstance(model.encoder.layer[0].query, LoRALinear)
    assert not isinstance(model.encoder.layer[3].fc, LoRALinear)


def test_inject_without_match_raises():
    with pytest.raises(ValueError):
        inject_lora(Tiny(), ["does_not_exist"])


def test_parameter_accounting_and_state_dict():
    model = Tiny(n=2, d=16)
    inject_lora(model, ["query"], rank=4)
    freeze_all(model)
    mark_lora_trainable(model)
    stats = count_parameters(model)
    assert stats["lora"] == 2 * (4 * 16 + 16 * 4)
    assert stats["trainable"] == stats["lora"]
    sd = lora_state_dict(model)
    assert len(sd) == 4 and all("lora_" in k for k in sd)


def test_merge_helpers_roundtrip():
    torch.manual_seed(0)
    model = Tiny(n=2)
    inject_lora(model, ["query", "value"], rank=2)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            nn.init.normal_(m.lora_B)
    model.eval()
    x = torch.randn(2, 16)
    ref = model(x)
    merge_lora(model)
    assert torch.allclose(model(x), ref, atol=1e-5)
    unmerge_lora(model)
    assert torch.allclose(model(x), ref, atol=1e-5)
