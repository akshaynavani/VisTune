"""Model tests use tiny randomly initialised configs, so no checkpoints are downloaded."""

import pytest
import torch
from torch import nn

transformers = pytest.importorskip("transformers")
from transformers import CLIPVisionConfig, CLIPVisionModel, ViTConfig, ViTModel  # noqa: E402

from vistune.config import LoRAConfig  # noqa: E402
from vistune.lora import LoRALinear, lora_modules  # noqa: E402
from vistune.models import (  # noqa: E402
    CLIPClassifier,
    CosineHead,
    ViTClassifier,
    apply_strategy,
    resolve_targets,
)

NUM_CLASSES = 5


def tiny_vit():
    cfg = ViTConfig(hidden_size=32, num_hidden_layers=4, num_attention_heads=4,
                    intermediate_size=64, image_size=32, patch_size=8)
    backbone = ViTModel(cfg, add_pooling_layer=False)
    return ViTClassifier(backbone, nn.Linear(32, NUM_CLASSES), cfg.num_hidden_layers)


def tiny_clip():
    cfg = CLIPVisionConfig(hidden_size=32, num_hidden_layers=4, num_attention_heads=4,
                           intermediate_size=64, image_size=32, patch_size=8, projection_dim=16)
    vision = CLIPVisionModel(cfg).vision_model
    projection = nn.Linear(32, 16, bias=False)
    head = CosineHead(torch.randn(NUM_CLASSES, 16), logit_scale=100.0)
    return CLIPClassifier(vision, projection, head, cfg.num_hidden_layers)


@pytest.mark.parametrize("builder", [tiny_vit, tiny_clip])
def test_forward_shape(builder):
    model = builder()
    logits = model(torch.randn(2, 3, 32, 32))
    assert logits.shape == (2, NUM_CLASSES)


@pytest.mark.parametrize("builder,preset,per_layer", [
    (tiny_vit, "qv", 2), (tiny_vit, "attn", 4), (tiny_vit, "mlp", 2), (tiny_vit, "all", 6),
    (tiny_clip, "qv", 2), (tiny_clip, "attn", 4), (tiny_clip, "mlp", 2), (tiny_clip, "all", 6),
])
def test_lora_presets_adapt_expected_modules(builder, preset, per_layer):
    model = builder()
    summary = apply_strategy(model, "lora", LoRAConfig(rank=2, alpha=4, targets=preset))
    assert len(summary["adapted_modules"]) == per_layer * model.num_layers
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert all("lora_" in n or n.startswith("head.") for n in trainable)


def test_lora_layer_subset():
    model = tiny_vit()
    apply_strategy(model, "lora", LoRAConfig(rank=2, targets="qv", layers=[-1]))
    assert len(lora_modules(model)) == 2
    assert isinstance(model.backbone.encoder.layer[3].attention.attention.query, LoRALinear)


def test_linear_probe_only_trains_head():
    model = tiny_vit()
    summary = apply_strategy(model, "linear_probe", LoRAConfig())
    head_params = sum(p.numel() for p in model.head.parameters())
    assert summary["params"]["trainable"] == head_params


def test_lora_does_not_change_initial_outputs():
    torch.manual_seed(0)
    model = tiny_vit().eval()
    x = torch.randn(2, 3, 32, 32)
    before = model(x)
    apply_strategy(model, "lora", LoRAConfig(rank=4, targets="all", dropout=0.0))
    model.eval()
    assert torch.allclose(model(x), before, atol=1e-5)


def test_unknown_preset_raises():
    with pytest.raises(KeyError):
        resolve_targets("vit", "nope")
