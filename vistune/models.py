"""Backbones (ViT, CLIP vision tower) behind a common classifier interface, plus the logic that
applies a fine-tuning strategy (linear probe / LoRA / full) to them."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import LoRAConfig, ModelConfig
from .lora import count_parameters, freeze_all, inject_lora, mark_lora_trainable
from .prompts import DEFAULT_TEMPLATES, text_classifier_weights

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TargetPreset:
    targets: tuple[str, ...]
    exclude: tuple[str, ...] = ()


# Module-name suffixes inside Hugging Face ViTModel / CLIPVisionTransformer encoder blocks.
TARGET_PRESETS: dict[str, dict[str, TargetPreset]] = {
    "vit": {
        "qv": TargetPreset(("attention.query", "attention.value")),
        "attn": TargetPreset(("attention.query", "attention.key", "attention.value",
                              "attention.output.dense")),
        "mlp": TargetPreset(("intermediate.dense", "output.dense"), exclude=("attention.output",)),
        "all": TargetPreset(("attention.query", "attention.key", "attention.value",
                             "attention.output.dense", "intermediate.dense", "output.dense")),
    },
    "clip": {
        "qv": TargetPreset(("self_attn.q_proj", "self_attn.v_proj")),
        "attn": TargetPreset(("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                              "self_attn.out_proj")),
        "mlp": TargetPreset(("mlp.fc1", "mlp.fc2")),
        "all": TargetPreset(("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                             "self_attn.out_proj", "mlp.fc1", "mlp.fc2")),
    },
}


def resolve_targets(family: str, targets) -> TargetPreset:
    if isinstance(targets, str):
        try:
            return TARGET_PRESETS[family][targets]
        except KeyError as exc:
            raise KeyError(f"Unknown target preset {targets!r} for {family}. "
                           f"Options: {sorted(TARGET_PRESETS[family])}") from exc
    return TargetPreset(tuple(targets))


class VisionClassifier(nn.Module):
    """Vision encoder + classification head. Subclasses implement `features`."""

    family: str = ""

    def __init__(self, backbone: nn.Module, head: nn.Module, num_layers: int):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.num_layers = num_layers

    def features(self, pixel_values: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(pixel_values))

    def head_parameters(self):
        return list(self.head.parameters())


class ViTClassifier(VisionClassifier):
    family = "vit"

    def features(self, pixel_values):
        out = self.backbone(pixel_values=pixel_values)
        return out.last_hidden_state[:, 0]  # CLS token (already layer-normed)


class CosineHead(nn.Module):
    """Linear classifier over L2-normalised features with a fixed temperature (CLIP style)."""

    def __init__(self, weight: torch.Tensor, logit_scale: float):
        super().__init__()
        self.weight = nn.Parameter(weight.clone())
        self.register_buffer("logit_scale", torch.tensor(float(logit_scale)))

    def forward(self, feats):
        feats = F.normalize(feats, dim=-1)
        return self.logit_scale * feats @ self.weight.t()


class CLIPClassifier(VisionClassifier):
    family = "clip"

    def __init__(self, backbone, projection, head, num_layers):
        super().__init__(backbone, head, num_layers)
        self.projection = projection

    def features(self, pixel_values):
        pooled = self.backbone(pixel_values=pixel_values).pooler_output
        return self.projection(pooled)


def _build_vit(cfg: ModelConfig, num_classes: int) -> ViTClassifier:
    from transformers import ViTModel

    backbone = ViTModel.from_pretrained(cfg.checkpoint, add_pooling_layer=False)
    hidden = backbone.config.hidden_size
    head = nn.Linear(hidden, num_classes)
    nn.init.trunc_normal_(head.weight, std=0.02)
    nn.init.zeros_(head.bias)
    return ViTClassifier(backbone, head, backbone.config.num_hidden_layers)


def _build_clip(cfg: ModelConfig, class_names: list[str], device) -> CLIPClassifier:
    from transformers import AutoTokenizer, CLIPModel

    clip = CLIPModel.from_pretrained(cfg.checkpoint)
    num_classes = len(class_names)
    if cfg.zero_shot_head:
        tokenizer = AutoTokenizer.from_pretrained(cfg.checkpoint)
        use_ensemble = cfg.prompt_template in (None, "", "ensemble")
        templates = DEFAULT_TEMPLATES if use_ensemble else [cfg.prompt_template]
        weights = text_classifier_weights(clip, tokenizer, class_names, templates, device=device)
        head: nn.Module = CosineHead(weights, clip.logit_scale.exp().item())
    else:
        head = nn.Linear(clip.config.projection_dim, num_classes)
    backbone = clip.vision_model
    projection = clip.visual_projection
    num_layers = clip.config.vision_config.num_hidden_layers
    # The text tower is no longer needed once the head exists.
    del clip
    return CLIPClassifier(backbone, projection, head, num_layers)


def build_model(model_cfg: ModelConfig, class_names: list[str], device="cpu") -> VisionClassifier:
    if model_cfg.family == "vit":
        return _build_vit(model_cfg, len(class_names))
    if model_cfg.family == "clip":
        return _build_clip(model_cfg, class_names, device)
    raise ValueError(f"Unknown model family {model_cfg.family!r}")


def apply_strategy(model: VisionClassifier, strategy: str, lora_cfg: LoRAConfig) -> dict:
    """Freeze/unfreeze and inject adapters according to the fine-tuning strategy.

    Returns a summary with parameter counts and the list of adapted modules.
    """
    adapted: list[str] = []
    if strategy == "full":
        for p in model.parameters():
            p.requires_grad_(True)
    elif strategy in ("linear_probe", "zero_shot"):
        freeze_all(model)
    elif strategy == "lora":
        preset = resolve_targets(model.family, lora_cfg.targets)
        report = inject_lora(
            model.backbone,
            targets=preset.targets,
            exclude=preset.exclude,
            rank=lora_cfg.rank,
            alpha=lora_cfg.alpha,
            dropout=lora_cfg.dropout,
            layers=lora_cfg.layers,
            num_layers=model.num_layers,
        )
        adapted = report.replaced
        freeze_all(model)
        mark_lora_trainable(model)
    else:
        raise ValueError(f"Unknown strategy {strategy!r}")

    if strategy != "zero_shot":
        for p in model.head_parameters():
            p.requires_grad_(True)

    stats = count_parameters(model)
    log.info("strategy=%s trainable=%d / %d (%.3f%%), adapted modules=%d", strategy,
             stats["trainable"], stats["total"], stats["trainable_pct"], len(adapted))
    return {"strategy": strategy, "params": stats, "adapted_modules": adapted}
