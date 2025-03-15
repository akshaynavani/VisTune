# VisTune — Implementation Plan

Parameter-efficient fine-tuning (LoRA) of CLIP and ViT vision backbones on domain-specific
image classification datasets, with an ablation harness that compares adapter configurations
on top-1 / top-5 accuracy.

## Key design decisions

- **Stack:** Python 3.10+, PyTorch, Hugging Face `transformers` for the backbones
  (`CLIPModel` / `ViTModel`), `torchvision` for datasets and transforms, PyYAML for configs.
- **LoRA implemented from scratch** (`vistune/lora.py`) rather than via `peft`, so the
  injection, merge/unmerge and adapter-only checkpointing logic is explicit and testable.
  A `LoRALinear` wraps an existing `nn.Linear`: `y = W x + (alpha / r) * B(A(dropout(x)))`,
  with `B` zero-initialised so training starts from the pretrained function.
- **Two backbone families behind one interface** (`VisionClassifier`):
  - `vit`: `ViTModel` + linear head on the CLS token.
  - `clip`: CLIP vision tower + projection. The head is either a fresh linear layer or is
    **initialised from CLIP text embeddings of the class prompts** ("a photo of a {label}"),
    giving a zero-shot starting point.
- **Datasets:** any `ImageFolder` directory, plus a few torchvision built-ins
  (Oxford Flowers102, Oxford-IIIT Pets, EuroSAT, DTD, FGVC Aircraft) as domain benchmarks.
  A `fake` dataset exists for smoke tests without downloads.
- **Ablations:** a YAML file declares a base config and a grid over adapter settings
  (rank, alpha, dropout, target modules, layer subset, head-only vs LoRA vs full fine-tune).
  Each run writes `metrics.json`; a report step aggregates into CSV + Markdown tables with
  trainable-parameter counts alongside top-1/top-5.

## Phases

### Phase 1 — Project scaffolding, config and data
Deliverables: package skeleton, typed config dataclasses with YAML loading/overrides,
dataset registry + transforms matching each backbone's preprocessing, seeding utilities.
Files: `pyproject.toml`, `requirements.txt`, `vistune/__init__.py`, `vistune/config.py`,
`vistune/data.py`, `vistune/utils.py`, `configs/base_vit.yaml`, `configs/base_clip.yaml`.

### Phase 2 — LoRA core
Deliverables: `LoRALinear` layer (merge/unmerge, dropout, scaling), module injection by
name pattern and layer index, freezing helpers, trainable-parameter accounting,
adapter-only state dict save/load.
Files: `vistune/lora.py`, `tests/test_lora.py`.

### Phase 3 — Backbones and classifier heads
Deliverables: `VisionClassifier` wrapping ViT and CLIP vision encoders, per-family target
module presets (`attn`, `qv`, `mlp`, `all`), CLIP zero-shot head initialisation from text
prompts, fine-tuning strategy application (`linear_probe`, `lora`, `full`).
Files: `vistune/models.py`, `vistune/prompts.py`, `tests/test_models.py`.

### Phase 4 — Training and evaluation
Deliverables: top-k accuracy metrics, train/eval loops with AMP, cosine LR schedule with
warmup, separate LR for head vs adapters, gradient clipping, best-checkpoint tracking,
JSON metrics, zero-shot evaluation for CLIP, CLI entry points `train` and `evaluate`.
Files: `vistune/metrics.py`, `vistune/engine.py`, `vistune/cli.py`, `tests/test_metrics.py`.

### Phase 5 — Ablation harness and reporting
Deliverables: grid expansion from YAML, sequential ablation runner with resume (skips
completed runs), results aggregation to CSV/Markdown, accuracy-vs-trainable-params plot,
example ablation configs.
Files: `vistune/ablation.py`, `vistune/report.py`, `configs/ablations/*.yaml`,
`tests/test_ablation.py`.

### Wrap-up
`.gitignore`, `README.md` with architecture diagram, run and test instructions.
