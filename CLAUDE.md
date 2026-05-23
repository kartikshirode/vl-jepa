# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

VL-JEPA is a vision-language pretraining model that combines:
- **JEPA** (Joint Embedding Predictive Architecture): predict masked patch representations in latent space from visible context, with an EMA target encoder.
- **Contrastive (InfoNCE)**: align CLS-pooled vision and text embeddings (CLIP-style).

Backbone: ViT-Tiny (timm) for vision, DistilBERT (Hugging Face) for text, MLP or Transformer predictor. Reference paper PDF is checked in at `2512.10942v1_copy.pdf`.

The repo is named `vl-jepa-jetson` because it was originally targeted at Jetson Orin Nano (8GB unified memory), but the current default config (`config_dgpu.yaml`) is tuned for a discrete GPU with 8GB+ VRAM. Both configurations coexist.

## Common commands

All commands assume the venv is activated. On Windows: `.venv\Scripts\activate`.

```powershell
# Smoke test (no dataset required, exercises every module + GPU path)
python test_quick.py

# Deeper component test
python scripts/test_implementation.py

# Verify CUDA is visible to PyTorch
python scripts/check_cuda.py

# Stage-1 training (full pretraining, all params trainable)
python train.py --config config_dgpu.yaml

# Stage-2 training (text encoder frozen, lower LR, early stop on mean recall)
python train.py --config config_dgpu.yaml --resume checkpoints/checkpoint_epoch_N.pth --stage2

# Resume Stage-1 from checkpoint
python train.py --config config_dgpu.yaml --resume checkpoints/checkpoint_epoch_N.pth

# Enable W&B logging
python train.py --config config_dgpu.yaml --wandb
```

There is no pytest suite, lint config, or formatter wired up. `test_quick.py` and `scripts/test_implementation.py` are plain Python scripts (no test framework), so "running a single test" means running the script that exercises the component you care about, or importing the module in a REPL.

## Architecture notes that span multiple files

### Two-encoder + EMA target pattern
`VLJEPAModel.__init__` (`vl_jepa/models/vl_jepa.py`) `deepcopy`s both the vision and text encoders into `target_vision_encoder` / `target_text_encoder` and freezes them. The target encoders are updated via `update_target_encoder()` (EMA, no grad) and **must be called manually after each optimizer step** in the training loop. If you add a new encoder, mirror it in the EMA target list and in the EMA update.

### Three forward modes share one entrypoint
`VLJEPAModel.forward(..., mode=...)` accepts `"jepa"`, `"contrastive"`, or `"both"`. The `"both"` branch is **not** equivalent to calling `forward_jepa` + `forward_contrastive` back-to-back; it deliberately encodes vision and text once and reuses the CLS tokens for contrastive, plus runs the predictor and target encoder for JEPA. When changing encoder outputs, update all three branches plus `forward_jepa` and `forward_contrastive` separately, since the standalone methods are still called from `train.py` paths.

The `"both"` branch hardcodes `loss = jepa_loss + 0.5 * contrastive_loss`. The yaml-configurable `jepa_loss_weight` / `contrastive_loss_weight` are applied in `train.py`, not inside the model. Don't add weighting inside `forward` without checking the trainer.

### JEPA loss skips CLS, normalizes, then masks
`compute_jepa_loss` drops index 0 (CLS), `F.layer_norm`s predicted and target patches, computes `smooth_l1_loss`, then averages only over `target_mask` positions. Patch grid is fixed at 14x14 = 196 (224 / 16). The mask layout must match.

### Mask generation lives outside the model
`vl_jepa/masks/multiblock.py` produces per-sample `context_mask` and `target_mask` tensors. They are generated in the dataloader / training loop and passed into `forward(...)`. The model does not slice patches with the masks; the masks only gate the JEPA loss. Be careful: the predictor sees all tokens, so masking is loss-side, not input-side.

### Two-stage training is a first-class concept
`train.py` has explicit `--stage2` plumbing that:
- Calls `model.freeze_text_encoder()` (freezes DistilBERT + verifies projection heads remain trainable; raises if anything is off).
- Uses `model.get_parameter_groups(lr, stage2=True)` so the optimizer never sees frozen params.
- Multiplies the configured LR by `--stage2_lr_factor` (default 0.5), skips warmup, caps epochs at `--stage2_epochs` (max 4), and runs early stopping on `mean_recall` (higher is better) instead of `val_loss` (lower is better) which is the Stage-1 selection metric.
- Saves to `checkpoints/stage2_epoch_N.pth` and `checkpoints/stage2_best.pth`.

When touching any of: optimizer construction, parameter freezing, checkpoint resume logic, or LR scheduling, check both stage branches in `train.py`.

### Config has two homes (intentional)
- `config_dgpu.yaml` at repo root: the default config the README points to. Larger batch, deeper predictor, `num_workers: 4`.
- `configs/config_coco_full.yaml`: Jetson-tuned (batch 1, gradient_accumulation_steps 16, `num_workers: 0`, `data_root: ./data`).

They diverge on `data.data_root` (`./vl_jepa/data/COCO2017` vs `./data`), `predictor.num_layers` (4 vs 3), and `text_encoder.max_length` (128 vs 64). When changing config schema, update both, and also update `vl_jepa/utils/config.py` if you add new keys.

### Dataset assumes a specific layout
`create_dataset` expects COCO 2017 at the path in `config.data.data_root` with `train2017/`, `val2017/`, and `annotations/captions_*.json` underneath. The `data/` and `vl_jepa/data/` directories are gitignored entirely. There's also a `dummy` dataset option for running training without real data.

## Things that will bite you

- `bitsandbytes` is optional and silently absent on ARM/Jetson. Train script prints a warning and falls back to standard `AdamW`. Don't make `adamw8bit` a hard requirement.
- `checkpoints/` is gitignored, so nothing is recoverable from git history; treat saved `.pth` files as the only source of truth for trained weights.
- `train.py` imports from `torch.cuda.amp` (`autocast`, `GradScaler`), which is the older API. If you upgrade to `torch.amp`, expect a deprecation churn across both train and eval paths.
- `mode="both"` reuses the CLS token directly from the **online** vision encoder for the contrastive head, not from the target encoder. Don't "fix" this without reading the paper.
- The Windows shell here is PowerShell; chained commands use `;` and `if ($?) { ... }`, not `&&`.
