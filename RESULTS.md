# VL-JEPA on COCO 2017: Results

Final pretrained model: **ViT-Tiny vision encoder + DistilBERT text encoder + 6-layer transformer predictor**, trained for 10 epochs on 591,753 image-caption pairs from COCO 2017 captions. Total wall time 10.8 hours on a Kaggle "GPU T4 x2" kernel (2× NVIDIA Tesla T4 16GB, DistributedDataParallel, effective global batch 64).

## Final retrieval metrics (COCO 5K val, Karpathy dedupe)

| Metric | Value |
|---|---|
| **i2t recall@1** | **50.30%** |
| i2t recall@5 | 79.33% |
| i2t recall@10 | 87.50% |
| **t2i recall@1** | **38.98%** |
| t2i recall@5 | 70.64% |
| t2i recall@10 | 81.50% |
| **mean recall (6-metric avg)** | **68.04%** |

Random baseline on COCO 5K is `1/5000 = 0.02%` for recall@1. Final i2t recall@1 is **2,515× the random baseline**.

![Training curves](docs/training_curves.png)

## Epoch-by-epoch trajectory

| Epoch | i2t R@1 | i2t R@5 | i2t R@10 | t2i R@1 | t2i R@5 | t2i R@10 | mean | JEPA loss (train) | Contrast loss (train) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 21.26 | 51.08 | 63.58 | 20.50 | 45.84 | 60.64 | 43.82 | 0.254 | 2.40 |
| 1 | 30.71 | 58.86 | 71.46 | 25.16 | 53.68 | 67.16 | 51.17 | 0.239 | 1.83 |
| 2 | 36.02 | 64.57 | 76.08 | 27.62 | 57.56 | 70.82 | 55.45 | 0.233 | 1.64 |
| 3 | 37.80 | 68.90 | 78.64 | 29.08 | 60.02 | 72.68 | 57.85 | 0.231 | 1.50 |
| 4 | 42.22 | 72.44 | 81.40 | 31.46 | 62.42 | 75.42 | 60.89 | 0.230 | 1.37 |
| 5 | 43.31 | 72.44 | 82.87 | 34.18 | 65.30 | 77.30 | 62.57 | 0.228 | 1.26 |
| 6 | 48.23 | 75.89 | 84.65 | 36.32 | 67.68 | 79.30 | 65.34 | 0.226 | 1.15 |
| 7 | 47.34 | 77.85 | 87.50 | 37.46 | 69.56 | 80.54 | 66.71 | 0.225 | 1.05 |
| 8 | 50.30 | 79.04 | 88.09 | 38.50 | 70.46 | 81.32 | 67.95 | 0.224 | 0.98 |
| **9** | **50.30** | **79.33** | **87.50** | **38.98** | **70.64** | **81.50** | **68.04** | **0.222** | **0.94** |

## Retrieval examples

![Retrieval examples](docs/retrieval_examples.png)

The grid above shows the model running on the COCO 5K val gallery: top half is text-query to image-retrieval, bottom half is image-query to caption-retrieval. Cosine-similarity scores in the 0.51-0.59 range are the typical operating range of the aligned embedding space after training.

## The story

This run is the result of fixing a contrastive collapse that pinned an earlier version of the same code at random baseline for four epochs straight.

### v7: total collapse

The earlier run (v7) used the same architecture but with `contrastive_loss_type: "infonce"` and the default base learning rate of 3e-4 applied uniformly to all parameters including the pretrained DistilBERT text encoder. By epoch 3:

- `i2t recall@1` sat at **0.098%** (5× random baseline, essentially nothing)
- Contrastive loss locked at exactly **`log(N) = log(32) = 3.4657`** for thousands of consecutive batches
- Forensic check on `checkpoint_epoch_3.pth`: the text encoder's CLS token row-norm standard deviation was **1.3e-06** across 16 different random inputs, meaning DistilBERT was emitting essentially one constant vector regardless of input
- Vision encoder CLS was similarly collapsed onto a single direction

The collapse was diagnosed by loading the checkpoint into [`scripts/diagnose_checkpoint.py`](scripts/diagnose_checkpoint.py) and inspecting the pre-projection encoder outputs on synthetic batches.

### Root cause (from the VL-JEPA paper's ablation table 5b)

Hitting a pretrained text encoder with the full base learning rate destroys its representations within a few epochs. The paper documents a sweet spot of **0.05× to 0.10×** the base LR for the text encoder. We had been using 1.0× (the default).

A second issue compounded the first: at our small effective batch (32 in v7, 64 in v15), InfoNCE's anti-collapse term is weak. When the cosine-similarity matrix flattens, the InfoNCE gradient through the logit_scale parameter vanishes, and there is no signal to push the encoders apart.

### v15: paper-grounded fixes

Three changes, no architectural modifications:

1. **`training.text_encoder_lr_multiplier: 0.05`** added as a separate parameter group in the optimizer. Text encoder now trains at `3e-4 × 0.05 = 1.5e-5`, 20× lower than the vision encoder and predictor.
2. **`model.contrastive_loss_type: "siglip"`** switched from default InfoNCE. SigLIP's sigmoid-per-pair formulation keeps gradients alive on uniform similarity matrices, so the heads can escape flat regions even at small batch.
3. **DistributedDataParallel with autograd-aware all-gather** across the two T4s, so SigLIP sees the full global batch of 64 negatives instead of a per-rank slice of 32.

Both fixes are tested in [`tests/test_text_lr_multiplier_and_siglip.py`](tests/test_text_lr_multiplier_and_siglip.py) and the DDP all-gather is verified end-to-end in [`scripts/smoke_ddp_cpu.py`](scripts/smoke_ddp_cpu.py) using `gloo` on CPU (the all-gather logic is backend-agnostic).

### v7 vs v15 head to head

| Property | v7 (broken) | v15 (fixed) |
|---|---|---|
| Contrastive loss | InfoNCE | SigLIP |
| Text encoder LR | 3.0e-4 (full base) | 1.5e-5 (0.05× base) |
| Effective batch | 32 | 64 (DDP, dual T4) |
| Final i2t recall@1 | 0.098% | **50.30%** |
| Text CLS row-norm std (random input probe) | 1.3e-06 (collapsed) | 0.56 (healthy) |
| Vision CLS row-norm std | 0.013 (collapsed) | 0.128 (healthy) |
| Contrastive loss at end of training | 3.4657 (= log(32), stuck) | 0.94 (descending) |

## Architecture and training details

| Component | Configuration |
|---|---|
| Vision encoder | `timm/vit_tiny_patch16_224`, 5.7M params, pretrained ImageNet init |
| Text encoder | `distilbert-base-uncased`, 66M params, HF pretrained init |
| Predictor | 6-layer transformer, hidden 384, 6 heads, 1M params |
| Vision projection | LayerNorm + Linear (192 → 256) |
| Text projection | LayerNorm + Linear (768 → 256) |
| EMA target encoders | deepcopy, frozen, momentum 0.996 → 1.0 |
| **Total** | **154.89M params, 83.01M trainable** |

Training setup:

| Knob | Value |
|---|---|
| Optimizer | AdamW (β = 0.9, 0.999, eps = 1e-8) |
| Base learning rate | 3.0e-4 |
| Text encoder LR | 1.5e-5 (0.05× base) |
| Weight decay | 0.05 |
| LR schedule | 1-epoch linear warmup, then cosine decay to 1e-6 |
| Per-rank batch | 32 |
| Effective global batch | 64 (2 ranks via DDP) |
| Mixed precision | fp16 with GradScaler |
| Gradient clip | 1.0 |
| JEPA loss weight | 1.0 |
| SigLIP loss weight | 0.5 |
| Mask scheme | I-JEPA multi-block: 1 context block (scale 0.85-1.0), 4 target blocks (scale 0.15-0.2) |
| Epochs | 10 |
| Wall time | 10.82 hours (Kaggle T4 x2) |

## Which checkpoint to use

The output bundle from Kaggle contains:

- **`checkpoint_epoch_9.pth`** ← the final model, what you want.
- `checkpoint_epoch_7.pth` — safety copy (kept by `keep_last_n_checkpoints: 2`).
- `best_model.pth` — **do not use**; it is the epoch-0 weights due to a Stage-1 selection bug in the version of `train.py` that produced this run. The bug has been fixed in commit-history after v15; new runs will save `best_model.pth` correctly based on `mean_recall`.

To verify a checkpoint:

```powershell
python scripts/diagnose_checkpoint.py checkpoints/final_model.pth configs/config_kaggle_t4.yaml
```

A healthy checkpoint should show non-trivial row-norm variance on both encoders and a cosine-similarity matrix with std > 0 on the random-input probe. (Note: cosine values can still be high on synthetic noise; the real check is the retrieval metrics on actual data.)
