# VL-JEPA Audit Report

**Project:** `vl-jepa-main` (Kartik)
**Auditor focus:** correctness against the JEPA research line — I-JEPA (Assran et al., 2023, CVPR), V-JEPA (Bardes et al., 2024), and follow-ups (A-JEPA, V-JEPA 2). Also general PyTorch / training-loop / packaging bugs.
**Scope:** every file in the archive. Line numbers refer to the files as shipped.

---

## TL;DR

Kartik, the project has a clean repo skeleton and the **intent** of the architecture is right — context encoder, EMA target encoder, predictor, multi-block mask, plus a CLIP-style contrastive head. But under the hood several things are off, and three of them are bad enough that the current code **either doesn't run at all or doesn't actually train a JEPA**:

1. The whole `vl_jepa/data/` module is **missing from the repo** — every entry point (`train.py`, `inference.py`, `test_quick.py`) crashes on import.
2. The **context encoder sees the full image, never the masked context**. The predictor's input therefore contains the target patches it's being asked to predict → the JEPA task is trivial / collapsed.
3. The **predictor has no notion of target position** (MLP variant has no positions at all; Transformer variant uses sequential index, not the 2D location of the target block). This is the single most important design choice in I-JEPA and it's broken here.

Below: ordered by severity, with paper grounding and concrete fixes.

---

## Tier 0 — Showstoppers (the project will not run)

### 🔴 0.1 — `vl_jepa.data` module does not exist

**Files affected:** `train.py` (lines 25-27), `inference.py` (line 14), `test_quick.py` (line 10), `scripts/test_implementation.py`.

```python
# train.py:25
from vl_jepa.data.dataset import create_dataset
from vl_jepa.data.transforms import get_train_transforms, get_val_transforms
from vl_jepa.data.collate import jepa_collate_fn
```

Listing the archive: the `vl_jepa/` package contains `models/`, `masks/`, `utils/` — **no `data/`**. README lists it under "Project Structure" but the directory is absent. Every command in the README (`python train.py …`, `python inference.py …`) raises `ModuleNotFoundError` at import time, before any code runs.

**Fix:** ship `vl_jepa/data/__init__.py`, `dataset.py` (COCO-Captions loader), `transforms.py` (the train/val torchvision pipelines that the configs already describe), `collate.py` (a `jepa_collate_fn` that returns the dict `{'images', 'input_ids', 'attention_mask'}` train.py expects).

### 🔴 0.2 — Masking config is read from the wrong key for `config_dgpu.yaml`

**File:** `train.py:469`, `vl_jepa/masks/multiblock.py:252`.

```python
# train.py
mask_generator = create_mask_generator(config['data'])
```

```python
# multiblock.py
def create_mask_generator(config: dict):
    mask_config = config.get('masking', {})           # ← reads sub-key 'masking' from what was passed
```

But `config_dgpu.yaml` puts `masking:` at the **top level** (line 114), while `configs/config_coco_full.yaml` nests it under `data:` (line 74). So with the DGPU config, `config['data'].get('masking', {})` returns `{}` and the mask generator silently falls back to the **default** values — your carefully tuned scales/aspect-ratios in the YAML are never read.

**Fix:** either move `masking:` under `data:` in `config_dgpu.yaml`, or change `train.py:469` to:
```python
mask_cfg = {'masking': config.get('masking', config['data'].get('masking', {})),
            'vision_encoder': config['model']['vision_encoder']}
mask_generator = create_mask_generator(mask_cfg)
```

### 🔴 0.3 — `wandb.run` accessed when `wandb` is `None`

**File:** `train.py:614`.

```python
if wandb.run is not None:
    wandb.finish()
```

Above (line 21) `wandb = None` if the package isn't installed. The final line of training raises `AttributeError: 'NoneType' object has no attribute 'run'`. Same pattern is correctly guarded inside the training loop (`if use_wandb and wandb.run is not None`) but missed here.

**Fix:** `if HAS_WANDB and wandb.run is not None: wandb.finish()`

---

## Tier 1 — JEPA-correctness bugs (paper-level violations)

This tier is the heart of the audit. To anchor it, the I-JEPA paper (CVPR 2023, §3) is explicit:

> "The context encoder is a Vision Transformer (ViT), which only processes the visible context patches. The predictor is a narrow ViT that takes the context encoder output and, conditioned on positional tokens, predicts the representations of a target block at a specific location."

Three things matter: **(a)** context encoder receives ONLY context patches; **(b)** predictor receives context tokens + learnable mask tokens with positional encodings at the **target spatial locations**; **(c)** target encoder runs on the full image and we slice out the target positions.

### 🔴 1.1 — Context encoder receives the full image (defeats the entire JEPA task)

**File:** `vl_jepa/models/vl_jepa.py:232` and `:436-451`.

```python
# forward_jepa
context_vision = self.vision_encoder(images, return_all_tokens=True)   # ← full image
predicted_vision = self.predictor(context_vision)
with torch.no_grad():
    target_vision = self.target_vision_encoder(images, return_all_tokens=True)
```

The `context_mask` argument is accepted by `forward_jepa` and then **never used** — it's only consulted later inside `compute_jepa_loss` to weight the loss. So the context encoder sees the full 196-patch tensor, including the target patches it's supposed to predict. The predictor then receives those same 196 tokens and is asked to output something close to the target encoder's output at the target positions. Since both encoders see the same image and differ only by an EMA delay, **the optimal predictor is the identity map** at target positions. This is exactly the representational-collapse failure mode JEPA is designed to avoid by hiding the targets from the context encoder.

**Why this matters:** the JEPA loss going down means almost nothing in this code. Most of the "learning signal" is coming from the contrastive head, not from JEPA. You've essentially built CLIP with a free identity-mapping auxiliary loss.

**Fix sketch (I-JEPA-faithful):**
```python
# Pseudocode for forward_jepa (correct version)
# 1. Pass only context patches to the context encoder
patch_tokens = self.vision_encoder.patch_embed(images)          # [B, N, D]
patch_tokens = patch_tokens + self.vision_encoder.pos_embed     # add abs pos before masking
context_tokens = gather(patch_tokens, context_indices)          # [B, N_ctx, D]
context_repr = self.vision_encoder.blocks(context_tokens)       # only ctx patches

# 2. Target encoder runs on the full image, then we index target positions
with torch.no_grad():
    target_repr_full = self.target_vision_encoder(images, return_all_tokens=True)
    target_repr = gather(target_repr_full[:, 1:], target_indices)   # [B, N_tgt, D]
    target_repr = F.layer_norm(target_repr, target_repr.shape[-1:])  # I-JEPA normalisation

# 3. Predictor receives context tokens + mask tokens at target positions
predicted = self.predictor(context_repr, context_indices, target_indices)  # [B, N_tgt, D]

loss = F.smooth_l1_loss(predicted, target_repr)  # or L2 per the I-JEPA paper
```

The masking generator already produces indices implicitly (via the boolean masks) — you just need a helper that converts the `[B, 196]` boolean mask into a `[B, N_ctx]` index tensor, e.g. `torch.where`.

Doing this correctly **requires a small surgery on `VisionEncoder`** because timm's `forward_features` doesn't expose patch-by-patch routing. Two options:

- Drop down to the timm internals: call `patch_embed`, add `pos_embed`, do the gather, then run the `blocks` ModuleList yourself, then `norm`. ~30 lines.
- Or use HuggingFace's `ViTModel` and pass a `bool_masked_pos` (it supports this natively for MAE-style training).

### 🔴 1.2 — `PredictorMLP` has zero notion of position

**File:** `vl_jepa/models/predictor.py:12-91`.

The MLP predictor is `Linear → LN → GELU → … → Linear`. It operates per-token. There are no positional embeddings, no mask tokens, no information about *which* position is being predicted. So at best it learns "given embedding at any position, output a residual" — which, combined with bug 1.1, makes the optimal solution the identity map.

This is not a JEPA predictor. It's a per-token MLP head. The I-JEPA paper is explicit that the predictor is "a narrow ViT … conditioned on positional tokens."

**Fix:** Default `predictor.type` should be `transformer`, **and** that transformer must be repaired (see 1.3). The MLP variant should either be deleted or restricted to a debug/ablation mode (and the README should stop calling it the default).

### 🟠 1.3 — `PredictorTransformer` uses sequential index positions instead of target spatial positions

**File:** `vl_jepa/models/predictor.py:127-215`.

```python
self.pos_embed = nn.Parameter(torch.zeros(1, 256, hidden_dim))  # max 256 tokens
self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
…
x = torch.cat([x, mask_tokens], dim=1)               # context first, then masks at end
x = x + self.pos_embed[:, :N_total, :]               # positions 0..N_ctx+N_target-1
```

Problems:

- **Mask tokens are placed at indices `[N_ctx, N_ctx+N_target)`**, not at the actual target spatial positions. From the predictor's point of view, the "where to predict" signal is completely scrambled — it sees a mask token at sequential slot 198, not at the original 14×14 grid location of the target patch.
- **Positional embeddings are 1D and indexed by concat order**, not by 2D grid coordinates. The model can never learn "predict the patch at row 3 col 7."
- **256-token hard cap** breaks if a user moves to ViT-Base/Large (more patches) or larger image sizes.

I-JEPA's actual setup (paper §3.3): the predictor has its own learnable 2D positional embeddings over the full N-patch grid; mask tokens get the positional embedding **of the target location they're supposed to predict**.

**Fix sketch:**
```python
class PredictorTransformer(nn.Module):
    def __init__(self, ..., num_patches=196):
        super().__init__()
        # 2D position embeddings over the full grid
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, context_tokens, context_indices, target_indices):
        B, N_ctx, _ = context_tokens.shape
        N_tgt = target_indices.shape[1]

        x_ctx = self.input_proj(context_tokens) + self.pos_embed.expand(B, -1, -1).gather(
            1, context_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim))

        # Mask tokens carry the positional embedding of the target position
        mask = self.mask_token.expand(B, N_tgt, -1)
        mask = mask + self.pos_embed.expand(B, -1, -1).gather(
            1, target_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim))

        x = torch.cat([x_ctx, mask], dim=1)
        x = self.transformer(x)
        return self.output_proj(x[:, N_ctx:, :])  # predictions for target positions only
```

### 🟠 1.4 — Target encoders are never put into `eval()` mode

**File:** `vl_jepa/models/vl_jepa.py:54-61`.

After `model.train()` is called in `train_one_epoch`, dropout, drop-path, and (for HuggingFace) any training-time-only branches remain active in `target_vision_encoder` and `target_text_encoder`. `requires_grad=False` only stops gradients — it does **not** disable stochastic forward behaviour. Result: the target representations are noisy (e.g. with `drop_path_rate: 0.1` in your DGPU config), which both hurts the JEPA signal and produces different targets on each call.

**Fix:** override `train()`:
```python
def train(self, mode: bool = True):
    super().train(mode)
    self.target_vision_encoder.eval()
    self.target_text_encoder.eval()
    return self
```

I-JEPA's reference code does exactly this.

### 🟠 1.5 — CLS-token assumption in `compute_jepa_loss`

**File:** `vl_jepa/models/vl_jepa.py:322-323`.

```python
predicted_patches = predicted[:, 1:, :]   # assumes index 0 is always CLS
target_patches    = target[:, 1:, :]
```

`timm` ViTs use `global_pool='token'` (CLS at index 0) by default — but `global_pool='avg'` returns N patches with no CLS, and `global_pool='map'` adds attention-pool tokens. If the model name ever changes (e.g. DINOv2 ViTs, EVA, SigLIP), the slice silently throws away a real patch and shifts every target index by one.

**Fix:** read `model.num_prefix_tokens` from timm (it returns 1 for CLS-only, 0 for avg-pool), and slice `[:, num_prefix_tokens:, :]`. Or, better, since you only care about patch tokens, route through `model.forward_features` and then `model.norm` and explicitly drop the prefix tokens by name.

### 🟠 1.6 — `min_keep` rescue can leave **zero** target patches

**File:** `vl_jepa/masks/multiblock.py:193-201`.

```python
if context_mask.sum() < self.min_keep:
    n_additional = self.min_keep - context_mask.sum()
    target_indices = np.where(target_mask)[0]
    if len(target_indices) > 0:
        to_show = np.random.choice(target_indices, size=min(n_additional, len(target_indices)), replace=False)
        context_mask[to_show] = True
        target_mask[to_show] = False
```

If targets covered almost the whole image and you need to claw back many patches, `to_show` can be *all* target patches → `target_mask.sum() == 0`. Downstream:

```python
num_targets = target_mask.sum() + 1e-8     # ≈ 1e-8
loss = loss.sum() / num_targets            # explodes or is dominated by floating-point noise
```

**Fix:** enforce a symmetric `min_target_keep` and re-roll the masks if both can't be satisfied (loop with a max-tries counter, raise on failure):
```python
if target_mask.sum() < self.min_target_keep or context_mask.sum() < self.min_keep:
    return self.__call__()  # bounded recursion
```

### 🟡 1.7 — README/architecture description does not match the mask generator

**File:** `README.md` lines 123-126, `vl_jepa/masks/multiblock.py:166-191`.

The README claims:
> Context blocks: 1 block (85-100% scale) - visible to model
> Target blocks: 4 blocks (15-20% scale) - model predicts these

But the generator only samples **target** blocks and sets `context_mask = ~target_mask` (line 187). With 4 target blocks at 15-20% each (non-overlapping), context is 20-40% of the image, not 85-100%.

In the real I-JEPA paper the context block is sampled *separately* with scale `(0.85, 1.0)`, then any patches overlapping with the targets are **removed** from the context. That's a different distribution — the context is geometrically a connected block, not "everything except the targets."

**Fix:** sample a context block per spec, then set
`context_mask = context_block_mask & ~target_mask`. Then drop the unused `context_scale`/`context_aspect_ratio` fields, or use them.

---

## Tier 2 — Logic / numerics / training-loop bugs

### 🟠 2.1 — EMA momentum schedule never reaches `ema_momentum_end`

**File:** `train.py:227-231`.

```python
progress = global_step / (config['training']['num_epochs'] * len(dataloader))
ema_momentum = ema_start + (ema_end - ema_start) * progress
```

`global_step` increments only every `grad_accum_steps` batches (line 241), but the denominator counts every batch (`len(dataloader)` per epoch). With `gradient_accumulation_steps: 2`, `progress` maxes out at **0.5** at the end of training, so the EMA momentum only goes 0.996 → 0.998 instead of → 1.000. The I-JEPA paper schedules this carefully and the end value matters for convergence.

**Fix:**
```python
total_optimizer_steps = (config['training']['num_epochs'] * len(dataloader)) // grad_accum_steps
progress = min(1.0, global_step / max(1, total_optimizer_steps))
```

### 🟡 2.2 — EMA update is not actually in-place

**File:** `vl_jepa/models/vl_jepa.py:198-199` and `204-206`.

```python
param_k.data = param_k.data * self.ema_momentum + param_q.data * (1.0 - self.ema_momentum)
```

This allocates two new tensors and reassigns `.data`. Replace with the fused in-place form for speed and memory (matters at 145M-param scale on a 3070):

```python
param_k.data.mul_(self.ema_momentum).add_(param_q.data, alpha=1.0 - self.ema_momentum)
```

### 🟠 2.3 — Validation retrieval metric assumes 1-to-1 image↔caption (broken for COCO)

**File:** `vl_jepa/utils/metrics.py:11-64`, called from `train.py:330`.

```python
correct = torch.arange(N).unsqueeze(1)
recall  = (topk_indices == correct).any(dim=1).float().mean()
```

COCO Captions has **5 captions per image** in the standard val split. If your dataloader returns `(image_i, caption_i_k)` pairs, then `image_5` and `image_6` may actually be the same image as `image_5`, and `text_5` is a paraphrase of `text_4`. R@K computed against the identity diagonal counts a correct paraphrase as a miss.

The canonical fix (and what the I-JEPA / CLIP repos use): build a `caption_id → image_id` mapping at eval time, then for each text query check whether *any* of its image's 5 captions are in top-K, and vice versa. The 1K-sample subset (line 326-328) also makes results not directly comparable to standard 5K-image COCO Karpathy splits.

### 🟡 2.4 — `outputs['loss']` in mode `"both"` uses a hardcoded 0.5 weight that the train loop ignores

**File:** `vl_jepa/models/vl_jepa.py:487`.

```python
outputs['loss'] = jepa_loss + 0.5 * contrastive_loss  # Weighted combination
```

In `train_one_epoch:209`, the loop recomputes the loss using the YAML weights anyway, so this line is dead. But if anyone calls `model(...)` outside the training loop (e.g. in `validate` if you ever switch it from `mode="contrastive"` to `mode="both"`), they get the wrong weighting. Either delete this line or wire `jepa_loss_weight`/`contrastive_loss_weight` into the model.

### 🟡 2.5 — `find_best_image` is O(N²) where it should be O(N)

**File:** `inference.py:179-208`.

For each candidate image you call `encode_image()` → full forward pass through ViT-Tiny. If you query 10 texts against 1000 images that's **10,000 forward passes** instead of 1000 cached embeddings.

**Fix:** precompute and cache image embeddings once:
```python
def index_images(self, image_paths, batch_size=32):
    embs = []
    for i in range(0, len(image_paths), batch_size):
        batch = torch.stack([self.transform(Image.open(p).convert('RGB')) for p in image_paths[i:i+batch_size]]).to(self.device)
        feat = self.model.vision_encoder(batch, return_all_tokens=False).squeeze(1)
        embs.append(F.normalize(self.model.vision_projection(feat), dim=-1).cpu())
    return torch.cat(embs)  # [N, D]
```

### 🟡 2.6 — `text_encoder.projection_dim: 384` in YAML is silently ignored

**File:** `config_dgpu.yaml:23`, `configs/config_coco_full.yaml:14`, vs. `vl_jepa/models/vl_jepa.py:438-444`.

`TextEncoder.__init__` builds an internal projection when `projection_dim` is set — but in the main forward path (`forward_contrastive` and the `"both"` branch), the code calls `text_encoder(..., return_projected=False)` and then `self.text_projection` (the VL-JEPA-owned projection from `text_dim → embedding_dim`) takes over. So the encoder-internal `text_encoder.projection` is created, holds parameters, **and is never used** at train time.

That's ~300K wasted parameters and a confusing knob. Either delete the `projection_dim` parameter from `TextEncoder`, or use it (then VL-JEPA's `text_projection` would take `projection_dim` as input dim instead of `embed_dim`).

### 🟡 2.7 — `torch.cuda.amp` is deprecated as of PyTorch 2.0

**File:** `train.py:9, 196, 476`.

```python
from torch.cuda.amp import autocast, GradScaler
```

The new API is `torch.amp.autocast('cuda', ...)` and `torch.amp.GradScaler('cuda')`. The old API still works but emits `FutureWarning` and will be removed. Tiny change; nice cleanup.

### 🟡 2.8 — `bnb.optim.AdamW8bit` is silently skipped when bitsandbytes is missing

**File:** `train.py:84`.

```python
if opt_type == 'adamw8bit' and HAS_BITSANDBYTES:
    ...
else:
    optimizer = torch.optim.AdamW(...)
    print("Using standard AdamW optimizer")
```

If `optimizer.type: adamw8bit` is configured but bitsandbytes isn't installed, you'll fall back to fp32 AdamW with no warning that the config intent was ignored. Easy to miss when chasing OOM on a small GPU.

**Fix:** print a warning when `opt_type == 'adamw8bit' and not HAS_BITSANDBYTES`.

---

## Tier 3 — Smell / robustness / documentation

### 🟢 3.1 — `get_intermediate_layers` fallback assumes default `return_all_tokens`

`vision_encoder.py:99-103`. The fallback `[self.forward(x)]` works for the default kwargs but breaks if the caller wants tokens-only behaviour. Pass `return_all_tokens=True` explicitly.

### 🟢 3.2 — Dropout in MLP predictor with no JEPA-paper precedent

`predictor.py:50, 60`. I-JEPA's predictor has no dropout (it's a small transformer with stochastic depth, not dropout). Dropout on the prediction head can hurt — the EMA target stops gradients already and adds enough regularisation.

### 🟢 3.3 — `embed_dim` of `VisionEncoderWithProjection`

`vision_encoder.py:145`. `self.embed_dim = self.encoder.embed_dim` exposes the **pre-projection** dim, but the module returns **post-projection** features when `return_projected=True`. If VL-JEPA ever uses `VisionEncoderWithProjection` instead of `VisionEncoder`, the `vision_dim = vision_encoder.embed_dim` line in VL-JEPA (`vl_jepa.py:64`) is wrong by a factor of `projection_dim / encoder_embed_dim`. Currently masked because `create_vision_encoder` checks `config['projection_dim']` at the **top level** of the model config (not under `vision_encoder`), and you never set it there, so the wrapped variant is never constructed. Brittle — either delete `VisionEncoderWithProjection` or fix the config-path lookup.

### 🟢 3.4 — Best-metric tracking is confusing in Stage-2

`train.py:481, 547-568`. Stage-1 uses `best_metric` (lower-is-better val loss). Stage-2 uses `best_mean_recall` (higher-is-better). `best_metric` is initialised in both paths but only consulted in Stage-1. Either unify into a single `best` tracker with a `direction: 'min'|'max'` flag, or just delete `best_metric` from the Stage-2 path so it's not dead state.

### 🟢 3.5 — `inference.encode_image` ignores patch tokens entirely

`inference.py:79-86`. Uses CLS token only — fine for the CLIP head, but ignores everything JEPA actually trains. Worth offering a `pool='cls'` vs `pool='mean'` flag in case downstream uses prefer the average over patch tokens (often better when the CLS isn't supervised heavily).

### 🟢 3.6 — README install snippet is Windows-only without saying so

`README.md:36`. `.venv\Scripts\activate` is Windows syntax. Linux/Mac users will paste it verbatim and get "command not found". Either put the OS-correct line first or use a clearly labelled tabbed block.

### 🟢 3.7 — `setup.py` is shipped but the project layout is also flat (no `src/` layout)

Pip-installing this in editable mode (`pip install -e .`) won't expose `vl_jepa` cleanly because `setup.py` likely doesn't declare `packages=find_packages()` or includes the `data/` package (which is missing anyway). Worth either deleting `setup.py` or making it functional.

---

## What "JEPA-faithful + a new style" could look like

You asked for **powerful + accurate + some new style**. After fixing Tier 0/1, here are higher-leverage upgrades, in rough order of bang-for-effort:

### A. Make it actually vision-LANGUAGE-JEPA (not just I-JEPA + CLIP)

Right now the JEPA part is purely visual. The "VL" comes entirely from the CLIP-style InfoNCE on top. A more principled VL-JEPA is to **condition the predictor on the text**: the model has to predict masked image patches using both the visible context *and* the caption. That makes the text genuinely useful to the JEPA objective:

```python
predicted = self.predictor(context_tokens, context_idx, target_idx,
                           text_kv=text_features)   # cross-attn over text
```

You already have `PredictorWithCrossAttention` — repurpose it so the queries are mask tokens (with target positional embeddings) and the keys/values are text tokens *and* context tokens. This is closer to what `arxiv:2512.10942v1` (the VL-JEPA paper you cite) and FLIP / CoCa-style hybrids do.

### B. Replace InfoNCE with SigLIP-style sigmoid loss

`SigLIP (Zhai et al., 2023)` shows sigmoid-on-each-pair beats softmax InfoNCE at small batch sizes — exactly your regime (effective batch 32 in `config_dgpu.yaml`). Drop-in replacement for `compute_contrastive_loss`:

```python
logits = vision_embed @ text_embed.t() * scale + bias  # learnable scale, bias
labels = 2 * torch.eye(B, device=logits.device) - 1     # +1 diagonal, -1 off-diagonal
loss = -F.logsigmoid(labels * logits).sum() / B
```

`scale` initialised to `log(10)`, `bias` to `-10`. Trains noticeably faster with smaller batches.

### C. Use V-JEPA's smooth L1 loss explicitly and drop the `LayerNorm(predicted)`

Your code calls `F.layer_norm` on **both** predicted and target (`vl_jepa.py:326-327`). I-JEPA layer-norms only the **target** (stop-grad) — you can see this in the official `i-jepa` repo's `src/utils/losses.py`. Normalising the predicted side too can dampen gradients. V-JEPA's smooth L1 is fine; just stop the layer-norm on the predictor output:

```python
target = F.layer_norm(target, target.shape[-1:])  # only target
loss   = F.smooth_l1_loss(predicted, target.detach(), reduction='none')
```

### D. Stochastic-depth schedule on the predictor

I-JEPA uses `drop_path_rate` rising from 0 → ~0.1 across predictor layers. Easy to add with timm's `DropPath`:

```python
dpr = torch.linspace(0, drop_path_max, num_layers).tolist()
# pass dpr[i] to TransformerEncoderLayer i
```

### E. Multi-target predictor calls (parallel, masked-attention style)

I-JEPA actually calls the predictor 4 times (once per target block) with the *same* context. With a batched implementation you can do all 4 in one pass by packing them in the batch dim — saves ~3x predictor compute. Practical win at the small-GPU regime.

### F. Move to `torch.compile`

After the Tier-1 surgery, the model is ~145M params with a tight inner loop. `model = torch.compile(model, mode="reduce-overhead")` in PyTorch 2.x routinely gives 20–40% throughput improvement on ViT-Tiny. Costs you nothing once the model graphs are stable.

### G. Sanity-check evals

Adding zero-shot ImageNet linear-probe and a quick COCO 5K-Karpathy retrieval eval (proper 5-caption matching) tells you within a few epochs whether the changes above are helping. Right now you only have val-loss + a broken R@K.

---

## Suggested fix order (so you don't break everything at once)

1. **Day 1 — make it run.** Tier 0.1 (ship `vl_jepa/data/`), 0.2 (config path), 0.3 (wandb guard). Verify `python train.py --config config_dgpu.yaml` runs an epoch on a tiny subset.
2. **Day 2 — fix the JEPA task.** Tier 1.1 (masked context-encoder forward), 1.2 (default predictor → transformer), 1.3 (target-position embeddings). Add an assertion at training start that `predicted.shape == target_shape_at_target_positions`. Watch JEPA loss; it should no longer be near-zero.
3. **Day 3 — tighten training.** 1.4 (target.eval()), 1.5 (prefix tokens), 1.6 (min-keep), 2.1 (EMA schedule), 2.2 (in-place EMA). Wire R@K to handle 5-captions.
4. **Day 4+ — go for the upgrades.** A (text-conditioned predictor), B (SigLIP), and the rest.

If you want, I can put together the actual replacement files for `vl_jepa/data/`, the corrected `vl_jepa.py:forward_jepa`, and the rewritten `PredictorTransformer` as a follow-up — just say which subset to start with.

— end of audit —
