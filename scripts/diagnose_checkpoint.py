"""
Diagnose a VL-JEPA checkpoint for contrastive collapse.

Reads a .pth checkpoint, inspects projection-head weight statistics, then
runs a small batch through both online encoders to measure:
  - Projection weight norms / std (collapse on the weight side)
  - Vision and text CLS embedding statistics (collapse on the feature side)
  - The cosine-similarity matrix between vision and text embeddings (the
    actual InfoNCE input). Contrastive loss = log(N) means this matrix is
    essentially constant; we can confirm that directly here.

Usage:
    python scripts/diagnose_checkpoint.py <path-to-checkpoint.pth>
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vl_jepa.models.vl_jepa import create_vl_jepa_model
from vl_jepa.utils.config import load_config


def main(ckpt_path: str, config_path: str):
    print(f"Loading checkpoint: {ckpt_path}")
    print(f"Using config:       {config_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print(f"  epoch        = {ckpt.get('epoch')}")
    print(f"  global_step  = {ckpt.get('global_step')}")
    print(f"  best_metric  = {ckpt.get('best_metric')}")

    config = load_config(config_path)
    model = create_vl_jepa_model(config)

    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt['state_dict']
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  missing keys     = {len(missing)} (first 3: {missing[:3]})")
    print(f"  unexpected keys  = {len(unexpected)} (first 3: {unexpected[:3]})")
    model.eval()

    print("\n--- Projection head weights ---")
    for name, p in model.named_parameters():
        if 'projection' in name:
            w = p.detach()
            print(f"  {name:50s} shape={tuple(w.shape)}  "
                  f"mean={w.mean().item():+.4e}  std={w.std().item():.4e}  "
                  f"norm={w.norm().item():.4e}")

    print("\n--- Optimizer scheduler state ---")
    sched_state = ckpt.get('scheduler_state_dict') or {}
    print(f"  last_lr (sched)   = {sched_state.get('_last_lr')}")
    print(f"  last_epoch        = {sched_state.get('last_epoch')}")
    print(f"  T_max             = {sched_state.get('T_max')}")
    print(f"  eta_min           = {sched_state.get('eta_min')}")
    print(f"  base_lrs          = {sched_state.get('base_lrs')}")

    print("\n--- Forward pass on synthetic batch (B=16) ---")
    torch.manual_seed(0)
    B = 16
    images = torch.randn(B, 3, 224, 224)
    text_ids = torch.randint(100, 30000, (B, 32))
    text_mask = torch.ones_like(text_ids)

    with torch.no_grad():
        v_cls_raw = model.vision_encoder(images, return_all_tokens=False).squeeze(1)
        t_cls_raw = model.text_encoder(text_ids, text_mask, return_all_tokens=False, return_projected=False)
        out = model(
            images=images,
            text_input_ids=text_ids,
            text_attention_mask=text_mask,
            mode='contrastive',
        )
    print(f"  vision encoder CLS (pre-proj)  std={v_cls_raw.std().item():.4e}  "
          f"row-norm std={v_cls_raw.norm(dim=1).std().item():.4e}  "
          f"off-diag cos={((v_cls_raw / v_cls_raw.norm(dim=1, keepdim=True)) @ (v_cls_raw / v_cls_raw.norm(dim=1, keepdim=True)).t()).fill_diagonal_(0).sum().item() / (B*B - B):+.4f}")
    print(f"  text   encoder CLS (pre-proj)  std={t_cls_raw.std().item():.4e}  "
          f"row-norm std={t_cls_raw.norm(dim=1).std().item():.4e}  "
          f"off-diag cos={((t_cls_raw / t_cls_raw.norm(dim=1, keepdim=True)) @ (t_cls_raw / t_cls_raw.norm(dim=1, keepdim=True)).t()).fill_diagonal_(0).sum().item() / (B*B - B):+.4f}")
    v = out['vision_embed']
    t = out['text_embed']
    print(f"  vision_embed  shape={tuple(v.shape)}  mean={v.mean().item():+.4e}  std={v.std().item():.4e}")
    print(f"  text_embed    shape={tuple(t.shape)}  mean={t.mean().item():+.4e}  std={t.std().item():.4e}")
    print(f"  vision row-norm std (across batch): {v.norm(dim=1).std().item():.4e}")
    print(f"  text   row-norm std (across batch): {t.norm(dim=1).std().item():.4e}")

    cos_sim = v @ t.t()  # already L2-normalized
    print(f"\n--- Cosine sim matrix (B x B) ---")
    print(f"  mean       = {cos_sim.mean().item():+.4f}")
    print(f"  std        = {cos_sim.std().item():.4e}")
    print(f"  diag mean  = {cos_sim.diag().mean().item():+.4f}")
    print(f"  off mean   = {((cos_sim.sum() - cos_sim.diag().sum()) / (B * B - B)).item():+.4f}")
    print(f"  min        = {cos_sim.min().item():+.4f}")
    print(f"  max        = {cos_sim.max().item():+.4f}")

    temperature = config['model'].get('temperature', 0.07)
    logits = cos_sim / temperature
    labels = torch.arange(B)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    print(f"\n--- InfoNCE (T={temperature}) on synthetic batch ---")
    print(f"  i2t loss   = {loss_i2t.item():.4f}")
    print(f"  t2i loss   = {loss_t2i.item():.4f}")
    print(f"  combined   = {((loss_i2t + loss_t2i) / 2).item():.4f}   (log(B)={torch.log(torch.tensor(float(B))).item():.4f})")

    print("\n--- Per-embedding cosine spread ---")
    v_pairs = v @ v.t()
    t_pairs = t @ t.t()
    v_off = (v_pairs.sum() - v_pairs.diag().sum()) / (B * B - B)
    t_off = (t_pairs.sum() - t_pairs.diag().sum()) / (B * B - B)
    print(f"  vision-vision off-diag mean = {v_off.item():+.4f}  (1.0 == all images collapsed to one direction)")
    print(f"  text-text     off-diag mean = {t_off.item():+.4f}  (1.0 == all captions collapsed to one direction)")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    ckpt = sys.argv[1]
    cfg = sys.argv[2] if len(sys.argv) > 2 else 'configs/config_kaggle_t4.yaml'
    main(ckpt, cfg)
