"""
One-off regression smoke runner used by the test-report agent.

Builds a tiny VL-JEPA from config_dgpu.yaml (pretrained=False so no downloads),
exercises jepa / both / siglip+cross-attention forward passes, builds the
optimizer, and round-trips a checkpoint through save + load. Prints PASS / FAIL
per check; non-zero exit on any failure.

Not part of the pytest suite (network model deps + checkpoint IO are too heavy
to run on every test invocation). Used as a manual gate.
"""

import os
import sys
import tempfile
import math
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vl_jepa.utils.config import load_config, validate_config
from vl_jepa.models.vl_jepa import create_vl_jepa_model
from vl_jepa.models import VisionEncoder, TextEncoder, VLJEPAModel
from vl_jepa.models.predictor import PredictorTransformer, PredictorWithCrossAttention
from vl_jepa.masks import MultiBlockMaskGenerator
from vl_jepa.utils.checkpoint import save_checkpoint, load_checkpoint


FAILS = []


def report(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILS.append((name, detail))


def _build_dgpu_model_no_download():
    cfg = load_config(ROOT / "config_dgpu.yaml")
    validate_config(cfg)
    # Force pretrained off so timm doesn't hit the network.
    cfg["model"]["vision_encoder"]["pretrained"] = False
    return cfg, create_vl_jepa_model(cfg)


def _make_batch(B=2):
    gen = MultiBlockMaskGenerator(input_size=224, patch_size=16)
    images = torch.randn(B, 3, 224, 224)
    ids = torch.randint(0, 1000, (B, 32))
    attn = torch.ones(B, 32)
    ctx, tgt = [], []
    for _ in range(B):
        _, _, ci, ti = gen()
        ctx.append(ci)
        tgt.append(ti)
    return images, ids, attn, torch.stack(ctx), torch.stack(tgt)


def check_jepa_forward():
    _, model = _build_dgpu_model_no_download()
    model.eval()
    images, ids, attn, ctx, tgt = _make_batch()
    with torch.no_grad():
        out = model(images, ids, attn, context_indices=ctx, target_indices=tgt, mode="jepa")
    loss = out["loss"].item()
    ok = math.isfinite(loss) and loss > 0
    report("jepa forward", ok, f"loss={loss:.4f}")


def check_both_forward():
    _, model = _build_dgpu_model_no_download()
    model.eval()
    images, ids, attn, ctx, tgt = _make_batch()
    with torch.no_grad():
        out = model(images, ids, attn, context_indices=ctx, target_indices=tgt, mode="both")
    jepa = out["jepa_loss"].item()
    cl = out["contrastive_loss"].item()
    ok = math.isfinite(jepa) and math.isfinite(cl) and jepa > 0 and cl > 0
    report("both forward", ok, f"jepa={jepa:.4f}, contrastive={cl:.4f}")


def check_siglip_cross_attention():
    # Build by hand so we can mix siglip + cross-attention predictor without
    # touching the dgpu yaml.
    torch.manual_seed(0)
    vision = VisionEncoder(pretrained=False, gradient_checkpointing=False)
    text = TextEncoder(gradient_checkpointing=False)
    predictor = PredictorWithCrossAttention(
        input_dim=192, text_dim=768, hidden_dim=384, output_dim=192,
        num_layers=2, num_heads=6, num_patches=196,
    )
    model = VLJEPAModel(vision, text, predictor, contrastive_loss_type="siglip")
    model.eval()
    images, ids, attn, ctx, tgt = _make_batch()
    with torch.no_grad():
        out = model(images, ids, attn, context_indices=ctx, target_indices=tgt, mode="both")
    jepa = out["jepa_loss"].item()
    cl = out["contrastive_loss"].item()
    ok = math.isfinite(jepa) and math.isfinite(cl) and jepa > 0
    report("siglip + cross-attention", ok, f"jepa={jepa:.4f}, contrastive={cl:.4f}")


def check_param_groups_integrity():
    torch.manual_seed(0)
    vision = VisionEncoder(pretrained=False, gradient_checkpointing=False)
    text = TextEncoder(gradient_checkpointing=False)
    predictor = PredictorTransformer(input_dim=192, hidden_dim=384, output_dim=192,
                                     num_layers=2, num_heads=6, num_patches=196)
    model = VLJEPAModel(vision, text, predictor, contrastive_loss_type="siglip")
    groups = model.get_parameter_groups(base_lr=1e-3, stage2=False)
    group_param_ids = {id(p) for g in groups for p in g["params"]}

    # SigLIP params present
    ok_siglip = id(model.siglip_logit_scale) in group_param_ids and \
                id(model.siglip_logit_bias) in group_param_ids
    report("siglip params in groups", ok_siglip)

    # Sum of numel across groups equals sum of trainable params on the model.
    n_group = sum(p.numel() for g in groups for p in g["params"])
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ok_count = n_group == n_train
    report("param-group integrity", ok_count, f"group={n_group}, trainable={n_train}")


def check_checkpoint_roundtrip():
    cfg, model = _build_dgpu_model_no_download()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ckpt.pth"
        save_checkpoint(
            model=model, optimizer=optimizer, scheduler=sched,
            epoch=1, global_step=42, best_metric=0.5,
            config=cfg, save_path=str(path), is_best=False,
        )

        # Rebuild fresh + load.
        cfg2, model2 = _build_dgpu_model_no_download()
        info = load_checkpoint(str(path), model2, optimizer=None, scheduler=None, device="cpu")
        ok_step = info["global_step"] == 42 and info["epoch"] == 1
        report("checkpoint round-trip metadata", ok_step,
               f"global_step={info['global_step']}, epoch={info['epoch']}")

        # Weight equality on a sample param.
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            if n1 != n2:
                report("checkpoint round-trip names", False, f"{n1} vs {n2}")
                return
            if not torch.allclose(p1.data, p2.data):
                report("checkpoint round-trip weights", False, f"diverged at {n1}")
                return
        report("checkpoint round-trip weights", True)


def main():
    check_jepa_forward()
    check_both_forward()
    check_siglip_cross_attention()
    check_param_groups_integrity()
    check_checkpoint_roundtrip()
    if FAILS:
        print(f"\n{len(FAILS)} check(s) failed.")
        sys.exit(1)
    print("\nAll regression smokes passed.")


if __name__ == "__main__":
    main()
