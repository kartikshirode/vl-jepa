"""
Phase 1 verification gate.

Tests that the JEPA objective is genuine:
  - The context encoder receives only the visible patches (not the full image).
  - The predictor returns one prediction per target position.
  - The target encoders stay in eval mode after model.train().
  - The loss on a random-init model is non-trivial (rules out identity collapse).
  - Gradients flow into the predictor but NOT into the target encoders.
"""

import torch
import pytest

from vl_jepa.models import VisionEncoder, TextEncoder, VLJEPAModel
from vl_jepa.models.predictor import PredictorTransformer
from vl_jepa.masks import MultiBlockMaskGenerator


def _build_tiny_model():
    """Small VL-JEPA model with no pretrained downloads."""
    vision = VisionEncoder(pretrained=False, gradient_checkpointing=False)
    text = TextEncoder(gradient_checkpointing=False)
    predictor = PredictorTransformer(
        input_dim=192, hidden_dim=384, output_dim=192,
        num_layers=2, num_heads=6, num_patches=196,
    )
    return VLJEPAModel(vision, text, predictor)


def _build_batch(batch_size=2):
    mask_gen = MultiBlockMaskGenerator(input_size=224, patch_size=16)
    images = torch.randn(batch_size, 3, 224, 224)
    ids = torch.randint(0, 1000, (batch_size, 32))
    attn = torch.ones(batch_size, 32)
    ctx_idxs, tgt_idxs = [], []
    for _ in range(batch_size):
        _, _, ci, ti = mask_gen()
        ctx_idxs.append(ci); tgt_idxs.append(ti)
    return images, ids, attn, torch.stack(ctx_idxs), torch.stack(tgt_idxs)


def test_context_encoder_input_size():
    """forward_context returns one token per context index, not per full patch."""
    vision = VisionEncoder(pretrained=False, gradient_checkpointing=False)
    images = torch.randn(2, 3, 224, 224)
    mask_gen = MultiBlockMaskGenerator()
    ctx_idx = torch.stack([mask_gen()[2] for _ in range(2)])
    with torch.no_grad():
        out = vision.forward_context(images, ctx_idx)
    assert out.shape == (2, ctx_idx.shape[1], vision.embed_dim), (
        f"context encoder output {tuple(out.shape)} != expected "
        f"(2, {ctx_idx.shape[1]}, {vision.embed_dim})"
    )
    # And explicitly NOT the full 196 patches.
    assert out.shape[1] < 196, "context encoder is still processing every patch"


def test_predictor_output_shape():
    """Predictor returns exactly N_tgt predictions, regardless of N_ctx."""
    predictor = PredictorTransformer(
        input_dim=192, hidden_dim=384, output_dim=192,
        num_layers=2, num_heads=6, num_patches=196,
    )
    B, N_ctx, N_tgt = 3, 50, 80
    ctx_tokens = torch.randn(B, N_ctx, 192)
    ctx_idx = torch.randint(0, 196, (B, N_ctx))
    tgt_idx = torch.randint(0, 196, (B, N_tgt))
    with torch.no_grad():
        out = predictor(ctx_tokens, ctx_idx, tgt_idx)
    assert out.shape == (B, N_tgt, 192)


def test_target_encoder_eval_mode():
    """train() must leave the EMA target encoders in eval()."""
    model = _build_tiny_model()
    model.train()
    assert not model.target_vision_encoder.training, "target vision encoder leaked into train mode"
    assert not model.target_text_encoder.training, "target text encoder leaked into train mode"


def test_predictor_not_identity():
    """
    With random initialization and a real I-JEPA mask, the JEPA loss should be
    non-trivial. The previous (broken) setup had the predictor collapse to the
    identity map and the loss vanish; here we assert a reasonable floor.
    """
    torch.manual_seed(0)
    model = _build_tiny_model()
    model.eval()
    images, ids, attn, ctx_idx, tgt_idx = _build_batch(batch_size=2)
    with torch.no_grad():
        out = model(
            images, ids, attn,
            context_indices=ctx_idx, target_indices=tgt_idx,
            mode='jepa',
        )
    loss = out['loss'].item()
    assert loss > 0.05, (
        f"JEPA loss {loss:.4f} too small for random init; possible identity collapse"
    )


def test_gradient_flow():
    """Gradients reach the predictor; target encoders stay frozen."""
    torch.manual_seed(0)
    model = _build_tiny_model()
    model.train()
    images, ids, attn, ctx_idx, tgt_idx = _build_batch(batch_size=2)
    out = model(
        images, ids, attn,
        context_indices=ctx_idx, target_indices=tgt_idx,
        mode='jepa',
    )
    out['loss'].backward()

    pred_grads = [p.grad for p in model.predictor.parameters() if p.grad is not None]
    assert pred_grads, "no gradients reached the predictor"
    assert any(g.abs().sum().item() > 0 for g in pred_grads), "predictor gradients all zero"

    for name, p in model.target_vision_encoder.named_parameters():
        assert p.grad is None, f"target_vision_encoder.{name} got a gradient"
    for name, p in model.target_text_encoder.named_parameters():
        assert p.grad is None, f"target_text_encoder.{name} got a gradient"


def test_mask_generator_disjoint():
    """Context and target masks are always disjoint after the carve-out."""
    mask_gen = MultiBlockMaskGenerator()
    for _ in range(20):
        ctx, tgt, _, _ = mask_gen()
        assert (ctx & tgt).sum().item() == 0, "context and target overlap"
        assert tgt.sum().item() > 0, "target mask is empty"
        assert ctx.sum().item() >= mask_gen.min_keep
