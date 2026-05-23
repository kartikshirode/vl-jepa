"""Phase-5 surface tests: SigLIP loss option and text-conditioned predictor."""

import torch
import pytest

from vl_jepa.models import VisionEncoder, TextEncoder, VLJEPAModel
from vl_jepa.models.predictor import PredictorTransformer, PredictorWithCrossAttention
from vl_jepa.masks import MultiBlockMaskGenerator


def _build_model(predictor, contrastive='infonce'):
    torch.manual_seed(0)
    return VLJEPAModel(
        vision_encoder=VisionEncoder(pretrained=False, gradient_checkpointing=False),
        text_encoder=TextEncoder(gradient_checkpointing=False),
        predictor=predictor,
        contrastive_loss_type=contrastive,
    )


def _build_inputs(B=2):
    gen = MultiBlockMaskGenerator()
    images = torch.randn(B, 3, 224, 224)
    ids = torch.randint(0, 1000, (B, 32))
    attn = torch.ones(B, 32)
    ctx_idxs, tgt_idxs = [], []
    for _ in range(B):
        _, _, ci, ti = gen()
        ctx_idxs.append(ci); tgt_idxs.append(ti)
    return images, ids, attn, torch.stack(ctx_idxs), torch.stack(tgt_idxs)


def test_siglip_loss_finite_and_grads_flow():
    """SigLIP variant: finite loss, scale and bias receive gradients."""
    predictor = PredictorTransformer(input_dim=192, hidden_dim=384, output_dim=192,
                                     num_layers=2, num_heads=6, num_patches=196)
    model = _build_model(predictor, contrastive='siglip')
    images, ids, attn, _, _ = _build_inputs()
    model.train()
    out = model(images, ids, attn, mode='contrastive')
    loss = out['loss']
    assert torch.isfinite(loss), f"SigLIP loss not finite: {loss}"
    loss.backward()
    assert model.siglip_logit_scale.grad is not None
    assert model.siglip_logit_bias.grad is not None


def test_text_conditioned_predictor_shape():
    """Cross-attention predictor returns one prediction per target index."""
    predictor = PredictorWithCrossAttention(
        input_dim=192, text_dim=768, hidden_dim=384, output_dim=192,
        num_layers=2, num_heads=6, num_patches=196,
    )
    model = _build_model(predictor)
    images, ids, attn, ctx_idx, tgt_idx = _build_inputs()
    model.eval()
    with torch.no_grad():
        out = model(images, ids, attn,
                    context_indices=ctx_idx, target_indices=tgt_idx, mode='jepa')
    assert out['predicted_vision'].shape == out['target_vision'].shape
    assert out['predicted_vision'].shape[1] == tgt_idx.shape[1]


def test_siglip_params_in_optimizer_groups():
    """SigLIP logit scale/bias must be in get_parameter_groups output so the
    optimizer actually updates them. Regression for the 'gradients flow but
    weights never move' bug."""
    predictor = PredictorTransformer(input_dim=192, hidden_dim=384, output_dim=192,
                                     num_layers=2, num_heads=6, num_patches=196)
    model = _build_model(predictor, contrastive='siglip')
    groups = model.get_parameter_groups(base_lr=1e-3, stage2=False)
    all_params = [p for g in groups for p in g['params']]
    assert any(p is model.siglip_logit_scale for p in all_params), \
        "siglip_logit_scale missing from optimizer param groups"
    assert any(p is model.siglip_logit_bias for p in all_params), \
        "siglip_logit_bias missing from optimizer param groups"

    # Same check under stage2.
    groups2 = model.get_parameter_groups(base_lr=1e-3, stage2=True)
    all_params2 = [p for g in groups2 for p in g['params']]
    assert any(p is model.siglip_logit_scale for p in all_params2)
    assert any(p is model.siglip_logit_bias for p in all_params2)


def test_drop_path_rate_does_not_break_forward():
    """DropPath in the predictor should not change output shape or break training."""
    predictor = PredictorTransformer(
        input_dim=192, hidden_dim=384, output_dim=192,
        num_layers=4, num_heads=6, num_patches=196,
        drop_path_rate=0.1,
    )
    model = _build_model(predictor)
    images, ids, attn, ctx_idx, tgt_idx = _build_inputs()
    model.train()
    out = model(images, ids, attn,
                context_indices=ctx_idx, target_indices=tgt_idx, mode='jepa')
    out['loss'].backward()
    assert out['predicted_vision'].shape[1] == tgt_idx.shape[1]
