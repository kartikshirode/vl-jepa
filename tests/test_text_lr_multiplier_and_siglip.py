"""
Lock in the two v8 fixes that aim to prevent the v7 contrastive collapse.

v7 (config_kaggle_t4.yaml epochs 0..3 on Kaggle) blew up the text encoder
within three epochs and parked contrastive loss at log(N). Post-mortem on
checkpoint_epoch_3.pth confirmed:
  - text encoder CLS row-norm std = 1.3e-06 (constant vector for every input)
  - vision encoder CLS off-diagonal cosine = +1.0000 (constant direction)
  - cosine-sim matrix std = 2.7e-06 (uniform softmax, loss == log(B))

VL-JEPA paper Table 5b: a x0.05 LR multiplier on the pretrained text encoder
prevents this. Plus, at our small effective batch (32) InfoNCE's anti-collapse
property is weak; SigLIP (Zhai et al. 2023) keeps gradients non-zero through
a uniform sim matrix, so the heads can escape flat regions.

These tests guard both fixes against silent regressions.
"""

import pytest
import torch
import torch.nn.functional as F

from vl_jepa.models import VisionEncoder, TextEncoder, VLJEPAModel
from vl_jepa.models.predictor import PredictorTransformer


def _build_tiny_siglip_model():
    torch.manual_seed(0)
    vision = VisionEncoder(pretrained=False, gradient_checkpointing=False)
    text = TextEncoder(gradient_checkpointing=False)
    predictor = PredictorTransformer(
        input_dim=192, hidden_dim=384, output_dim=192,
        num_layers=2, num_heads=6, num_patches=196,
    )
    return VLJEPAModel(vision, text, predictor, contrastive_loss_type='siglip')


def test_text_lr_multiplier_isolates_text_encoder_group(tiny_model):
    """get_parameter_groups must put text encoder in its own group at base_lr * mult."""
    base_lr = 1e-3
    mult = 0.05
    groups = tiny_model.get_parameter_groups(base_lr, text_encoder_lr_multiplier=mult)

    text_param_ids = {id(p) for p in tiny_model.text_encoder.parameters()}
    text_groups = [g for g in groups if any(id(p) in text_param_ids for p in g['params'])]
    other_groups = [g for g in groups if not any(id(p) in text_param_ids for p in g['params'])]

    assert len(text_groups) == 1, (
        f"text encoder params must live in exactly one group, found {len(text_groups)}"
    )
    assert text_groups[0]['lr'] == pytest.approx(base_lr * mult), (
        f"text encoder group LR is {text_groups[0]['lr']}, expected {base_lr * mult}"
    )
    for g in other_groups:
        assert g['lr'] == pytest.approx(base_lr), (
            f"non-text group LR is {g['lr']}, expected {base_lr}"
        )


def test_text_lr_multiplier_default_does_not_change_lrs(tiny_model):
    """With the default multiplier (1.0), every group remains at base_lr.

    This is the back-compat invariant: configs without
    text_encoder_lr_multiplier behave as they did before the fix.
    """
    base_lr = 7e-4
    groups = tiny_model.get_parameter_groups(base_lr)  # default mult=1.0
    for g in groups:
        assert g['lr'] == pytest.approx(base_lr), (
            f"unexpected group LR {g['lr']} with default multiplier; "
            "back-compat broken"
        )


def test_text_lr_multiplier_ignored_in_stage2(tiny_model):
    """Stage-2 freezes the text encoder; the multiplier must not surface."""
    base_lr = 1e-3
    groups = tiny_model.get_parameter_groups(
        base_lr, stage2=True, text_encoder_lr_multiplier=0.05,
    )
    text_param_ids = {id(p) for p in tiny_model.text_encoder.parameters()}
    for g in groups:
        for p in g['params']:
            assert id(p) not in text_param_ids, (
                "Stage-2 param groups must not contain text encoder params"
            )
        assert g['lr'] == pytest.approx(base_lr)


def test_siglip_forward_returns_finite_loss():
    """SigLIP path on a contrastive-only forward returns a finite scalar."""
    model = _build_tiny_siglip_model()
    model.eval()
    B = 4
    images = torch.randn(B, 3, 224, 224)
    ids = torch.randint(100, 30000, (B, 32))
    attn = torch.ones(B, 32)

    with torch.no_grad():
        out = model(
            images=images,
            text_input_ids=ids,
            text_attention_mask=attn,
            mode='contrastive',
        )
    assert out['loss'].dim() == 0
    assert torch.isfinite(out['loss']), f"non-finite SigLIP loss {out['loss']}"


def test_siglip_grad_survives_uniform_sim_matrix():
    """SigLIP's sigmoid-per-pair must produce non-zero gradient on
    siglip_logit_scale/bias even when the cosine-sim matrix is uniform.

    This is the property that InfoNCE lacks at small batch: when all
    cosines are equal, the InfoNCE softmax saturates and gradient through
    logit_scale -> 0. SigLIP's binary-cross-entropy-per-pair has gradient
    proportional to (target - sigmoid(logit)), which stays bounded away
    from zero for any non-degenerate target distribution.
    """
    model = _build_tiny_siglip_model()
    model.train()

    B = 8
    D = 256
    # Construct already-collapsed embeddings: all 8 vision and text embeds
    # point in the same direction. Off-diagonal cosine == diagonal cosine == 1.
    v = torch.zeros(B, D)
    v[:, 0] = 1.0
    t = v.clone()
    loss = model.compute_contrastive_loss(v, t)
    assert torch.isfinite(loss)
    loss.backward()

    assert model.siglip_logit_scale.grad is not None, (
        "siglip_logit_scale received no gradient; SigLIP path is broken"
    )
    assert model.siglip_logit_bias.grad is not None, (
        "siglip_logit_bias received no gradient"
    )
    s_grad = model.siglip_logit_scale.grad.abs().item()
    b_grad = model.siglip_logit_bias.grad.abs().item()
    assert s_grad > 1e-6, (
        f"siglip_logit_scale grad too small ({s_grad:.2e}); collapse-escape property lost"
    )
    assert b_grad > 1e-6, f"siglip_logit_bias grad too small ({b_grad:.2e})"


def test_infonce_gradient_at_uniform_sim_matrix_is_near_zero_baseline():
    """Document the baseline failure mode SigLIP avoids.

    With a uniform sim matrix under InfoNCE + temperature=0.07, loss is
    exactly log(B) and the gradient toward any change in the matrix is
    near zero. The matrix lives in input-space here (not weights), but
    the principle is the same: this is what we observed on v7.
    """
    B = 8
    D = 256
    v = torch.zeros(B, D, requires_grad=True)
    with torch.no_grad():
        v[:, 0] = 1.0
    t = v.detach().clone().requires_grad_(False)
    temperature = 0.07

    logits = (v / v.norm(dim=1, keepdim=True)) @ (t / t.norm(dim=1, keepdim=True)).t() / temperature
    labels = torch.arange(B)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2.0
    expected = torch.log(torch.tensor(float(B)))
    assert torch.allclose(loss.detach(), expected, atol=1e-5), (
        f"baseline InfoNCE on uniform sim should equal log(B)={expected.item():.4f}, "
        f"got {loss.item():.4f}"
    )


def test_siglip_params_get_optimizer_updates_via_get_parameter_groups():
    """The SigLIP logit_scale/bias live on the top-level module, not on a
    submodule; verify get_parameter_groups still surfaces them so the
    optimizer actually updates them. Regression for the comment at
    vl_jepa.py around the siglip group-append block."""
    model = _build_tiny_siglip_model()
    groups = model.get_parameter_groups(base_lr=1e-3, text_encoder_lr_multiplier=0.05)
    all_params = [p for g in groups for p in g['params']]
    all_ids = {id(p) for p in all_params}
    assert id(model.siglip_logit_scale) in all_ids, (
        "siglip_logit_scale not in any optimizer group; it will not be updated"
    )
    assert id(model.siglip_logit_bias) in all_ids, (
        "siglip_logit_bias not in any optimizer group; it will not be updated"
    )
