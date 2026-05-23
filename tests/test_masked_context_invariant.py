"""
Masked-context invariant.

This is the JEPA correctness contract: the context encoder must NEVER see the
target patches. If a future refactor accidentally feeds the full image through
the context path, the model has a trivial identity solution and the loss
collapses. The test perturbs only the target pixels and asserts the
forward_context output is byte-identical (it should be, since those patches
never touch the context branch).

Build a small ViT-Tiny with pretrained=False so this stays CPU-cheap.
"""

import torch
import pytest

from vl_jepa.models import VisionEncoder
from vl_jepa.masks import MultiBlockMaskGenerator


def _patch_indices_to_pixel_box(idx: int, patch_size: int, grid: int):
    """Convert flat patch index (row-major) to (top, left, h, w) pixel box."""
    r, c = divmod(idx, grid)
    top = r * patch_size
    left = c * patch_size
    return top, left, patch_size, patch_size


def test_context_output_shape():
    """forward_context returns [B, N_ctx, D] aligned with the index input."""
    vision = VisionEncoder(pretrained=False, gradient_checkpointing=False)
    vision.eval()
    mask_gen = MultiBlockMaskGenerator(input_size=224, patch_size=16)
    B = 2
    ctx_idx = torch.stack([mask_gen()[2] for _ in range(B)])
    images = torch.randn(B, 3, 224, 224)
    with torch.no_grad():
        out = vision.forward_context(images, ctx_idx)
    assert out.shape == (B, ctx_idx.shape[1], vision.embed_dim)


def test_context_unchanged_when_target_pixels_perturbed():
    """
    Take a single image and a single context/target mask. Run forward_context
    once, then overwrite every pixel inside any target patch with noise (and
    leave context patches untouched). forward_context output should be
    bit-identical (within fp tolerance) because the context branch only ever
    reads context pixels.

    If this ever fails: someone has wired the full image through the context
    encoder, the predictor can trivially copy from its input, and the JEPA
    objective has collapsed. Hard regression gate.
    """
    torch.manual_seed(0)
    vision = VisionEncoder(pretrained=False, gradient_checkpointing=False)
    vision.eval()

    mask_gen = MultiBlockMaskGenerator(input_size=224, patch_size=16)
    ctx_mask, tgt_mask, ctx_idx, _ = mask_gen()

    # Disjoint by construction; assert it so the test premise holds.
    assert (ctx_mask & tgt_mask).sum().item() == 0

    grid = vision.img_size // vision.patch_size  # 14
    patch_size = vision.patch_size
    img = torch.randn(1, 3, vision.img_size, vision.img_size)
    img_perturbed = img.clone()

    # Stomp every target patch with noise. Context patches stay byte-equal.
    tgt_indices = torch.where(tgt_mask)[0].tolist()
    for flat_idx in tgt_indices:
        top, left, h, w = _patch_indices_to_pixel_box(flat_idx, patch_size, grid)
        img_perturbed[0, :, top:top + h, left:left + w] = torch.randn(3, h, w) * 10.0

    ctx_idx_b = ctx_idx.unsqueeze(0)  # [1, N_ctx]

    with torch.no_grad():
        out_orig = vision.forward_context(img, ctx_idx_b)
        out_pert = vision.forward_context(img_perturbed, ctx_idx_b)

    # ViT-Tiny is deterministic in eval mode; allclose with a strict tol is fine.
    assert torch.allclose(out_orig, out_pert, atol=1e-5, rtol=1e-4), (
        "forward_context output changed when only target-region pixels were "
        "perturbed. The context branch is leaking target patches; the JEPA "
        "task has collapsed."
    )


def test_context_DOES_change_when_context_pixels_perturbed():
    """
    Sanity counterpart to the test above: if we perturb a CONTEXT patch, the
    output MUST change. Otherwise the previous test is trivially true (e.g. if
    forward_context were returning zeros) and gives false confidence.
    """
    torch.manual_seed(0)
    vision = VisionEncoder(pretrained=False, gradient_checkpointing=False)
    vision.eval()

    mask_gen = MultiBlockMaskGenerator(input_size=224, patch_size=16)
    ctx_mask, tgt_mask, ctx_idx, _ = mask_gen()

    grid = vision.img_size // vision.patch_size
    patch_size = vision.patch_size
    img = torch.randn(1, 3, vision.img_size, vision.img_size)
    img_perturbed = img.clone()

    # Perturb one context patch.
    ctx_indices = torch.where(ctx_mask)[0].tolist()
    flat_idx = ctx_indices[0]
    top, left, h, w = _patch_indices_to_pixel_box(flat_idx, patch_size, grid)
    img_perturbed[0, :, top:top + h, left:left + w] += 5.0

    ctx_idx_b = ctx_idx.unsqueeze(0)

    with torch.no_grad():
        out_orig = vision.forward_context(img, ctx_idx_b)
        out_pert = vision.forward_context(img_perturbed, ctx_idx_b)

    assert not torch.allclose(out_orig, out_pert, atol=1e-4), (
        "Perturbing a context patch did not change forward_context output. "
        "Either context indexing is wrong or the encoder ignores its input."
    )
