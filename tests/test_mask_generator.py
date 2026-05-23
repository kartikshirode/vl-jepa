"""Tests for the multi-block mask generator."""

import pytest
import torch

from vl_jepa.masks import MultiBlockMaskGenerator


def test_returns_four_tensors():
    gen = MultiBlockMaskGenerator()
    out = gen()
    assert len(out) == 4
    ctx_mask, tgt_mask, ctx_idx, tgt_idx = out
    assert ctx_mask.dtype == torch.bool
    assert tgt_mask.dtype == torch.bool
    assert ctx_idx.dtype == torch.long
    assert tgt_idx.dtype == torch.long


def test_disjoint_and_nonempty():
    gen = MultiBlockMaskGenerator()
    for _ in range(50):
        ctx, tgt, _, _ = gen()
        assert (ctx & tgt).sum().item() == 0
        assert tgt.sum().item() > 0
        assert ctx.sum().item() >= gen.min_keep


def test_fixed_sizes_across_batch():
    """Whole point of fixed counts: collation must work without padding."""
    gen = MultiBlockMaskGenerator()
    sizes = set()
    for _ in range(32):
        _, _, ci, ti = gen()
        sizes.add((ci.shape[0], ti.shape[0]))
    assert len(sizes) == 1, f"mask shapes drift across samples: {sizes}"


def test_indices_within_range():
    gen = MultiBlockMaskGenerator()
    for _ in range(20):
        _, _, ci, ti = gen()
        assert ci.min().item() >= 0 and ci.max().item() < gen.total_patches
        assert ti.min().item() >= 0 and ti.max().item() < gen.total_patches


def test_retry_raises_on_impossible_constraints():
    """Asking for a 100% target scale with no overlap must raise, not silently emit garbage."""
    gen = MultiBlockMaskGenerator(
        target_scale=(1.0, 1.0),
        num_target_blocks=2,
        allow_overlap=False,
        min_keep=50,
        max_tries=3,
    )
    with pytest.raises(RuntimeError):
        gen.sample()
