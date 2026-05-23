"""Shared pytest fixtures for the VL-JEPA test suite."""

import torch
import pytest

from vl_jepa.models import VisionEncoder, TextEncoder, VLJEPAModel
from vl_jepa.models.predictor import PredictorTransformer
from vl_jepa.masks import MultiBlockMaskGenerator


@pytest.fixture(scope="session")
def tiny_model():
    """A small VL-JEPA built without any pretrained downloads."""
    torch.manual_seed(0)
    vision = VisionEncoder(pretrained=False, gradient_checkpointing=False)
    text = TextEncoder(gradient_checkpointing=False)
    predictor = PredictorTransformer(
        input_dim=192, hidden_dim=384, output_dim=192,
        num_layers=2, num_heads=6, num_patches=196,
    )
    return VLJEPAModel(vision, text, predictor)


@pytest.fixture(scope="session")
def mask_gen():
    return MultiBlockMaskGenerator(input_size=224, patch_size=16)


@pytest.fixture
def dummy_batch(mask_gen):
    """A small batch of images, tokens, and stacked mask indices."""
    B = 2
    images = torch.randn(B, 3, 224, 224)
    ids = torch.randint(0, 1000, (B, 32))
    attn = torch.ones(B, 32)
    ctx_idxs, tgt_idxs = [], []
    for _ in range(B):
        _, _, ci, ti = mask_gen()
        ctx_idxs.append(ci); tgt_idxs.append(ti)
    return {
        'images': images,
        'input_ids': ids,
        'attention_mask': attn,
        'context_indices': torch.stack(ctx_idxs),
        'target_indices': torch.stack(tgt_idxs),
    }
