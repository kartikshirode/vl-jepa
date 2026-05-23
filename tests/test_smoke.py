"""Smoke checks: module imports load and a tiny forward runs end to end."""

import torch
import pytest


def test_imports():
    from vl_jepa.models import VisionEncoder, TextEncoder, PredictorMLP, VLJEPAModel
    from vl_jepa.models.predictor import PredictorTransformer, PredictorWithCrossAttention
    from vl_jepa.models.vl_jepa import create_vl_jepa_model
    from vl_jepa.masks import MultiBlockMaskGenerator
    from vl_jepa.data import get_train_transforms, get_val_transforms, jepa_collate_fn


def test_jepa_forward(tiny_model, dummy_batch):
    tiny_model.eval()
    with torch.no_grad():
        out = tiny_model(
            images=dummy_batch['images'],
            text_input_ids=dummy_batch['input_ids'],
            text_attention_mask=dummy_batch['attention_mask'],
            context_indices=dummy_batch['context_indices'],
            target_indices=dummy_batch['target_indices'],
            mode='jepa',
        )
    assert 'loss' in out
    assert out['loss'].dim() == 0
    assert out['predicted_vision'].shape == out['target_vision'].shape


def test_contrastive_forward(tiny_model, dummy_batch):
    tiny_model.eval()
    with torch.no_grad():
        out = tiny_model(
            images=dummy_batch['images'],
            text_input_ids=dummy_batch['input_ids'],
            text_attention_mask=dummy_batch['attention_mask'],
            mode='contrastive',
        )
    assert out['vision_embed'].shape[-1] == out['text_embed'].shape[-1]
    assert out['loss'].dim() == 0


def test_both_mode_returns_jepa_and_contrastive_losses(tiny_model, dummy_batch):
    tiny_model.eval()
    with torch.no_grad():
        out = tiny_model(
            images=dummy_batch['images'],
            text_input_ids=dummy_batch['input_ids'],
            text_attention_mask=dummy_batch['attention_mask'],
            context_indices=dummy_batch['context_indices'],
            target_indices=dummy_batch['target_indices'],
            mode='both',
        )
    assert 'jepa_loss' in out and 'contrastive_loss' in out
    # mode='both' must NOT define a hardcoded outputs['loss']; trainer
    # computes the weighted combination from config.
    assert 'loss' not in out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_forward(tiny_model, dummy_batch):
    model = tiny_model.cuda()
    batch = {k: v.cuda() for k, v in dummy_batch.items()}
    with torch.no_grad():
        out = model(
            images=batch['images'],
            text_input_ids=batch['input_ids'],
            text_attention_mask=batch['attention_mask'],
            context_indices=batch['context_indices'],
            target_indices=batch['target_indices'],
            mode='jepa',
        )
    assert out['loss'].device.type == 'cuda'
    # Move model back to CPU so other tests reusing the session fixture stay on CPU.
    model.cpu()
