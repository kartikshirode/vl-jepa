"""Checkpoint save / load round-trip."""

import json
import torch

from vl_jepa.models import VisionEncoder, TextEncoder, VLJEPAModel
from vl_jepa.models.predictor import PredictorTransformer
from vl_jepa.utils.checkpoint import save_checkpoint, load_checkpoint


def _build_pair():
    """Two identical-architecture models. We train weights on the first one
    and reload into the second to verify state_dict round-trips."""
    torch.manual_seed(0)
    models = []
    for _ in range(2):
        v = VisionEncoder(pretrained=False, gradient_checkpointing=False)
        t = TextEncoder(gradient_checkpointing=False)
        p = PredictorTransformer(
            input_dim=192, hidden_dim=384, output_dim=192,
            num_layers=2, num_heads=6, num_patches=196,
        )
        models.append(VLJEPAModel(v, t, p))
    return models


def test_roundtrip_preserves_state(tmp_path):
    src, dst = _build_pair()

    # Perturb the source so it differs from the freshly initialized dst.
    with torch.no_grad():
        for p in src.parameters():
            p.add_(0.05)

    # Sanity: parameters now differ.
    pre_load_diff = sum((a - b).abs().sum().item() for a, b in zip(src.parameters(), dst.parameters()))
    assert pre_load_diff > 0

    optimizer = torch.optim.AdamW(src.parameters(), lr=1e-4)
    config = {'training': {'num_epochs': 1}, 'note': 'roundtrip test'}
    ckpt = tmp_path / "ckpt.pth"
    save_checkpoint(
        model=src, optimizer=optimizer, scheduler=None,
        epoch=3, global_step=42, best_metric=0.123,
        config=config, save_path=ckpt, is_best=False,
    )

    info = load_checkpoint(str(ckpt), dst, optimizer=None, scheduler=None, device='cpu')
    assert info['epoch'] == 3
    assert info['global_step'] == 42
    assert info['config']['note'] == 'roundtrip test'

    # After loading, every parameter must match the source.
    for a, b in zip(src.state_dict().values(), dst.state_dict().values()):
        assert torch.equal(a, b), "state_dict drift after load"


def test_sidecar_json_written(tmp_path):
    src, _ = _build_pair()
    optimizer = torch.optim.AdamW(src.parameters(), lr=1e-4)
    ckpt = tmp_path / "ckpt.pth"
    save_checkpoint(
        model=src, optimizer=optimizer, scheduler=None,
        epoch=0, global_step=0, best_metric=0.0,
        config={'foo': 'bar'}, save_path=ckpt, is_best=False,
    )
    sidecar = ckpt.parent / (ckpt.name + ".config.json")
    assert sidecar.exists()
    with open(sidecar, 'r', encoding='utf-8') as f:
        data = json.load(f)
    assert data['foo'] == 'bar'


def test_legacy_projection_keys_filtered(tmp_path):
    """A checkpoint that still carries text_encoder.projection.* keys
    must load with strict=False (phase-3 removed those layers)."""
    src, dst = _build_pair()
    optimizer = torch.optim.AdamW(src.parameters(), lr=1e-4)
    ckpt = tmp_path / "ckpt.pth"

    # Inject fake legacy keys into the saved state by mutating after save.
    save_checkpoint(
        model=src, optimizer=optimizer, scheduler=None,
        epoch=0, global_step=0, best_metric=0.0,
        config={}, save_path=ckpt, is_best=False,
    )
    payload = torch.load(ckpt, map_location='cpu', weights_only=True)
    payload['model_state_dict']['text_encoder.projection.weight'] = torch.randn(8, 8)
    payload['model_state_dict']['target_text_encoder.projection.bias'] = torch.zeros(8)
    torch.save(payload, ckpt)

    info = load_checkpoint(str(ckpt), dst, device='cpu')
    assert info['epoch'] == 0
