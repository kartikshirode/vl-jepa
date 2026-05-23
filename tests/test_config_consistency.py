"""
Catch schema drift between the two yaml configs early. If config_dgpu.yaml
and configs/config_coco_full.yaml stop sharing the same top-level keys, the
trainer will read a missing setting at the wrong time.
"""

import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def test_both_configs_have_same_top_level_sections():
    dgpu = _load(REPO_ROOT / "config_dgpu.yaml")
    coco = _load(REPO_ROOT / "configs" / "config_coco_full.yaml")
    expected = {'model', 'training', 'data', 'logging'}
    assert expected <= dgpu.keys(), f"config_dgpu missing keys: {expected - dgpu.keys()}"
    assert expected <= coco.keys(), f"config_coco_full missing keys: {expected - coco.keys()}"


def test_masking_lives_under_data_in_both_configs():
    """Pre-phase-0 the dgpu config had masking at top level; the coco config
    had it under data. The dgpu trainer therefore silently used default
    mask config. Lock both to live under data."""
    dgpu = _load(REPO_ROOT / "config_dgpu.yaml")
    coco = _load(REPO_ROOT / "configs" / "config_coco_full.yaml")
    assert 'masking' in dgpu['data'], "dgpu config: masking must live under data:"
    assert 'masking' in coco['data'], "coco config: masking must live under data:"
    assert 'masking' not in dgpu, "dgpu config still has a top-level masking key"
    assert 'masking' not in coco, "coco config still has a top-level masking key"


def test_predictor_default_is_transformer():
    """MLP predictor has no positional info; it must NOT be the default after phase-1."""
    for cfg_path in [REPO_ROOT / "config_dgpu.yaml",
                     REPO_ROOT / "configs" / "config_coco_full.yaml"]:
        cfg = _load(cfg_path)
        assert cfg['model']['predictor']['type'] == 'transformer', (
            f"{cfg_path}: predictor.type should be 'transformer'"
        )


def test_text_encoder_projection_dim_removed():
    """phase-3 deleted the dead internal projection. The config keys go too."""
    for cfg_path in [REPO_ROOT / "config_dgpu.yaml",
                     REPO_ROOT / "configs" / "config_coco_full.yaml"]:
        cfg = _load(cfg_path)
        te = cfg['model']['text_encoder']
        assert 'projection_dim' not in te, f"{cfg_path}: leftover projection_dim"
