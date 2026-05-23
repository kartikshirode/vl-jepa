"""Lock-in for the checkpoint retention helper.

Without retention, a 20-epoch run with save_every=2 generates 10 checkpoint
files at ~1 GB each, easily blowing the /kaggle/working 20 GB cap. The
helper keeps only the highest-epoch N files and leaves best_model.pth (or
any non-epoch_* file) alone.
"""

import importlib.util
from pathlib import Path

import pytest


def _load_helper():
    """Pull the helper out of train.py without importing the whole module.

    train.py expects argparse + a config to be wired before main() runs; we
    only want the pure function. Importing via spec to extract the symbol.
    """
    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("train_module", repo_root / "train.py")
    mod = importlib.util.module_from_spec(spec)
    # Stop execution of __main__ block when imported as a library.
    mod.__name__ = "train_module"
    spec.loader.exec_module(mod)
    return mod._prune_old_checkpoints


def _touch(p: Path, size_bytes: int = 16) -> None:
    p.write_bytes(b"x" * size_bytes)


def test_prune_keeps_latest_n_only(tmp_path):
    prune = _load_helper()
    # Make 7 per-epoch checkpoints (epochs 0, 2, 4, 6, 8, 10, 12)
    for e in range(0, 13, 2):
        _touch(tmp_path / f"checkpoint_epoch_{e}.pth", 1024)
    # Plus a best_model that should never be deleted
    _touch(tmp_path / "best_model.pth", 1024)

    prune(tmp_path, "checkpoint_epoch_*.pth", keep_last_n=2)

    remaining = sorted(p.name for p in tmp_path.glob("checkpoint_epoch_*.pth"))
    assert remaining == ["checkpoint_epoch_10.pth", "checkpoint_epoch_12.pth"], remaining
    # best_model survives
    assert (tmp_path / "best_model.pth").exists()


def test_prune_noop_when_keep_is_none(tmp_path):
    prune = _load_helper()
    for e in range(0, 6, 2):
        _touch(tmp_path / f"checkpoint_epoch_{e}.pth")

    prune(tmp_path, "checkpoint_epoch_*.pth", keep_last_n=None)

    # All three still here
    assert len(list(tmp_path.glob("checkpoint_epoch_*.pth"))) == 3


def test_prune_noop_when_count_below_keep(tmp_path):
    prune = _load_helper()
    _touch(tmp_path / "checkpoint_epoch_2.pth")
    _touch(tmp_path / "checkpoint_epoch_4.pth")

    prune(tmp_path, "checkpoint_epoch_*.pth", keep_last_n=5)

    assert len(list(tmp_path.glob("checkpoint_epoch_*.pth"))) == 2


def test_prune_handles_stage2_pattern(tmp_path):
    prune = _load_helper()
    for e in range(0, 5):
        _touch(tmp_path / f"stage2_epoch_{e}.pth")
    # A best file that uses the same prefix but no epoch number
    _touch(tmp_path / "stage2_best.pth")

    prune(tmp_path, "stage2_epoch_*.pth", keep_last_n=1)

    remaining = sorted(p.name for p in tmp_path.glob("stage2_*.pth"))
    assert remaining == ["stage2_best.pth", "stage2_epoch_4.pth"], remaining
