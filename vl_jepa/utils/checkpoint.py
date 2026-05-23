"""
Checkpoint management utilities
"""

import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Any
import json
import warnings


def _config_sidecar_path(ckpt_path: Path) -> Path:
    """Sibling JSON file that holds the run config for a checkpoint."""
    return ckpt_path.with_suffix(ckpt_path.suffix + ".config.json")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    global_step: int,
    best_metric: float,
    config: Dict,
    save_path: str,
    is_best: bool = False,
):
    """
    Save model checkpoint atomically. The run config is written as a sibling
    `.config.json` file so the checkpoint payload itself stays loadable with
    `torch.load(..., weights_only=True)`.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'best_metric': best_metric,
    }

    tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, save_path)
    print(f"Checkpoint saved to {save_path}")

    sidecar = _config_sidecar_path(save_path)
    with open(sidecar, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    if is_best:
        best_path = save_path.parent / 'best_model.pth'
        best_tmp = best_path.with_suffix(best_path.suffix + ".tmp")
        torch.save(checkpoint, best_tmp)
        os.replace(best_tmp, best_path)
        best_sidecar = _config_sidecar_path(best_path)
        with open(best_sidecar, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        print(f"Best model saved to {best_path}")


def _torch_load_safely(checkpoint_path: Path, device: str) -> Dict[str, Any]:
    """
    Try a weights_only load first (safe). On failure, fall back to the legacy
    pickle load with a loud warning so old checkpoints still work.
    """
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as e:
        warnings.warn(
            f"weights_only=True load failed for {checkpoint_path} ({e}). "
            "Falling back to legacy pickle load. Re-save this checkpoint to "
            "drop the warning.",
            RuntimeWarning,
        )
        return torch.load(checkpoint_path, map_location=device, weights_only=False)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = 'cuda',
) -> Dict[str, Any]:
    """Load a checkpoint into the supplied model/optimizer/scheduler."""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = _torch_load_safely(checkpoint_path, device)

    state = checkpoint['model_state_dict']
    # Phase-3: drop legacy internal text_encoder.projection.* and
    # target_text_encoder.projection.* keys. They were unused at train time
    # but were saved into older checkpoints, so a clean model now mismatches.
    legacy_prefixes = ('text_encoder.projection.', 'target_text_encoder.projection.')
    removed = [k for k in state if k.startswith(legacy_prefixes)]
    for k in removed:
        state.pop(k)
    if removed:
        print(f"Dropped {len(removed)} legacy projection keys from checkpoint (e.g. {removed[0]})")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Note: {len(missing)} missing keys in checkpoint (e.g. {missing[0]})")
    if unexpected:
        print(f"Note: {len(unexpected)} unexpected keys in checkpoint (e.g. {unexpected[0]})")
    print("Model weights loaded")

    if optimizer is not None and checkpoint.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state loaded")

    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Scheduler state loaded")

    # Pull config from sibling JSON if it exists, else from inside the payload
    # (legacy checkpoints), else empty.
    sidecar = _config_sidecar_path(checkpoint_path)
    if sidecar.exists():
        with open(sidecar, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    else:
        cfg = checkpoint.get('config', {}) or {}

    info = {
        'epoch': checkpoint.get('epoch', 0),
        'global_step': checkpoint.get('global_step', 0),
        'best_metric': checkpoint.get('best_metric', float('inf')),
        'config': cfg,
    }

    print(f"Checkpoint loaded: epoch={info['epoch']}, step={info['global_step']}")

    return info


def save_model_only(model: nn.Module, save_path: str):
    """Save model weights only (atomic, safe-load friendly)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
    torch.save(model.state_dict(), tmp_path)
    os.replace(tmp_path, save_path)
    print(f"Model weights saved to {save_path}")


def load_model_only(model: nn.Module, checkpoint_path: str, device: str = 'cuda'):
    """Load model weights only."""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = _torch_load_safely(checkpoint_path, device)
    model.load_state_dict(state_dict)
    print(f"Model weights loaded from {checkpoint_path}")


if __name__ == "__main__":
    print("Checkpoint utilities loaded successfully!")
