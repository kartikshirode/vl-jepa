"""
Checkpoint management utilities
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Any
import json


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
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: LR scheduler state
        epoch: Current epoch
        global_step: Global training step
        best_metric: Best validation metric so far
        config: Training configuration
        save_path: Path to save checkpoint
        is_best: Whether this is the best model
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare checkpoint
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'best_metric': best_metric,
        'config': config,
    }
    
    # Save checkpoint
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")
    
    # Save best model separately
    if is_best:
        best_path = save_path.parent / 'best_model.pth'
        torch.save(checkpoint, best_path)
        print(f"Best model saved to {best_path}")
    
    # Save config as JSON
    config_path = save_path.parent / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = 'cuda',
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load checkpoint to
        
    Returns:
        Dictionary with checkpoint info (epoch, step, etc.)
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model weights loaded")
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state loaded")
    
    # Load scheduler state
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Scheduler state loaded")
    
    # Return checkpoint info
    info = {
        'epoch': checkpoint.get('epoch', 0),
        'global_step': checkpoint.get('global_step', 0),
        'best_metric': checkpoint.get('best_metric', float('inf')),
        'config': checkpoint.get('config', {}),
    }
    
    print(f"Checkpoint loaded: epoch={info['epoch']}, step={info['global_step']}")
    
    return info


def save_model_only(model: nn.Module, save_path: str):
    """
    Save only model weights (for inference/deployment).
    
    Args:
        model: Model to save
        save_path: Path to save model
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")


def load_model_only(model: nn.Module, checkpoint_path: str, device: str = 'cuda'):
    """
    Load only model weights.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load to
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Model weights loaded from {checkpoint_path}")


if __name__ == "__main__":
    print("Checkpoint utilities loaded successfully!")
