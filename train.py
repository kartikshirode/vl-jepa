"""
Training script for VL-JEPA on Jetson Orin Nano
Optimized for low memory with FP16, gradient accumulation, and 8-bit AdamW
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import argparse
from pathlib import Path
import time
from tqdm import tqdm
from typing import Optional, Dict

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None

# Import VL-JEPA components
from vl_jepa.models.vl_jepa import create_vl_jepa_model
from vl_jepa.data.dataset import create_dataset
from vl_jepa.data.transforms import get_train_transforms, get_val_transforms
from vl_jepa.data.collate import jepa_collate_fn
from vl_jepa.masks.multiblock import create_mask_generator
from vl_jepa.utils.config import load_config, save_config, print_config
from vl_jepa.utils.logger import setup_logger
from vl_jepa.utils.checkpoint import save_checkpoint, load_checkpoint
from vl_jepa.utils.metrics import AverageMeter, compute_retrieval_metrics

try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False
    print("Warning: bitsandbytes not found. Using standard AdamW.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train VL-JEPA model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    return parser.parse_args()


def create_optimizer(model: nn.Module, config: Dict) -> torch.optim.Optimizer:
    """Create optimizer from config"""
    opt_config = config['training']['optimizer']
    opt_type = opt_config.get('type', 'adamw')
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    betas = tuple(opt_config.get('betas', [0.9, 0.999]))
    eps = opt_config.get('eps', 1e-8)
    
    if opt_type == 'adamw8bit' and HAS_BITSANDBYTES:
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        print("Using 8-bit AdamW optimizer")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        print("Using standard AdamW optimizer")
    
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, config: Dict, steps_per_epoch: int):
    """Create learning rate scheduler"""
    sched_config = config['training']['scheduler']
    sched_type = sched_config.get('type', 'cosine')
    
    num_epochs = config['training']['num_epochs']
    warmup_epochs = config['training'].get('warmup_epochs', 10)
    min_lr = config['training'].get('min_lr', 1e-6)
    
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = num_epochs * steps_per_epoch
    
    if sched_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=min_lr,
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10 * steps_per_epoch,
            gamma=0.1,
        )
    
    return scheduler, warmup_steps


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    mask_generator,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    epoch: int,
    config: Dict,
    logger,
    global_step: int,
    warmup_steps: int,
) -> tuple:
    """Train for one epoch"""
    model.train()
    
    loss_meter = AverageMeter()
    jepa_loss_meter = AverageMeter()
    contrastive_loss_meter = AverageMeter()
    
    batch_size = config['training']['batch_size']
    grad_accum_steps = config['training']['gradient_accumulation_steps']
    log_every = config['logging'].get('log_every', 10)
    grad_clip = config['training'].get('gradient_clip', 1.0)
    empty_cache_every = config['training'].get('empty_cache_every', 100)
    use_wandb = config['logging'].get('use_wandb', False)
    
    # EMA momentum schedule
    ema_start = config['training'].get('ema_momentum_start', 0.996)
    ema_end = config['training'].get('ema_momentum_end', 1.0)
    
    # Loss weights
    jepa_loss_weight = config['training'].get('jepa_loss_weight', 1.0)
    contrastive_loss_weight = config['training'].get('contrastive_loss_weight', 0.5)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        images = batch['images'].cuda()
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        
        # Generate masks (context = visible, target = to predict)
        context_masks = []
        target_masks = []
        for _ in range(images.shape[0]):
            ctx_mask, tgt_mask = mask_generator()
            context_masks.append(ctx_mask)
            target_masks.append(tgt_mask)
        
        context_mask = torch.stack(context_masks).cuda()
        target_mask = torch.stack(target_masks).cuda()
        
        # Forward pass with mixed precision
        with autocast(enabled=True):
            outputs = model(
                images=images,
                text_input_ids=input_ids,
                text_attention_mask=attention_mask,
                context_mask=context_mask,
                target_mask=target_mask,
                mode="both",
            )
            
            # Compute weighted loss: JEPA + contrastive
            jepa_loss = outputs['jepa_loss']
            contrastive_loss = outputs['contrastive_loss']
            loss = jepa_loss_weight * jepa_loss + contrastive_loss_weight * contrastive_loss
            loss = loss / grad_accum_steps  # Scale loss for gradient accumulation
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Update weights every grad_accum_steps
        if (batch_idx + 1) % grad_accum_steps == 0:
            # Unscale gradients and clip
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Update EMA target encoder
            progress = global_step / (config['training']['num_epochs'] * len(dataloader))
            ema_momentum = ema_start + (ema_end - ema_start) * progress
            model.ema_momentum = ema_momentum
            model.update_target_encoder()
            
            # Learning rate warmup
            if global_step < warmup_steps:
                lr_scale = min(1.0, float(global_step + 1) / warmup_steps)
                for pg in optimizer.param_groups:
                    pg['lr'] = config['training']['learning_rate'] * lr_scale
            else:
                scheduler.step()
            
            global_step += 1
        
        # Update meters
        loss_meter.update(loss.item() * grad_accum_steps, images.size(0))
        if 'jepa_loss' in outputs:
            jepa_loss_meter.update(outputs['jepa_loss'].item(), images.size(0))
        if 'contrastive_loss' in outputs:
            contrastive_loss_meter.update(outputs['contrastive_loss'].item(), images.size(0))
        
        # Logging
        if batch_idx % log_every == 0:
            lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{loss_meter.avg:.4f}",
                'jepa': f"{jepa_loss_meter.avg:.4f}",
                'contrast': f"{contrastive_loss_meter.avg:.4f}",
                'lr': f"{lr:.2e}",
            })
            
            if use_wandb and wandb.run is not None:
                wandb.log({
                    'train/loss': loss_meter.avg,
                    'train/jepa_loss': jepa_loss_meter.avg,
                    'train/contrastive_loss': contrastive_loss_meter.avg,
                    'train/lr': lr,
                    'train/ema_momentum': model.ema_momentum,
                    'train/epoch': epoch,
                    'train/step': global_step,
                })
        
        # Clear CUDA cache periodically
        if batch_idx % empty_cache_every == 0:
            torch.cuda.empty_cache()
    
    logger.info(f"Epoch {epoch} - Loss: {loss_meter.avg:.4f}, JEPA Loss: {jepa_loss_meter.avg:.4f}, Contrastive Loss: {contrastive_loss_meter.avg:.4f}")
    
    return loss_meter.avg, global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    epoch: int,
    config: Dict,
    logger,
) -> Dict[str, float]:
    """Validate model"""
    model.eval()
    
    loss_meter = AverageMeter()
    
    # Collect embeddings for retrieval
    all_image_embeds = []
    all_text_embeds = []
    
    use_wandb = config['logging'].get('use_wandb', False)
    
    pbar = tqdm(dataloader, desc=f"Validation Epoch {epoch}")
    
    for batch in pbar:
        images = batch['images'].cuda()
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        
        # Forward pass
        outputs = model(
            images=images,
            text_input_ids=input_ids,
            text_attention_mask=attention_mask,
            mode="contrastive",
        )
        
        # Collect embeddings
        all_image_embeds.append(outputs['vision_embed'].cpu())
        all_text_embeds.append(outputs['text_embed'].cpu())
        
        if 'loss' in outputs:
            loss_meter.update(outputs['loss'].item(), images.size(0))
    
    # Compute retrieval metrics
    image_embeds = torch.cat(all_image_embeds, dim=0)
    text_embeds = torch.cat(all_text_embeds, dim=0)
    
    # Limit to first 1000 samples for faster evaluation
    if image_embeds.shape[0] > 1000:
        image_embeds = image_embeds[:1000]
        text_embeds = text_embeds[:1000]
    
    metrics = compute_retrieval_metrics(image_embeds, text_embeds)
    metrics['val_loss'] = loss_meter.avg
    
    logger.info(f"Validation Epoch {epoch} - Loss: {loss_meter.avg:.4f}")
    logger.info(f"Retrieval Metrics: {metrics}")
    
    if use_wandb and wandb.run is not None:
        wandb.log({f'val/{k}': v for k, v in metrics.items()})
        wandb.log({'val/epoch': epoch})
    
    return metrics


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    print("Configuration:")
    print_config(config)
    
    # Setup logger
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger = setup_logger(
        name="vl_jepa",
        log_file=log_dir / f"train_{time.strftime('%Y%m%d_%H%M%S')}.log",
    )
    
    # Setup device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'
    
    logger.info(f"Using device: {device}")
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Initialize wandb
    if args.wandb and HAS_WANDB and config['logging'].get('use_wandb', False):
        wandb.init(
            project=config['logging'].get('wandb_project', 'vl-jepa'),
            entity=config['logging'].get('wandb_entity', None),
            config=config,
            name=f"vl_jepa_{time.strftime('%Y%m%d_%H%M%S')}",
        )
    
    # Create model
    logger.info("Creating model...")
    model = create_vl_jepa_model(config)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {num_params / 1e6:.2f}M")
    logger.info(f"Trainable parameters: {num_trainable / 1e6:.2f}M")
    
    # Create datasets
    logger.info("Creating datasets...")
    train_transform = get_train_transforms(config['data'])
    val_transform = get_val_transforms(config['data'])
    
    # Get tokenizer from model
    tokenizer = model.text_encoder.tokenizer
    
    train_dataset = create_dataset(
        config,
        split='train',
        transform=train_transform,
        tokenizer=tokenizer,
    )
    
    # Try to create validation dataset (optional)
    val_dataset = None
    try:
        val_dataset = create_dataset(
            config,
            split='val',
            transform=val_transform,
            tokenizer=tokenizer,
        )
        logger.info(f"Val dataset: {len(val_dataset)} samples")
    except FileNotFoundError:
        logger.warning("Validation dataset not found, skipping validation")
    
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    
    # Create dataloaders
    num_workers = config['data'].get('num_workers', 2)
    train_loader_kwargs = {
        'batch_size': config['training']['batch_size'],
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': config['data'].get('pin_memory', True),
        'collate_fn': jepa_collate_fn,
    }
    # Only add these for multiprocessing (num_workers > 0)
    if num_workers > 0:
        train_loader_kwargs['persistent_workers'] = config['data'].get('persistent_workers', True)
        train_loader_kwargs['prefetch_factor'] = config['data'].get('prefetch_factor', 2)
    
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'] * 2,
            shuffle=False,
            num_workers=config['data'].get('num_workers', 2),
            pin_memory=config['data'].get('pin_memory', True),
            collate_fn=jepa_collate_fn,
        )
    
    # Create mask generator
    mask_generator = create_mask_generator(config['data'])
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler, warmup_steps = create_scheduler(optimizer, config, len(train_loader))
    
    # Create gradient scaler for mixed precision
    scaler = GradScaler(enabled=True)
    
    # Load checkpoint if resuming
    start_epoch = 0
    global_step = 0
    best_metric = float('inf')
    
    if args.resume:
        checkpoint_info = load_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            device=device,
        )
        start_epoch = checkpoint_info['epoch'] + 1
        global_step = checkpoint_info['global_step']
        best_metric = checkpoint_info['best_metric']
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    checkpoint_dir = Path(config['training'].get('checkpoint_dir', 'checkpoints'))
    checkpoint_dir.mkdir(exist_ok=True)
    save_every = config['training'].get('save_every', 5)
    
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, num_epochs):
        # Train
        train_loss, global_step = train_one_epoch(
            model=model,
            dataloader=train_loader,
            mask_generator=mask_generator,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            config=config,
            logger=logger,
            global_step=global_step,
            warmup_steps=warmup_steps,
        )
        
        # Validate (if val_loader exists)
        val_metrics = {'val_loss': float('inf')}
        if val_loader is not None:
            val_metrics = validate(
                model=model,
                dataloader=val_loader,
                epoch=epoch,
                config=config,
                logger=logger,
            )
        
        # Save checkpoint
        is_best = val_metrics['val_loss'] < best_metric
        if is_best:
            best_metric = val_metrics['val_loss']
        
        if (epoch + 1) % save_every == 0 or is_best:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
                config=config,
                save_path=checkpoint_dir / f"checkpoint_epoch_{epoch}.pth",
                is_best=is_best,
            )
    
    logger.info("Training completed!")
    
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
