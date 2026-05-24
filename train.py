"""
Training script for VL-JEPA on Jetson Orin Nano
Optimized for low memory with FP16, gradient accumulation, and 8-bit AdamW
"""

import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast, GradScaler
import argparse
from pathlib import Path
import time
from tqdm import tqdm
from typing import Optional, Dict


def _setup_distributed():
    """Initialize torch.distributed when launched via torchrun.

    torchrun sets RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT in
    the environment. When those are missing (plain `python train.py ...`),
    we stay in single-process mode and the rest of the script behaves
    exactly as it did before. The helper returns (rank, local_rank,
    world_size, is_distributed) so the caller doesn't have to recheck
    env vars repeatedly.

    The NCCL backend is used on CUDA (faster all-reduce); gloo is the
    fallback for CPU-only torchrun runs, which we don't expect in
    practice but keep working for tests.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        world_size = int(os.environ["WORLD_SIZE"])
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, True
    return 0, 0, 1, False


def _is_main_process() -> bool:
    """True on rank 0 of a DDP run, or always in single-process mode."""
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def _unwrap(model: nn.Module) -> nn.Module:
    """Return the underlying module from a DDP wrapper, or the model itself."""
    return model.module if isinstance(model, DDP) else model


def _barrier():
    """Synchronize all ranks. No-op outside DDP."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

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
from vl_jepa.utils.config import load_config, save_config, print_config, validate_config
from vl_jepa.utils.logger import setup_logger
from vl_jepa.utils.checkpoint import save_checkpoint, load_checkpoint
from vl_jepa.utils.metrics import AverageMeter, compute_retrieval_metrics

try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False
    print("Warning: bitsandbytes not found. Using standard AdamW.")


def _prune_old_checkpoints(checkpoint_dir, pattern: str, keep_last_n) -> None:
    """Delete older per-epoch checkpoint files, keeping only the latest N.

    Per-epoch checkpoints are ~1 GB each at our model size, so a 20-epoch run
    with save_every=2 generates 12 GB of redundant state when the resume
    mechanism only ever needs the most recent file. This function trims the
    glob match in `checkpoint_dir` down to the highest `keep_last_n` epochs
    by parsing the epoch number out of each filename.

    No-op when keep_last_n is None or <= 0 (the default), so local runs
    that do not set the config field keep the original "save everything"
    behavior.

    `best_model.pth` and `stage2_best.pth` are not matched by typical
    epoch_* patterns and are left untouched.
    """
    if keep_last_n is None or keep_last_n <= 0:
        return
    import re
    from pathlib import Path as _Path

    ckpts = list(_Path(checkpoint_dir).glob(pattern))
    if len(ckpts) <= keep_last_n:
        return

    epoch_re = re.compile(r"epoch_(\d+)\.pth$")

    def _epoch_of(p):
        m = epoch_re.search(p.name)
        return int(m.group(1)) if m else -1

    ckpts.sort(key=_epoch_of)
    to_delete = ckpts[:-keep_last_n]
    deleted = 0
    for p in to_delete:
        try:
            p.unlink()
            deleted += 1
        except OSError:
            # Best effort; if a file is locked we just leave it.
            pass
    if deleted:
        print(f"Pruned {deleted} old checkpoint(s) from {checkpoint_dir}; "
              f"keeping last {keep_last_n}.")


def _worker_init(worker_id: int):
    """Seed numpy and Python random per worker so mask sampling is distinct.

    PyTorch seeds the torch RNG per worker but does not consistently seed
    numpy across all versions. Without this init fn, every worker can replay
    the same numpy sequence and the per-sample JEPA mask becomes deterministic
    per dataset index, collapsing mask diversity over a multi-day run.
    """
    import numpy as _np
    import random as _random
    base = torch.initial_seed()
    seed = (base + worker_id) % (2 ** 32)
    _np.random.seed(seed)
    _random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train VL-JEPA model")
    parser.add_argument("--config", type=str, default="config_dgpu.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--stage2", action="store_true", help="Stage-2 training: freeze text encoder, lower LR")
    parser.add_argument("--stage2_lr_factor", type=float, default=0.5, help="LR multiplier for Stage-2 (default: 0.5)")
    parser.add_argument("--stage2_epochs", type=int, default=None, help="Number of epochs for Stage-2 (overrides training.stage2_epochs in config; default: 4)")
    parser.add_argument("--early_stop_patience", type=int, default=2, help="Early stopping patience based on mean recall")
    parser.add_argument("--compile", action="store_true", help="torch.compile the model (production runs only; ~30s compile cost per code change)")
    return parser.parse_args()


def create_optimizer(model: nn.Module, config: Dict, stage2: bool = False, stage2_lr_factor: float = 0.5) -> torch.optim.Optimizer:
    """Create optimizer from config

    Args:
        model: The VL-JEPA model
        config: Configuration dictionary
        stage2: If True, use Stage-2 settings (frozen text encoder, lower LR)
        stage2_lr_factor: Learning rate multiplier for Stage-2
    """
    opt_config = config['training']['optimizer']
    opt_type = opt_config.get('type', 'adamw')
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    betas = tuple(opt_config.get('betas', [0.9, 0.999]))
    eps = opt_config.get('eps', 1e-8)
    # Per VL-JEPA paper Table 5b, the pretrained text encoder needs an LR
    # multiplier of x0.05 to x0.10 to avoid destroying its CLS representations
    # in the first few epochs. Default to 1.0 so existing configs are unchanged.
    text_lr_mult = float(config['training'].get('text_encoder_lr_multiplier', 1.0))

    # Stage-2: Use reduced learning rate
    if stage2:
        lr = lr * stage2_lr_factor
        print(f"Stage-2: Using reduced LR = {lr:.2e} (factor: {stage2_lr_factor})")

    # Get parameter groups (excludes frozen text encoder in Stage-2)
    if hasattr(model, 'get_parameter_groups'):
        param_groups = model.get_parameter_groups(
            lr,
            stage2=stage2,
            text_encoder_lr_multiplier=text_lr_mult,
        )
        if not stage2 and text_lr_mult != 1.0:
            print(f"Stage-1: text encoder LR = {lr * text_lr_mult:.2e} "
                  f"(base {lr:.2e} x {text_lr_mult})")
    else:
        # Fallback: only trainable parameters
        param_groups = [p for p in model.parameters() if p.requires_grad]
    
    if opt_type == 'adamw8bit' and HAS_BITSANDBYTES:
        optimizer = bnb.optim.AdamW8bit(
            param_groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
        )
        print("Using 8-bit AdamW optimizer")
    else:
        if opt_type == 'adamw8bit' and not HAS_BITSANDBYTES:
            import warnings
            warnings.warn(
                "Config requested optimizer.type='adamw8bit' but bitsandbytes "
                "is not installed; falling back to torch.optim.AdamW (fp32). "
                "Install bitsandbytes (`pip install bitsandbytes`) to honor "
                "the config.",
                RuntimeWarning,
            )
        optimizer = torch.optim.AdamW(
            param_groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
        )
        print("Using standard AdamW optimizer")

    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict,
    steps_per_epoch: int,
    num_epochs_override: int = None,
    warmup_epochs_override: int = None,
):
    """
    Create LR scheduler from config.

    num_epochs_override / warmup_epochs_override let Stage-2 use a
    shorter run length than config['training']['num_epochs'] without
    polluting the cosine schedule horizon with the unused tail.
    """
    sched_config = config['training']['scheduler']
    sched_type = sched_config.get('type', 'cosine')

    num_epochs = num_epochs_override if num_epochs_override is not None else config['training']['num_epochs']
    warmup_epochs = warmup_epochs_override if warmup_epochs_override is not None else config['training'].get('warmup_epochs', 10)
    min_lr = config['training'].get('min_lr', 1e-6)

    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = num_epochs * steps_per_epoch

    if sched_type == 'cosine':
        # Clamp so warmup_epochs >= num_epochs doesn't yield T_max <= 0.
        t_max = max(1, total_steps - warmup_steps)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=min_lr,
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, 10 * steps_per_epoch), gamma=0.1,
        )

    return scheduler, warmup_steps


def _optimizer_step(
    model,
    optimizer,
    scheduler,
    scaler,
    grad_clip: float,
    warmup_steps: int,
    global_step: int,
    total_optimizer_steps: int,
    ema_start: float,
    ema_end: float,
    base_lr: float,
) -> int:
    """One optimizer step plus EMA + LR scheduling. Returns the next global_step."""
    scaler.unscale_(optimizer)
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Detect optimizer-step skip from GradScaler. When inf/nan grads are
    # present, scaler.step() returns without calling optimizer.step() and
    # scaler.update() reduces the loss scale. EMA + scheduler advancing on a
    # skipped step subtly desyncs the cosine schedule from real progress.
    before_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    after_scale = scaler.get_scale()
    optimizer.zero_grad()
    skipped = after_scale < before_scale

    if not skipped:
        # EMA schedule must be in OPTIMIZER STEPS, not micro-batches.
        # `model` may be wrapped in DistributedDataParallel; the EMA target
        # encoders and the ema_momentum attribute live on the underlying
        # VLJEPAModel, so always go through _unwrap.
        underlying = _unwrap(model)
        progress = min(1.0, global_step / max(1, total_optimizer_steps))
        underlying.ema_momentum = ema_start + (ema_end - ema_start) * progress
        underlying.update_target_encoder()

        if global_step < warmup_steps:
            lr_scale = min(1.0, float(global_step + 1) / max(1, warmup_steps))
            # Multiply each group's base_lr stored in optimizer.defaults rather
            # than re-reading the config, so Stage-2's reduced LR is preserved
            # if warmup is ever enabled for Stage-2.
            for pg in optimizer.param_groups:
                pg['lr'] = pg.get('initial_lr', base_lr) * lr_scale
        else:
            scheduler.step()

        return global_step + 1

    # On skip, leave global_step / scheduler / EMA exactly where they were.
    return global_step


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
    device: str = 'cuda',
    stage2: bool = False,
) -> tuple:
    """Train for one epoch"""
    model.train()
    
    # Stage-2 safety check: verify text encoder is still frozen
    if stage2:
        verify_info = model.verify_frozen_text_encoder()
        if not verify_info['all_frozen']:
            raise RuntimeError(f"Stage-2 ERROR: Text encoder unfrozen! {verify_info}")
    
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
    
    # Loss weights - LOCKED, do not change during training
    jepa_loss_weight = config['training'].get('jepa_loss_weight', 1.0)
    contrastive_loss_weight = config['training'].get('contrastive_loss_weight', 0.5)
    
    # tqdm should only render on rank 0; other ranks would interleave their
    # progress bars on stdout and make logs unreadable.
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not _is_main_process())
    use_amp = device.startswith('cuda')

    # Total optimizer steps (not micro-batches) drive the EMA schedule.
    # Use ceiling division so the partial-tail flush at end-of-epoch counts.
    steps_per_epoch = max(1, -(-len(dataloader) // grad_accum_steps))
    total_optimizer_steps = max(1, config['training']['num_epochs'] * steps_per_epoch)

    pending_grad = False  # True between backward() and optimizer.step()
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        images = batch['images'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Masks are produced inside the dataset (worker-parallel) and arrive
        # in the collated batch dict. Fall back to the legacy main-loop path
        # only if the dataset wasn't configured with a mask generator.
        if 'context_indices' in batch:
            context_mask = batch['context_mask'].to(device)
            target_mask = batch['target_mask'].to(device)
            context_indices = batch['context_indices'].to(device)
            target_indices = batch['target_indices'].to(device)
        else:
            ctx_masks, tgt_masks, ctx_idxs, tgt_idxs = [], [], [], []
            for _ in range(images.shape[0]):
                cm, tm, ci, ti = mask_generator()
                ctx_masks.append(cm); tgt_masks.append(tm)
                ctx_idxs.append(ci); tgt_idxs.append(ti)
            context_mask = torch.stack(ctx_masks).to(device)
            target_mask = torch.stack(tgt_masks).to(device)
            context_indices = torch.stack(ctx_idxs).to(device)
            target_indices = torch.stack(tgt_idxs).to(device)

        # Forward pass with mixed precision (no-op on CPU)
        with autocast(device_type='cuda' if device.startswith('cuda') else 'cpu', enabled=use_amp):
            outputs = model(
                images=images,
                text_input_ids=input_ids,
                text_attention_mask=attention_mask,
                context_indices=context_indices,
                target_indices=target_indices,
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
        pending_grad = True

        # Update weights every grad_accum_steps
        if (batch_idx + 1) % grad_accum_steps == 0:
            global_step = _optimizer_step(
                model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                grad_clip=grad_clip, warmup_steps=warmup_steps,
                global_step=global_step, total_optimizer_steps=total_optimizer_steps,
                ema_start=ema_start, ema_end=ema_end,
                base_lr=config['training']['learning_rate'],
            )
            pending_grad = False
        
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
        
        # Clear CUDA cache periodically (CUDA only)
        if device.startswith('cuda') and batch_idx % empty_cache_every == 0:
            torch.cuda.empty_cache()

    # Flush any partial gradient-accumulation tail at the end of the epoch.
    # Without this, dataloaders whose length isn't a multiple of grad_accum
    # drop the last micro-batches' gradients.
    if pending_grad:
        global_step = _optimizer_step(
            model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            grad_clip=grad_clip, warmup_steps=warmup_steps,
            global_step=global_step, total_optimizer_steps=total_optimizer_steps,
            ema_start=ema_start, ema_end=ema_end,
            base_lr=config['training']['learning_rate'],
        )
        pending_grad = False

    logger.info(f"Epoch {epoch} - Loss: {loss_meter.avg:.4f}, JEPA Loss: {jepa_loss_meter.avg:.4f}, Contrastive Loss: {contrastive_loss_meter.avg:.4f}")

    return loss_meter.avg, global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    epoch: int,
    config: Dict,
    logger,
    device: str = 'cuda',
) -> Dict[str, float]:
    """Validate model"""
    model.eval()

    loss_meter = AverageMeter()

    # Collect embeddings for retrieval. Also accumulate image_ids per step so
    # we can hand them to compute_retrieval_metrics. On COCO val, each sample
    # is one (image, caption) pair and the same image_id repeats across the
    # ~5 captions per image; without ids, the metric collapses to a broken
    # diagonal that scores roughly 1/5 of the true recall.
    all_image_embeds = []
    all_text_embeds = []
    all_image_ids = []

    use_wandb = config['logging'].get('use_wandb', False)
    use_amp = device.startswith('cuda')

    pbar = tqdm(dataloader, desc=f"Validation Epoch {epoch}")

    for batch in pbar:
        images = batch['images'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Forward pass under autocast on CUDA. Without this, validation runs
        # in FP32 at 2x the train batch size, which is the biggest single OOM
        # hazard on 8 GB cards.
        with autocast(device_type='cuda' if device.startswith('cuda') else 'cpu', enabled=use_amp):
            outputs = model(
                images=images,
                text_input_ids=input_ids,
                text_attention_mask=attention_mask,
                mode="contrastive",
            )

        # Collect embeddings
        all_image_embeds.append(outputs['vision_embed'].cpu())
        all_text_embeds.append(outputs['text_embed'].cpu())

        # Collate passes image_ids through as a Python list per batch.
        if 'image_ids' in batch:
            all_image_ids.extend(batch['image_ids'])

        if 'loss' in outputs:
            loss_meter.update(outputs['loss'].item(), images.size(0))

    # Compute retrieval metrics
    image_embeds = torch.cat(all_image_embeds, dim=0)
    text_embeds = torch.cat(all_text_embeds, dim=0)

    # Cap via config (default: no cap). Was hardcoded to 1000.
    eval_cfg = config.get('evaluation', {})
    max_eval_samples = eval_cfg.get('max_eval_samples', None)
    if max_eval_samples is not None and image_embeds.shape[0] > max_eval_samples:
        image_embeds = image_embeds[:max_eval_samples]
        text_embeds = text_embeds[:max_eval_samples]
        all_image_ids = all_image_ids[:max_eval_samples]

    # Build the id tensors when the batches carried image_ids. Each sample is
    # (image_i, caption_i), so text_image_ids per row equals image_ids per row.
    # Then dedupe the image gallery to one row per unique image_id. COCO has
    # ~5 captions per image, so without dedup each image appears 5 times in
    # image_embeds with identical embeddings (model is deterministic in eval).
    # Those 5 ties fill 5 top-K slots in t2i and depress t2i_recall@5 close to
    # t2i_recall@1. The Karpathy COCO 5K convention dedupes the image gallery,
    # which is what compute_retrieval_metrics expects when N_img != N_txt.
    if len(all_image_ids) == image_embeds.shape[0] and all_image_ids:
        import numpy as np
        ids_np = np.asarray([int(x) for x in all_image_ids])
        text_image_ids = torch.from_numpy(ids_np).long()
        unique_ids, first_idx = np.unique(ids_np, return_index=True)
        image_embeds = image_embeds[torch.from_numpy(first_idx).long()]
        image_ids = torch.from_numpy(unique_ids).long()
    else:
        image_ids = None
        text_image_ids = None

    metrics = compute_retrieval_metrics(
        image_embeds, text_embeds,
        image_ids=image_ids, text_image_ids=text_image_ids,
    )
    metrics['val_loss'] = loss_meter.avg
    
    logger.info(f"Validation Epoch {epoch} - Loss: {loss_meter.avg:.4f}")
    logger.info(f"Retrieval Metrics: {metrics}")
    
    if use_wandb and wandb.run is not None:
        wandb.log({f'val/{k}': v for k, v in metrics.items()})
        wandb.log({'val/epoch': epoch})
    
    return metrics


def main():
    args = parse_args()

    # Distributed setup. When launched via torchrun (RANK/WORLD_SIZE present
    # in env), this initializes the process group and pins each rank to its
    # own GPU; when launched plainly (`python train.py ...`), it stays in
    # single-process mode. Everything downstream branches off these values.
    rank, local_rank, world_size, is_distributed = _setup_distributed()
    is_main = (rank == 0)

    # Load + sanity-check the config so missing keys surface here, not mid-epoch.
    config = load_config(args.config)
    validate_config(config)
    if is_main:
        print("Configuration:")
        print_config(config)
        if is_distributed:
            print(f"Distributed: rank {rank} / world_size {world_size}, "
                  f"local_rank {local_rank}, effective_batch = "
                  f"{config['training']['batch_size']} x {world_size} = "
                  f"{config['training']['batch_size'] * world_size}")

    # Setup logger. Only rank 0 emits to stdout / file; other ranks get a
    # silent logger so duplicate lines do not flood the log.
    log_dir = Path("logs")
    if is_main:
        log_dir.mkdir(exist_ok=True)
        logger = setup_logger(
            name="vl_jepa",
            log_file=log_dir / f"train_{time.strftime('%Y%m%d_%H%M%S')}.log",
        )
    else:
        # Minimal stub so the rest of the code can call logger.info etc.
        import logging as _logging
        logger = _logging.getLogger(f"vl_jepa_silent_rank{rank}")
        logger.addHandler(_logging.NullHandler())
        logger.setLevel(_logging.CRITICAL)

    # Setup device. DDP: each rank owns one GPU at index local_rank.
    if is_distributed and torch.cuda.is_available():
        device = f"cuda:{local_rank}"
    elif args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'
    else:
        device = args.device

    logger.info(f"Using device: {device}")
    if device.startswith('cuda'):
        logger.info(f"GPU: {torch.cuda.get_device_name(device)}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")

    # Initialize wandb only on rank 0; other ranks would create duplicate runs.
    if is_main and args.wandb and HAS_WANDB and config['logging'].get('use_wandb', False):
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

    # Cache the tokenizer reference BEFORE any wrapper (torch.compile or DDP)
    # potentially obscures attribute access. The dataset construction below
    # needs the live tokenizer object.
    tokenizer = model.text_encoder.tokenizer

    # Optional torch.compile. Adds ~30s on first forward; only worth it for
    # production runs where the graph is stable. Wrap AFTER moving to device
    # and AFTER grabbing the tokenizer, BEFORE the DDP wrap.
    if args.compile:
        if not hasattr(torch, 'compile'):
            logger.warning("--compile requested but torch.compile is unavailable in this PyTorch build")
        else:
            logger.warning(
                "--compile enabled. torch.compile interacts poorly with EMA, "
                "deepcopy and dynamic shapes. For multi-day runs, verify on a "
                "single-epoch dry run first."
            )
            logger.info("torch.compile(mode='reduce-overhead') enabled")
            model = torch.compile(model, mode='reduce-overhead')

    # Stage-2: Freeze text encoder. This MUST happen before DDP wraps the
    # model, otherwise DDP's bucket setup picks up requires_grad=True
    # for parameters we are about to freeze, and the buckets are static.
    if args.stage2:
        # Stage-2 fine-tunes a Stage-1 checkpoint; without --resume we'd be
        # silently training from scratch with frozen text, which is useless.
        if not args.resume:
            raise SystemExit(
                "--stage2 requires --resume <stage1-checkpoint>. Refusing to "
                "start Stage-2 fine-tuning from random weights."
            )
        logger.info("=" * 50)
        logger.info("STAGE-2 TRAINING MODE")
        logger.info("=" * 50)
        logger.info(f"Stage-2: Will load Stage-1 weights from {args.resume}")
        freeze_info = model.freeze_text_encoder()
        logger.info(f"Frozen text encoder parameters: {freeze_info['frozen_text_params'] / 1e6:.2f}M")
        logger.info(f"Trainable parameters: {freeze_info['trainable_params'] / 1e6:.2f}M")
        logger.info("Text encoder weights are FROZEN - no gradients will flow through DistilBERT")
        logger.info(f"Stage-2 epochs (CLI override): {args.stage2_epochs}")
        logger.info(f"Stage-2 LR factor: {args.stage2_lr_factor}")
        logger.info(f"Early stopping patience: {args.early_stop_patience} (based on mean_recall)")
        
        # Verify freeze was successful
        verify_info = model.verify_frozen_text_encoder()
        if verify_info['all_frozen']:
            logger.info(f"✓ Verified: All {verify_info['total_count']} text encoder parameters are frozen")
        else:
            logger.error(f"✗ ERROR: Only {verify_info['frozen_count']}/{verify_info['total_count']} text params frozen!")
            raise RuntimeError("Text encoder freeze verification failed!")
        logger.info("=" * 50)
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {num_params / 1e6:.2f}M")
    logger.info(f"Trainable parameters: {num_trainable / 1e6:.2f}M")

    # DDP wrap. Must be AFTER .to(device), AFTER tokenizer capture, AFTER any
    # Stage-2 parameter freezing. We don't need find_unused_parameters=True:
    # in mode='both' every trainable parameter participates in the backward
    # graph (target encoders are no_grad and outside DDP's bucket tracking).
    if is_distributed:
        # device_ids must be a single-GPU list when each rank owns one GPU.
        model = DDP(
            model,
            device_ids=[local_rank] if device.startswith('cuda') else None,
            output_device=local_rank if device.startswith('cuda') else None,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
        logger.info(f"DDP wrapped on local_rank={local_rank}, world_size={world_size}")
    
    # Create datasets. Mask generation now lives in Dataset.__getitem__ so
    # it can parallelize across DataLoader workers; validation doesn't need
    # masks (uses contrastive mode only).
    logger.info("Creating datasets...")
    train_transform = get_train_transforms(config['data'])
    val_transform = get_val_transforms(config['data'])

    # tokenizer was captured pre-compile above.
    # Pass the full config so the mask generator can read both data.masking
    # (block scales, counts) and model.vision_encoder (image_size, patch_size).
    mask_generator = create_mask_generator(config)

    train_dataset = create_dataset(
        config,
        split='train',
        transform=train_transform,
        tokenizer=tokenizer,
        mask_generator=mask_generator,
    )

    val_dataset = None
    try:
        val_dataset = create_dataset(
            config,
            split='val',
            transform=val_transform,
            tokenizer=tokenizer,
            mask_generator=None,  # contrastive-only path
        )
        logger.info(f"Val dataset: {len(val_dataset)} samples")
    except FileNotFoundError:
        logger.warning("Validation dataset not found, skipping validation")

    logger.info(f"Train dataset: {len(train_dataset)} samples")
    
    # Create dataloaders
    num_workers = config['data'].get('num_workers', 2)
    train_loader_kwargs = {
        'batch_size': config['training']['batch_size'],
        'num_workers': num_workers,
        'pin_memory': config['data'].get('pin_memory', True),
        'collate_fn': jepa_collate_fn,
    }
    # Only add these for multiprocessing (num_workers > 0)
    if num_workers > 0:
        train_loader_kwargs['persistent_workers'] = config['data'].get('persistent_workers', True)
        train_loader_kwargs['prefetch_factor'] = config['data'].get('prefetch_factor', 2)
        train_loader_kwargs['worker_init_fn'] = _worker_init

    # Under DDP we use DistributedSampler so each rank sees a non-overlapping
    # slice of the dataset. drop_last=True keeps per-rank batch sizes uniform,
    # which the all_gather in compute_contrastive_loss requires (tensors of
    # mismatched shape would crash NCCL). The sampler's set_epoch(epoch) is
    # called at the top of each train epoch to reshuffle.
    train_sampler = None
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank,
            shuffle=True, drop_last=True,
        )
        train_loader_kwargs['sampler'] = train_sampler
        train_loader_kwargs['shuffle'] = False
    else:
        train_loader_kwargs['shuffle'] = True

    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    
    val_loader = None
    if val_dataset is not None and is_main:
        # Only rank 0 runs validation. compute_contrastive_loss skips the
        # all-gather in eval mode, so non-rank-0 processes do not need to
        # iterate the val loader. They will block at a barrier while rank 0
        # validates and saves the checkpoint. This is the simplest correct
        # design; the alternative (all ranks validate, gather embeddings)
        # would save ~30s of wall time per epoch but adds substantial code.
        val_bs_factor = config['training'].get('val_batch_size_factor', 1)
        val_num_workers = config['data'].get('num_workers', 2)
        val_loader_kwargs = {
            'batch_size': config['training']['batch_size'] * val_bs_factor,
            'shuffle': False,
            'num_workers': val_num_workers,
            'pin_memory': config['data'].get('pin_memory', True),
            'collate_fn': jepa_collate_fn,
        }
        if val_num_workers > 0:
            val_loader_kwargs['worker_init_fn'] = _worker_init
        val_loader = DataLoader(val_dataset, **val_loader_kwargs)
    
    # Mask generator already built and passed into create_dataset above;
    # the trainer no longer needs a separate reference (kept as a no-op
    # arg into train_one_epoch for back-compat).

    # Create optimizer and scheduler (with Stage-2 settings if applicable).
    # steps_per_epoch must be in optimizer steps (post grad-accum), not in
    # micro-batches, otherwise the cosine T_max is sized too long and the LR
    # never reaches min_lr by end of training.
    grad_accum_steps = config['training']['gradient_accumulation_steps']
    steps_per_epoch_opt = max(1, -(-len(train_loader) // grad_accum_steps))
    # Optimizer construction reads model.get_parameter_groups, which lives on
    # VLJEPAModel, not on the DDP wrapper -- pass the unwrapped module.
    optimizer = create_optimizer(_unwrap(model), config, stage2=args.stage2, stage2_lr_factor=args.stage2_lr_factor)
    scheduler, warmup_steps = create_scheduler(optimizer, config, steps_per_epoch_opt)

    # Create gradient scaler for mixed precision (no-op on CPU)
    scaler = GradScaler(device=device, enabled=device.startswith('cuda'))

    # Load checkpoint if resuming
    start_epoch = 0
    global_step = 0
    best_metric = float('inf') if not args.stage2 else 0.0  # Stage-2 uses mean_recall (higher is better)
    best_mean_recall = 0.0  # Track best mean recall for Stage-2

    if args.resume:
        # load_checkpoint calls model.load_state_dict; the state dict was
        # saved from the underlying VLJEPAModel (DDP unwraps on save), so it
        # must be loaded into the unwrapped module too.
        checkpoint_info = load_checkpoint(
            args.resume,
            _unwrap(model),
            optimizer=None if args.stage2 else optimizer,  # Don't load optimizer state for Stage-2
            scheduler=None if args.stage2 else scheduler,  # Don't load scheduler state for Stage-2
            device=device,
        )
        if args.stage2:
            # For Stage-2, start fresh epoch count but keep model weights
            start_epoch = 0
            global_step = 0
            logger.info(f"Stage-2: Loaded model weights from checkpoint (epoch {checkpoint_info['epoch']})")
            logger.info("Stage-2: Reset optimizer and scheduler for fine-tuning")
        else:
            start_epoch = checkpoint_info['epoch'] + 1
            global_step = checkpoint_info['global_step']
            best_metric = checkpoint_info['best_metric']
            logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop. Stage-2 epoch count: CLI > config > default 4.
    if args.stage2:
        num_epochs = (
            args.stage2_epochs
            if args.stage2_epochs is not None
            else config['training'].get('stage2_epochs', 4)
        )
    else:
        num_epochs = config['training']['num_epochs']
    checkpoint_dir = Path(config['training'].get('checkpoint_dir', 'checkpoints'))
    checkpoint_dir.mkdir(exist_ok=True)
    save_every = config['training'].get('save_every', 5)
    
    # Early stopping for Stage-2
    early_stop_patience = args.early_stop_patience
    epochs_without_improvement = 0
    
    logger.info("Starting training...")
    if args.stage2:
        logger.info(f"Stage-2: Training for {num_epochs} epochs with early stopping (patience={early_stop_patience})")
        logger.info(f"Stage-2: Loss weights LOCKED at jepa={config['training'].get('jepa_loss_weight', 1.0)}, contrastive={config['training'].get('contrastive_loss_weight', 0.5)}")
    
    for epoch in range(start_epoch, num_epochs):
        # DistributedSampler must be told the current epoch so its internal
        # RNG advances; without this, every epoch sees the same shuffle.
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

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
            warmup_steps=warmup_steps if not args.stage2 else 0,  # No warmup for Stage-2
            device=device,
            stage2=args.stage2,  # Pass stage2 flag for safety checks
        )

        # Validate (if val_loader exists). Only rank 0 has a val_loader (see
        # dataloader setup above) and only rank 0 holds the eval graph for
        # the unwrapped model; other ranks reach the barrier below and wait.
        val_metrics = None
        if is_main and val_loader is not None:
            val_metrics = validate(
                model=_unwrap(model),
                dataloader=val_loader,
                epoch=epoch,
                config=config,
                logger=logger,
                device=device,
            )

        # Determine if this is the best model. When no val_loader is
        # available, fall back to train_loss for Stage-1 and disable early
        # stopping for Stage-2 (we can't measure mean_recall).
        if val_metrics is None:
            val_metrics = {'val_loss': train_loss, 'mean_recall': 0.0}
            no_val = True
        else:
            no_val = False

        if args.stage2:
            if no_val:
                # No val_loader: skip mean_recall tracking and early stopping.
                is_best = True
                logger.warning("Stage-2 without val_loader: saving every epoch, early stopping disabled.")
            else:
                # Stage-2: Use mean_recall (higher is better)
                current_mean_recall = val_metrics.get('mean_recall', 0.0)
                is_best = current_mean_recall > best_mean_recall
                if is_best:
                    best_mean_recall = current_mean_recall
                    epochs_without_improvement = 0
                    logger.info(f"Stage-2: New best mean_recall = {best_mean_recall:.2f}%")
                else:
                    epochs_without_improvement += 1
                    logger.info(f"Stage-2: No improvement for {epochs_without_improvement} epoch(s)")
                if epochs_without_improvement >= early_stop_patience:
                    logger.info(f"Stage-2: Early stopping triggered (no improvement for {early_stop_patience} epochs)")
                    break
        else:
            # Stage-1: val_loss (lower is better). When no val_loader, val_loss
            # falls back to train_loss above, so progress still updates best.
            is_best = val_metrics['val_loss'] < best_metric
            if is_best:
                best_metric = val_metrics['val_loss']
        
        # Save checkpoint. Only rank 0 writes to disk; other ranks block on
        # the barrier so they do not race ahead into the next epoch while
        # rank 0 is still serializing ~1 GB to /kaggle/working.
        if is_main:
            keep_last_n = config['training'].get('keep_last_n_checkpoints')
            # Always save from the unwrapped module so the state dict keys
            # don't have the DDP "module." prefix; load_checkpoint expects
            # the unwrapped keys.
            ckpt_model = _unwrap(model)
            if args.stage2:
                # Stage-2: Always save each epoch, mark best
                save_checkpoint(
                    model=ckpt_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    best_metric=best_mean_recall,
                    config=config,
                    save_path=checkpoint_dir / f"stage2_epoch_{epoch}.pth",
                    is_best=is_best,
                )
                _prune_old_checkpoints(checkpoint_dir, "stage2_epoch_*.pth", keep_last_n)
                if is_best:
                    save_checkpoint(
                        model=ckpt_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        global_step=global_step,
                        best_metric=best_mean_recall,
                        config=config,
                        save_path=checkpoint_dir / "stage2_best.pth",
                        is_best=True,
                    )
            else:
                if (epoch + 1) % save_every == 0 or is_best:
                    save_checkpoint(
                        model=ckpt_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        global_step=global_step,
                        best_metric=best_metric,
                        config=config,
                        save_path=checkpoint_dir / f"checkpoint_epoch_{epoch}.pth",
                        is_best=is_best,
                    )
                    _prune_old_checkpoints(checkpoint_dir, "checkpoint_epoch_*.pth", keep_last_n)
        # Resync ranks at the end of every epoch. Without this, rank 1 may
        # start the next epoch's forward while rank 0 is still saving, which
        # is fine semantically but produces confusing tqdm interleaving.
        _barrier()

    logger.info("Training completed!")
    if args.stage2:
        logger.info(f"Stage-2 Best Mean Recall: {best_mean_recall:.2f}%")

    if is_main and HAS_WANDB and wandb is not None and wandb.run is not None:
        wandb.finish()

    # Clean shutdown of the process group. Skipping this leaves NCCL state
    # in a state that some PyTorch versions warn about on exit.
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
