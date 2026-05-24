"""
2-rank DDP smoke on CPU (gloo backend). Verifies the all-gather + DDP
contrastive-loss path that will run on Kaggle's dual T4.

Run via plain python (uses mp.spawn instead of torchrun because Windows
PyTorch builds lack libuv support that torchrun's rendezvous needs):
    python scripts/smoke_ddp_cpu.py

Without 2 physical GPUs locally we can't exercise the cuda:0 / cuda:1 path
the way Kaggle will, but the DDP code paths that matter (init_process_group,
DistributedSampler, autograd-aware all_gather, eval-mode gather gating)
all behave the same under gloo + CPU. NCCL is only an optimization on top.

What this smoke checks per rank:
  - dist.init_process_group succeeds, dist.get_world_size() == 2
  - _gather_for_contrastive on a rank-distinct tensor produces a [B_global, D]
    tensor where rows reflect both ranks (i.e. all_gather actually concatenated)
  - The autograd Function backward returns the correct slice (so each rank
    only gets gradient for its own local rows)
  - VLJEPAModel.compute_contrastive_loss returns the same value on both ranks
    in training mode (i.e. they see the same global sim matrix)
  - In eval mode the loss differs across ranks (because the gather is gated
    off and each rank only sees its own slice). This is the property that
    lets rank 0 validate alone without deadlocking.
"""

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vl_jepa.models.vl_jepa import (
    _gather_for_contrastive, _AllGatherWithGrad, _ddp_is_active,
)
from vl_jepa.models import VisionEncoder, TextEncoder, VLJEPAModel
from vl_jepa.models.predictor import PredictorTransformer


def log(msg: str):
    rank = dist.get_rank() if dist.is_initialized() else -1
    print(f"[rank {rank}] {msg}", flush=True)


def _worker(rank: int, world_size: int):
    # Windows PyTorch wheels are built without libuv support, but c10d's
    # default TCPStore wants it. USE_LIBUV=0 forces the non-libuv path.
    os.environ["USE_LIBUV"] = "0"
    # Set the env vars init_process_group reads when the user didn't go
    # through torchrun.
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29503"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # gloo backend so we don't need NCCL / 2 GPUs locally.
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    log(f"init_process_group ok, world_size={dist.get_world_size()}")
    assert _ddp_is_active(), "DDP should be active after init"

    # 1) Naked all-gather check.
    D = 8
    local = torch.full((4, D), float(rank), requires_grad=True)
    gathered = _AllGatherWithGrad.apply(local)
    assert gathered.shape == (8, D), f"expected [8, {D}], got {tuple(gathered.shape)}"
    # First 4 rows should be rank 0's tensor (all 0s), next 4 should be rank 1's (all 1s).
    assert (gathered[0:4] == 0.0).all() and (gathered[4:8] == 1.0).all(), (
        f"all_gather concat order wrong on rank {rank}: head={gathered[0].mean()}, "
        f"tail={gathered[-1].mean()}"
    )
    # Backward should give this rank only its slice.
    target = torch.arange(8 * D, dtype=torch.float).reshape(8, D)
    loss = ((gathered - target) ** 2).sum()
    loss.backward()
    assert local.grad is not None
    # The slice we got back must correspond to this rank's rows in `gathered`.
    expected_slice_norm = ((torch.full((4, D), float(rank)) - target[rank*4:(rank+1)*4]) * 2).norm().item()
    actual_grad_norm = local.grad.norm().item()
    log(f"all_gather grad slice norm = {actual_grad_norm:.4f}  "
        f"expected ~ {expected_slice_norm:.4f}")
    assert abs(actual_grad_norm - expected_slice_norm) < 1e-4, "wrong backward slice"

    # 2) Real model: compute_contrastive_loss must agree across ranks in
    #    training mode (gather on), differ in eval mode (gather off).
    torch.manual_seed(0)  # SAME seed on both ranks so encoders + projections are bit-identical
    vision = VisionEncoder(pretrained=False, gradient_checkpointing=False)
    text = TextEncoder(gradient_checkpointing=False)
    predictor = PredictorTransformer(
        input_dim=192, hidden_dim=384, output_dim=192,
        num_layers=2, num_heads=6, num_patches=196,
    )
    model = VLJEPAModel(vision, text, predictor, contrastive_loss_type='siglip')

    # Synthetic per-rank embeddings. We make them differ per rank so the gather
    # has something to actually do.
    torch.manual_seed(100 + rank)
    B = 4
    v_local = F.normalize(torch.randn(B, 256), dim=-1)
    t_local = F.normalize(torch.randn(B, 256), dim=-1)

    # Training mode: should gather. Loss must be equal across ranks.
    model.train()
    loss_train = model.compute_contrastive_loss(v_local.clone(), t_local.clone())
    all_losses = [torch.zeros_like(loss_train) for _ in range(world_size)]
    dist.all_gather(all_losses, loss_train)
    train_vals = [l.item() for l in all_losses]
    log(f"training-mode contrastive loss = {loss_train.item():.6f}  (all ranks: {train_vals})")
    assert abs(train_vals[0] - train_vals[1]) < 1e-5, (
        f"training-mode loss differs across ranks: {train_vals}; "
        "all_gather did not synchronize embeddings"
    )

    # Eval mode: should NOT gather. Loss may differ across ranks.
    model.eval()
    loss_eval = model.compute_contrastive_loss(v_local.clone(), t_local.clone())
    all_losses_eval = [torch.zeros_like(loss_eval) for _ in range(world_size)]
    dist.all_gather(all_losses_eval, loss_eval)
    eval_vals = [l.item() for l in all_losses_eval]
    log(f"eval-mode contrastive loss = {loss_eval.item():.6f}  (all ranks: {eval_vals})")
    # The local v_local/t_local differ per rank, so the local-only loss must differ too.
    assert abs(eval_vals[0] - eval_vals[1]) > 1e-3, (
        f"eval-mode loss is identical across ranks ({eval_vals}); the "
        "self.training gate failed and the gather is still active"
    )

    log("DDP smoke PASSED")
    dist.destroy_process_group()


if __name__ == "__main__":
    import torch.multiprocessing as mp
    # spawn 2 worker processes. The smoke is self-contained per rank, so
    # there's nothing the parent process needs to do besides wait.
    mp.spawn(_worker, args=(2,), nprocs=2, join=True)
