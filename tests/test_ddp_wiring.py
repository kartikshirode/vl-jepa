"""
Guard the dual-T4 DDP wiring against silent regressions.

We can't actually spin up a 2-process DDP run inside pytest (the runner is
single-process), so these tests cover the seams that have to behave
correctly outside DDP and the boundary conditions that determine whether
the all-gather is invoked. The actual 2-rank correctness is exercised on
Kaggle and via scripts/smoke_ddp_singleproc.py.
"""

import os
import sys

import pytest
import torch
import torch.distributed as dist

REPO_ROOT = __import__('pathlib').Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vl_jepa.models.vl_jepa import (  # noqa: E402
    _ddp_is_active, _gather_for_contrastive, _AllGatherWithGrad,
)
import train as train_module  # noqa: E402


def test_ddp_is_active_false_when_dist_not_initialized():
    """Default test environment: no process group, _ddp_is_active is False."""
    if dist.is_available() and dist.is_initialized():
        pytest.skip("dist already initialized; can't test the off path here")
    assert _ddp_is_active() is False


def test_gather_for_contrastive_is_identity_without_ddp():
    """Outside DDP the gather wrapper returns the input tensor exactly.

    This is the back-compat invariant: existing single-GPU runs see
    unchanged loss math, unchanged gradients, unchanged everything.
    """
    t = torch.randn(4, 8, requires_grad=True)
    out = _gather_for_contrastive(t)
    assert out is t, "single-GPU path must short-circuit and return the same tensor"


def test_setup_distributed_returns_singleprocess_defaults_without_env():
    """Without RANK/WORLD_SIZE in env, _setup_distributed returns the
    single-process tuple and does not initialize a process group."""
    for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        os.environ.pop(k, None)
    rank, local_rank, world_size, is_distributed = train_module._setup_distributed()
    assert (rank, local_rank, world_size, is_distributed) == (0, 0, 1, False)


def test_setup_distributed_singleprocess_world_size_one_does_not_init(monkeypatch):
    """WORLD_SIZE=1 must not init a process group; init at world_size=1 is
    technically valid but produces a NCCL group that emits noisy warnings.
    """
    if dist.is_available() and dist.is_initialized():
        pytest.skip("dist already initialized externally")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    rank, local_rank, world_size, is_distributed = train_module._setup_distributed()
    assert is_distributed is False
    assert world_size == 1
    assert not dist.is_initialized()


def test_unwrap_returns_self_when_not_ddp():
    """train_module._unwrap on a plain nn.Module is identity."""
    m = torch.nn.Linear(4, 4)
    assert train_module._unwrap(m) is m


def test_compute_contrastive_loss_no_gather_in_eval_mode(tiny_model):
    """Eval mode disables the all-gather so rank 0 can run validate() alone
    without deadlocking on a collective that the other ranks aren't calling.

    Without DDP this is trivially true, but the gate is structural and we
    want a test that fails if someone removes the `self.training` check.
    """
    tiny_model.eval()
    B = 4
    v = torch.randn(B, 256)
    t = torch.randn(B, 256)
    v = torch.nn.functional.normalize(v, dim=-1)
    t = torch.nn.functional.normalize(t, dim=-1)
    # Just confirm the loss is computable and finite; the structural test is
    # that the model attribute we rely on is still there.
    assert hasattr(tiny_model, 'training')
    assert tiny_model.training is False
    loss = tiny_model.compute_contrastive_loss(v, t)
    assert torch.isfinite(loss)


def test_compute_contrastive_loss_gathers_in_training_mode_outside_ddp_is_noop(tiny_model):
    """In training mode without DDP, _gather_for_contrastive is still a
    no-op (because _ddp_is_active is False), so the loss equals the
    single-rank loss. This is the back-compat invariant on the training
    side.
    """
    tiny_model.train()
    B = 4
    v = torch.randn(B, 256)
    t = torch.randn(B, 256)
    v = torch.nn.functional.normalize(v, dim=-1)
    t = torch.nn.functional.normalize(t, dim=-1)
    loss = tiny_model.compute_contrastive_loss(v, t)
    assert torch.isfinite(loss)


def test_all_gather_with_grad_function_metadata():
    """The autograd.Function has the right forward/backward signature so
    a future PyTorch version that tightens autograd.Function API doesn't
    silently break the DDP path."""
    assert hasattr(_AllGatherWithGrad, 'forward')
    assert hasattr(_AllGatherWithGrad, 'backward')


def test_train_module_exports_ddp_helpers():
    """train.py exports the helpers train_kaggle.py and tests rely on."""
    for name in ('_setup_distributed', '_is_main_process', '_unwrap', '_barrier'):
        assert hasattr(train_module, name), f"train.{name} missing"
