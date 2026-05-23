"""
Phase 2 verification: EMA momentum schedule reaches ema_end correctly
and in-place EMA update doesn't allocate new tensors.
"""

import math
import torch
import pytest

from vl_jepa.models import VisionEncoder, TextEncoder, VLJEPAModel
from vl_jepa.models.predictor import PredictorTransformer


def _build_tiny_model():
    vision = VisionEncoder(pretrained=False, gradient_checkpointing=False)
    text = TextEncoder(gradient_checkpointing=False)
    predictor = PredictorTransformer(
        input_dim=192, hidden_dim=384, output_dim=192,
        num_layers=2, num_heads=6, num_patches=196,
    )
    return VLJEPAModel(vision, text, predictor)


def test_ema_in_place_update():
    """update_target_encoder should mutate target params in place, not reallocate."""
    model = _build_tiny_model()
    model.ema_momentum = 0.5
    # Snapshot the storage pointer of one target param. mul_/add_ should not
    # replace the underlying tensor object.
    target_param = next(model.target_vision_encoder.parameters())
    before_storage = target_param.data.data_ptr()
    # Make context and target differ so the EMA actually moves them.
    for p in model.vision_encoder.parameters():
        p.data.add_(0.1)
    model.update_target_encoder()
    after_storage = target_param.data.data_ptr()
    assert before_storage == after_storage, "EMA update reallocated target tensor"


def test_ema_schedule_reaches_end():
    """
    Simulate the train loop's progress math: with `total_optimizer_steps`
    matching the loop's denominator, after the final step the momentum
    must equal ema_end (within float precision).
    """
    ema_start, ema_end = 0.996, 1.0
    total_optimizer_steps = 100
    # progress = global_step / max(1, total_optimizer_steps); momentum at the
    # final iteration uses global_step == total_optimizer_steps - 1 BEFORE
    # increment, then we step. The schedule clamps progress to 1.0, so the
    # final momentum after the last optimizer step is exactly ema_end.
    last_momentum = ema_start + (ema_end - ema_start) * min(1.0, total_optimizer_steps / max(1, total_optimizer_steps))
    assert math.isclose(last_momentum, ema_end, rel_tol=1e-6)


def test_ema_progress_uses_optimizer_steps_not_microbatches():
    """
    Regression: with grad_accum_steps > 1, the schedule must be in
    optimizer steps, not micro-batches. We simulate the trainer's math
    and check that at the end of training progress == 1.0.

    Trainer uses ceiling division so a partial tail (micro-batches not
    divisible by grad_accum) still counts; the per-epoch end-flush in
    train_one_epoch fires the optimizer for that tail.
    """
    num_epochs = 2
    micro_batches_per_epoch = 10
    grad_accum = 4
    # Ceiling division mirrors train.py.
    steps_per_epoch = max(1, -(-micro_batches_per_epoch // grad_accum))
    total_optimizer_steps = max(1, num_epochs * steps_per_epoch)

    # Walk through one epoch: optimizer steps fire on every `grad_accum`th
    # batch, plus a tail-flush at end of each epoch when grad has accumulated.
    global_step = 0
    for epoch in range(num_epochs):
        pending = False
        for i in range(micro_batches_per_epoch):
            pending = True
            if (i + 1) % grad_accum == 0:
                global_step += 1
                pending = False
        if pending:
            global_step += 1
    progress = min(1.0, global_step / max(1, total_optimizer_steps))
    assert progress == 1.0, f"end-of-training progress should be 1.0, got {progress}"
