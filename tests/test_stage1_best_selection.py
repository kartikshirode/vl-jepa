"""
Regression: Stage-1 must select the best checkpoint by mean_recall (higher
is better), not val_loss (lower is better).

The v15 run on Kaggle exposed why this matters. Under SigLIP, val_loss climbed
monotonically (3.43 epoch 0 -> 4.21 epoch 9) because SigLIP couples its loss
magnitude to logit_scale and the per-rank eval-mode loss is computed without
the all-gather (different sample distribution). Meanwhile mean_recall climbed
correctly (43.82 -> 68.04). The old `is_best = val_loss < best_metric` check
fired only for epoch 0 in v15, so best_model.pth was saved with the WORST
retrieval weights of the entire run. Switching to mean_recall fixes that.

These tests inspect the source rather than running a full epoch end-to-end --
training for an epoch costs minutes even in the smallest fixture configuration.
The structural assertions catch any future regression that re-introduces a
val_loss-based selector.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _train_source() -> str:
    return (ROOT / "train.py").read_text(encoding="utf-8")


def test_stage1_uses_mean_recall_for_is_best():
    """The Stage-1 branch of the is_best check must read mean_recall, not val_loss."""
    src = _train_source()

    # Find the else (Stage-1) branch that follows the args.stage2 check around
    # the per-epoch save block. We grep for the comment header so the test
    # tolerates light refactors but fails loud if someone reverts the metric.
    assert "Stage-1: mean_recall (higher is better)" in src, (
        "Stage-1 selection metric comment is missing; the comparison may have "
        "regressed to val_loss. Check train.py around the post-validation block."
    )
    # And confirm the actual comparison is on mean_recall, not val_loss.
    assert re.search(r"current_mean_recall\s*=\s*val_metrics\.get\(['\"]mean_recall['\"]", src), (
        "current_mean_recall assignment from val_metrics missing in train.py"
    )
    assert "is_best = current_mean_recall > best_metric" in src, (
        "Stage-1 is_best comparison must use 'current_mean_recall > best_metric'"
    )


def test_stage1_best_metric_initialized_for_higher_is_better():
    """best_metric must start at 0.0, not float('inf'), now that the comparison
    is mean_recall > best_metric. With +inf as the initializer no epoch ever
    qualifies as best."""
    src = _train_source()
    # The old initializer is gone, the new one is present.
    bad_pattern = re.search(r"best_metric\s*=\s*float\(['\"]inf['\"]\)", src)
    assert bad_pattern is None, (
        "best_metric is still initialized to float('inf') somewhere in train.py. "
        "With mean_recall-based selection this would prevent any epoch from "
        "being marked best."
    )
    assert re.search(r"best_metric\s*=\s*0\.0", src), (
        "best_metric should be initialized to 0.0 for the mean_recall-based "
        "selection to work."
    )


def test_old_val_loss_selection_is_gone():
    """The explicit `is_best = val_metrics['val_loss'] < best_metric` line was
    the bug. It must not exist anywhere in train.py."""
    src = _train_source()
    bad = re.search(r"is_best\s*=\s*val_metrics\[['\"]val_loss['\"]\]\s*<\s*best_metric", src)
    assert bad is None, (
        "Found a val_loss-based is_best comparison in train.py. This was the "
        "v15 best_model.pth bug. Use mean_recall instead."
    )
