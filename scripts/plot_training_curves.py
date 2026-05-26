"""
Parse a VL-JEPA train log and produce a single PNG with the training-side
losses on one panel and the validation-side retrieval metrics on another.

Intended as the headline figure for the README. Reads either the local
checkpoints/logs/*.log file or a Kaggle output bundle. Designed to be
re-runnable as more epochs land (the parser is tolerant of partial logs).

Usage:
    python scripts/plot_training_curves.py <log-path> [-o <out.png>]
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Two log line shapes the trainer emits per epoch:
#   "Epoch 9 - Loss: 0.6912, JEPA Loss: 0.2223, Contrastive Loss: 0.9378"
#   "Retrieval Metrics: {'i2t_recall@1': 50.295..., ..., 'mean_recall': 68.04...}"
EPOCH_TRAIN_RE = re.compile(
    r"Epoch\s+(\d+)\s+-\s+Loss:\s+([0-9.]+),\s+JEPA Loss:\s+([0-9.]+),\s+Contrastive Loss:\s+([0-9.]+)"
)
RETRIEVAL_RE = re.compile(r"Retrieval Metrics:\s+(\{.*?\})")


def parse_log(log_path: Path):
    train_rows = {}    # epoch -> (loss, jepa, contrastive)
    retrieval_rows = []  # list of dicts as they appear in order

    text = log_path.read_text(encoding="utf-8", errors="replace")
    for m in EPOCH_TRAIN_RE.finditer(text):
        epoch = int(m.group(1))
        train_rows[epoch] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))

    # Validation lines appear in epoch order; index them by encounter.
    for m in RETRIEVAL_RE.finditer(text):
        try:
            metrics = eval(m.group(1), {"__builtins__": {}}, {})  # safe: ints+floats only
        except Exception:
            continue
        retrieval_rows.append(metrics)

    return train_rows, retrieval_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log_path", help="Path to train log (logs/train_*.log)")
    p.add_argument(
        "-o", "--out", default="docs/training_curves.png",
        help="Output PNG path (default: docs/training_curves.png)",
    )
    args = p.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        sys.exit(f"log not found: {log_path}")

    train, retrieval = parse_log(log_path)
    if not train:
        sys.exit("No training-epoch lines parsed. Check the log format.")
    if not retrieval:
        print("WARNING: no validation lines parsed; only training panel will render.")

    epochs = sorted(train.keys())
    combined = [train[e][0] for e in epochs]
    jepa = [train[e][1] for e in epochs]
    contrast = [train[e][2] for e in epochs]

    val_epochs = list(range(len(retrieval)))
    i2t_r1 = [r.get("i2t_recall@1", float("nan")) for r in retrieval]
    t2i_r1 = [r.get("t2i_recall@1", float("nan")) for r in retrieval]
    mean_r = [r.get("mean_recall", float("nan")) for r in retrieval]

    fig, (ax_loss, ax_recall) = plt.subplots(
        1, 2, figsize=(12, 4.5), gridspec_kw={"wspace": 0.3},
    )

    # Loss panel
    ax_loss.plot(epochs, combined, marker="o", label="combined (jepa+0.5*contrast)", color="#222222", linewidth=2)
    ax_loss.plot(epochs, jepa, marker="s", label="JEPA loss", color="#1f77b4")
    ax_loss.plot(epochs, contrast, marker="^", label="SigLIP contrast loss", color="#d62728")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training losses")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(loc="upper right", fontsize=9)
    ax_loss.set_xticks(epochs)

    # Recall panel
    ax_recall.plot(val_epochs, i2t_r1, marker="o", label="i2t recall@1", color="#1f77b4", linewidth=2)
    ax_recall.plot(val_epochs, t2i_r1, marker="s", label="t2i recall@1", color="#d62728", linewidth=2)
    ax_recall.plot(val_epochs, mean_r, marker="^", label="mean recall (all R@K)", color="#2ca02c", linewidth=2)
    ax_recall.set_xlabel("Epoch")
    ax_recall.set_ylabel("Recall (%)")
    ax_recall.set_title("COCO 5K validation retrieval")
    ax_recall.grid(True, alpha=0.3)
    ax_recall.legend(loc="lower right", fontsize=9)
    ax_recall.set_xticks(val_epochs)
    ax_recall.set_ylim(0, max(85, max([m for m in mean_r if m == m] + [50])))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("VL-JEPA pretraining on COCO 2017 (ViT-Tiny + DistilBERT, dual T4, 10 epochs)", fontsize=11)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")

    if retrieval:
        print(f"\nFinal-epoch retrieval (epoch {val_epochs[-1]}):")
        for k, v in retrieval[-1].items():
            if k.endswith("loss"):
                continue
            print(f"  {k:20s} = {v:.2f}")


if __name__ == "__main__":
    main()
