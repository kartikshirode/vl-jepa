"""Kaggle Notebooks entrypoint for VL-JEPA training on a single P100.

This script is the `code_file` referenced by `kernel-metadata.json`. It runs
inside the Kaggle kernel runtime and handles environment setup, dependency
install, repo clone, and the training launch. It assumes the kernel has the
`awsaf49/coco-2017-dataset` dataset attached and a GPU accelerator enabled.

The local laptop config and code stay untouched; this script and the matching
`configs/config_kaggle_p100.yaml` are the only Kaggle-specific surface.
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_URL = "https://github.com/kartikshirode/vl-jepa.git"
REPO_DIR = Path("/kaggle/working/vl-jepa")
KAGGLE_CONFIG_REL = "configs/config_kaggle_p100.yaml"
DATA_ROOT = Path("/kaggle/input/coco-2017-dataset/coco2017")

# Silence noisy warnings before any imports that trigger them.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def banner(msg: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n{msg}\n{bar}", flush=True)


def pip_install(*pkgs: str) -> None:
    """Quiet pip install. Skips on already-installed packages."""
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *pkgs],
    )


def check_gpu() -> None:
    import torch

    if not torch.cuda.is_available():
        print(
            "ERROR: No CUDA device visible. Enable a GPU accelerator in the "
            "kernel settings (Settings -> Accelerator -> GPU P100).",
            file=sys.stderr,
        )
        sys.exit(1)
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} | VRAM: {props.total_memory / 1024**3:.1f} GB")
    print(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")
    if "P100" not in props.name:
        print(
            f"WARNING: expected an NVIDIA P100, got '{props.name}'. The "
            f"config_kaggle_p100.yaml batch size and worker counts were sized "
            f"for P100; other accelerators may need tuning."
        )


def check_dataset() -> None:
    train_dir = DATA_ROOT / "train2017"
    val_dir = DATA_ROOT / "val2017"
    ann_dir = DATA_ROOT / "annotations"
    missing = [str(p) for p in (train_dir, val_dir, ann_dir) if not p.exists()]
    if missing:
        print(
            "ERROR: COCO 2017 dataset is not mounted where expected.\n"
            f"  Missing: {missing}\n"
            "Add the 'awsaf49/coco-2017-dataset' dataset to the kernel "
            "(Add Input -> Search 'coco-2017-dataset' -> select 'awsaf49').",
            file=sys.stderr,
        )
        sys.exit(1)
    n_train = sum(1 for _ in train_dir.iterdir())
    n_val = sum(1 for _ in val_dir.iterdir())
    print(f"COCO 2017 at {DATA_ROOT}")
    print(f"  train2017: {n_train:,} files")
    print(f"  val2017:   {n_val:,} files")
    print(f"  annotations: {[p.name for p in ann_dir.iterdir()]}")


def clone_repo() -> None:
    if REPO_DIR.exists():
        print(f"Repo already at {REPO_DIR}; pulling latest commit on main...")
        subprocess.check_call(
            ["git", "-C", str(REPO_DIR), "fetch", "--depth", "1", "origin", "main"],
        )
        subprocess.check_call(
            ["git", "-C", str(REPO_DIR), "reset", "--hard", "origin/main"],
        )
    else:
        print(f"Cloning {REPO_URL} -> {REPO_DIR} ...")
        subprocess.check_call(
            ["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)],
        )
    head = subprocess.check_output(
        ["git", "-C", str(REPO_DIR), "log", "-1", "--oneline"], text=True,
    ).strip()
    print(f"Repo HEAD: {head}")


def install_missing_deps() -> None:
    """Install packages that the Kaggle base image doesn't already provide.

    Kaggle ships torch, torchvision, transformers, timm, numpy, pillow, pyyaml,
    tqdm by default. The following are missing on at least some Kaggle images
    and are the ones train.py + vl_jepa actually need at runtime. Anything
    optional (bitsandbytes, wandb, tensorboard, opencv, matplotlib) is
    intentionally skipped; train.py gracefully degrades when they are absent.
    """
    pip_install("einops", "omegaconf", "pycocotools", "accelerate")


def copy_outputs_to_kaggle_root() -> None:
    """Mirror logs into /kaggle/working/logs/ so they sit beside checkpoints.

    train.py writes logs to a relative `logs/` directory under its cwd, which
    inside the cloned repo means `/kaggle/working/vl-jepa/logs/`. Kaggle's
    output capture grabs the full `/kaggle/working/` tree, but having the logs
    one click deeper than the checkpoints is just noise. This copies them up.
    """
    src = REPO_DIR / "logs"
    dst = Path("/kaggle/working/logs")
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for log_file in src.glob("*.log"):
        shutil.copy2(log_file, dst / log_file.name)
    print(f"Copied {sum(1 for _ in src.glob('*.log'))} log files to {dst}")


def run_training() -> int:
    os.chdir(REPO_DIR)
    cmd = [sys.executable, "train.py", "--config", KAGGLE_CONFIG_REL]
    banner(f"Launching: {' '.join(cmd)}")
    t0 = time.time()
    try:
        rc = subprocess.call(cmd)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received; train.py was sent SIGINT.")
        rc = 130
    elapsed = time.time() - t0
    banner(f"Training exited with code {rc} after {elapsed / 60:.1f} min")
    return rc


def main() -> int:
    banner("VL-JEPA on Kaggle P100 — environment check")
    check_gpu()
    check_dataset()

    banner("Installing missing dependencies")
    install_missing_deps()

    banner("Fetching repository")
    clone_repo()

    rc = run_training()
    copy_outputs_to_kaggle_root()
    return rc


if __name__ == "__main__":
    sys.exit(main())
