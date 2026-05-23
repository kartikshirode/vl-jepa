"""Kaggle Notebooks entrypoint for VL-JEPA training on a single T4.

This script is the `code_file` referenced by `kernel-metadata.json`. It runs
inside the Kaggle kernel runtime and handles environment setup, dependency
install, repo clone, and the training launch. It assumes the kernel has the
`awsaf49/coco-2017-dataset` dataset attached and a GPU accelerator enabled
(use "GPU T4 x2" - the script only addresses the first T4).

Why T4 and not P100: Kaggle's current PyTorch (>= 2.10) dropped support for
Pascal (sm_60), so a P100 kernel fails to launch CUDA. T4 is Turing (sm_75)
and supported.

The local laptop config and code stay untouched; this script and the matching
`configs/config_kaggle_t4.yaml` are the only Kaggle-specific surface.
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_URL = "https://github.com/kartikshirode/vl-jepa.git"
REPO_DIR = Path("/kaggle/working/vl-jepa")
KAGGLE_CONFIG_REL = "configs/config_kaggle_t4.yaml"
KAGGLE_INPUT = Path("/kaggle/input")
# Dataset hosts pick their own internal layout. Try common shapes for the
# awsaf49/coco-2017-dataset and fall back to listing /kaggle/input to help
# diagnose if none match.
DATA_ROOT_CANDIDATES = [
    Path("/kaggle/input/coco-2017-dataset/coco2017"),
    Path("/kaggle/input/coco-2017-dataset"),
    Path("/kaggle/input/coco2017"),
]

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
            "kernel settings (Settings -> Accelerator -> GPU T4 x2).",
            file=sys.stderr,
        )
        sys.exit(1)
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} | VRAM: {props.total_memory / 1024**3:.1f} GB")
    print(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")
    if "T4" not in props.name:
        print(
            f"WARNING: expected an NVIDIA T4, got '{props.name}'. The "
            f"config_kaggle_t4.yaml batch size and worker counts were sized "
            f"for T4 (16 GB, Turing sm_75). If this is a P100, training will "
            f"crash because Kaggle's PyTorch dropped Pascal (sm_60) support."
        )


def check_dataset() -> Path:
    """Find the COCO 2017 mount and return the data_root path.

    Returns the matched data_root path so the caller can pass it into the
    training config as an override (avoiding a hardcoded path in the yaml
    that may drift from what Kaggle actually mounts).
    """
    for candidate in DATA_ROOT_CANDIDATES:
        if (
            (candidate / "train2017").exists()
            and (candidate / "val2017").exists()
            and (candidate / "annotations").exists()
        ):
            n_train = sum(1 for _ in (candidate / "train2017").iterdir())
            n_val = sum(1 for _ in (candidate / "val2017").iterdir())
            print(f"COCO 2017 at {candidate}")
            print(f"  train2017: {n_train:,} files")
            print(f"  val2017:   {n_val:,} files")
            print(f"  annotations: {[p.name for p in (candidate / 'annotations').iterdir()]}")
            return candidate

    # No candidate matched. Dump what's actually under /kaggle/input so we
    # can see what the dataset attach produced.
    print("ERROR: COCO 2017 dataset not found at any expected path.", file=sys.stderr)
    print(f"Tried: {[str(p) for p in DATA_ROOT_CANDIDATES]}", file=sys.stderr)
    if KAGGLE_INPUT.exists():
        print(f"\nContents of {KAGGLE_INPUT}:", file=sys.stderr)
        for top in sorted(KAGGLE_INPUT.iterdir()):
            print(f"  {top}", file=sys.stderr)
            if top.is_dir():
                try:
                    children = sorted(top.iterdir())[:8]
                except Exception:
                    children = []
                for child in children:
                    marker = "/" if child.is_dir() else ""
                    print(f"    {child.name}{marker}", file=sys.stderr)
    print(
        "\nIf the awsaf49/coco-2017-dataset is attached but at a different "
        "path, add that path to DATA_ROOT_CANDIDATES in train_kaggle.py.",
        file=sys.stderr,
    )
    sys.exit(1)


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


def patch_config_for_runtime(data_root: Path) -> Path:
    """Write a runtime-patched copy of the YAML config.

    The committed config has data.data_root pinned to what Kaggle SHOULD mount,
    but check_dataset() may have discovered the dataset at a slightly different
    path. Override data.data_root with the real path so train.py finds the
    files. Returns the path to the patched config.
    """
    import yaml

    src = REPO_DIR / KAGGLE_CONFIG_REL
    dst = Path("/kaggle/working/config_kaggle_t4_runtime.yaml")
    with open(src, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["data_root"] = str(data_root)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Patched config written to {dst} with data_root={data_root}")
    return dst


def run_training(config_path: Path) -> int:
    os.chdir(REPO_DIR)
    cmd = [sys.executable, "train.py", "--config", str(config_path)]
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
    banner("VL-JEPA on Kaggle T4 — environment check")
    check_gpu()
    data_root = check_dataset()

    banner("Installing missing dependencies")
    install_missing_deps()

    banner("Fetching repository")
    clone_repo()

    banner("Patching config with discovered dataset path")
    runtime_config = patch_config_for_runtime(data_root)

    rc = run_training(runtime_config)
    copy_outputs_to_kaggle_root()
    return rc


if __name__ == "__main__":
    sys.exit(main())
