"""Kaggle Notebooks entrypoint for VL-JEPA training on a single T4.

This script is the `code_file` referenced by `kernel-metadata.json`. It runs
inside the Kaggle kernel runtime and handles environment setup, dependency
install, repo clone, optional checkpoint resume, and the training launch.

Requirements (set in the web UI on first run, then sticky):
  - Accelerator: "GPU T4 x2" (this script only uses the first T4)
  - Internet: enabled
  - Input dataset: awsaf49/coco-2017-dataset

Auto-resume flow (manual one-click attach between sessions):
  - Session 1 (no prior output): no checkpoint mounted, starts at epoch 0.
  - Each session saves checkpoint_epoch_N.pth into /kaggle/working/checkpoints/
    every `save_every` epochs (configured as 2). Old checkpoints are
    pruned to the latest `keep_last_n_checkpoints` (configured as 2) to
    keep /kaggle/working well under its 20 GB cap.
  - Session 2+: before triggering, open the kernel page in a browser and
    click "Add Input" -> "Notebook Output" -> pick the previous version
    of THIS kernel -> Save. That mounts the previous /kaggle/working/ at
    /kaggle/input/<slug>/. (Kaggle's CLI rejects a kernel listing itself
    in kernel_sources, so this step is unavoidably manual.) Then click
    "Save Version" -> "Save & Run All (Commit)".
  - This script's find_resume_checkpoint scans the candidate paths,
    locates the highest-epoch checkpoint, and passes it to train.py via
    --resume. train.py restores model + optimizer + scheduler + EMA +
    epoch counter automatically; no edits needed.

Why T4 and not P100: Kaggle's PyTorch (>= 2.10) dropped Pascal (sm_60), so
a P100 kernel fails to launch CUDA. T4 is Turing (sm_75) and supported.

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
KAGGLE_WORKING = Path("/kaggle/working")
# COCO dataset candidates. Kaggle mounts datasets at either the modern
# /kaggle/input/datasets/<owner>/<slug>/ layout or the legacy /kaggle/input/<slug>/,
# and the awsaf49/coco-2017-dataset packs everything under a coco2017/ subdir.
DATA_ROOT_CANDIDATES = [
    Path("/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017"),
    Path("/kaggle/input/coco-2017-dataset/coco2017"),
    Path("/kaggle/input/coco-2017-dataset"),
    Path("/kaggle/input/coco2017"),
]
# Resume-checkpoint candidates. When this kernel re-runs with itself in
# kernel_sources, the previous version's /kaggle/working/checkpoints/ ends up
# under /kaggle/input/<slug>/checkpoints/. Path layout matches the same dual
# convention as datasets.
CHECKPOINT_CANDIDATES = [
    Path("/kaggle/input/vl-jepa-coco-pretraining-p100/checkpoints"),
    Path("/kaggle/input/datasets/kartikshirode/vl-jepa-coco-pretraining-p100/checkpoints"),
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


def find_resume_checkpoint():
    """Locate the latest checkpoint_epoch_*.pth from a previous kernel run.

    Returns the path to the highest-epoch checkpoint, or None if no previous
    run output is mounted. With kernel_sources self-referencing this kernel,
    Kaggle mounts the most recent COMPLETED run's /kaggle/working/checkpoints/
    under /kaggle/input/.

    First-session runs return None (nothing to resume) and training starts
    fresh. Subsequent sessions return the latest checkpoint and training
    resumes from there. Best-by-val checkpoints (best_model.pth) are
    intentionally NOT used for resume because they may be older than the
    most recent per-epoch save.
    """
    import re

    epoch_pat = re.compile(r"checkpoint_epoch_(\d+)\.pth$")
    best = None
    best_epoch = -1
    for d in CHECKPOINT_CANDIDATES:
        if not d.exists():
            continue
        for p in d.iterdir():
            m = epoch_pat.search(p.name)
            if not m:
                continue
            epoch = int(m.group(1))
            if epoch > best_epoch:
                best_epoch = epoch
                best = p

    if best is not None:
        size_mb = best.stat().st_size / (1024 * 1024)
        print(
            f"Found resume checkpoint at {best} "
            f"(epoch {best_epoch}, {size_mb:.0f} MB)"
        )
    else:
        print("No previous-session checkpoint found; starting fresh.")
        attached = [str(d) for d in CHECKPOINT_CANDIDATES if d.parent.exists()]
        if attached:
            print(f"  (Looked under: {attached})")
    return best


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


def run_training(config_path: Path, resume_from=None) -> int:
    os.chdir(REPO_DIR)
    cmd = [sys.executable, "train.py", "--config", str(config_path)]
    if resume_from is not None:
        cmd.extend(["--resume", str(resume_from)])
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

    banner("Looking for previous-session checkpoint to resume from")
    resume_ckpt = find_resume_checkpoint()

    banner("Installing missing dependencies")
    install_missing_deps()

    banner("Fetching repository")
    clone_repo()

    banner("Patching config with discovered dataset path")
    runtime_config = patch_config_for_runtime(data_root)

    if resume_ckpt is not None:
        banner(f"RESUMING from epoch {resume_ckpt.stem.rsplit('_', 1)[-1]}")
    else:
        banner("STARTING FRESH (epoch 0)")
    rc = run_training(runtime_config, resume_from=resume_ckpt)
    copy_outputs_to_kaggle_root()
    return rc


if __name__ == "__main__":
    sys.exit(main())
