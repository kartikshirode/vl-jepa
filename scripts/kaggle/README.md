# Kaggle Notebooks — VL-JEPA training ops guide

This directory holds everything needed to run a full VL-JEPA pretraining job
on a free Kaggle Notebook using one of the two T4 GPUs in the "T4 x2"
accelerator slot. The local laptop config
(`config_dgpu.yaml`) is untouched so you can compare runs head-to-head and
debug the Kaggle setup independently.

## What's here (and what's not)

- `train_kaggle.py` — entrypoint the Kaggle kernel runs. Installs missing
  deps, clones the public repo, sanity-checks the GPU and dataset, launches
  `train.py` with `configs/config_kaggle_p100.yaml`. Tracked in git.
- `kernel-metadata.template.json` — placeholder manifest. Copy it to
  `kernel-metadata.json` and fill in your handle. Tracked in git.
- `kernel-metadata.json` — your filled-in manifest. **Gitignored** because it
  carries your Kaggle username and the kernel privacy flag. You keep this
  file local; it never goes to the public repo.
- This README.

The actual training config lives at `configs/config_kaggle_p100.yaml` in the
repo root (not here) so it sits next to `config_dgpu.yaml` for diffing.

## One-time setup

1. **Install the Kaggle CLI** (any Python env will do):

   ```powershell
   pip install --user kaggle
   ```

2. **Generate a Kaggle API token.** Go to <https://www.kaggle.com/settings>,
   scroll to "API", click "Create New API Token". A `kaggle.json` downloads.

3. **Place the token where the CLI looks for it.** On Windows:

   ```powershell
   New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.kaggle" | Out-Null
   Move-Item -Path "$env:USERPROFILE\Downloads\kaggle.json" -Destination "$env:USERPROFILE\.kaggle\kaggle.json" -Force
   ```

4. **Verify auth:**

   ```powershell
   kaggle config view
   kaggle kernels list --mine | Select-Object -First 5
   ```

   If `kernels list` returns without an auth error, you're set.

5. **Create your local manifest** (one-time, file is gitignored):

   ```powershell
   Copy-Item scripts/kaggle/kernel-metadata.template.json scripts/kaggle/kernel-metadata.json
   ```

   Then edit `scripts/kaggle/kernel-metadata.json`:

   - `id`: replace `<KAGGLE_USERNAME>/<KERNEL_SLUG>` with
     `<your-handle>/<a-slug-you-pick>`. The slug is lowercase, hyphen-
     separated, alphanumeric. Example: `yourhandle/vl-jepa-coco-p100`.
   - `title`: any display name you want.
   - `is_private`: defaults to `true`. Leave that way to keep the kernel
     unlisted on Kaggle (still works for you and anyone you share with).

Through the rest of this guide, replace `<KAGGLE_KERNEL_ID>` with the `id`
value you set in the manifest.

## Submitting the kernel

From the repo root:

```powershell
kaggle kernels push -p scripts/kaggle
```

The first push creates the kernel. Subsequent pushes push a new version with
the latest `train_kaggle.py`. Kaggle prints the kernel URL on success.

**On the first push only**, open the kernel URL in your browser and confirm
the GPU accelerator is set to "GPU T4 x2" (not "GPU P100" or "None") under
Settings -> Accelerator. `kernel-metadata.json`'s `enable_gpu: true` enables
a GPU but doesn't pick which one; that's a one-time click. Subsequent pushes
remember the choice.

## Monitoring

```powershell
# Status: queued / running / complete / error
kaggle kernels status <KAGGLE_KERNEL_ID>

# Pull whatever has been written so far (also works while the kernel is running)
kaggle kernels output <KAGGLE_KERNEL_ID> -p .\kaggle_outputs --quiet
```

`kaggle kernels output` downloads everything in `/kaggle/working/` to the
local path. Mid-run it captures whatever's been written so far — checkpoints,
partial logs.

## Multi-session runs (the 12-hour wall)

Kaggle kills any session that exceeds 12 hours. At ~4 it/s on T4 with batch
32, one epoch is ~80 min (train + val), so a full 20-epoch run takes
~27 hours and spans 2-3 sessions.

The training side is fully prepared for resume: per-epoch checkpoints save
to `/kaggle/working/checkpoints/` (kept to the latest 2 by
`keep_last_n_checkpoints: 2`), and `train_kaggle.py` will pick up the
highest-epoch checkpoint from `/kaggle/input/<slug>/checkpoints/` if one
is mounted and pass it to `train.py --resume`.

What requires a manual click: Kaggle's CLI does not allow a kernel to
list itself in `kernel_sources`, so the previous run's output has to be
attached via the web UI between sessions. The flow is:

1. **Session 1**: open the kernel page, click "Save Version" ->
   "Save & Run All (Commit)". No prior output exists, so the script
   starts at epoch 0. Session runs until the 12-hour cap (~9 epochs).
2. **Session 2** (after Session 1 finishes / times out): on the kernel
   page, click "Add Input" (right panel) -> "Notebook Output" tab -> pick
   the version that just finished -> "Add". This mounts that version's
   `/kaggle/working/` at `/kaggle/input/<slug>/`. Then click "Save
   Version" -> "Save & Run All (Commit)". The script auto-detects the
   checkpoint and resumes from where Session 1 left off.
3. **Session 3 (if needed)**: same as Session 2 but pick Session 2's
   version as the input.

Save frequency is every 2 epochs (`save_every: 2`), so each session loses
at most two epochs of work if it ends mid-checkpoint. With 4 it/s and
20 epochs, expect 2-3 sessions and roughly one manual attach click
between each pair.

## Pulling the final outputs

Once `kaggle kernels status` reports `complete`:

```powershell
kaggle kernels output <KAGGLE_KERNEL_ID> -p .\kaggle_outputs
```

You'll get:

- `kaggle_outputs/checkpoints/checkpoint_epoch_*.pth` — per-epoch saves (last 2 kept by `keep_last_n_checkpoints`).
- `kaggle_outputs/checkpoints/best_model.pth` — best by `mean_recall` (higher is better; the val_loss-based selector was replaced after the v15 run, see RESULTS.md).
- `kaggle_outputs/logs/train_*.log` — full training log.

To resume locally from a Kaggle checkpoint:

```powershell
python train.py --config config_dgpu.yaml --resume kaggle_outputs/checkpoints/checkpoint_epoch_N.pth
```

## Expected timings

A single T4 at the configured `batch_size: 32` is roughly 1.5-2.5x the local 4060 Laptop
on this workload. Rough estimates:

| Step | Wall time |
|---|---|
| Cold start (dep install + repo clone + GPU warm-up) | 1-3 min |
| Per training epoch | 20-30 min |
| Per validation epoch | 2-3 min |
| 20 epochs total | **7-11 hours** |

Comfortable margin under the 12-hour session cap. If `kaggle kernels status`
reports the job hit the wall, the last `checkpoint_epoch_N.pth` is fine to
resume from in a second push by editing `training.resume_from` in the
config or passing `--resume` via a manual `train.py` invocation in a
follow-up notebook.

## Comparing against the local laptop run

Both configs use:

- `effective_batch_size = 32` (laptop: 16 x 2 accum; Kaggle: 32 x 1)
- `learning_rate = 3e-4`, `weight_decay = 0.05`, `warmup_ratio ~ 10%`
- Same masking, same predictor, same losses.

So per-epoch loss curves should be near-identical (modulo data-loader
ordering and FP16 numeric noise). The Kaggle run finishes faster wall-clock;
the optimization trajectory is the same.

## Troubleshooting

- **`kaggle kernels push` says the kernel slug already exists** — that's
  fine, it's an update. Pass `--version-notes "..."` to label the new
  version if you want.
- **Kernel ends in `error` status with "GPU not available"** — Kaggle GPU
  pool is throttled; resubmit after a few minutes.
- **Training dies on first batch with OOM** — drop `data.num_workers` to 1
  or reduce `training.batch_size` to 24 in the Kaggle config, commit, push,
  resubmit.
- **`kaggle.json` not found** — check `kaggle config view` reports the
  config dir. On Windows it's `C:\Users\<you>\.kaggle\`. The file must be
  named exactly `kaggle.json`.
- **Output directory empty after run** — `kaggle kernels output` only pulls
  files in `/kaggle/working/`. Anything written to the cloned repo's
  `logs/` is copied up by `train_kaggle.py`'s
  `copy_outputs_to_kaggle_root` step. If that step didn't run (e.g.,
  training errored before the copy), the logs are still inside the repo
  subdirectory of the output bundle.
- **`kernel-metadata.json` missing** — you forgot the one-time setup step
  5. Copy from the template, fill in your handle.
