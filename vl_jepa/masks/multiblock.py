"""
Multi-block masking strategy for JEPA.

Follows the I-JEPA recipe:
  - Sample one rectangular context block at `context_scale`.
  - Sample several rectangular target blocks at `target_scale`.
  - Remove any patch that landed in a target from the context, so the context
    encoder genuinely never sees what it's being asked to predict.

The generator returns both boolean masks (for visualization) and 1D index
tensors with FIXED N_ctx and N_tgt across the batch (padded by random
sampling from the eligible pool when blocks come out short). Fixed counts
make collation easy: no padding logic in the model.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import random


@dataclass
class MaskSample:
    """One sample's worth of masks plus their flattened indices."""
    context_mask: torch.Tensor   # bool [num_patches]
    target_mask: torch.Tensor    # bool [num_patches]
    context_indices: torch.Tensor  # long [N_ctx]
    target_indices: torch.Tensor   # long [N_tgt]


class MultiBlockMaskGenerator:
    """Multi-block I-JEPA mask sampler with fixed output sizes."""

    def __init__(
        self,
        input_size: int = 224,
        patch_size: int = 16,
        num_context_blocks: int = 1,
        num_target_blocks: int = 4,
        context_scale: Tuple[float, float] = (0.85, 1.0),
        target_scale: Tuple[float, float] = (0.15, 0.2),
        context_aspect_ratio: Tuple[float, float] = (1.0, 1.0),
        target_aspect_ratio: Tuple[float, float] = (0.75, 1.5),
        allow_overlap: bool = False,
        min_keep: int = 10,
        max_tries: int = 20,
    ):
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_patches = (input_size // patch_size)
        self.total_patches = self.num_patches ** 2

        self.num_context_blocks = num_context_blocks
        self.num_target_blocks = num_target_blocks
        self.context_scale = context_scale
        self.target_scale = target_scale
        self.context_aspect_ratio = context_aspect_ratio
        self.target_aspect_ratio = target_aspect_ratio
        self.allow_overlap = allow_overlap
        self.min_keep = min_keep
        self.max_tries = max_tries

        # Fix per-sample sizes for batchability. Use the midpoint of each
        # scale range times the patch count, with a floor so we always have
        # something to predict.
        mean_ctx_scale = 0.5 * (context_scale[0] + context_scale[1])
        mean_tgt_scale = 0.5 * (target_scale[0] + target_scale[1])
        # Total target coverage across all blocks (before deduping overlap).
        target_total = mean_tgt_scale * num_target_blocks
        # Context block area, with the targets carved out.
        ctx_after_carve = max(mean_ctx_scale - target_total, 0.2)

        self.n_target_fixed = max(int(round(target_total * self.total_patches)), 1)
        self.n_context_fixed = max(int(round(ctx_after_carve * self.total_patches)), min_keep)

        # Cap to physical limits.
        self.n_target_fixed = min(self.n_target_fixed, self.total_patches - min_keep)
        self.n_context_fixed = min(self.n_context_fixed, self.total_patches - self.n_target_fixed)

    # ----- block geometry helpers -----

    def _sample_block_size(
        self,
        scale: Tuple[float, float],
        aspect_ratio: Tuple[float, float],
    ) -> Tuple[int, int]:
        _scale = random.uniform(scale[0], scale[1])
        _ar = random.uniform(aspect_ratio[0], aspect_ratio[1])
        area = max(int(_scale * self.total_patches), 1)
        h = int(round(np.sqrt(area / _ar)))
        w = int(round(np.sqrt(area * _ar)))
        h = min(max(h, 1), self.num_patches)
        w = min(max(w, 1), self.num_patches)
        return h, w

    def _sample_block_position(
        self,
        block_h: int,
        block_w: int,
        occupied: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[int, int]]:
        max_top = self.num_patches - block_h
        max_left = self.num_patches - block_w
        if max_top < 0 or max_left < 0:
            return None
        for _ in range(100):
            top = random.randint(0, max_top)
            left = random.randint(0, max_left)
            if occupied is not None and not self.allow_overlap:
                if occupied[top:top + block_h, left:left + block_w].any():
                    continue
            return top, left
        return None

    def _try_sample(self) -> Optional[MaskSample]:
        """One attempt. Returns None if it can't satisfy constraints."""
        # Sample target blocks first.
        target_grid = np.zeros((self.num_patches, self.num_patches), dtype=bool)
        for _ in range(self.num_target_blocks):
            h, w = self._sample_block_size(self.target_scale, self.target_aspect_ratio)
            pos = self._sample_block_position(h, w, target_grid if not self.allow_overlap else None)
            if pos is None:
                continue
            top, left = pos
            target_grid[top:top + h, left:left + w] = True

        # Sample one context block at context_scale (per I-JEPA), then carve out targets.
        context_grid = np.zeros((self.num_patches, self.num_patches), dtype=bool)
        for _ in range(self.num_context_blocks):
            h, w = self._sample_block_size(self.context_scale, self.context_aspect_ratio)
            pos = self._sample_block_position(h, w, None)  # context blocks can overlap each other
            if pos is None:
                continue
            top, left = pos
            context_grid[top:top + h, left:left + w] = True

        # Carve targets out of context (this is the core JEPA invariant).
        context_grid = context_grid & ~target_grid

        ctx_flat = context_grid.flatten()
        tgt_flat = target_grid.flatten()

        n_ctx_avail = int(ctx_flat.sum())
        n_tgt_avail = int(tgt_flat.sum())

        if n_tgt_avail == 0 or n_ctx_avail < self.min_keep:
            return None

        # Pad / trim to fixed sizes.
        target_indices = np.where(tgt_flat)[0]
        context_indices = np.where(ctx_flat)[0]

        # Trim or expand target indices.
        if len(target_indices) >= self.n_target_fixed:
            target_indices = np.random.choice(target_indices, size=self.n_target_fixed, replace=False)
        else:
            # Pad from patches that are neither in current context nor target.
            outside = np.where(~ctx_flat & ~tgt_flat)[0]
            need = self.n_target_fixed - len(target_indices)
            if len(outside) < need:
                # Last-ditch: take from context too, but only if we don't break min_keep.
                spare = len(context_indices) - self.min_keep
                if spare < need - len(outside):
                    return None
                extra_from_ctx = np.random.choice(context_indices, size=need - len(outside), replace=False)
                target_indices = np.concatenate([target_indices, outside, extra_from_ctx])
                context_indices = np.setdiff1d(context_indices, extra_from_ctx, assume_unique=False)
            else:
                pick = np.random.choice(outside, size=need, replace=False)
                target_indices = np.concatenate([target_indices, pick])

        # Trim or expand context indices.
        if len(context_indices) >= self.n_context_fixed:
            context_indices = np.random.choice(context_indices, size=self.n_context_fixed, replace=False)
        else:
            outside = np.setdiff1d(
                np.arange(self.total_patches),
                np.concatenate([context_indices, target_indices]),
                assume_unique=False,
            )
            need = self.n_context_fixed - len(context_indices)
            if len(outside) < need:
                return None
            pick = np.random.choice(outside, size=need, replace=False)
            context_indices = np.concatenate([context_indices, pick])

        # Rebuild canonical boolean masks from the final index sets.
        final_ctx_mask = np.zeros(self.total_patches, dtype=bool)
        final_tgt_mask = np.zeros(self.total_patches, dtype=bool)
        final_ctx_mask[context_indices] = True
        final_tgt_mask[target_indices] = True

        return MaskSample(
            context_mask=torch.from_numpy(final_ctx_mask),
            target_mask=torch.from_numpy(final_tgt_mask),
            context_indices=torch.from_numpy(np.sort(context_indices)).long(),
            target_indices=torch.from_numpy(np.sort(target_indices)).long(),
        )

    def sample(self) -> MaskSample:
        """Sample a valid mask, retrying up to `max_tries` times."""
        for _ in range(self.max_tries):
            out = self._try_sample()
            if out is not None:
                return out
        raise RuntimeError(
            f"MultiBlockMaskGenerator could not satisfy constraints after "
            f"{self.max_tries} tries. Check context_scale / target_scale / "
            f"num_target_blocks / min_keep settings."
        )

    def __call__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backwards-friendly call: returns 4-tuple of (ctx_mask, tgt_mask, ctx_idx, tgt_idx)."""
        s = self.sample()
        return s.context_mask, s.target_mask, s.context_indices, s.target_indices

    def visualize_masks(
        self,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> np.ndarray:
        """Render a [H, W] int array (0 masked, 1 context, 2 target, 3 overlap)."""
        H = W = self.num_patches
        vis = np.zeros((H, W), dtype=np.int32)
        context_2d = context_mask.reshape(H, W).numpy()
        target_2d = target_mask.reshape(H, W).numpy()
        vis[context_2d] = 1
        vis[target_2d] = 2
        vis[context_2d & target_2d] = 3
        return vis


def create_mask_generator(config: dict) -> MultiBlockMaskGenerator:
    """Build a MultiBlockMaskGenerator from a config dict."""
    mask_config = config.get('masking', {})
    vision_config = config.get('vision_encoder', {})

    input_size = vision_config.get('image_size', 224)
    patch_size = vision_config.get('patch_size', 16)

    return MultiBlockMaskGenerator(
        input_size=input_size,
        patch_size=patch_size,
        num_context_blocks=mask_config.get('num_context_blocks', 1),
        num_target_blocks=mask_config.get('num_target_blocks', 4),
        context_scale=tuple(mask_config.get('context_scale', [0.85, 1.0])),
        target_scale=tuple(mask_config.get('target_scale', [0.15, 0.2])),
        context_aspect_ratio=tuple(mask_config.get('context_aspect_ratio', [1.0, 1.0])),
        target_aspect_ratio=tuple(mask_config.get('target_aspect_ratio', [0.75, 1.5])),
        allow_overlap=mask_config.get('allow_overlap', False),
        min_keep=mask_config.get('min_keep', 10),
    )


if __name__ == "__main__":
    print("Testing MultiBlockMaskGenerator...")
    mask_gen = MultiBlockMaskGenerator(
        input_size=224,
        patch_size=16,
        num_context_blocks=1,
        num_target_blocks=4,
        context_scale=(0.85, 1.0),
        target_scale=(0.15, 0.2),
        allow_overlap=False,
    )
    ctx_mask, tgt_mask, ctx_idx, tgt_idx = mask_gen()
    print(f"Context mask shape: {ctx_mask.shape}")
    print(f"Target mask shape: {tgt_mask.shape}")
    print(f"Context patches: {ctx_mask.sum().item()} / {mask_gen.total_patches}")
    print(f"Target patches: {tgt_mask.sum().item()} / {mask_gen.total_patches}")
    print(f"Disjoint (no overlap): {(ctx_mask & tgt_mask).sum().item() == 0}")
    print(f"Context indices shape: {ctx_idx.shape}")
    print(f"Target indices shape: {tgt_idx.shape}")
    print(f"Fixed N_ctx = {mask_gen.n_context_fixed}, N_tgt = {mask_gen.n_target_fixed}")

    # Same sizes across a batch?
    print("\nBatch consistency check...")
    sizes = set()
    for _ in range(16):
        _, _, ci, ti = mask_gen()
        sizes.add((ci.shape[0], ti.shape[0]))
    print(f"Distinct (N_ctx, N_tgt) shapes across 16 samples: {sizes}")
    assert len(sizes) == 1, "Mask sizes drifted across samples; collation will break."
    print("Mask generator test passed!")
