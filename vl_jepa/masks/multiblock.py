"""
Multi-block masking strategy for JEPA
Based on I-JEPA paper masking approach
"""

import torch
import numpy as np
from typing import Tuple, List, Optional
import random


class MultiBlockMaskGenerator:
    """
    Generate context and target block masks for JEPA training.
    
    Context blocks: Large blocks of visible patches (what the model sees)
    Target blocks: Smaller blocks to predict (what the model predicts)
    
    Args:
        input_size: Input image size (e.g., 224)
        patch_size: Size of each patch (e.g., 16)
        num_context_blocks: Number of context blocks (typically 1)
        num_target_blocks: Number of target blocks (typically 4)
        context_scale: Scale range for context blocks (e.g., [0.85, 1.0])
        target_scale: Scale range for target blocks (e.g., [0.15, 0.2])
        context_aspect_ratio: Aspect ratio range for context blocks
        target_aspect_ratio: Aspect ratio range for target blocks
        allow_overlap: Whether target blocks can overlap with context
        min_keep: Minimum number of patches to keep visible
    """
    
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
    
    def _sample_block_size(
        self,
        scale: Tuple[float, float],
        aspect_ratio: Tuple[float, float],
    ) -> Tuple[int, int]:
        """
        Sample block size (height, width) in terms of patches.
        
        Args:
            scale: Scale range [min, max] relative to image
            aspect_ratio: Aspect ratio range [min, max]
            
        Returns:
            (height, width) in number of patches
        """
        # Sample scale and aspect ratio
        _scale = random.uniform(scale[0], scale[1])
        _aspect_ratio = random.uniform(aspect_ratio[0], aspect_ratio[1])
        
        # Calculate area in patches
        area = int(_scale * self.total_patches)
        
        # Calculate height and width maintaining aspect ratio
        h = int(round(np.sqrt(area / _aspect_ratio)))
        w = int(round(np.sqrt(area * _aspect_ratio)))
        
        # Clip to valid range
        h = min(max(h, 1), self.num_patches)
        w = min(max(w, 1), self.num_patches)
        
        return h, w
    
    def _sample_block_position(
        self,
        block_h: int,
        block_w: int,
        occupied_mask: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[int, int]]:
        """
        Sample top-left position for a block.
        
        Args:
            block_h: Block height in patches
            block_w: Block width in patches
            occupied_mask: Mask of already occupied patches [H, W]
            
        Returns:
            (top, left) position or None if no valid position
        """
        # Valid range for top-left corner
        max_top = self.num_patches - block_h
        max_left = self.num_patches - block_w
        
        if max_top < 0 or max_left < 0:
            return None
        
        # Try to find valid position
        max_attempts = 100
        for _ in range(max_attempts):
            top = random.randint(0, max_top)
            left = random.randint(0, max_left)
            
            # Check if position overlaps with occupied regions
            if occupied_mask is not None and not self.allow_overlap:
                block_region = occupied_mask[top:top+block_h, left:left+block_w]
                if block_region.any():
                    continue  # Overlap detected, try again
            
            return top, left
        
        # Could not find valid position
        return None
    
    def _create_block_mask(
        self,
        blocks: List[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        """
        Create binary mask from block definitions.
        
        Args:
            blocks: List of (top, left, height, width) tuples
            
        Returns:
            Binary mask [H, W] where 1 = visible/target, 0 = masked
        """
        mask = np.zeros((self.num_patches, self.num_patches), dtype=bool)
        
        for top, left, h, w in blocks:
            mask[top:top+h, left:left+w] = True
        
        return mask
    
    def __call__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate context and target masks.
        
        Following I-JEPA/VL-JEPA strategy:
        - First generate target blocks (what to predict)
        - Context is everything EXCEPT targets (what model sees)
        
        Returns:
            context_mask: Boolean mask for context (visible) patches [N] - True = visible
            target_mask: Boolean mask for target (predict) patches [N] - True = predict
        """
        # Storage for blocks
        target_blocks = []
        
        # Generate target blocks FIRST (what we want to predict)
        occupied_mask = np.zeros((self.num_patches, self.num_patches), dtype=bool)
        
        for _ in range(self.num_target_blocks):
            h, w = self._sample_block_size(self.target_scale, self.target_aspect_ratio)
            pos = self._sample_block_position(h, w, occupied_mask if not self.allow_overlap else None)
            
            if pos is not None:
                top, left = pos
                target_blocks.append((top, left, h, w))
                occupied_mask[top:top+h, left:left+w] = True
        
        # Create target mask
        target_mask = self._create_block_mask(target_blocks)
        
        # Context mask is the INVERSE of target mask (visible = not masked)
        # In JEPA: context sees everything except targets
        context_mask = ~target_mask
        
        # Flatten to 1D
        context_mask = context_mask.flatten()  # [H*W]
        target_mask = target_mask.flatten()  # [H*W]
        
        # Ensure minimum patches are visible in context
        if context_mask.sum() < self.min_keep:
            # Randomly select additional patches from targets to show
            n_additional = self.min_keep - context_mask.sum()
            target_indices = np.where(target_mask)[0]
            if len(target_indices) > 0:
                to_show = np.random.choice(target_indices, size=min(n_additional, len(target_indices)), replace=False)
                context_mask[to_show] = True
                target_mask[to_show] = False  # Remove from targets
        
        # Convert to torch tensors
        context_mask = torch.from_numpy(context_mask)
        target_mask = torch.from_numpy(target_mask)
        
        return context_mask, target_mask
    
    def visualize_masks(
        self,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> np.ndarray:
        """
        Create visualization of masks.
        
        Args:
            context_mask: Context mask [N]
            target_mask: Target mask [N]
            
        Returns:
            Visualization array [H, W] where:
                0 = masked (neither context nor target)
                1 = context (visible)
                2 = target (to predict)
                3 = overlap (both context and target)
        """
        H = W = self.num_patches
        
        vis = np.zeros((H, W), dtype=np.int32)
        
        context_2d = context_mask.reshape(H, W).numpy()
        target_2d = target_mask.reshape(H, W).numpy()
        
        vis[context_2d] = 1
        vis[target_2d] = 2
        vis[context_2d & target_2d] = 3
        
        return vis


def create_mask_generator(config: dict) -> MultiBlockMaskGenerator:
    """
    Factory function to create mask generator from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        MultiBlockMaskGenerator instance
    """
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
    
    # Create mask generator
    mask_gen = MultiBlockMaskGenerator(
        input_size=224,
        patch_size=16,
        num_context_blocks=1,
        num_target_blocks=4,
        context_scale=(0.85, 1.0),
        target_scale=(0.15, 0.2),
        allow_overlap=False,
    )
    
    # Generate masks
    context_mask, target_mask = mask_gen()
    
    print(f"Context mask shape: {context_mask.shape}")
    print(f"Target mask shape: {target_mask.shape}")
    print(f"Context patches: {context_mask.sum().item()} / {mask_gen.total_patches}")
    print(f"Target patches: {target_mask.sum().item()} / {mask_gen.total_patches}")
    print(f"Overlap patches: {(context_mask & target_mask).sum().item()}")
    
    # Visualize
    vis = mask_gen.visualize_masks(context_mask, target_mask)
    print(f"Visualization shape: {vis.shape}")
    print(f"Unique values: {np.unique(vis)}")
    
    # Test batch generation
    print("\nTesting batch generation...")
    batch_size = 4
    context_masks = []
    target_masks = []
    
    for _ in range(batch_size):
        ctx, tgt = mask_gen()
        context_masks.append(ctx)
        target_masks.append(tgt)
    
    context_batch = torch.stack(context_masks)
    target_batch = torch.stack(target_masks)
    
    print(f"Context batch shape: {context_batch.shape}")
    print(f"Target batch shape: {target_batch.shape}")
    
    print("\nMask generator test passed!")
