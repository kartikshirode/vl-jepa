"""
Vision Encoder using ViT-Tiny for VL-JEPA
Optimized for Jetson Orin Nano with gradient checkpointing
"""

import torch
import torch.nn as nn
import timm
from typing import Optional, Tuple


class VisionEncoder(nn.Module):
    """
    Vision Transformer (ViT-Tiny) encoder for image processing.
    
    Args:
        model_name: Name of the ViT model (default: vit_tiny_patch16_224)
        pretrained: Whether to load ImageNet pretrained weights
        hidden_dim: Hidden dimension size (192 for ViT-Tiny)
        img_size: Input image size
        patch_size: Size of image patches
        num_classes: Number of output classes (0 for feature extraction)
        gradient_checkpointing: Enable gradient checkpointing to save memory
    """
    
    def __init__(
        self,
        model_name: str = "vit_tiny_patch16_224",
        pretrained: bool = True,
        hidden_dim: int = 192,
        img_size: int = 224,
        patch_size: int = 16,
        num_classes: int = 0,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Create ViT model from timm
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,  # Remove classification head
            img_size=img_size,
        )

        # Enable gradient checkpointing for memory efficiency
        if gradient_checkpointing and hasattr(self.model, 'set_grad_checkpointing'):
            self.model.set_grad_checkpointing(enable=True)

        # Get embedding dimension from model
        self.embed_dim = self.model.embed_dim
        # Number of prefix tokens (CLS, dist) varies by ViT variant; timm exposes it.
        self.num_prefix_tokens = getattr(self.model, 'num_prefix_tokens', 1)
        
    def forward(
        self, 
        x: torch.Tensor,
        return_all_tokens: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass through vision encoder.
        
        Args:
            x: Input images [B, 3, H, W]
            return_all_tokens: If True, return all patch tokens. If False, return only CLS token.
            
        Returns:
            Encoded features [B, N, D] where N is number of patches (or 1 if return_all_tokens=False)
        """
        # Forward through ViT
        if return_all_tokens:
            # Get all patch embeddings (including CLS token)
            features = self.model.forward_features(x)
            return features  # [B, N+1, D] where N = num_patches
        else:
            # Get only global features (CLS token)
            features = self.model.forward_features(x)
            # Take only CLS token (first token)
            return features[:, 0:1, :]  # [B, 1, D]
    
    def forward_context(
        self,
        x: torch.Tensor,
        context_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the encoder over only the visible context patches.

        This is the I-JEPA context path: patchify, add positional embedding,
        gather the visible patches by index, then run them through the
        transformer blocks. The context encoder never sees the target patches,
        so the predictor can't collapse to the identity solution.

        Args:
            x: Images [B, 3, H, W]
            context_indices: LongTensor [B, N_ctx] with indices into the
                14x14 patch grid (range 0..195 for 224px / 16px patches).

        Returns:
            [B, N_ctx, D] context patch representations after the final norm.
        """
        B = x.shape[0]
        D = self.embed_dim
        N_ctx = context_indices.shape[1]

        # 1. Patch embedding -> [B, 196, D]
        patches = self.model.patch_embed(x)

        # 2. Positional embedding. timm stores pos_embed as either
        # [1, num_patches, D] (no prefix) or [1, num_patches + num_prefix, D].
        # We want the per-patch slice without any prefix token positions.
        pos_embed = self.model.pos_embed
        if pos_embed.shape[1] == patches.shape[1] + self.num_prefix_tokens:
            patch_pos = pos_embed[:, self.num_prefix_tokens:, :]
        else:
            patch_pos = pos_embed
        patches = patches + patch_pos

        if hasattr(self.model, 'pos_drop'):
            patches = self.model.pos_drop(patches)

        # 3. Gather context patches by index.
        idx = context_indices.unsqueeze(-1).expand(B, N_ctx, D)
        context_tokens = torch.gather(patches, dim=1, index=idx)

        # 4. Some timm variants apply a pre-norm before blocks; keep it if present.
        if hasattr(self.model, 'norm_pre'):
            context_tokens = self.model.norm_pre(context_tokens)

        # 5. Run transformer blocks (no CLS prepended).
        context_tokens = self.model.blocks(context_tokens)

        # 6. Final norm.
        context_tokens = self.model.norm(context_tokens)

        return context_tokens

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: int = 1,
    ) -> list:
        """
        Get intermediate layer outputs for analysis/visualization.
        
        Args:
            x: Input images [B, 3, H, W]
            n: Number of last layers to return
            
        Returns:
            List of intermediate features
        """
        if hasattr(self.model, 'get_intermediate_layers'):
            return self.model.get_intermediate_layers(x, n)
        else:
            # Fallback: return the final patch tokens. The default forward
            # would collapse to a CLS-only tensor when return_all_tokens=False,
            # which is not what an intermediate-layers consumer expects.
            return [self.forward(x, return_all_tokens=True)]
    
    def get_num_patches(self) -> int:
        """Get number of patches per image"""
        return self.num_patches
    
    def get_patch_size(self) -> int:
        """Get patch size"""
        return self.patch_size


class VisionEncoderWithProjection(nn.Module):
    """
    Vision encoder with an additional projection head.
    Useful for matching dimensions with text encoder.
    """
    
    def __init__(
        self,
        model_name: str = "vit_tiny_patch16_224",
        pretrained: bool = True,
        hidden_dim: int = 192,
        projection_dim: int = 256,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        
        # Vision encoder
        self.encoder = VisionEncoder(
            model_name=model_name,
            pretrained=pretrained,
            hidden_dim=hidden_dim,
            gradient_checkpointing=gradient_checkpointing,
        )
        
        # Projection head
        self.projection = nn.Sequential(
            nn.LayerNorm(self.encoder.embed_dim),
            nn.Linear(self.encoder.embed_dim, projection_dim),
        )
        
        self.projection_dim = projection_dim
        self.embed_dim = self.encoder.embed_dim  # Expose embed_dim for parent model
    
    def forward(
        self,
        x: torch.Tensor,
        return_all_tokens: bool = True,
        return_projected: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass with optional projection.
        
        Args:
            x: Input images [B, 3, H, W]
            return_all_tokens: Return all patch tokens or just CLS
            return_projected: Apply projection head
            
        Returns:
            Encoded and optionally projected features
        """
        # Encode
        features = self.encoder(x, return_all_tokens=return_all_tokens)
        
        # Project if requested
        if return_projected:
            features = self.projection(features)
        
        return features


def create_vision_encoder(config: dict) -> nn.Module:
    """
    Factory function to create vision encoder from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        VisionEncoder or VisionEncoderWithProjection
    """
    vision_config = config.get('vision_encoder', {})
    
    model_name = vision_config.get('type', 'vit_tiny_patch16_224')
    pretrained = vision_config.get('pretrained', True)
    hidden_dim = vision_config.get('hidden_dim', 192)
    gradient_checkpointing = vision_config.get('gradient_checkpointing', True)
    
    # Check if we need projection
    if 'projection_dim' in config:
        return VisionEncoderWithProjection(
            model_name=model_name,
            pretrained=pretrained,
            hidden_dim=hidden_dim,
            projection_dim=config['projection_dim'],
            gradient_checkpointing=gradient_checkpointing,
        )
    else:
        return VisionEncoder(
            model_name=model_name,
            pretrained=pretrained,
            hidden_dim=hidden_dim,
            gradient_checkpointing=gradient_checkpointing,
        )


if __name__ == "__main__":
    # Test vision encoder
    print("Testing VisionEncoder...")
    
    model = VisionEncoder(
        model_name="vit_tiny_patch16_224",
        pretrained=False,  # For testing without download
        gradient_checkpointing=True,
    )
    
    # Test input
    x = torch.randn(2, 3, 224, 224)
    
    # Forward pass
    with torch.no_grad():
        output = model(x, return_all_tokens=True)
        print(f"Output shape (all tokens): {output.shape}")  # [2, 197, 192]
        
        output_cls = model(x, return_all_tokens=False)
        print(f"Output shape (CLS only): {output_cls.shape}")  # [2, 1, 192]
    
    print(f"Number of patches: {model.get_num_patches()}")
    print(f"Patch size: {model.get_patch_size()}")
    print(f"Embedding dimension: {model.embed_dim}")
    
    # Test with projection
    print("\nTesting VisionEncoderWithProjection...")
    model_proj = VisionEncoderWithProjection(
        model_name="vit_tiny_patch16_224",
        pretrained=False,
        projection_dim=256,
    )
    
    with torch.no_grad():
        output_proj = model_proj(x)
        print(f"Projected output shape: {output_proj.shape}")  # [2, 197, 256]
    
    print("\nVision encoder test passed!")
