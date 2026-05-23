"""
Predictor Network for VL-JEPA
Predicts masked patch representations from context
"""

import torch
import torch.nn as nn
from typing import Optional
import math


class PredictorMLP(nn.Module):
    """
    MLP-based predictor for JEPA.
    Takes context embeddings and predicts target embeddings.
    
    Args:
        input_dim: Input dimension from encoder
        hidden_dim: Hidden dimension of MLP
        output_dim: Output dimension (should match encoder output)
        num_layers: Number of MLP layers
        dropout: Dropout probability
        use_layer_norm: Use layer normalization
    """
    
    def __init__(
        self,
        input_dim: int = 192,
        hidden_dim: int = 256,
        output_dim: int = 192,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Build MLP layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with truncated normal distribution"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through predictor.
        
        Args:
            x: Context features [B, N, D] or [B, D]
            
        Returns:
            Predicted features with same shape as input
        """
        return self.mlp(x)


class PredictorTransformer(nn.Module):
    """
    JEPA predictor: a narrow transformer that takes context patch embeddings
    plus mask tokens at the target spatial positions, and outputs the predicted
    representations at those target positions.

    The positional embedding is sized to the full patch grid and indexed by
    the actual 2D grid location of each token (context or target), so the
    model has a real signal about where to predict.

    Args:
        input_dim: Input dim from the context encoder.
        hidden_dim: Predictor working width.
        output_dim: Output dim (matches target encoder output).
        num_layers: Transformer depth.
        num_heads: Attention heads.
        mlp_ratio: FFN ratio.
        dropout: Dropout.
        num_patches: Total number of patch positions in the grid (e.g. 196
            for 224 / 16). Sized at construction; no hidden 256-token cap.
    """

    def __init__(
        self,
        input_dim: int = 192,
        hidden_dim: int = 384,
        output_dim: int = 192,
        num_layers: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_patches: int = 196,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_patches = num_patches

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Positional embedding over the full patch grid, indexed by position id.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        # Learned mask token, same value for every target slot before pos add.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def _gather_pos(self, indices: torch.Tensor) -> torch.Tensor:
        """Gather positional embedding rows at the given grid indices."""
        B, N = indices.shape
        D = self.hidden_dim
        pos = self.pos_embed.expand(B, -1, -1)  # [B, num_patches, D]
        idx = indices.unsqueeze(-1).expand(B, N, D)
        return torch.gather(pos, dim=1, index=idx)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            context_tokens: [B, N_ctx, input_dim] from the context encoder.
            context_indices: [B, N_ctx] grid positions of those tokens.
            target_indices:  [B, N_tgt] grid positions to predict.

        Returns:
            [B, N_tgt, output_dim] predictions at the target positions.
        """
        B = context_tokens.shape[0]
        N_ctx = context_indices.shape[1]
        N_tgt = target_indices.shape[1]

        ctx = self.input_proj(context_tokens) + self._gather_pos(context_indices)
        mask = self.mask_token.expand(B, N_tgt, -1) + self._gather_pos(target_indices)

        x = torch.cat([ctx, mask], dim=1)
        x = self.transformer(x)
        x = self.norm(x)
        # Only the predictions at the target slots matter.
        return self.output_proj(x[:, N_ctx:, :])


class PredictorWithCrossAttention(nn.Module):
    """
    Predictor with cross-attention between vision and text.
    For multimodal prediction tasks.
    
    Args:
        vision_dim: Vision feature dimension
        text_dim: Text feature dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        num_layers: Number of layers
        num_heads: Number of attention heads
    """
    
    def __init__(
        self,
        vision_dim: int = 192,
        text_dim: int = 768,
        hidden_dim: int = 384,
        output_dim: int = 192,
        num_layers: int = 4,
        num_heads: int = 6,
    ):
        super().__init__()
        
        # Project vision and text to same dimension
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward with cross-attention between vision and text.
        
        Args:
            vision_features: Vision features [B, N_v, D_v]
            text_features: Text features [B, N_t, D_t]
            
        Returns:
            Fused predictions [B, N_v, D_out]
        """
        # Project to hidden dimension
        v = self.vision_proj(vision_features)  # [B, N_v, hidden_dim]
        t = self.text_proj(text_features)  # [B, N_t, hidden_dim]
        
        # Cross-attention: vision queries, text keys/values
        x = v
        for attn_layer, norm_layer in zip(self.cross_attention_layers, self.layer_norms):
            # Cross attention
            attn_out, _ = attn_layer(
                query=x,
                key=t,
                value=t,
            )
            # Residual + norm
            x = norm_layer(x + attn_out)
        
        # Output projection
        out = self.output_proj(x)
        
        return out


def create_predictor(config: dict, predictor_type: str = "transformer") -> nn.Module:
    """
    Factory function to create a predictor from config.

    Args:
        config: Full model config dict (expects 'predictor' and 'vision_encoder' keys).
        predictor_type: One of 'transformer' (default), 'mlp', or 'cross_attention'.
    """
    predictor_config = config.get('predictor', {})
    vision_config = config.get('vision_encoder', {})

    input_dim = predictor_config.get('input_dim', 192)
    hidden_dim = predictor_config.get('hidden_dim', 384)
    output_dim = predictor_config.get('output_dim', 192)
    num_layers = predictor_config.get('num_layers', 6)
    dropout = predictor_config.get('dropout', 0.0)

    img_size = vision_config.get('image_size', 224)
    patch_size = vision_config.get('patch_size', 16)
    num_patches = (img_size // patch_size) ** 2

    if predictor_type == "transformer":
        return PredictorTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            num_heads=predictor_config.get('num_heads', 6),
            mlp_ratio=predictor_config.get('mlp_ratio', 4.0),
            dropout=dropout,
            num_patches=num_patches,
        )
    elif predictor_type == "mlp":
        # Kept for ablation only. Note: has no positional signal, so the JEPA
        # task collapses to identity if used as the default with a masked
        # context encoder. Don't use this for real training.
        return PredictorMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=max(num_layers, 2),
            dropout=dropout,
        )
    elif predictor_type == "cross_attention":
        return PredictorWithCrossAttention(
            vision_dim=input_dim,
            text_dim=predictor_config.get('text_dim', 768),
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
        )
    else:
        raise ValueError(f"Unknown predictor type: {predictor_type}")


if __name__ == "__main__":
    # Test MLP predictor
    print("Testing PredictorMLP...")
    predictor_mlp = PredictorMLP(
        input_dim=192,
        hidden_dim=256,
        output_dim=192,
        num_layers=3,
    )
    
    x = torch.randn(2, 196, 192)  # [batch, num_patches, dim]
    with torch.no_grad():
        out = predictor_mlp(x)
        print(f"MLP output shape: {out.shape}")  # [2, 196, 192]
    
    # Test Transformer predictor
    print("\nTesting PredictorTransformer...")
    predictor_trans = PredictorTransformer(
        input_dim=192,
        hidden_dim=384,
        output_dim=192,
        num_layers=4,
        num_patches=196,
    )

    context = torch.randn(2, 100, 192)
    ctx_idx = torch.randint(0, 196, (2, 100))
    tgt_idx = torch.randint(0, 196, (2, 50))

    with torch.no_grad():
        out = predictor_trans(context, ctx_idx, tgt_idx)
        print(f"Transformer output shape: {out.shape}")  # [2, 50, 192]
    
    # Test cross-attention predictor
    print("\nTesting PredictorWithCrossAttention...")
    predictor_cross = PredictorWithCrossAttention(
        vision_dim=192,
        text_dim=768,
        hidden_dim=384,
        output_dim=192,
    )
    
    vision_feat = torch.randn(2, 196, 192)
    text_feat = torch.randn(2, 128, 768)
    
    with torch.no_grad():
        out = predictor_cross(vision_feat, text_feat)
        print(f"Cross-attention output shape: {out.shape}")  # [2, 196, 192]
    
    print("\nPredictor tests passed!")
