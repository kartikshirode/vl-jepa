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
        drop_path_rate: float = 0.0,
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

        # Build encoder layers individually so we can attach per-layer DropPath
        # following the I-JEPA recipe (rate increases linearly with depth).
        try:
            from timm.layers import DropPath  # newer timm
        except ImportError:
            try:
                from timm.models.layers import DropPath  # older timm
            except ImportError:
                DropPath = None

        self.layers = nn.ModuleList()
        self.drop_paths = nn.ModuleList()
        dpr = torch.linspace(0.0, drop_path_rate, num_layers).tolist() if drop_path_rate > 0 else [0.0] * num_layers
        for rate in dpr:
            self.layers.append(nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=int(hidden_dim * mlp_ratio),
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            ))
            if DropPath is not None and rate > 0:
                self.drop_paths.append(DropPath(rate))
            else:
                self.drop_paths.append(nn.Identity())

        # Kept as a no-op alias for the old single transformer attribute so
        # forward() reads naturally.
        self.transformer = None
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
        for layer, dp in zip(self.layers, self.drop_paths):
            # nn.TransformerEncoderLayer doesn't expose its residuals, so we
            # wrap the whole layer in DropPath. Functionally close enough to
            # I-JEPA's per-residual DropPath for our use case (Tiny ViT scale).
            x = x + dp(layer(x) - x) if isinstance(dp, nn.Identity) is False else layer(x)
        x = self.norm(x)
        return self.output_proj(x[:, N_ctx:, :])


class PredictorWithCrossAttention(nn.Module):
    """
    Text-conditioned JEPA predictor.

    Same I-JEPA-style interface as PredictorTransformer (takes context tokens,
    context indices, and target indices; outputs predictions at the target
    positions), but each transformer block alternates self-attention over the
    {context + mask token} sequence with cross-attention into the text tokens.
    The model can therefore use the caption as side-information when
    predicting masked patches, which is what "VL"-JEPA is supposed to mean.

    Args:
        input_dim:  Vision context-encoder output dim.
        text_dim:   Text encoder output dim.
        hidden_dim: Predictor working width.
        output_dim: Prediction dim (matches target encoder output dim).
        num_layers: Self+cross block count.
        num_heads:  Attention heads per block.
        mlp_ratio:  FFN ratio inside each block.
        dropout:    Dropout inside attention and FFN.
        num_patches: Total patch positions in the grid.
    """

    def __init__(
        self,
        input_dim: int = 192,
        text_dim: int = 768,
        hidden_dim: int = 384,
        output_dim: int = 192,
        num_layers: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_patches: int = 196,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.self_attns = nn.ModuleList()
        self.cross_attns = nn.ModuleList()
        self.norm1 = nn.ModuleList()
        self.norm2 = nn.ModuleList()
        self.norm3 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for _ in range(num_layers):
            self.self_attns.append(nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True))
            self.cross_attns.append(nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True))
            self.norm1.append(nn.LayerNorm(hidden_dim))
            self.norm2.append(nn.LayerNorm(hidden_dim))
            self.norm3.append(nn.LayerNorm(hidden_dim))
            self.ffns.append(nn.Sequential(
                nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
                nn.GELU(),
                nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
            ))

        self.norm_out = nn.LayerNorm(hidden_dim)
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
        B, N = indices.shape
        D = self.hidden_dim
        return torch.gather(self.pos_embed.expand(B, -1, -1), 1, indices.unsqueeze(-1).expand(B, N, D))

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
        text_features: torch.Tensor = None,
        text_attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B = context_tokens.shape[0]
        N_ctx = context_indices.shape[1]
        N_tgt = target_indices.shape[1]

        ctx = self.input_proj(context_tokens) + self._gather_pos(context_indices)
        mask = self.mask_token.expand(B, N_tgt, -1) + self._gather_pos(target_indices)
        x = torch.cat([ctx, mask], dim=1)

        if text_features is None:
            # Caller used cross-attention predictor without supplying text:
            # behave like a self-attention-only predictor in that case.
            t = None
            kp_mask = None
        else:
            t = self.text_proj(text_features)
            # PyTorch MHA expects True == "ignore this key".
            kp_mask = (text_attention_mask == 0) if text_attention_mask is not None else None

        for sa, ca, n1, n2, n3, ffn in zip(
            self.self_attns, self.cross_attns, self.norm1, self.norm2, self.norm3, self.ffns,
        ):
            # Self-attention block (pre-norm).
            h = n1(x)
            x = x + sa(h, h, h, need_weights=False)[0]
            # Cross-attention into text, if available.
            if t is not None:
                h = n2(x)
                x = x + ca(h, t, t, key_padding_mask=kp_mask, need_weights=False)[0]
            # FFN.
            x = x + ffn(n3(x))

        x = self.norm_out(x)
        return self.output_proj(x[:, N_ctx:, :])


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
            drop_path_rate=predictor_config.get('drop_path_rate', 0.0),
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
    elif predictor_type in ("cross_attention", "cross_attention_text"):
        return PredictorWithCrossAttention(
            input_dim=input_dim,
            text_dim=predictor_config.get('text_dim', 768),
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            num_heads=predictor_config.get('num_heads', 6),
            mlp_ratio=predictor_config.get('mlp_ratio', 4.0),
            dropout=dropout,
            num_patches=num_patches,
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
