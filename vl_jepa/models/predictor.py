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


def _get_drop_path_cls():
    """Locate timm's DropPath across timm versions; return None if unavailable."""
    try:
        from timm.layers import DropPath
        return DropPath
    except ImportError:
        try:
            from timm.models.layers import DropPath
            return DropPath
        except ImportError:
            return None


class _PredictorBlock(nn.Module):
    """Pre-norm transformer block with separate DropPath on attn and FFN residuals.

    This is the I-JEPA-faithful shape: each of the two residual branches gets
    its own stochastic-depth gate, so the two paths drop independently.
    nn.TransformerEncoderLayer hides its residuals, so we hand-roll the block.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        ffn_dim = int(hidden_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        DropPath = _get_drop_path_cls()
        if DropPath is not None and drop_path > 0:
            self.drop_path1 = DropPath(drop_path)
            self.drop_path2 = DropPath(drop_path)
        else:
            self.drop_path1 = nn.Identity()
            self.drop_path2 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.drop_path1(self.attn(h, h, h, need_weights=False)[0])
        x = x + self.drop_path2(self.ffn(self.norm2(x)))
        return x


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
        drop_path_rate: Maximum stochastic-depth rate; ramps linearly from 0
            at the first layer to this value at the last (I-JEPA recipe).
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

        # Per-layer DropPath rate, linear ramp 0 -> drop_path_rate.
        if drop_path_rate > 0:
            dpr = torch.linspace(0.0, drop_path_rate, num_layers).tolist()
        else:
            dpr = [0.0] * num_layers

        self.blocks = nn.ModuleList([
            _PredictorBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=rate,
            )
            for rate in dpr
        ])

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
        for block in self.blocks:
            x = block(x)
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
        input_dim=192,
        text_dim=768,
        hidden_dim=384,
        output_dim=192,
        num_patches=196,
    )

    context_feat = torch.randn(2, 100, 192)
    ctx_idx_c = torch.randint(0, 196, (2, 100))
    tgt_idx_c = torch.randint(0, 196, (2, 50))
    text_feat = torch.randn(2, 32, 768)
    text_mask = torch.ones(2, 32)

    with torch.no_grad():
        out = predictor_cross(context_feat, ctx_idx_c, tgt_idx_c,
                              text_features=text_feat,
                              text_attention_mask=text_mask)
        print(f"Cross-attention output shape: {out.shape}")  # [2, 50, 192]

    print("\nPredictor tests passed!")
