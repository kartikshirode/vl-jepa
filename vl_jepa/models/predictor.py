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
    Transformer-based predictor for JEPA.
    More powerful than MLP but uses more memory.
    
    Args:
        input_dim: Input dimension from encoder
        hidden_dim: Hidden dimension of transformer
        output_dim: Output dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        dropout: Dropout probability
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
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, hidden_dim))  # Max 256 tokens
        
        # Mask tokens (learnable parameters for masked positions)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for better stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
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
    
    def forward(
        self,
        context_tokens: torch.Tensor,
        target_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through predictor.
        
        Args:
            context_tokens: Context patch embeddings [B, N_ctx, D]
            target_positions: Indices of target positions to predict [B, N_target]
                            If None, predicts all positions
            
        Returns:
            Predicted target embeddings [B, N_target, D] or [B, N_ctx, D]
        """
        B, N_ctx, D = context_tokens.shape
        
        # Project input
        x = self.input_proj(context_tokens)  # [B, N_ctx, hidden_dim]
        
        # Add mask tokens for target positions if specified
        if target_positions is not None:
            N_target = target_positions.shape[1]
            mask_tokens = self.mask_token.expand(B, N_target, -1)
            # Concatenate context and mask tokens
            x = torch.cat([x, mask_tokens], dim=1)  # [B, N_ctx + N_target, hidden_dim]
            N_total = N_ctx + N_target
        else:
            N_total = N_ctx
        
        # Add positional embeddings
        x = x + self.pos_embed[:, :N_total, :]
        
        # Transformer
        x = self.transformer(x)  # [B, N_total, hidden_dim]
        
        # Extract target predictions
        if target_positions is not None:
            # Get only the mask token predictions
            x = x[:, N_ctx:, :]  # [B, N_target, hidden_dim]
        
        # Output projection
        out = self.output_proj(x)
        
        return out


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


def create_predictor(config: dict, predictor_type: str = "mlp") -> nn.Module:
    """
    Factory function to create predictor from config.
    
    Args:
        config: Configuration dictionary
        predictor_type: Type of predictor ("mlp", "transformer", "cross_attention")
        
    Returns:
        Predictor module
    """
    predictor_config = config.get('predictor', {})
    
    input_dim = predictor_config.get('input_dim', 192)
    hidden_dim = predictor_config.get('hidden_dim', 256)
    output_dim = predictor_config.get('output_dim', 192)
    num_layers = predictor_config.get('num_layers', 3)
    dropout = predictor_config.get('dropout', 0.1)
    
    if predictor_type == "mlp":
        return PredictorMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif predictor_type == "transformer":
        return PredictorTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
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
    )
    
    context = torch.randn(2, 100, 192)
    target_pos = torch.randint(0, 196, (2, 50))
    
    with torch.no_grad():
        out = predictor_trans(context, target_pos)
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
