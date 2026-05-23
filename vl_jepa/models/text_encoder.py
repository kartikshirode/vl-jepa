"""
Text Encoder using DistilBERT for VL-JEPA
Optimized for Jetson Orin Nano with gradient checkpointing
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Optional, Dict


class TextEncoder(nn.Module):
    """
    Text encoder using DistilBERT.

    The previous version accepted a `projection_dim` arg that built an
    internal projection head, but the main VL-JEPA training path used
    `self.text_projection` on the parent module instead and called this
    encoder with `return_projected=False`. The internal projection was
    dead weight. It's been removed; VL-JEPA's text_projection is the
    single projection in the trainable graph.

    Args:
        model_name: HuggingFace model name (default: distilbert-base-uncased).
        max_length: Maximum sequence length.
        gradient_checkpointing: Enable gradient checkpointing for memory.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        gradient_checkpointing: bool = True,
        **legacy_kwargs,
    ):
        super().__init__()

        # Friendly migration: accept and ignore `projection_dim` / `hidden_dim`
        # from older configs without raising, but mention what happened once.
        if 'projection_dim' in legacy_kwargs and legacy_kwargs['projection_dim'] is not None:
            import warnings
            warnings.warn(
                "TextEncoder.projection_dim is no longer used. VL-JEPA's "
                "text_projection at the parent level handles dim alignment.",
                DeprecationWarning,
            )

        self.model_name = model_name
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, config=config)

        # Guard against transformers builds that lack gradient_checkpointing_enable.
        if gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

        self.embed_dim = self.model.config.hidden_size
        self.hidden_dim = self.embed_dim  # back-compat alias
        # No internal projection: this module returns raw encoder features.
        self.projection = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_all_tokens: bool = False,
        return_projected: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass through text encoder.
        
        Args:
            input_ids: Token IDs [B, L]
            attention_mask: Attention mask [B, L]
            return_all_tokens: If True, return all token embeddings. If False, return only [CLS] token.
            return_projected: If True and projection exists, apply projection
            
        Returns:
            Encoded text features [B, D] or [B, L, D]
        """
        # Forward through BERT/DistilBERT
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        # Get features
        if return_all_tokens:
            # All token embeddings
            features = outputs.last_hidden_state  # [B, L, D]
        else:
            # CLS token embedding (first token)
            features = outputs.last_hidden_state[:, 0, :]  # [B, D]
        
        # Apply projection if requested and available
        if return_projected and self.projection is not None:
            features = self.projection(features)
        
        return features
    
    def tokenize(
        self,
        texts: list,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize text inputs.
        
        Args:
            texts: List of text strings
            padding: Padding strategy
            truncation: Whether to truncate
            return_tensors: Return format
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        return self.tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=self.max_length,
            return_tensors=return_tensors,
        )


class TextEncoderWithPooling(nn.Module):
    """
    Text encoder with mean pooling instead of CLS token.
    Sometimes gives better results for retrieval tasks.
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.encoder = TextEncoder(
            model_name=model_name,
            max_length=max_length,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.embed_dim = self.encoder.embed_dim
        # No internal projection: VL-JEPA's text_projection runs after this.
        self.projection = None
    
    def mean_pooling(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean pooling with attention mask.
        
        Args:
            token_embeddings: Token embeddings [B, L, D]
            attention_mask: Attention mask [B, L]
            
        Returns:
            Pooled features [B, D]
        """
        # Expand attention mask to match embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum embeddings
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        
        # Divide by the sum of attention mask
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_projected: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass with mean pooling.
        
        Args:
            input_ids: Token IDs [B, L]
            attention_mask: Attention mask [B, L]
            return_projected: Apply projection
            
        Returns:
            Pooled and optionally projected features [B, D]
        """
        # Get all token embeddings
        token_embeddings = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_all_tokens=True,
            return_projected=False,
        )
        
        # Mean pooling
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        pooled = self.mean_pooling(token_embeddings, attention_mask)
        
        # Apply projection
        if return_projected and self.projection is not None:
            pooled = self.projection(pooled)
        
        return pooled
    
    def tokenize(self, texts: list, **kwargs) -> Dict[str, torch.Tensor]:
        """Tokenize using underlying encoder's tokenizer"""
        return self.encoder.tokenize(texts, **kwargs)


def create_text_encoder(config: dict, use_pooling: bool = False) -> nn.Module:
    """Factory function to create a TextEncoder from a model config."""
    text_config = config.get('text_encoder', {})

    model_name = text_config.get('type', 'distilbert-base-uncased')
    max_length = text_config.get('max_length', 128)
    gradient_checkpointing = text_config.get('gradient_checkpointing', True)

    if use_pooling:
        return TextEncoderWithPooling(
            model_name=model_name,
            max_length=max_length,
            gradient_checkpointing=gradient_checkpointing,
        )
    else:
        return TextEncoder(
            model_name=model_name,
            max_length=max_length,
            gradient_checkpointing=gradient_checkpointing,
        )


if __name__ == "__main__":
    # Test text encoder
    print("Testing TextEncoder...")
    
    model = TextEncoder(
        model_name="distilbert-base-uncased",
        projection_dim=256,
        max_length=128,
        gradient_checkpointing=True,
    )
    
    # Test input
    texts = [
        "A dog playing in the park",
        "A cat sitting on a chair",
    ]
    
    # Tokenize
    tokens = model.tokenize(texts)
    print(f"Input IDs shape: {tokens['input_ids'].shape}")
    print(f"Attention mask shape: {tokens['attention_mask'].shape}")
    
    # Forward pass
    with torch.no_grad():
        # CLS token only
        output_cls = model(
            input_ids=tokens['input_ids'],
            attention_mask=tokens['attention_mask'],
            return_all_tokens=False,
            return_projected=True,
        )
        print(f"Output shape (CLS, projected): {output_cls.shape}")  # [2, 256]
        
        # All tokens
        output_all = model(
            input_ids=tokens['input_ids'],
            attention_mask=tokens['attention_mask'],
            return_all_tokens=True,
            return_projected=False,
        )
        print(f"Output shape (all tokens, no projection): {output_all.shape}")  # [2, 128, 768]
    
    # Test mean pooling version
    print("\nTesting TextEncoderWithPooling...")
    model_pool = TextEncoderWithPooling(
        model_name="distilbert-base-uncased",
        projection_dim=256,
    )
    
    with torch.no_grad():
        output_pool = model_pool(
            input_ids=tokens['input_ids'],
            attention_mask=tokens['attention_mask'],
            return_projected=True,
        )
        print(f"Pooled output shape: {output_pool.shape}")  # [2, 256]
    
    print("\nText encoder test passed!")
