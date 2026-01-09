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
    Text encoder using DistilBERT for processing text captions.
    
    Args:
        model_name: HuggingFace model name (default: distilbert-base-uncased)
        hidden_dim: Hidden dimension of the model
        projection_dim: Dimension to project text features to (optional)
        max_length: Maximum sequence length
        gradient_checkpointing: Enable gradient checkpointing
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        hidden_dim: int = 768,
        projection_dim: Optional[int] = None,
        max_length: int = 128,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.projection_dim = projection_dim
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model
        config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, config=config)
        
        # Enable gradient checkpointing for memory efficiency
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Get actual hidden size from model
        self.embed_dim = self.model.config.hidden_size
        
        # Optional projection layer
        if projection_dim is not None:
            self.projection = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, projection_dim),
            )
        else:
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
        projection_dim: Optional[int] = None,
        max_length: int = 128,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        
        self.encoder = TextEncoder(
            model_name=model_name,
            projection_dim=None,  # We'll add projection after pooling
            max_length=max_length,
            gradient_checkpointing=gradient_checkpointing,
        )
        
        # Projection after pooling
        if projection_dim is not None:
            self.projection = nn.Sequential(
                nn.LayerNorm(self.encoder.embed_dim),
                nn.Linear(self.encoder.embed_dim, projection_dim),
            )
        else:
            self.projection = None
        
        self.projection_dim = projection_dim
    
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
    """
    Factory function to create text encoder from config.
    
    Args:
        config: Configuration dictionary
        use_pooling: Use mean pooling instead of CLS token
        
    Returns:
        TextEncoder or TextEncoderWithPooling
    """
    text_config = config.get('text_encoder', {})
    
    model_name = text_config.get('type', 'distilbert-base-uncased')
    projection_dim = text_config.get('projection_dim', None)
    max_length = text_config.get('max_length', 128)
    gradient_checkpointing = text_config.get('gradient_checkpointing', True)
    
    if use_pooling:
        return TextEncoderWithPooling(
            model_name=model_name,
            projection_dim=projection_dim,
            max_length=max_length,
            gradient_checkpointing=gradient_checkpointing,
        )
    else:
        return TextEncoder(
            model_name=model_name,
            projection_dim=projection_dim,
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
