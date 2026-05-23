"""
Main VL-JEPA Model
Combines vision encoder, text encoder, and predictor with EMA target encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import copy

from .vision_encoder import VisionEncoder
from .text_encoder import TextEncoder
from .predictor import PredictorMLP, PredictorWithCrossAttention


class VLJEPAModel(nn.Module):
    """
    Vision-Language Joint Embedding Predictive Architecture (VL-JEPA).
    
    Architecture:
        - Context Encoder: Encodes visible patches and text
        - Target Encoder: EMA copy of context encoder, encodes masked patches
        - Predictor: Predicts target representations from context
    
    Args:
        vision_encoder: Vision encoder (ViT)
        text_encoder: Text encoder (BERT/DistilBERT)
        predictor: Predictor network
        embedding_dim: Dimension of shared embedding space
        ema_momentum: EMA momentum for target encoder (0.996 recommended)
        temperature: Temperature for contrastive loss
    """
    
    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder: nn.Module,
        predictor: nn.Module,
        embedding_dim: int = 256,
        ema_momentum: float = 0.996,
        temperature: float = 0.07,
        contrastive_loss_type: str = 'infonce',
    ):
        super().__init__()
        # Optional contrastive loss variant. 'infonce' is the legacy CLIP-style
        # softmax loss. 'siglip' is the SigLIP sigmoid-per-pair loss from
        # Zhai et al. 2023, which trains better at small effective batch.
        self.contrastive_loss_type = contrastive_loss_type
        
        # Context encoders (trainable)
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        
        # Predictor (trainable)
        self.predictor = predictor
        
        # Target encoders (EMA, not trainable)
        self.target_vision_encoder = copy.deepcopy(vision_encoder)
        self.target_text_encoder = copy.deepcopy(text_encoder)
        
        # Freeze target encoders
        for param in self.target_vision_encoder.parameters():
            param.requires_grad = False
        for param in self.target_text_encoder.parameters():
            param.requires_grad = False
        
        # Projection heads for contrastive learning (optional)
        vision_dim = vision_encoder.embed_dim
        text_dim = text_encoder.embed_dim
        
        self.vision_projection = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, embedding_dim),
        )
        
        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, embedding_dim),
        )
        
        # Parameters
        self.ema_momentum = ema_momentum
        self.temperature = temperature
        self.embedding_dim = embedding_dim

        # SigLIP needs learnable scale and bias (Zhai et al. 2023 defaults).
        if contrastive_loss_type == 'siglip':
            import math
            self.siglip_logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
            self.siglip_logit_bias = nn.Parameter(torch.tensor(-10.0))
        
        # Initialize projections
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights"""
        for m in [self.vision_projection, self.text_projection]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.trunc_normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
    
    def freeze_text_encoder(self):
        """
        Freeze the ENTIRE text encoder for Stage-2 training.
        This includes all layers, LayerNorms, embeddings - everything in DistilBERT.
        
        IMPORTANT: This does NOT freeze projection heads (vision_projection, text_projection).
        Projection heads remain trainable in Stage-2.
        """
        # Freeze ALL parameters in text encoder (including LayerNorms, embeddings, etc.)
        frozen_count = 0
        for name, param in self.text_encoder.named_parameters():
            param.requires_grad = False
            frozen_count += 1
        
        # Also freeze target text encoder (should already be frozen, but ensure it)
        for param in self.target_text_encoder.parameters():
            param.requires_grad = False
        
        # VERIFY: Projection heads should still be trainable
        vision_proj_trainable = all(p.requires_grad for p in self.vision_projection.parameters())
        text_proj_trainable = all(p.requires_grad for p in self.text_projection.parameters())
        
        if not vision_proj_trainable or not text_proj_trainable:
            raise RuntimeError("ERROR: Projection heads were accidentally frozen!")
        
        # Count frozen vs trainable parameters
        frozen_params = sum(p.numel() for p in self.text_encoder.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'frozen_text_params': frozen_params,
            'frozen_layer_count': frozen_count,
            'trainable_params': trainable_params,
            'vision_proj_trainable': vision_proj_trainable,
            'text_proj_trainable': text_proj_trainable,
        }
    
    def unfreeze_text_encoder(self):
        """Unfreeze the text encoder (for Stage-1 or fine-tuning)."""
        for param in self.text_encoder.parameters():
            param.requires_grad = True

    def train(self, mode: bool = True):
        """
        Override train() so the EMA target encoders stay in eval mode.
        requires_grad=False stops gradients, but it does NOT disable dropout
        or stochastic depth. Without this override, drop_path inside the
        target ViT produces noisy targets that hurt the JEPA signal.

        Loops over named_children so any future target_* attribute is caught
        automatically; missing one would silently regress this.
        """
        super().train(mode)
        for name, child in self.named_children():
            if name.startswith('target_'):
                child.eval()
        return self
    
    def get_trainable_parameters(self):
        """
        Get only trainable parameters for optimizer.
        Use this in Stage-2 to avoid passing frozen params to optimizer.
        """
        return [p for p in self.parameters() if p.requires_grad]
    
    def get_parameter_groups(self, base_lr: float, stage2: bool = False):
        """
        Get parameter groups with different learning rates.
        
        Args:
            base_lr: Base learning rate
            stage2: If True, text encoder is frozen and excluded
            
        Returns:
            List of parameter groups for optimizer
        """
        if stage2:
            # Stage-2: Only vision encoder, predictor, and projection heads
            groups = [
                {'params': list(self.vision_encoder.parameters()), 'lr': base_lr},
                {'params': list(self.predictor.parameters()), 'lr': base_lr},
                {'params': list(self.vision_projection.parameters()), 'lr': base_lr},
                {'params': list(self.text_projection.parameters()), 'lr': base_lr},
            ]
        else:
            # Stage-1: All parameters
            groups = [
                {'params': list(self.vision_encoder.parameters()), 'lr': base_lr},
                {'params': list(self.text_encoder.parameters()), 'lr': base_lr},
                {'params': list(self.predictor.parameters()), 'lr': base_lr},
                {'params': list(self.vision_projection.parameters()), 'lr': base_lr},
                {'params': list(self.text_projection.parameters()), 'lr': base_lr},
            ]

        # SigLIP logit scale/bias are nn.Parameters attached to the top-level
        # module, so they don't live inside any of the submodule groups above.
        # Without this they'd receive gradients but never get a weight update.
        if hasattr(self, 'siglip_logit_scale') and hasattr(self, 'siglip_logit_bias'):
            groups.append({
                'params': [self.siglip_logit_scale, self.siglip_logit_bias],
                'lr': base_lr,
                'weight_decay': 0.0,
            })

        return groups
    
    def verify_frozen_text_encoder(self):
        """
        Verify that text encoder is properly frozen.
        Returns True if all text encoder params have requires_grad=False.
        """
        text_params = list(self.text_encoder.parameters())
        frozen_count = sum(1 for p in text_params if not p.requires_grad)
        total_count = len(text_params)
        
        all_frozen = frozen_count == total_count
        return {
            'all_frozen': all_frozen,
            'frozen_count': frozen_count,
            'total_count': total_count,
        }

    @torch.no_grad()
    def update_target_encoder(self):
        """
        EMA update of the target encoders from the online encoders.
        Call after every optimizer step.

        In-place (mul_ / add_) to avoid allocating two new tensors per
        parameter every step.
        """
        m = self.ema_momentum
        for param_q, param_k in zip(
            self.vision_encoder.parameters(),
            self.target_vision_encoder.parameters(),
        ):
            param_k.data.mul_(m).add_(param_q.data, alpha=1.0 - m)
        for param_q, param_k in zip(
            self.text_encoder.parameters(),
            self.target_text_encoder.parameters(),
        ):
            param_k.data.mul_(m).add_(param_q.data, alpha=1.0 - m)
    
    def _gather_target_patches(
        self,
        target_full: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Slice target encoder output to just the target patch positions."""
        num_prefix = getattr(self.vision_encoder, 'num_prefix_tokens', 1)
        patch_tokens = target_full[:, num_prefix:, :]  # [B, num_patches, D]
        B, N_tgt = target_indices.shape
        D = patch_tokens.shape[-1]
        idx = target_indices.unsqueeze(-1).expand(B, N_tgt, D)
        return torch.gather(patch_tokens, dim=1, index=idx)

    def forward_jepa(
        self,
        images: torch.Tensor,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        context_indices: Optional[torch.Tensor] = None,
        target_indices: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        I-JEPA forward pass.

        The context encoder sees ONLY the visible context patches (indexed by
        context_indices). The target encoder runs on the full image and we
        gather just the target patch positions. The predictor takes the
        context tokens and produces predictions at the target positions.

        Args:
            images: [B, 3, H, W]
            text_input_ids / text_attention_mask: unused in pure JEPA mode,
                accepted for signature compatibility.
            context_indices: [B, N_ctx] grid indices of visible patches.
            target_indices: [B, N_tgt] grid indices to predict.
            context_mask / target_mask: optional boolean masks for downstream
                visualization; not used by the loss.

        Returns:
            Dict with predicted_vision, target_vision (both [B, N_tgt, D]).
        """
        if context_indices is None or target_indices is None:
            raise ValueError("forward_jepa requires context_indices and target_indices.")

        # Design note on multi-target handling:
        # The I-JEPA paper samples 4 target blocks per image and runs the
        # predictor once per block (same context, different target). The
        # equivalent batched form is to stack [B, num_blocks, n_per_block]
        # into the batch dim and run one predictor pass. We instead pre-merge
        # all target patches across blocks into a single [B, N_tgt] tensor in
        # the mask generator, which is one predictor pass already and saves
        # the replication of context tokens. The only semantic delta versus
        # paper-faithful per-block calls is that target tokens of different
        # blocks share self-attention; for ViT-Tiny scale this has not shown
        # a measurable downside.
        # Context encoder sees only visible patches.
        context_repr = self.vision_encoder.forward_context(images, context_indices)
        assert context_repr.shape[1] == context_indices.shape[1], (
            f"Context encoder returned {context_repr.shape[1]} tokens for "
            f"{context_indices.shape[1]} context indices."
        )

        # Predict at target positions. The cross-attention predictor wants
        # text features as side-information; everyone else ignores them.
        from .predictor import PredictorWithCrossAttention
        if isinstance(self.predictor, PredictorWithCrossAttention) and text_input_ids is not None:
            text_features = self.text_encoder(
                text_input_ids, text_attention_mask,
                return_all_tokens=True, return_projected=False,
            )
            predicted_vision = self.predictor(
                context_repr, context_indices, target_indices,
                text_features=text_features, text_attention_mask=text_attention_mask,
            )
        else:
            predicted_vision = self.predictor(context_repr, context_indices, target_indices)

        # Target encoder runs on the full image (no grad), then gather targets.
        with torch.no_grad():
            target_full = self.target_vision_encoder(images, return_all_tokens=True)
            target_vision = self._gather_target_patches(target_full, target_indices)

        return {
            'predicted_vision': predicted_vision,
            'target_vision': target_vision.detach(),
            'context_repr': context_repr,
            'context_indices': context_indices,
            'target_indices': target_indices,
            'context_mask': context_mask,
            'target_mask': target_mask,
        }
    
    def forward_contrastive(
        self,
        images: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for contrastive learning (like CLIP).
        
        Args:
            images: Input images [B, 3, H, W]
            text_input_ids: Text token IDs [B, L]
            text_attention_mask: Text attention mask [B, L]
            
        Returns:
            Dictionary with vision and text embeddings
        """
        # Encode vision (CLS token only)
        vision_features = self.vision_encoder(images, return_all_tokens=False)  # [B, 1, D_v]
        vision_features = vision_features.squeeze(1)  # [B, D_v]
        
        # Encode text (CLS token only)
        text_features = self.text_encoder(
            text_input_ids,
            text_attention_mask,
            return_all_tokens=False,
            return_projected=False
        )  # [B, D_t]
        
        # Project to shared embedding space
        vision_embed = self.vision_projection(vision_features)  # [B, embedding_dim]
        text_embed = self.text_projection(text_features)  # [B, embedding_dim]
        
        # Normalize
        vision_embed = F.normalize(vision_embed, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1)
        
        return {
            'vision_embed': vision_embed,
            'text_embed': text_embed,
        }
    
    def compute_jepa_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,  # unused, kept for back-compat
    ) -> torch.Tensor:
        """
        Smooth L1 loss between predicted and target representations.

        Both `predicted` and `target` are already aligned to the target
        positions (shape [B, N_tgt, D]); no CLS slicing, no mask broadcasting.

        Per I-JEPA: layer-norm the target side only, not the predicted side.
        Normalizing both dampens gradients.
        """
        assert predicted.shape == target.shape, (
            f"predicted {tuple(predicted.shape)} vs target {tuple(target.shape)}"
        )
        target = F.layer_norm(target, target.shape[-1:])
        return F.smooth_l1_loss(predicted, target.detach(), reduction='mean')
    
    def compute_contrastive_loss(
        self,
        vision_embed: torch.Tensor,
        text_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Contrastive loss between vision and text embeddings.

        Two variants are selectable via self.contrastive_loss_type:
          - 'infonce' (default): CLIP-style symmetric softmax cross entropy.
          - 'siglip':            sigmoid-per-pair loss (Zhai et al. 2023).
                                 Trains better at small effective batch
                                 (our DGPU profile sits at batch=32).
        """
        B = vision_embed.shape[0]
        if self.contrastive_loss_type == 'siglip':
            logits = vision_embed @ text_embed.t() * self.siglip_logit_scale.exp() + self.siglip_logit_bias
            # Labels: +1 on the diagonal (matched pair), -1 off-diagonal.
            labels = 2.0 * torch.eye(B, device=logits.device) - 1.0
            # SigLIP normalizes by B (per pair-row), not B*B.
            return -F.logsigmoid(labels * logits).sum() / B

        # Default: InfoNCE.
        logits = (vision_embed @ text_embed.t()) / self.temperature
        labels = torch.arange(B, device=logits.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        return (loss_i2t + loss_t2i) / 2.0
    
    def forward(
        self,
        images: torch.Tensor,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        context_indices: Optional[torch.Tensor] = None,
        target_indices: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        mode: str = "jepa",
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: [B, 3, H, W]
            text_input_ids / text_attention_mask: required for contrastive and both.
            context_indices / target_indices: required for jepa and both.
            context_mask / target_mask: optional, for visualization only.
            mode: "jepa", "contrastive", or "both".
        """
        if mode == "jepa":
            outputs = self.forward_jepa(
                images=images,
                context_indices=context_indices,
                target_indices=target_indices,
                context_mask=context_mask,
                target_mask=target_mask,
            )
            jepa_loss = self.compute_jepa_loss(
                outputs['predicted_vision'],
                outputs['target_vision'],
            )
            outputs['jepa_loss'] = jepa_loss
            outputs['loss'] = jepa_loss

        elif mode == "contrastive":
            outputs = self.forward_contrastive(images, text_input_ids, text_attention_mask)
            contrastive_loss = self.compute_contrastive_loss(
                outputs['vision_embed'],
                outputs['text_embed'],
            )
            outputs['contrastive_loss'] = contrastive_loss
            outputs['loss'] = contrastive_loss

        elif mode == "both":
            # When the predictor is cross-attention based, forward_jepa() would
            # call text_encoder once (full token sequence for cross-attn) and
            # then forward_contrastive() would call it again (CLS pooled). That
            # doubles the text encoder work for every "both" batch. Detect this
            # case and encode text once.
            is_cross_attn = isinstance(self.predictor, PredictorWithCrossAttention)

            if is_cross_attn and text_input_ids is not None:
                # Single text pass: get all tokens (for cross-attn into predictor)
                # plus the CLS pool (for contrastive). text_encoder returns the
                # full sequence including CLS at position 0 when
                # return_all_tokens=True.
                text_tokens_full = self.text_encoder(
                    text_input_ids, text_attention_mask,
                    return_all_tokens=True, return_projected=False,
                )
                # Index 0 == CLS for DistilBERT / BERT-family encoders.
                text_cls = text_tokens_full[:, 0, :]

                # JEPA path: context encoder over visible patches, predictor
                # receives both the patches and the precomputed text tokens.
                context_repr = self.vision_encoder.forward_context(images, context_indices)
                predicted_vision = self.predictor(
                    context_repr, context_indices, target_indices,
                    text_features=text_tokens_full,
                    text_attention_mask=text_attention_mask,
                )
                with torch.no_grad():
                    target_full = self.target_vision_encoder(images, return_all_tokens=True)
                    target_vision = self._gather_target_patches(target_full, target_indices)

                jepa_out = {
                    'predicted_vision': predicted_vision,
                    'target_vision': target_vision.detach(),
                    'context_repr': context_repr,
                    'context_indices': context_indices,
                    'target_indices': target_indices,
                    'context_mask': context_mask,
                    'target_mask': target_mask,
                }
                jepa_loss = self.compute_jepa_loss(predicted_vision, target_vision)

                # Contrastive path: reuse the text CLS we already have; only
                # vision still needs its CLS-only pooled forward.
                vision_features = self.vision_encoder(images, return_all_tokens=False).squeeze(1)
                vision_embed = F.normalize(self.vision_projection(vision_features), dim=-1)
                text_embed = F.normalize(self.text_projection(text_cls), dim=-1)
                contrastive_loss = self.compute_contrastive_loss(vision_embed, text_embed)
            else:
                # Non-text-conditioned predictor: original two-pass path is fine
                # because forward_jepa skips the text encoder entirely.
                jepa_out = self.forward_jepa(
                    images=images,
                    context_indices=context_indices,
                    target_indices=target_indices,
                    context_mask=context_mask,
                    target_mask=target_mask,
                )
                jepa_loss = self.compute_jepa_loss(
                    jepa_out['predicted_vision'],
                    jepa_out['target_vision'],
                )
                contrastive_out = self.forward_contrastive(images, text_input_ids, text_attention_mask)
                vision_embed = contrastive_out['vision_embed']
                text_embed = contrastive_out['text_embed']
                contrastive_loss = self.compute_contrastive_loss(vision_embed, text_embed)

            outputs = {
                **jepa_out,
                'vision_embed': vision_embed,
                'text_embed': text_embed,
                'jepa_loss': jepa_loss,
                'contrastive_loss': contrastive_loss,
            }
            # The trainer recomputes the combined loss from config weights;
            # don't add a dead outputs['loss'] here.

        else:
            raise ValueError(f"Unknown mode: {mode}")

        return outputs


def create_vl_jepa_model(config: dict) -> VLJEPAModel:
    """
    Factory function to create VL-JEPA model from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        VLJEPAModel instance
    """
    from .vision_encoder import create_vision_encoder
    from .text_encoder import create_text_encoder
    from .predictor import create_predictor
    
    # Create encoders
    vision_encoder = create_vision_encoder(config['model'])
    text_encoder = create_text_encoder(config['model'])
    
    # Create predictor (default: transformer with 2D positional embeddings)
    predictor_type = config['model']['predictor'].get('type', 'transformer')
    predictor = create_predictor(config['model'], predictor_type=predictor_type)
    
    # Create model
    model = VLJEPAModel(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        predictor=predictor,
        embedding_dim=config['model'].get('embedding_dim', 256),
        ema_momentum=config['training'].get('ema_momentum_start', 0.996),
        temperature=config['model'].get('temperature', 0.07),
        contrastive_loss_type=config['model'].get('contrastive_loss_type', 'infonce'),
    )
    
    return model


if __name__ == "__main__":
    print("Testing VLJEPAModel...")
    
    # Create simple config
    config = {
        'model': {
            'vision_encoder': {
                'type': 'vit_tiny_patch16_224',
                'pretrained': False,
                'hidden_dim': 192,
                'gradient_checkpointing': False,
            },
            'text_encoder': {
                'type': 'distilbert-base-uncased',
                'projection_dim': None,
                'max_length': 128,
                'gradient_checkpointing': False,
            },
            'predictor': {
                'type': 'mlp',
                'input_dim': 192,
                'hidden_dim': 256,
                'output_dim': 192,
                'num_layers': 3,
            },
            'embedding_dim': 256,
            'temperature': 0.07,
        },
        'training': {
            'ema_momentum_start': 0.996,
        }
    }
    
    # Create model
    model = create_vl_jepa_model(config)
    print(f"Model created successfully!")
    
    # Test inputs
    images = torch.randn(2, 3, 224, 224)
    text_input_ids = torch.randint(0, 1000, (2, 128))
    text_attention_mask = torch.ones(2, 128)
    # Fixed sizes per batch, like the real mask generator returns.
    context_indices = torch.stack([torch.randperm(196)[:80] for _ in range(2)])
    target_indices = torch.stack([torch.randperm(196)[:40] for _ in range(2)])

    # Test JEPA mode
    with torch.no_grad():
        outputs = model(
            images=images,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            context_indices=context_indices,
            target_indices=target_indices,
            mode="jepa",
        )
        print(f"JEPA loss: {outputs['loss'].item():.4f}")
    
    # Test contrastive mode
    with torch.no_grad():
        outputs = model(
            images=images,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            mode="contrastive",
        )
        print(f"Contrastive loss: {outputs['loss'].item():.4f}")
    
    # Test EMA update
    model.update_target_encoder()
    print("EMA update successful!")
    
    print("\nVL-JEPA model test passed!")
