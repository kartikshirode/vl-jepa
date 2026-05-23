"""
Quick test script to verify VL-JEPA implementation
Tests all components without requiring full dataset
"""

import torch
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_vision_encoder():
    """Test vision encoder"""
    print("\n" + "="*80)
    print("Testing Vision Encoder (ViT-Tiny)")
    print("="*80)
    
    from vl_jepa.models.vision_encoder import VisionEncoder
    
    model = VisionEncoder(
        model_name="vit_tiny_patch16_224",
        pretrained=False,  # Don't download for quick test
        gradient_checkpointing=True,
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        output = model(x, return_all_tokens=True)
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Expected: [2, 197, 192] (batch, patches+cls, dim)")
    print(f"✓ Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    assert output.shape == (2, 197, 192), f"Unexpected output shape: {output.shape}"
    print("✓ Vision encoder test PASSED")
    
    return True


def test_text_encoder():
    """Test text encoder"""
    print("\n" + "="*80)
    print("Testing Text Encoder (DistilBERT)")
    print("="*80)
    
    from vl_jepa.models.text_encoder import TextEncoder
    
    model = TextEncoder(
        model_name="distilbert-base-uncased",
        max_length=128,
        gradient_checkpointing=True,
    )
    
    # Test tokenization
    texts = [
        "A dog playing in the park",
        "A cat sitting on a chair",
    ]
    
    tokens = model.tokenize(texts)
    
    print(f"✓ Tokenized {len(texts)} texts")
    print(f"✓ Input IDs shape: {tokens['input_ids'].shape}")
    print(f"✓ Attention mask shape: {tokens['attention_mask'].shape}")
    
    # Test forward pass. The internal projection was removed; the encoder
    # now returns raw DistilBERT hidden_size (768).
    with torch.no_grad():
        output = model(
            input_ids=tokens['input_ids'],
            attention_mask=tokens['attention_mask'],
            return_all_tokens=False,
            return_projected=False,
        )

    expected_dim = model.embed_dim
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Expected: [2, {expected_dim}] (batch, embed_dim)")
    print(f"✓ Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    assert output.shape == (2, expected_dim), f"Unexpected output shape: {output.shape}"
    print("✓ Text encoder test PASSED")
    
    return True


def test_predictor():
    """Test predictor"""
    print("\n" + "="*80)
    print("Testing Predictor (MLP)")
    print("="*80)
    
    from vl_jepa.models.predictor import PredictorMLP
    
    model = PredictorMLP(
        input_dim=192,
        hidden_dim=256,
        output_dim=192,
        num_layers=3,
    )
    
    # Test forward pass
    x = torch.randn(2, 196, 192)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    assert output.shape == x.shape, f"Output shape {output.shape} doesn't match input {x.shape}"
    print("✓ Predictor test PASSED")
    
    return True


def test_masking():
    """Test masking strategy"""
    print("\n" + "="*80)
    print("Testing Multi-Block Masking")
    print("="*80)
    
    from vl_jepa.masks.multiblock import MultiBlockMaskGenerator
    
    mask_gen = MultiBlockMaskGenerator(
        input_size=224,
        patch_size=16,
        num_context_blocks=1,
        num_target_blocks=4,
        context_scale=(0.85, 1.0),
        target_scale=(0.15, 0.2),
        allow_overlap=False,
    )
    
    # Generate masks. Generator returns (ctx_mask, tgt_mask, ctx_idx, tgt_idx).
    context_mask, target_mask, ctx_idx, tgt_idx = mask_gen()

    print(f"✓ Context mask shape: {context_mask.shape}")
    print(f"✓ Target mask shape: {target_mask.shape}")
    print(f"✓ Context indices shape: {ctx_idx.shape}")
    print(f"✓ Target indices shape: {tgt_idx.shape}")
    print(f"✓ Total patches: {mask_gen.total_patches}")
    print(f"✓ Context patches: {context_mask.sum().item()}")
    print(f"✓ Target patches: {target_mask.sum().item()}")
    print(f"✓ Overlap patches: {(context_mask & target_mask).sum().item()}")
    
    assert context_mask.shape[0] == mask_gen.total_patches
    assert target_mask.shape[0] == mask_gen.total_patches
    print("✓ Masking test PASSED")
    
    return True


def test_vl_jepa_model():
    """Test full VL-JEPA model"""
    print("\n" + "="*80)
    print("Testing VL-JEPA Model (Full Integration)")
    print("="*80)
    
    from vl_jepa.models.vl_jepa import VLJEPAModel
    from vl_jepa.models.vision_encoder import VisionEncoder
    from vl_jepa.models.text_encoder import TextEncoder
    from vl_jepa.models.predictor import PredictorMLP
    
    # Create components
    vision_encoder = VisionEncoder(
        model_name="vit_tiny_patch16_224",
        pretrained=False,
        gradient_checkpointing=False,  # Disable for testing
    )
    
    text_encoder = TextEncoder(
        model_name="distilbert-base-uncased",
        max_length=128,
        gradient_checkpointing=False,  # Disable for testing
    )
    
    predictor = PredictorMLP(
        input_dim=192,
        hidden_dim=256,
        output_dim=192,
        num_layers=3,
    )
    
    # Create model
    model = VLJEPAModel(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        predictor=predictor,
        embedding_dim=256,
        ema_momentum=0.996,
    )
    
    print(f"✓ Model created")
    print(f"✓ Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Test forward pass
    images = torch.randn(2, 3, 224, 224)
    text_input_ids = torch.randint(0, 1000, (2, 128))
    text_attention_mask = torch.ones(2, 128)
    context_mask = torch.rand(2, 196) > 0.15  # 85% visible (context)
    target_mask = torch.rand(2, 196) > 0.80   # 20% target patches
    
    with torch.no_grad():
        # Test JEPA mode
        outputs = model(
            images=images,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            context_mask=context_mask,
            target_mask=target_mask,
            mode="jepa",
        )
        
        print(f"✓ JEPA forward pass completed")
        print(f"✓ Loss: {outputs['loss'].item():.4f}")
        
        # Test contrastive mode
        outputs = model(
            images=images,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            mode="contrastive",
        )
        
        print(f"✓ Contrastive forward pass completed")
        print(f"✓ Loss: {outputs['loss'].item():.4f}")
        
        # Test EMA update
        model.update_target_encoder()
        print(f"✓ EMA update completed")
    
    print("✓ VL-JEPA model test PASSED")
    
    return True


def test_transforms():
    """Test data transforms"""
    print("\n" + "="*80)
    print("Testing Data Transforms")
    print("="*80)
    
    from vl_jepa.data.transforms import get_train_transforms, get_val_transforms
    from PIL import Image
    import numpy as np
    
    config = {
        'image_transforms': {
            'resize': 256,
            'random_resized_crop': 224,
            'random_crop_scale': [0.2, 1.0],
            'random_crop_ratio': [0.75, 1.333],
            'horizontal_flip': 0.5,
            'color_jitter': {
                'brightness': 0.4,
                'contrast': 0.4,
                'saturation': 0.4,
                'hue': 0.1,
            },
            'gaussian_blur': {
                'kernel_size': 23,
                'sigma': [0.1, 2.0],
                'p': 0.5,
            },
            'normalize': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
            },
        }
    }
    
    # Create dummy image
    img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    
    # Test train transform
    train_transform = get_train_transforms(config)
    img_train = train_transform(img)
    
    print(f"✓ Train transform output shape: {img_train.shape}")
    assert img_train.shape == (3, 224, 224)
    
    # Test val transform
    val_transform = get_val_transforms(config)
    img_val = val_transform(img)
    
    print(f"✓ Val transform output shape: {img_val.shape}")
    assert img_val.shape == (3, 224, 224)
    
    print("✓ Transforms test PASSED")
    
    return True


def test_memory_usage():
    """Test memory usage"""
    print("\n" + "="*80)
    print("Testing Memory Usage")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("⚠ CUDA not available, skipping memory test")
        return True
    
    from vl_jepa.models.vl_jepa import VLJEPAModel
    from vl_jepa.models.vision_encoder import VisionEncoder
    from vl_jepa.models.text_encoder import TextEncoder
    from vl_jepa.models.predictor import PredictorMLP
    
    # Create model
    vision_encoder = VisionEncoder(
        model_name="vit_tiny_patch16_224",
        pretrained=False,
        gradient_checkpointing=True,
    )
    
    text_encoder = TextEncoder(
        model_name="distilbert-base-uncased",
        projection_dim=None,
        max_length=128,
        gradient_checkpointing=True,
    )
    
    predictor = PredictorMLP(
        input_dim=192,
        hidden_dim=256,
        output_dim=192,
        num_layers=3,
    )
    
    model = VLJEPAModel(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        predictor=predictor,
    ).cuda()
    
    # Measure memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Forward pass
    images = torch.randn(2, 3, 224, 224).cuda()
    text_input_ids = torch.randint(0, 1000, (2, 128)).cuda()
    text_attention_mask = torch.ones(2, 128).cuda()
    
    with torch.no_grad():
        outputs = model(
            images=images,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            mode="jepa",
        )
    
    memory_allocated = torch.cuda.memory_allocated() / 1e9
    memory_reserved = torch.cuda.memory_reserved() / 1e9
    max_memory = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"✓ Memory allocated: {memory_allocated:.2f} GB")
    print(f"✓ Memory reserved: {memory_reserved:.2f} GB")
    print(f"✓ Peak memory: {max_memory:.2f} GB")
    
    if memory_allocated > 6.0:
        print("⚠ Warning: Memory usage exceeds 6GB (may not fit on 8GB Jetson)")
    else:
        print("✓ Memory usage is acceptable for Jetson Orin Nano")
    
    print("✓ Memory test PASSED")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("VL-JEPA Implementation Test Suite")
    print("="*80)
    
    tests = [
        ("Vision Encoder", test_vision_encoder),
        ("Text Encoder", test_text_encoder),
        ("Predictor", test_predictor),
        ("Masking", test_masking),
        ("VL-JEPA Model", test_vl_jepa_model),
        ("Transforms", test_transforms),
        ("Memory Usage", test_memory_usage),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} test FAILED with error:")
            print(f"  {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{name:.<50} {status}")
    
    all_passed = all(success for _, success in results)
    
    print("="*80)
    if all_passed:
        print("✓ All tests PASSED! Implementation is ready.")
    else:
        print("✗ Some tests FAILED. Please check the errors above.")
    print("="*80)
    
    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
