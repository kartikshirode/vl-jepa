"""Quick test script for VL-JEPA components"""

import torch
print('Testing VL-JEPA components...')

# Test model imports
from vl_jepa.models import VisionEncoder, TextEncoder, PredictorMLP, VLJEPAModel
from vl_jepa.models.vl_jepa import create_vl_jepa_model
from vl_jepa.masks import MultiBlockMaskGenerator
from vl_jepa.data import get_train_transforms, get_val_transforms, jepa_collate_fn
print('1. All imports OK')

# Test mask generator
mask_gen = MultiBlockMaskGenerator(input_size=224, patch_size=16)
ctx_mask, tgt_mask = mask_gen()
print(f'2. Mask generator OK - Context: {ctx_mask.sum()}/196, Target: {tgt_mask.sum()}/196')

# Test vision encoder
vision_enc = VisionEncoder(pretrained=False, gradient_checkpointing=False)
x = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    v_out = vision_enc(x)
print(f'3. Vision encoder OK - Output: {v_out.shape}')

# Test text encoder
text_enc = TextEncoder(gradient_checkpointing=False)
ids = torch.randint(0, 1000, (2, 128))
mask = torch.ones(2, 128)
with torch.no_grad():
    t_out = text_enc(ids, mask)
print(f'4. Text encoder OK - Output: {t_out.shape}')

# Test predictor
predictor = PredictorMLP(input_dim=192, hidden_dim=256, output_dim=192)
with torch.no_grad():
    p_out = predictor(v_out)
print(f'5. Predictor OK - Output: {p_out.shape}')

# Test full model
model = VLJEPAModel(vision_enc, text_enc, predictor)
ctx_masks = torch.stack([mask_gen()[0] for _ in range(2)])
tgt_masks = torch.stack([mask_gen()[1] for _ in range(2)])
with torch.no_grad():
    out = model(x, ids, mask, context_mask=ctx_masks, target_mask=tgt_masks, mode='jepa')
print(f'6. VL-JEPA model OK - JEPA loss: {out["loss"].item():.4f}')

# Test contrastive mode
with torch.no_grad():
    out_c = model(x, ids, mask, mode='contrastive')
print(f'7. Contrastive mode OK - Loss: {out_c["loss"].item():.4f}')

# Test data transforms
config = {'image_transforms': {'resize': 256, 'random_resized_crop': 224}}
train_tf = get_train_transforms(config)
val_tf = get_val_transforms(config)
print(f'8. Data transforms OK')

# Check CUDA
if torch.cuda.is_available():
    print(f'\nCUDA Available: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
    
    # Test on GPU
    model_gpu = model.cuda()
    x_gpu = x.cuda()
    ids_gpu = ids.cuda()
    mask_gpu = mask.cuda()
    ctx_gpu = ctx_masks.cuda()
    tgt_gpu = tgt_masks.cuda()
    
    with torch.no_grad():
        out_gpu = model_gpu(x_gpu, ids_gpu, mask_gpu, context_mask=ctx_gpu, target_mask=tgt_gpu, mode='jepa')
    print(f'9. GPU test OK - JEPA loss: {out_gpu["loss"].item():.4f}')
else:
    print('\nNo CUDA available')

print('\n✓ All tests passed!')
