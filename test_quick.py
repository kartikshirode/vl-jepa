"""Quick test script for VL-JEPA components."""

import sys
import platform

if platform.system() == "Windows":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch
print('Testing VL-JEPA components...')

from vl_jepa.models import VisionEncoder, TextEncoder, PredictorMLP, VLJEPAModel
from vl_jepa.models.predictor import PredictorTransformer
from vl_jepa.models.vl_jepa import create_vl_jepa_model
from vl_jepa.masks import MultiBlockMaskGenerator
from vl_jepa.data import get_train_transforms, get_val_transforms, jepa_collate_fn
print('1. All imports OK')

# Test mask generator (now returns 4-tuple: masks + indices)
mask_gen = MultiBlockMaskGenerator(input_size=224, patch_size=16)
ctx_mask, tgt_mask, ctx_idx, tgt_idx = mask_gen()
assert (ctx_mask & tgt_mask).sum().item() == 0, "context and target overlap"
print(f'2. Mask generator OK - Context: {ctx_mask.sum().item()}/196, Target: {tgt_mask.sum().item()}/196, N_ctx={ctx_idx.shape[0]}, N_tgt={tgt_idx.shape[0]}')

# Test vision encoder (full forward and context forward)
vision_enc = VisionEncoder(pretrained=False, gradient_checkpointing=False)
x = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    v_out = vision_enc(x)
    ctx_idx_batch = torch.stack([mask_gen()[2] for _ in range(2)])
    v_ctx = vision_enc.forward_context(x, ctx_idx_batch)
assert v_ctx.shape == (2, ctx_idx_batch.shape[1], 192), f"unexpected {v_ctx.shape}"
print(f'3. Vision encoder OK - full: {v_out.shape}, context: {v_ctx.shape}')

# Test text encoder
text_enc = TextEncoder(gradient_checkpointing=False)
ids = torch.randint(0, 1000, (2, 128))
mask = torch.ones(2, 128)
with torch.no_grad():
    t_out = text_enc(ids, mask)
print(f'4. Text encoder OK - Output: {t_out.shape}')

# Test transformer predictor (2D positional gather)
predictor = PredictorTransformer(input_dim=192, hidden_dim=384, output_dim=192, num_layers=2, num_heads=6, num_patches=196)
tgt_idx_batch = torch.stack([mask_gen()[3] for _ in range(2)])
with torch.no_grad():
    p_out = predictor(v_ctx, ctx_idx_batch, tgt_idx_batch)
assert p_out.shape == (2, tgt_idx_batch.shape[1], 192), f"unexpected {p_out.shape}"
print(f'5. Predictor OK - Output: {p_out.shape}')

# Test full VL-JEPA model in jepa mode
model = VLJEPAModel(vision_enc, text_enc, predictor)
with torch.no_grad():
    out = model(
        x, ids, mask,
        context_indices=ctx_idx_batch,
        target_indices=tgt_idx_batch,
        mode='jepa',
    )
print(f'6. VL-JEPA model OK - JEPA loss: {out["loss"].item():.4f}')

# train() puts target encoders in eval mode
model.train()
assert not model.target_vision_encoder.training, "target vision encoder still training"
assert not model.target_text_encoder.training, "target text encoder still training"
print('   target encoders correctly forced to eval()')

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
    model_gpu = model.cuda()
    x_gpu = x.cuda(); ids_gpu = ids.cuda(); mask_gpu = mask.cuda()
    ctx_gpu = ctx_idx_batch.cuda(); tgt_gpu = tgt_idx_batch.cuda()
    with torch.no_grad():
        out_gpu = model_gpu(
            x_gpu, ids_gpu, mask_gpu,
            context_indices=ctx_gpu, target_indices=tgt_gpu,
            mode='jepa',
        )
    print(f'9. GPU test OK - JEPA loss: {out_gpu["loss"].item():.4f}')
else:
    print('\nNo CUDA available')

print('\n✓ All tests passed!')
