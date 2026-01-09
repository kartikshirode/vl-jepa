# VL-JEPA: Vision-Language Joint Embedding Predictive Architecture

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)

Implementation of VL-JEPA for vision-language pretraining with JEPA (Joint Embedding Predictive Architecture) and contrastive learning.

## 🚀 Features

- **JEPA Training**: Self-supervised learning with masked patch prediction in representation space
- **Contrastive Learning**: Vision-language alignment with InfoNCE loss
- **Memory Optimized**: FP16 mixed precision, gradient checkpointing support
- **Flexible Architecture**: ViT-Tiny/Small + DistilBERT + MLP/Transformer Predictor
- **Multi-Modal**: Vision-language pretraining with COCO Captions

## 📋 Requirements

### Hardware
- NVIDIA GPU with 8GB+ VRAM (RTX 3060, 4060, etc.)

### Software
- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/mandarwagh9/vl-jepa-jetson.git
cd vl-jepa-jetson

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## 📊 Dataset Preparation

### COCO Captions

1. Download from [COCO Dataset](https://cocodataset.org/#download):
   - `train2017.zip` (~18GB)
   - `val2017.zip` (~1GB)
   - `annotations_trainval2017.zip` (~241MB)

2. Extract to `./vl_jepa/data/COCO2017/`:
```
vl_jepa/data/COCO2017/
├── train2017/
├── val2017/
└── annotations/
    ├── captions_train2017.json
    └── captions_val2017.json
```

## 🎯 Training

### Start Training

```bash
# Start training with DGPU config
python train.py --config config_dgpu.yaml

# Resume from checkpoint
python train.py --config config_dgpu.yaml --resume checkpoints/checkpoint_epoch_5.pth

# With Weights & Biases logging
python train.py --config config_dgpu.yaml --wandb
```

### Configuration

Edit `config_dgpu.yaml` to adjust:
- **batch_size**: 16 (default, adjust based on VRAM)
- **gradient_accumulation_steps**: 2 (effective batch = 32)
- **learning_rate**: 3e-4
- **num_epochs**: 100
- **Loss weights**: `jepa_loss_weight: 1.0`, `contrastive_loss_weight: 0.5`

### Training Metrics

The training uses combined loss:
```
Total Loss = jepa_loss_weight × JEPA Loss + contrastive_loss_weight × Contrastive Loss
```

- **JEPA Loss**: Smooth L1 loss between predicted and target patch representations
- **Contrastive Loss**: InfoNCE loss for vision-language alignment

## 🧪 Model Architecture

```
VL-JEPA Model (~145M parameters, ~73M trainable)
├── Vision Encoder (ViT-Tiny)
│   ├── Patch Embedding: 16x16 patches
│   ├── Hidden Dim: 192
│   ├── Layers: 12
│   ├── Attention Heads: 3
│   └── Parameters: ~5.7M
├── Text Encoder (DistilBERT)
│   ├── Hidden Dim: 768
│   ├── Layers: 6
│   ├── Max Length: 128 tokens
│   └── Parameters: ~66M
├── Predictor (MLP)
│   ├── Hidden: 384
│   ├── Layers: 4
│   └── Parameters: ~1M
├── Projection Heads
│   └── Embedding Dim: 256
└── Target Encoders (EMA, frozen)
    └── Momentum: 0.996 → 1.0
```

## 🎨 Masking Strategy

JEPA-style multi-block masking:
- **Context blocks**: 1 block (85-100% scale) - visible to model
- **Target blocks**: 4 blocks (15-20% scale) - model predicts these
- **Patch grid**: 14×14 = 196 patches

## 📈 Evaluation Metrics

- **Image→Text Retrieval**: R@1, R@5, R@10
- **Text→Image Retrieval**: R@1, R@5, R@10  
- **Mean Recall**: Average across all retrieval metrics
- **Validation Loss**: Contrastive loss on validation set

## 📁 Project Structure

```
vl-jepa/
├── vl_jepa/                  # Main package
│   ├── models/               # Model implementations
│   │   ├── vision_encoder.py    # ViT encoder
│   │   ├── text_encoder.py      # DistilBERT encoder
│   │   ├── predictor.py         # MLP/Transformer predictor
│   │   └── vl_jepa.py           # Main VL-JEPA model
│   ├── data/                 # Data loading
│   │   ├── dataset.py           # COCO dataset
│   │   ├── transforms.py        # Image augmentations
│   │   └── collate.py           # Batch collation
│   ├── masks/                # Masking strategy
│   │   └── multiblock.py        # Multi-block masks
│   └── utils/                # Utilities
│       ├── config.py            # Config management
│       ├── logger.py            # Logging
│       ├── checkpoint.py        # Checkpointing
│       └── metrics.py           # Evaluation metrics
├── scripts/                  # Utility scripts
├── checkpoints/              # Model checkpoints
├── train.py                  # Training script
├── inference.py              # Inference script
├── config_dgpu.yaml          # DGPU configuration
└── requirements.txt          # Dependencies
```

## 🔧 Troubleshooting

### Out of Memory

```yaml
# Reduce batch size
training:
  batch_size: 8
  gradient_accumulation_steps: 4  # Keep effective batch size
```

### Slow Training

- Check GPU utilization: `nvidia-smi`
- Increase `num_workers` in config
- Enable `pin_memory: true`

## 📚 References

- [VL-JEPA Paper](https://arxiv.org/abs/2512.10942v1)
- [I-JEPA: A Path Towards Autonomous Machine Intelligence](https://arxiv.org/abs/2301.08243)
- [COCO Dataset](https://cocodataset.org/)

## 📄 License

MIT License - see LICENSE file for details.
