#!/bin/bash
# VL-JEPA Full Setup Script for Jetson Orin Nano
# Run this after SSH'ing into your Jetson

set -e  # Exit on error

echo "=============================================="
echo "VL-JEPA Setup for Jetson Orin Nano"
echo "=============================================="

# Navigate to project
cd ~/VL-JEPA-T1
echo "✓ In project directory: $(pwd)"

# Step 1: Check PyTorch
echo ""
echo "Step 1: Checking PyTorch..."
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# Step 2: Install dependencies
echo ""
echo "Step 2: Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install timm transformers pillow pyyaml tqdm numpy scipy tensorboard wandb pycocotools accelerate einops omegaconf opencv-python pandas

# Note: bitsandbytes may not work on ARM, skip if fails
pip3 install bitsandbytes || echo "Warning: bitsandbytes failed (expected on ARM), will use standard AdamW"

echo "✓ Dependencies installed"

# Step 3: Configure Jetson power mode
echo ""
echo "Step 3: Configuring Jetson power mode..."
echo "mandar" | sudo -S nvpmodel -m 0 2>/dev/null || echo "Power mode already set or requires manual sudo"
echo "mandar" | sudo -S jetson_clocks 2>/dev/null || echo "jetson_clocks requires manual sudo"
echo "✓ Power configuration attempted"

# Step 4: Test implementation
echo ""
echo "Step 4: Testing implementation..."
python3 scripts/test_implementation.py

# Step 5: Create data directories
echo ""
echo "Step 5: Setting up data directories..."
mkdir -p data/images/train2017
mkdir -p data/images/val2017
mkdir -p data/annotations
mkdir -p checkpoints
mkdir -p logs
mkdir -p runs
echo "✓ Directories created"

# Step 6: Download COCO dataset (optional - uncomment to download)
echo ""
echo "Step 6: Dataset setup..."
echo "To download COCO dataset, run these commands manually:"
echo "  cd ~/VL-JEPA-T1/data"
echo "  wget http://images.cocodataset.org/zips/train2017.zip"
echo "  wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
echo "  unzip train2017.zip -d images/"
echo "  unzip annotations_trainval2017.zip"
echo ""
echo "Or for a quick test with dummy data, we'll create a minimal dataset..."

# Create minimal test dataset for verification
python3 << 'EOF'
import os
import json
from PIL import Image
import numpy as np

print("Creating minimal test dataset...")

# Create dummy images
os.makedirs("data/images/train2017", exist_ok=True)
os.makedirs("data/annotations", exist_ok=True)

annotations = {"images": [], "annotations": []}
ann_id = 1

for i in range(100):  # Create 100 dummy images
    img_id = i + 1
    filename = f"{img_id:012d}.jpg"
    
    # Create random image
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    img.save(f"data/images/train2017/{filename}")
    
    annotations["images"].append({
        "id": img_id,
        "file_name": filename,
        "height": 224,
        "width": 224
    })
    
    # Add caption
    annotations["annotations"].append({
        "id": ann_id,
        "image_id": img_id,
        "caption": f"A test image number {img_id} with random content for training verification."
    })
    ann_id += 1

with open("data/annotations/captions_train2017.json", "w") as f:
    json.dump(annotations, f)

print(f"✓ Created {len(annotations['images'])} dummy images for testing")
EOF

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. For real training, download COCO dataset (commands above)"
echo "2. Or test with dummy data now:"
echo "   python3 train.py --config config.yaml"
echo ""
echo "To monitor training:"
echo "   tegrastats  # Jetson stats"
echo "   tensorboard --logdir runs/  # Training curves"
echo ""
