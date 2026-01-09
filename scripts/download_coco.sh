#!/bin/bash
# Download COCO 2017 dataset for VL-JEPA training

set -e

DATA_DIR="./data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "=== Downloading COCO 2017 Dataset ==="
echo "This will download ~20GB of data"

# Download train images (18GB)
if [ ! -f "train2017.zip" ] && [ ! -d "train2017" ]; then
    echo "Downloading train2017 images..."
    wget -c http://images.cocodataset.org/zips/train2017.zip
fi

# Download val images (1GB)  
if [ ! -f "val2017.zip" ] && [ ! -d "val2017" ]; then
    echo "Downloading val2017 images..."
    wget -c http://images.cocodataset.org/zips/val2017.zip
fi

# Download annotations (241MB)
if [ ! -f "annotations_trainval2017.zip" ] && [ ! -d "annotations" ]; then
    echo "Downloading annotations..."
    wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
fi

# Extract files
if [ -f "train2017.zip" ] && [ ! -d "train2017" ]; then
    echo "Extracting train2017..."
    unzip -q train2017.zip
    rm train2017.zip
fi

if [ -f "val2017.zip" ] && [ ! -d "val2017" ]; then
    echo "Extracting val2017..."
    unzip -q val2017.zip
    rm val2017.zip
fi

if [ -f "annotations_trainval2017.zip" ] && [ ! -d "annotations" ]; then
    echo "Extracting annotations..."
    unzip -q annotations_trainval2017.zip
    rm annotations_trainval2017.zip
fi

echo ""
echo "=== Download Complete ==="
echo "Train images: $(ls train2017 2>/dev/null | wc -l) files"
echo "Val images: $(ls val2017 2>/dev/null | wc -l) files"
echo "Annotations: $(ls annotations 2>/dev/null)"
