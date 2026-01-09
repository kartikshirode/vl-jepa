#!/usr/bin/env python3
"""Create a minimal test dataset for VL-JEPA training verification."""

import os
import json
from PIL import Image
import numpy as np

def main():
    print("Creating test dataset...")
    
    # Create directories
    os.makedirs("data/images/train2017", exist_ok=True)
    os.makedirs("data/annotations", exist_ok=True)
    
    annotations = {"images": [], "annotations": []}
    ann_id = 1
    num_images = 100
    
    for i in range(num_images):
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
            "caption": f"A test image number {img_id} with random colorful patterns for training verification."
        })
        ann_id += 1
        
        if (i + 1) % 20 == 0:
            print(f"  Created {i + 1}/{num_images} images...")
    
    # Save annotations
    with open("data/annotations/captions_train2017.json", "w") as f:
        json.dump(annotations, f)
    
    print(f"\n✓ Created {len(annotations['images'])} test images")
    print(f"✓ Saved annotations to data/annotations/captions_train2017.json")
    print("\nDataset ready for training!")

if __name__ == "__main__":
    main()
