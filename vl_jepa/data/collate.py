"""
Collate functions for VL-JEPA data loading.
"""

import torch
from typing import Dict, List, Any, Optional


def jepa_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for JEPA training batches.
    
    Handles batching of images, text tokens, and optional masks.
    
    Args:
        batch: List of sample dictionaries, each containing:
            - 'image': Image tensor [3, H, W]
            - 'input_ids': Token IDs [L]
            - 'attention_mask': Attention mask [L]
            - Optional: 'caption', 'image_path', etc.
            
    Returns:
        Batched dictionary with:
            - 'images': [B, 3, H, W]
            - 'input_ids': [B, L]
            - 'attention_mask': [B, L]
    """
    images = torch.stack([sample['image'] for sample in batch], dim=0)
    input_ids = torch.stack([sample['input_ids'] for sample in batch], dim=0)
    attention_mask = torch.stack([sample['attention_mask'] for sample in batch], dim=0)

    result = {
        'images': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    }

    # Stack per-sample mask indices when the dataset provides them. The mask
    # generator now lives inside Dataset.__getitem__ so DataLoader workers
    # parallelize it for free instead of running on the trainer's main loop.
    for key in ('context_mask', 'target_mask', 'context_indices', 'target_indices'):
        if key in batch[0]:
            result[key] = torch.stack([sample[key] for sample in batch], dim=0)

    if 'caption' in batch[0]:
        result['captions'] = [sample['caption'] for sample in batch]
    if 'image_id' in batch[0]:
        result['image_ids'] = [sample['image_id'] for sample in batch]
    if 'image_path' in batch[0]:
        result['image_paths'] = [sample['image_path'] for sample in batch]

    return result


def multimodal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Extended collate function for multimodal tasks.
    
    Supports variable length sequences and multiple captions per image.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched dictionary
    """
    # Separate images and texts
    images = []
    all_input_ids = []
    all_attention_masks = []
    image_to_text_map = []  # Maps each image to its corresponding text indices
    
    text_idx = 0
    for i, sample in enumerate(batch):
        images.append(sample['image'])
        
        # Handle multiple captions per image
        if isinstance(sample['input_ids'], list):
            num_captions = len(sample['input_ids'])
            for j in range(num_captions):
                all_input_ids.append(sample['input_ids'][j])
                all_attention_masks.append(sample['attention_mask'][j])
            image_to_text_map.append(list(range(text_idx, text_idx + num_captions)))
            text_idx += num_captions
        else:
            all_input_ids.append(sample['input_ids'])
            all_attention_masks.append(sample['attention_mask'])
            image_to_text_map.append([text_idx])
            text_idx += 1
    
    # Stack tensors
    images = torch.stack(images, dim=0)
    input_ids = torch.stack(all_input_ids, dim=0)
    attention_mask = torch.stack(all_attention_masks, dim=0)
    
    return {
        'images': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'image_to_text_map': image_to_text_map,
    }


def create_collate_fn(mode: str = 'jepa') -> callable:
    """Pick the right collate function for the training mode."""
    if mode == 'jepa':
        return jepa_collate_fn
    elif mode == 'multimodal':
        return multimodal_collate_fn
    else:
        raise ValueError(f"Unknown collate mode: {mode}")


if __name__ == "__main__":
    # Test collate function
    print("Testing collate functions...")
    
    # Create dummy batch
    batch = [
        {
            'image': torch.randn(3, 224, 224),
            'input_ids': torch.randint(0, 1000, (128,)),
            'attention_mask': torch.ones(128),
            'caption': "A dog playing in the park",
            'image_id': 12345,
        },
        {
            'image': torch.randn(3, 224, 224),
            'input_ids': torch.randint(0, 1000, (128,)),
            'attention_mask': torch.ones(128),
            'caption': "A cat sitting on a chair",
            'image_id': 67890,
        },
    ]
    
    # Test JEPA collate
    collated = jepa_collate_fn(batch)
    
    print(f"Images shape: {collated['images'].shape}")  # [2, 3, 224, 224]
    print(f"Input IDs shape: {collated['input_ids'].shape}")  # [2, 128]
    print(f"Attention mask shape: {collated['attention_mask'].shape}")  # [2, 128]
    print(f"Captions: {collated['captions']}")
    print(f"Image IDs: {collated['image_ids']}")
    
    print("\nCollate function test passed!")
