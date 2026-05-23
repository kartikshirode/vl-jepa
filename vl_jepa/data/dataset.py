"""
Dataset implementations for VL-JEPA training.
Supports COCO Captions, Flickr30k, and generic image-text datasets.
"""

import logging
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import json
import os
import random
from typing import Dict, Any, Optional, Callable, List, Union

_logger = logging.getLogger("vl_jepa.data")


def _safe_load_image(
    image_path: Path,
    failure_counter: List[int],
    total_samples: int,
    warning_budget: int = 10,
    failure_ratio_limit: float = 0.01,
) -> Image.Image:
    """
    Open an image, falling back to a black 224x224 for one-off corruption.
    Rate-limits warnings and raises if too many samples in the same dataset
    instance have failed. `failure_counter` is a 1-element list used as a
    cheap mutable counter across calls.
    """
    try:
        return Image.open(image_path).convert('RGB')
    except Exception as e:
        failure_counter[0] += 1
        if failure_counter[0] <= warning_budget:
            _logger.warning("Failed to load %s (%s). Falling back to black image.", image_path, e)
        elif failure_counter[0] == warning_budget + 1:
            _logger.warning("Further image load failures will be summarized at epoch end.")
        # Abort training when failures become a meaningful fraction of the dataset.
        if total_samples > 0 and failure_counter[0] / total_samples > failure_ratio_limit:
            raise RuntimeError(
                f"More than {failure_ratio_limit:.1%} of samples failed to "
                f"load ({failure_counter[0]} / {total_samples}). Stopping to "
                f"avoid training on mostly-black data."
            )
        return Image.new('RGB', (224, 224), color='black')


class ImageTextDataset(Dataset):
    """
    Generic image-text dataset.
    
    Expects data in format:
        - images_dir: Directory containing images
        - annotations: JSON file with list of {"image": "filename", "caption": "text"}
        
    Args:
        images_dir: Path to images directory
        annotations_file: Path to annotations JSON
        transform: Image transform
        tokenizer: Text tokenizer
        max_length: Maximum text sequence length
    """
    
    def __init__(
        self,
        images_dir: str,
        annotations_file: str,
        transform: Optional[Callable] = None,
        tokenizer: Optional[Any] = None,
        max_length: int = 128,
    ):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)

        # If annotations is a dict with 'annotations' key, extract it
        if isinstance(self.annotations, dict) and 'annotations' in self.annotations:
            self.annotations = self.annotations['annotations']

        # Mutable counter for image-load failures (shared with _safe_load_image).
        self._failure_counter = [0]
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ann = self.annotations[idx]
        
        # Load image
        image_file = ann.get('image', ann.get('file_name', ann.get('image_id')))
        image_path = self.images_dir / str(image_file)
        
        # Handle case where image_id is used
        if not image_path.exists() and 'image_id' in ann:
            # Try with .jpg extension
            image_path = self.images_dir / f"{ann['image_id']:012d}.jpg"
        
        image = _safe_load_image(image_path, self._failure_counter, len(self.annotations))

        if self.transform is not None:
            image = self.transform(image)

        # Get caption
        caption = ann.get('caption', ann.get('text', ''))
        
        # Tokenize
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                caption,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt',
            )
            input_ids = tokens['input_ids'].squeeze(0)
            attention_mask = tokens['attention_mask'].squeeze(0)
        else:
            input_ids = torch.zeros(self.max_length, dtype=torch.long)
            attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        
        return {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'caption': caption,
            'image_id': str(image_file),
            'image_path': str(image_path),
        }


class COCOCaptionsDataset(Dataset):
    """
    COCO Captions dataset.
    
    Expects COCO format with:
        - images/train2017 or images/val2017
        - annotations/captions_train2017.json or captions_val2017.json
        
    Args:
        data_root: Root directory containing 'images' and 'annotations' folders
        split: 'train' or 'val'
        transform: Image transform
        tokenizer: Text tokenizer
        max_length: Maximum text sequence length
        max_samples: Maximum number of samples (for memory-limited training)
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        tokenizer: Optional[Any] = None,
        max_length: int = 128,
        max_samples: Optional[int] = None,
        mask_generator: Optional[Any] = None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_generator = mask_generator

        # Determine paths based on split
        if split == 'train':
            self.images_dir = self.data_root / 'images' / 'train2017'
            annotations_file = self.data_root / 'annotations' / 'captions_train2017.json'
        elif split == 'val':
            self.images_dir = self.data_root / 'images' / 'val2017'
            annotations_file = self.data_root / 'annotations' / 'captions_val2017.json'
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train' or 'val'.")
        
        # Alternative path structure (images directly in train2017/val2017)
        if not self.images_dir.exists():
            self.images_dir = self.data_root / f'{split}2017'
        
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        if not annotations_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
        
        # Load annotations
        print(f"Loading COCO {split} annotations from {annotations_file}...")
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # Build image_id to filename mapping
        self.image_id_to_file = {
            img['id']: img['file_name'] 
            for img in coco_data['images']
        }
        
        # Get annotations (caption per image)
        self.annotations = coco_data['annotations']
        
        # Limit samples if specified. Use a local RNG so we don't leak our
        # determinism choice into the global random state.
        if max_samples is not None and len(self.annotations) > max_samples:
            print(f"Limiting dataset from {len(self.annotations)} to {max_samples} samples")
            local_rng = random.Random(42)
            self.annotations = local_rng.sample(self.annotations, max_samples)

        # Mutable counter (single-element list) for image-load failures.
        self._failure_counter = [0]

        print(f"Loaded {len(self.annotations)} image-caption pairs")
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ann = self.annotations[idx]
        
        # Get image path
        image_id = ann['image_id']
        image_file = self.image_id_to_file.get(image_id, f"{image_id:012d}.jpg")
        image_path = self.images_dir / image_file
        
        # Load image (failure-tolerant; aborts past 1% failure rate).
        image = _safe_load_image(image_path, self._failure_counter, len(self.annotations))

        if self.transform is not None:
            image = self.transform(image)

        # Get caption
        caption = ann['caption']

        # Tokenize
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                caption,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt',
            )
            input_ids = tokens['input_ids'].squeeze(0)
            attention_mask = tokens['attention_mask'].squeeze(0)
        else:
            input_ids = torch.zeros(self.max_length, dtype=torch.long)
            attention_mask = torch.zeros(self.max_length, dtype=torch.long)

        sample = {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'caption': caption,
            'image_id': image_id,
            'image_path': str(image_path),
        }
        if self.mask_generator is not None:
            cm, tm, ci, ti = self.mask_generator()
            sample.update({
                'context_mask': cm,
                'target_mask': tm,
                'context_indices': ci,
                'target_indices': ti,
            })
        return sample


class Flickr30kDataset(Dataset):
    """
    Flickr30k dataset.
    
    Expects format:
        - flickr30k-images/ containing images
        - results.csv or captions.txt with image-caption pairs
        
    Args:
        data_root: Root directory containing images and captions
        split: 'train', 'val', or 'test'
        transform: Image transform
        tokenizer: Text tokenizer
        max_length: Maximum text sequence length
        max_samples: Maximum number of samples
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        tokenizer: Optional[Any] = None,
        max_length: int = 128,
        max_samples: Optional[int] = None,
        mask_generator: Optional[Any] = None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_generator = mask_generator

        # Find images directory
        self.images_dir = self.data_root / 'flickr30k-images'
        if not self.images_dir.exists():
            self.images_dir = self.data_root / 'images'
        if not self.images_dir.exists():
            self.images_dir = self.data_root

        self.annotations = self._load_captions()

        # P3.2: require an explicit split file. Falling back to "use all
        # images" silently mixes train/val/test, which has burned every
        # serious user of this loader at least once. Karpathy splits live at
        # https://github.com/karpathy/neuraltalk2 (the json) or
        # https://github.com/li-xirong/coco-cn (txt forms).
        split_file = self.data_root / f'{split}.txt'
        if not split_file.exists():
            raise FileNotFoundError(
                f"Flickr30k split file not found: {split_file}\n"
                "Download the Karpathy splits (train.txt / val.txt / test.txt) "
                "and place them in the data root. Refusing to silently use "
                "all images for split='{split}'."
            )
        with open(split_file, 'r', encoding='utf-8') as f:
            split_images = set(line.strip() for line in f)
        self.annotations = [
            ann for ann in self.annotations
            if ann['image'] in split_images
        ]

        if max_samples is not None and len(self.annotations) > max_samples:
            local_rng = random.Random(42)
            self.annotations = local_rng.sample(self.annotations, max_samples)

        self._failure_counter = [0]
        print(f"Loaded {len(self.annotations)} Flickr30k {split} samples")
    
    def _load_captions(self) -> List[Dict[str, str]]:
        """Load captions from various possible formats."""
        annotations = []
        
        # Try CSV format (results.csv)
        csv_file = self.data_root / 'results.csv'
        if csv_file.exists():
            import csv
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f, delimiter='|')
                for row in reader:
                    annotations.append({
                        'image': row['image_name'].strip(),
                        'caption': row['comment'].strip(),
                    })
            return annotations
        
        # Try token format (Flickr30k.token.txt)
        token_file = self.data_root / 'Flickr30k.token.txt'
        if token_file.exists():
            with open(token_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        image_id = parts[0].split('#')[0]
                        caption = parts[1]
                        annotations.append({
                            'image': image_id,
                            'caption': caption,
                        })
            return annotations
        
        # Try JSON format
        json_file = self.data_root / 'captions.json'
        if json_file.exists():
            with open(json_file, 'r') as f:
                return json.load(f)
        
        raise FileNotFoundError(
            f"No caption file found in {self.data_root}. "
            "Expected results.csv, Flickr30k.token.txt, or captions.json"
        )
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ann = self.annotations[idx]
        
        # Load image
        image_file = ann['image']
        image_path = self.images_dir / image_file
        
        image = _safe_load_image(image_path, self._failure_counter, len(self.annotations))

        if self.transform is not None:
            image = self.transform(image)

        caption = ann['caption']

        if self.tokenizer is not None:
            tokens = self.tokenizer(
                caption,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt',
            )
            input_ids = tokens['input_ids'].squeeze(0)
            attention_mask = tokens['attention_mask'].squeeze(0)
        else:
            input_ids = torch.zeros(self.max_length, dtype=torch.long)
            attention_mask = torch.zeros(self.max_length, dtype=torch.long)

        sample = {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'caption': caption,
            'image_path': str(image_path),
        }
        if self.mask_generator is not None:
            cm, tm, ci, ti = self.mask_generator()
            sample.update({
                'context_mask': cm,
                'target_mask': tm,
                'context_indices': ci,
                'target_indices': ti,
            })
        return sample


class DummyDataset(Dataset):
    """
    Dummy dataset for testing without real data.
    Generates random images and captions.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        image_size: int = 224,
        tokenizer: Optional[Any] = None,
        max_length: int = 128,
        transform: Optional[Callable] = None,
        mask_generator: Optional[Any] = None,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform
        self.mask_generator = mask_generator

        self.captions = [
            f"A synthetic image number {i} for testing purposes"
            for i in range(num_samples)
        ]

        _logger.info("Created dummy dataset with %d samples", num_samples)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Generate random image
        image = torch.randn(3, self.image_size, self.image_size)
        
        # Get caption
        caption = self.captions[idx]
        
        # Tokenize
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                caption,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt',
            )
            input_ids = tokens['input_ids'].squeeze(0)
            attention_mask = tokens['attention_mask'].squeeze(0)
        else:
            input_ids = torch.zeros(self.max_length, dtype=torch.long)
            attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        
        sample = {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'caption': caption,
            'image_id': idx,
        }
        if self.mask_generator is not None:
            cm, tm, ci, ti = self.mask_generator()
            sample.update({
                'context_mask': cm,
                'target_mask': tm,
                'context_indices': ci,
                'target_indices': ti,
            })
        return sample


def create_dataset(
    config: Dict[str, Any],
    split: str = 'train',
    transform: Optional[Callable] = None,
    tokenizer: Optional[Any] = None,
    mask_generator: Optional[Any] = None,
) -> Dataset:
    """
    Factory function to create dataset from config.
    
    Args:
        config: Configuration dictionary with 'data' section
        split: 'train', 'val', or 'test'
        transform: Image transform (optional, will create default if None)
        tokenizer: Text tokenizer (optional)
        
    Returns:
        Dataset instance
    """
    data_config = config.get('data', config)
    
    dataset_name = data_config.get('dataset_name', 'coco_captions').lower()
    data_root = data_config.get('data_root', './data')
    max_length = config.get('model', {}).get('text_encoder', {}).get('max_length', 128)
    max_samples = data_config.get('max_samples', None)
    
    # Adjust max_samples based on split
    if split == 'val' and max_samples is not None:
        max_samples = min(max_samples // 10, 5000)  # Use 10% for validation
    
    print(f"Creating {dataset_name} dataset (split={split})...")
    
    if dataset_name == 'coco_captions' or dataset_name == 'coco':
        return COCOCaptionsDataset(
            data_root=data_root,
            split=split,
            transform=transform,
            tokenizer=tokenizer,
            max_length=max_length,
            max_samples=max_samples,
            mask_generator=mask_generator,
        )

    elif dataset_name == 'flickr30k' or dataset_name == 'flickr':
        return Flickr30kDataset(
            data_root=data_root,
            split=split,
            transform=transform,
            tokenizer=tokenizer,
            max_length=max_length,
            max_samples=max_samples,
            mask_generator=mask_generator,
        )

    elif dataset_name == 'dummy' or dataset_name == 'test':
        return DummyDataset(
            num_samples=max_samples or 1000,
            tokenizer=tokenizer,
            max_length=max_length,
            transform=transform,
            mask_generator=mask_generator,
        )
    
    elif dataset_name == 'custom' or dataset_name == 'generic':
        images_dir = data_config.get('images_dir', data_root)
        annotations_file = data_config.get('annotations_file')
        
        if annotations_file is None:
            raise ValueError("'annotations_file' required for custom dataset")
        
        return ImageTextDataset(
            images_dir=images_dir,
            annotations_file=annotations_file,
            transform=transform,
            tokenizer=tokenizer,
            max_length=max_length,
        )
    
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            "Supported: coco_captions, flickr30k, dummy, custom"
        )


if __name__ == "__main__":
    from transformers import AutoTokenizer
    print("Testing dataset implementations...")

    # Test dummy dataset
    print("\n1. Testing DummyDataset...")
    dummy_ds = DummyDataset(num_samples=10)
    sample = dummy_ds[0]
    print(f"   Image shape: {sample['image'].shape}")
    print(f"   Caption: {sample['caption']}")

    # Test with tokenizer
    print("\n2. Testing DummyDataset with tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        dummy_ds_tok = DummyDataset(num_samples=10, tokenizer=tokenizer)
        sample = dummy_ds_tok[0]
        print(f"   Input IDs shape: {sample['input_ids'].shape}")
        print(f"   Attention mask shape: {sample['attention_mask'].shape}")
    except Exception as e:
        print(f"   Skipping tokenizer test: {e}")
    
    # Test create_dataset factory
    print("\n3. Testing create_dataset factory...")
    config = {
        'data': {
            'dataset_name': 'dummy',
            'max_samples': 100,
        },
        'model': {
            'text_encoder': {
                'max_length': 64,
            }
        }
    }
    ds = create_dataset(config, split='train')
    print(f"   Dataset length: {len(ds)}")
    
    print("\nDataset tests passed!")
