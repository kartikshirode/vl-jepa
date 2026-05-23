"""
Image transforms and augmentations for VL-JEPA training.
Optimized for memory efficiency on Jetson devices.
"""

import torch
from torchvision import transforms
from typing import Dict, Any, Optional, Tuple
import random
from PIL import Image, ImageFilter, ImageOps


class GaussianBlur:
    """Gaussian blur augmentation with random sigma."""
    
    def __init__(self, sigma: Tuple[float, float] = (0.1, 2.0)):
        self.sigma = sigma
    
    def __call__(self, x: Image.Image) -> Image.Image:
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))


class Solarization:
    """Solarization augmentation."""
    
    def __init__(self, threshold: int = 128):
        self.threshold = threshold
    
    def __call__(self, x: Image.Image) -> Image.Image:
        return ImageOps.solarize(x, threshold=self.threshold)


class RandomGrayscale:
    """Convert image to grayscale with probability p."""
    
    def __init__(self, p: float = 0.2):
        self.p = p
    
    def __call__(self, x: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return x.convert('L').convert('RGB')
        return x


def get_train_transforms(config: Dict[str, Any]) -> transforms.Compose:
    """
    Get training image transforms with data augmentation.
    
    Based on JEPA/CLIP training recipes:
    - Random Resized Crop
    - Random Horizontal Flip
    - Color Jitter
    - Gaussian Blur (optional)
    - Normalization
    
    Args:
        config: Data configuration dictionary
        
    Returns:
        Composed transforms
    """
    image_config = config.get('image_transforms', {})
    
    # Get parameters with defaults
    resize = image_config.get('resize', 256)
    crop_size = image_config.get('random_resized_crop', 224)
    scale = tuple(image_config.get('random_crop_scale', [0.2, 1.0]))
    ratio = tuple(image_config.get('random_crop_ratio', [0.75, 1.333]))
    
    # Color jitter parameters
    color_jitter = image_config.get('color_jitter', {})
    brightness = color_jitter.get('brightness', 0.4)
    contrast = color_jitter.get('contrast', 0.4)
    saturation = color_jitter.get('saturation', 0.4)
    hue = color_jitter.get('hue', 0.1)
    color_jitter_prob = color_jitter.get('prob', 0.8)
    
    # Additional augmentations
    horizontal_flip = image_config.get('random_horizontal_flip', 0.5)
    gaussian_blur_prob = image_config.get('gaussian_blur_prob', 0.1)
    grayscale_prob = image_config.get('grayscale_prob', 0.2)
    
    # ImageNet normalization
    mean = image_config.get('normalize_mean', [0.485, 0.456, 0.406])
    std = image_config.get('normalize_std', [0.229, 0.224, 0.225])
    
    # Build transforms list
    transform_list = [
        # Random Resized Crop
        transforms.RandomResizedCrop(
            crop_size,
            scale=scale,
            ratio=ratio,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        # Random horizontal flip
        transforms.RandomHorizontalFlip(p=horizontal_flip),
    ]
    
    # Color jitter with probability
    if color_jitter_prob > 0:
        transform_list.append(
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                )
            ], p=color_jitter_prob)
        )
    
    # Grayscale with probability
    if grayscale_prob > 0:
        transform_list.append(transforms.RandomGrayscale(p=grayscale_prob))
    
    # Gaussian blur with probability
    if gaussian_blur_prob > 0:
        transform_list.append(
            transforms.RandomApply([GaussianBlur(sigma=(0.1, 2.0))], p=gaussian_blur_prob)
        )
    
    # Convert to tensor and normalize
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    return transforms.Compose(transform_list)


def get_val_transforms(config: Dict[str, Any]) -> transforms.Compose:
    """
    Get validation/inference image transforms.
    
    Simple resize and center crop without augmentation.
    
    Args:
        config: Data configuration dictionary
        
    Returns:
        Composed transforms
    """
    image_config = config.get('image_transforms', {})
    
    # Get parameters with defaults
    resize = image_config.get('resize', 256)
    crop_size = image_config.get('random_resized_crop', 224)
    
    # ImageNet normalization
    mean = image_config.get('normalize_mean', [0.485, 0.456, 0.406])
    std = image_config.get('normalize_std', [0.229, 0.224, 0.225])
    
    return transforms.Compose([
        transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_augmentation_transforms(
    strength: str = 'normal',
    image_size: int = 224,
) -> transforms.Compose:
    """
    Get predefined augmentation transforms by strength level.
    
    Args:
        strength: One of 'light', 'normal', 'strong'
        image_size: Target image size
        
    Returns:
        Composed transforms
    """
    # ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if strength == 'light':
        # Light augmentation for fine-tuning
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    
    elif strength == 'normal':
        # Standard augmentation for training
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    
    elif strength == 'strong':
        # Strong augmentation for self-supervised learning
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.5),
            transforms.RandomApply([Solarization()], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    
    else:
        raise ValueError(f"Unknown augmentation strength: {strength}")


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize image tensor for visualization.
    
    Args:
        tensor: Normalized image tensor [C, H, W] or [B, C, H, W]
        
    Returns:
        Denormalized tensor with values in [0, 1]
    """
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    
    if tensor.dim() == 4:
        # Batch of images
        mean = mean.view(1, 3, 1, 1).to(tensor.device)
        std = std.view(1, 3, 1, 1).to(tensor.device)
    else:
        # Single image
        mean = mean.view(3, 1, 1).to(tensor.device)
        std = std.view(3, 1, 1).to(tensor.device)
    
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)


if __name__ == "__main__":
    # Test transforms
    print("Testing image transforms...")
    
    # Create dummy config
    config = {
        'image_transforms': {
            'resize': 256,
            'random_resized_crop': 224,
            'random_crop_scale': [0.2, 1.0],
            'random_horizontal_flip': 0.5,
            'color_jitter': {
                'brightness': 0.4,
                'contrast': 0.4,
                'saturation': 0.4,
                'hue': 0.1,
                'prob': 0.8,
            },
            'gaussian_blur_prob': 0.1,
            'grayscale_prob': 0.2,
        }
    }
    
    # Get transforms
    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms(config)
    
    print(f"Train transforms: {train_transform}")
    print(f"Val transforms: {val_transform}")
    
    # Create a dummy image and apply transforms
    dummy_image = Image.new('RGB', (256, 256), color='red')
    
    train_output = train_transform(dummy_image)
    val_output = val_transform(dummy_image)
    
    print(f"\nTrain output shape: {train_output.shape}")  # [3, 224, 224]
    print(f"Val output shape: {val_output.shape}")  # [3, 224, 224]
    print(f"Train output range: [{train_output.min():.3f}, {train_output.max():.3f}]")
    
    # Test denormalization
    denorm_output = denormalize(train_output)
    print(f"Denormalized range: [{denorm_output.min():.3f}, {denorm_output.max():.3f}]")
    
    print("\nTransforms test passed!")
