"""
Inference script for VL-JEPA model
Use trained model for image-text retrieval and similarity computation
"""

import torch
import torch.nn.functional as F
from PIL import Image
import argparse
from pathlib import Path
from typing import List, Tuple

from vl_jepa.models.vl_jepa import create_vl_jepa_model
from vl_jepa.data.transforms import get_val_transforms
from vl_jepa.utils.config import load_config
from vl_jepa.utils.checkpoint import load_checkpoint


class VLJEPAInference:
    """
    Inference wrapper for VL-JEPA model.
    """
    
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = 'cuda',
    ):
        """
        Initialize inference model.
        
        Args:
            config_path: Path to config file
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
        """
        self.device = device
        
        # Load config
        self.config = load_config(config_path)
        
        # Create model
        print("Loading model...")
        self.model = create_vl_jepa_model(self.config)
        
        # Load checkpoint
        load_checkpoint(
            checkpoint_path,
            self.model,
            device=device,
        )
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # Get transforms and tokenizer
        self.transform = get_val_transforms(self.config['data'])
        self.tokenizer = self.model.text_encoder.tokenizer
        
        print("Model loaded successfully!")
    
    @torch.no_grad()
    def encode_image(self, image_path: str) -> torch.Tensor:
        """
        Encode image to embedding.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image embedding [D]
        """
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Encode image
        vision_features = self.model.vision_encoder(image_tensor, return_all_tokens=False)
        vision_features = vision_features.squeeze(1)  # Remove sequence dim
        
        # Project to shared space
        vision_embed = self.model.vision_projection(vision_features)
        vision_embed = F.normalize(vision_embed, dim=-1)
        
        return vision_embed.squeeze(0).cpu()
    
    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text to embedding.
        
        Args:
            text: Text string
            
        Returns:
            Text embedding [D]
        """
        # Tokenize text
        tokens = self.tokenizer(
            [text],
            padding='max_length',
            truncation=True,
            max_length=self.config['model']['text_encoder']['max_length'],
            return_tensors='pt',
        )
        
        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)
        
        # Encode text
        text_features = self.model.text_encoder(
            input_ids,
            attention_mask,
            return_all_tokens=False,
            return_projected=False,
        )
        
        # Project to shared space
        text_embed = self.model.text_projection(text_features)
        text_embed = F.normalize(text_embed, dim=-1)
        
        return text_embed.squeeze(0).cpu()
    
    @torch.no_grad()
    def compute_similarity(
        self,
        image_path: str,
        text: str,
    ) -> float:
        """
        Compute similarity between image and text.
        
        Args:
            image_path: Path to image
            text: Text string
            
        Returns:
            Similarity score (0-1)
        """
        image_embed = self.encode_image(image_path)
        text_embed = self.encode_text(text)
        
        similarity = torch.dot(image_embed, text_embed).item()
        
        return similarity
    
    @torch.no_grad()
    def find_best_text(
        self,
        image_path: str,
        texts: List[str],
    ) -> Tuple[str, float]:
        """
        Find best matching text for an image.
        
        Args:
            image_path: Path to image
            texts: List of candidate texts
            
        Returns:
            (best_text, similarity_score)
        """
        image_embed = self.encode_image(image_path)
        
        best_text = None
        best_score = -1.0
        
        for text in texts:
            text_embed = self.encode_text(text)
            similarity = torch.dot(image_embed, text_embed).item()
            
            if similarity > best_score:
                best_score = similarity
                best_text = text
        
        return best_text, best_score
    
    @torch.no_grad()
    def find_best_image(
        self,
        text: str,
        image_paths: List[str],
    ) -> Tuple[str, float]:
        """
        Find best matching image for a text.
        
        Args:
            text: Text query
            image_paths: List of candidate image paths
            
        Returns:
            (best_image_path, similarity_score)
        """
        text_embed = self.encode_text(text)
        
        best_image = None
        best_score = -1.0
        
        for image_path in image_paths:
            image_embed = self.encode_image(image_path)
            similarity = torch.dot(image_embed, text_embed).item()
            
            if similarity > best_score:
                best_score = similarity
                best_image = image_path
        
        return best_image, best_score


def main():
    parser = argparse.ArgumentParser(description="VL-JEPA Inference")
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--image", type=str, help="Path to image")
    parser.add_argument("--text", type=str, help="Text query")
    parser.add_argument("--mode", type=str, default="similarity", 
                       choices=["similarity", "image2text", "text2image"],
                       help="Inference mode")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    args = parser.parse_args()
    
    # Initialize model
    model = VLJEPAInference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    
    if args.mode == "similarity":
        # Compute image-text similarity
        if not args.image or not args.text:
            print("Error: --image and --text required for similarity mode")
            return
        
        similarity = model.compute_similarity(args.image, args.text)
        print(f"\nImage: {args.image}")
        print(f"Text: {args.text}")
        print(f"Similarity: {similarity:.4f}")
    
    elif args.mode == "image2text":
        # Find best text for image
        if not args.image:
            print("Error: --image required for image2text mode")
            return
        
        # Example texts (you can modify this)
        texts = [
            "A dog playing in the park",
            "A cat sitting on a chair",
            "A person riding a bicycle",
            "A beautiful sunset over the ocean",
            "A car driving on the highway",
        ]
        
        best_text, score = model.find_best_text(args.image, texts)
        
        print(f"\nImage: {args.image}")
        print(f"Best matching text: {best_text}")
        print(f"Similarity: {score:.4f}")
        print("\nAll candidates:")
        for text in texts:
            sim = model.compute_similarity(args.image, text)
            marker = "→" if text == best_text else " "
            print(f"  {marker} {sim:.4f} | {text}")
    
    elif args.mode == "text2image":
        # Find best image for text
        if not args.text:
            print("Error: --text required for text2image mode")
            return
        
        # Example: search in a directory
        image_dir = Path("data/images/val2017")
        if not image_dir.exists():
            print(f"Error: Image directory not found: {image_dir}")
            print("Please provide a directory with images")
            return
        
        # Get first 100 images
        image_paths = list(image_dir.glob("*.jpg"))[:100]
        
        if not image_paths:
            print(f"No images found in {image_dir}")
            return
        
        print(f"Searching {len(image_paths)} images...")
        best_image, score = model.find_best_image(args.text, [str(p) for p in image_paths])
        
        print(f"\nText query: {args.text}")
        print(f"Best matching image: {best_image}")
        print(f"Similarity: {score:.4f}")


if __name__ == "__main__":
    main()
