#!/usr/bin/env python3
"""
Zero-shot ImageNet evaluation for a trained VL-JEPA checkpoint.

Builds text embeddings for the standard 1000 ImageNet class templates,
encodes a directory of validation images, then reports top-1 and top-5
accuracy. Writes a JSON summary to the run directory.

Usage:
    python scripts/eval_zeroshot_imagenet.py \
        --config config_dgpu.yaml \
        --checkpoint checkpoints/best_model.pth \
        --imagenet_val /path/to/imagenet/val \
        --out runs/zeroshot_imagenet.json

The class index -> name map can be a JSON {"0": "tench", ...} or one of
the standard imagenet_classes.txt files (one class per line).
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from vl_jepa.models.vl_jepa import create_vl_jepa_model
from vl_jepa.data.transforms import get_val_transforms
from vl_jepa.utils.config import load_config
from vl_jepa.utils.checkpoint import load_checkpoint


# Subset of OpenAI CLIP's 80 ImageNet templates. Adding more usually helps
# noticeably; this list is kept short so the script ships small.
TEMPLATES = [
    "a photo of a {}.",
    "a bad photo of a {}.",
    "a low resolution photo of a {}.",
    "a photo of the {}.",
    "a cropped photo of a {}.",
    "a close-up photo of a {}.",
    "a bright photo of a {}.",
    "a dark photo of a {}.",
]


def load_class_names(path: Path):
    if path.suffix.lower() == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        return [mapping[str(i)] for i in range(len(mapping))]
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


@torch.no_grad()
def build_text_classifier(model, tokenizer, class_names, device, max_length):
    """Average text embeddings across templates for each class. [num_classes, D]."""
    embeds = []
    for name in tqdm(class_names, desc="text classifier"):
        prompts = [tpl.format(name) for tpl in TEMPLATES]
        toks = tokenizer(prompts, padding='max_length', truncation=True,
                         max_length=max_length, return_tensors='pt')
        toks = {k: v.to(device) for k, v in toks.items()}
        text_features = model.text_encoder(toks['input_ids'], toks['attention_mask'],
                                            return_all_tokens=False, return_projected=False)
        text_embed = F.normalize(model.text_projection(text_features), dim=-1)
        embeds.append(text_embed.mean(dim=0))
    return torch.stack(embeds, dim=0)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--imagenet_val", required=True, help="Root with class subdirs")
    parser.add_argument("--class_names", default=None, help="JSON or txt with class names")
    parser.add_argument("--out", default="runs/zeroshot_imagenet.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu'
    config = load_config(args.config)
    model = create_vl_jepa_model(config).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    transform = get_val_transforms(config['data'])
    tokenizer = model.text_encoder.tokenizer
    max_length = config['model']['text_encoder']['max_length']

    val_root = Path(args.imagenet_val)
    class_dirs = sorted([d for d in val_root.iterdir() if d.is_dir()])
    if args.class_names:
        class_names = load_class_names(Path(args.class_names))
    else:
        class_names = [d.name for d in class_dirs]
    if len(class_names) != len(class_dirs):
        sys.exit(f"Class count mismatch: {len(class_names)} names vs {len(class_dirs)} dirs")

    text_classifier = build_text_classifier(model, tokenizer, class_names, device, max_length)

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for label, cls_dir in enumerate(tqdm(class_dirs, desc="val images")):
        paths = sorted([p for p in cls_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
        for start in range(0, len(paths), args.batch_size):
            chunk = paths[start:start + args.batch_size]
            batch = torch.stack([transform(Image.open(p).convert('RGB')) for p in chunk]).to(device)
            v = model.vision_encoder(batch, return_all_tokens=False).squeeze(1)
            v = F.normalize(model.vision_projection(v), dim=-1)
            scores = v @ text_classifier.t()  # [B, num_classes]
            top1 = scores.topk(1, dim=1).indices.squeeze(1)
            top5 = scores.topk(5, dim=1).indices
            correct_top1 += (top1 == label).sum().item()
            correct_top5 += (top5 == label).any(dim=1).sum().item()
            total += batch.shape[0]

    out = {
        'num_classes': len(class_names),
        'num_images': total,
        'top1': 100.0 * correct_top1 / max(1, total),
        'top5': 100.0 * correct_top5 / max(1, total),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
