#!/usr/bin/env python3
"""
COCO Karpathy-split retrieval evaluation for a trained VL-JEPA checkpoint.

Karpathy splits are the standard for reporting image-text retrieval on
COCO Captions. Each image in val/test has 5 captions; a text query is
correct if its source image is in the top-K, and an image query is
correct if any of its 5 captions is in the top-K. This script applies
those semantics via the multi-caption R@K path in
vl_jepa.utils.metrics.compute_retrieval_metrics.

Usage:
    python scripts/eval_coco_karpathy.py \
        --config config_dgpu.yaml \
        --checkpoint checkpoints/best_model.pth \
        --karpathy dataset_coco.json \
        --images_root /path/to/coco/images \
        --split test \
        --out runs/coco_karpathy.json
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from vl_jepa.models.vl_jepa import create_vl_jepa_model
from vl_jepa.data.transforms import get_val_transforms
from vl_jepa.utils.config import load_config
from vl_jepa.utils.checkpoint import load_checkpoint
from vl_jepa.utils.metrics import compute_retrieval_metrics


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--karpathy", required=True, help="dataset_coco.json from Karpathy splits")
    parser.add_argument("--images_root", required=True, help="root with train2014/val2014 etc.")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--out", default="runs/coco_karpathy.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu'

    with open(args.karpathy, 'r', encoding='utf-8') as f:
        data = json.load(f)
    items = [x for x in data['images'] if x['split'] == args.split]

    config = load_config(args.config)
    model = create_vl_jepa_model(config).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    transform = get_val_transforms(config['data'])
    tokenizer = model.text_encoder.tokenizer
    max_length = config['model']['text_encoder']['max_length']

    # Build per-image embeddings.
    image_paths = []
    image_ids = []
    for it in items:
        # Karpathy COCO entries use filename like 'COCO_val2014_000000037986.jpg'
        # and a filepath relative to images_root.
        image_paths.append(Path(args.images_root) / it['filepath'] / it['filename'])
        image_ids.append(it['imgid'])

    image_embeds = []
    for start in tqdm(range(0, len(image_paths), args.batch_size), desc="images"):
        chunk = image_paths[start:start + args.batch_size]
        batch = torch.stack([transform(Image.open(p).convert('RGB')) for p in chunk]).to(device)
        v = model.vision_encoder(batch, return_all_tokens=False).squeeze(1)
        v = F.normalize(model.vision_projection(v), dim=-1)
        image_embeds.append(v.cpu())
    image_embeds = torch.cat(image_embeds, dim=0)
    image_ids_t = torch.tensor(image_ids)

    # Build per-caption embeddings.
    text_embeds = []
    text_image_ids = []
    for it in tqdm(items, desc="captions"):
        captions = [s['raw'] for s in it['sentences']]
        toks = tokenizer(captions, padding='max_length', truncation=True,
                         max_length=max_length, return_tensors='pt')
        toks = {k: v.to(device) for k, v in toks.items()}
        t = model.text_encoder(toks['input_ids'], toks['attention_mask'],
                               return_all_tokens=False, return_projected=False)
        t = F.normalize(model.text_projection(t), dim=-1)
        text_embeds.append(t.cpu())
        text_image_ids.extend([it['imgid']] * len(captions))
    text_embeds = torch.cat(text_embeds, dim=0)
    text_image_ids_t = torch.tensor(text_image_ids)

    metrics = compute_retrieval_metrics(
        image_embeds, text_embeds,
        topk=(1, 5, 10),
        image_ids=image_ids_t,
        text_image_ids=text_image_ids_t,
    )

    out = {
        'split': args.split,
        'num_images': image_embeds.shape[0],
        'num_captions': text_embeds.shape[0],
        **metrics,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
