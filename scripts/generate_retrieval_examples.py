"""
Generate a visual retrieval-examples grid using the trained VL-JEPA model.

For the README headline figure: shows the model doing actual image<->text
retrieval on the COCO 5K validation split. Two halves:
  - text-to-image: a few captions, each with its top-5 retrieved images
  - image-to-text: a few images, each with its top-5 retrieved captions

Both halves use the same model and the same val gallery.

Usage:
    python scripts/generate_retrieval_examples.py \
        --checkpoint kaggle_outputs_v13/checkpoints/checkpoint_epoch_9.pth \
        --config configs/config_kaggle_t4.yaml \
        --data-root vl_jepa/data/COCO2017 \
        --out docs/retrieval_examples.png
"""
import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vl_jepa.models.vl_jepa import create_vl_jepa_model
from vl_jepa.data.transforms import get_val_transforms
from vl_jepa.utils.config import load_config
from vl_jepa.utils.checkpoint import load_checkpoint


# Pick a handful of query captions and a handful of query images for the grid.
# These are chosen to span a variety of scenes (people, animals, vehicles,
# food, sport) so the figure reads well without having to scan many rows.
QUERY_CAPTIONS = [
    "A red double-decker bus driving on a city street",
    "A small dog sitting on a wooden bench",
    "A person riding a surfboard on a wave",
    "A plate of pasta with tomato sauce on a table",
    "Two children playing soccer in a green field",
]

# Image ids picked from COCO val to span similar variety.
QUERY_IMAGE_IDS = [
    397133,   # a baseball player
    37777,    # a giraffe scene
    252219,   # a kitchen / food
    87038,    # a dog
    174482,   # a snowboard / outdoor
]


def load_inference(config_path: str, checkpoint_path: str, device: str):
    config = load_config(config_path)
    model = create_vl_jepa_model(config)
    load_checkpoint(checkpoint_path, model, device=device)
    model = model.to(device).eval()
    transform = get_val_transforms(config["data"])
    tokenizer = model.text_encoder.tokenizer
    max_len = config["model"]["text_encoder"]["max_length"]
    return model, transform, tokenizer, max_len


@torch.no_grad()
def encode_images(model, transform, image_paths, device, batch_size=64):
    """Return [N, D] L2-normalized vision embeddings."""
    all_embeds = []
    for start in range(0, len(image_paths), batch_size):
        chunk = image_paths[start:start + batch_size]
        tensors = []
        for p in chunk:
            img = Image.open(p).convert("RGB")
            tensors.append(transform(img))
        batch = torch.stack(tensors, dim=0).to(device)
        feats = model.vision_encoder(batch, return_all_tokens=False).squeeze(1)
        proj = model.vision_projection(feats)
        all_embeds.append(F.normalize(proj, dim=-1).cpu())
        if (start // batch_size) % 10 == 0:
            print(f"  encoded {start + len(chunk)}/{len(image_paths)} images", flush=True)
    return torch.cat(all_embeds, dim=0)


@torch.no_grad()
def encode_texts(model, tokenizer, texts, device, max_len, batch_size=256):
    """Return [N, D] L2-normalized text embeddings."""
    all_embeds = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        tokens = tokenizer(
            chunk, padding="max_length", truncation=True,
            max_length=max_len, return_tensors="pt",
        )
        ids = tokens["input_ids"].to(device)
        mask = tokens["attention_mask"].to(device)
        feats = model.text_encoder(ids, mask, return_all_tokens=False, return_projected=False)
        proj = model.text_projection(feats)
        all_embeds.append(F.normalize(proj, dim=-1).cpu())
    return torch.cat(all_embeds, dim=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--data-root", required=True, help="COCO 2017 root with val2017/ and annotations/")
    p.add_argument("--out", default="docs/retrieval_examples.png")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-images", type=int, default=2000, help="Limit val gallery for speed")
    args = p.parse_args()

    data_root = Path(args.data_root)
    val_dir = data_root / "val2017"
    ann_path = data_root / "annotations" / "captions_val2017.json"
    if not val_dir.exists() or not ann_path.exists():
        sys.exit(f"COCO val not found at {data_root}")

    print(f"Loading model on {args.device}...", flush=True)
    model, transform, tokenizer, max_len = load_inference(args.config, args.checkpoint, args.device)

    print(f"Loading COCO val annotations from {ann_path}...", flush=True)
    with open(ann_path, "r", encoding="utf-8") as f:
        anns = json.load(f)
    image_records = {img["id"]: img["file_name"] for img in anns["images"]}
    # One caption per image_id, picked deterministically (first encountered).
    captions_by_image = {}
    for c in anns["annotations"]:
        captions_by_image.setdefault(c["image_id"], c["caption"])

    # Trim the gallery for tractable encoding time. Use the first
    # `max_images` ids; deterministic given the JSON order.
    image_ids = list(image_records.keys())[: args.max_images]
    image_paths = [str(val_dir / image_records[i]) for i in image_ids]
    captions = [captions_by_image[i] for i in image_ids if i in captions_by_image]
    # Align image_ids with captions index.
    image_ids_with_caps = [i for i in image_ids if i in captions_by_image]
    image_paths_with_caps = [str(val_dir / image_records[i]) for i in image_ids_with_caps]

    print(f"Indexing {len(image_paths_with_caps)} images + {len(captions)} captions from val...", flush=True)
    img_embeds = encode_images(model, transform, image_paths_with_caps, args.device)
    cap_embeds = encode_texts(model, tokenizer, captions, args.device, max_len)
    print(f"  image embeddings: {tuple(img_embeds.shape)}")
    print(f"  caption embeddings: {tuple(cap_embeds.shape)}")

    # --- text -> image queries ---
    print("\nText -> Image retrieval:")
    t2i_results = []  # (caption, [top5 image paths], [top5 scores])
    for cap in QUERY_CAPTIONS:
        emb = encode_texts(model, tokenizer, [cap], args.device, max_len)[0]
        scores = (img_embeds @ emb).numpy()
        top5 = np.argsort(scores)[::-1][:5]
        paths = [image_paths_with_caps[i] for i in top5]
        scs = [float(scores[i]) for i in top5]
        t2i_results.append((cap, paths, scs))
        print(f"  '{cap[:60]}...' -> top1 score {scs[0]:.3f}")

    # --- image -> text queries ---
    print("\nImage -> Text retrieval:")
    i2t_results = []  # (image_path, [top5 captions], [top5 scores])
    for qid in QUERY_IMAGE_IDS:
        if qid not in image_records:
            print(f"  image_id {qid} not in val set; skipping")
            continue
        qpath = str(val_dir / image_records[qid])
        qemb = encode_images(model, transform, [qpath], args.device)[0]
        scores = (cap_embeds @ qemb).numpy()
        top5 = np.argsort(scores)[::-1][:5]
        caps_top = [captions[i] for i in top5]
        scs = [float(scores[i]) for i in top5]
        i2t_results.append((qpath, caps_top, scs))
        print(f"  image {qid} -> top1 score {scs[0]:.3f}: '{caps_top[0][:60]}...'")

    # --- render the figure ---
    n_t2i = len(t2i_results)
    n_i2t = len(i2t_results)
    n_rows = n_t2i + n_i2t
    fig = plt.figure(figsize=(16, 3.2 * n_rows))
    plt.suptitle(
        "VL-JEPA retrieval on COCO 5K val (ViT-Tiny + DistilBERT, 10 epochs)\n"
        f"Final metrics: i2t_recall@1 = 50.30%, mean_recall = 68.04%",
        fontsize=13, y=0.995,
    )

    # text->image rows
    for row_idx, (cap, paths, scs) in enumerate(t2i_results):
        # Left: caption text
        ax_cap = plt.subplot2grid((n_rows, 7), (row_idx, 0), colspan=2)
        ax_cap.axis("off")
        ax_cap.text(
            0.02, 0.5, f"TEXT QUERY:\n\n\"{cap}\"",
            wrap=True, fontsize=10, va="center", ha="left",
            transform=ax_cap.transAxes,
        )
        # Right: 5 retrieved images
        for k, (p, s) in enumerate(zip(paths, scs)):
            ax = plt.subplot2grid((n_rows, 7), (row_idx, 2 + k))
            img = Image.open(p).convert("RGB")
            ax.imshow(img)
            ax.set_title(f"#{k+1}  sim={s:.3f}", fontsize=9)
            ax.axis("off")

    # image->text rows
    for row_idx, (qpath, caps_top, scs) in enumerate(i2t_results):
        row = n_t2i + row_idx
        ax_img = plt.subplot2grid((n_rows, 7), (row, 0), colspan=2)
        img = Image.open(qpath).convert("RGB")
        ax_img.imshow(img)
        ax_img.set_title("IMAGE QUERY", fontsize=10)
        ax_img.axis("off")
        # Captions list
        ax_caps = plt.subplot2grid((n_rows, 7), (row, 2), colspan=5)
        ax_caps.axis("off")
        lines = [f"#{k+1}  sim={s:.3f}  {c}" for k, (c, s) in enumerate(zip(caps_top, scs))]
        ax_caps.text(
            0.02, 0.5, "\n".join(lines),
            fontsize=10, va="center", ha="left",
            transform=ax_caps.transAxes, family="monospace",
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
