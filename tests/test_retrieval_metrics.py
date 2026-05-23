"""
Phase 2 verification: retrieval R@K handles the COCO 5-captions-per-image case.

The pre-Phase-2 implementation matched only the diagonal of the similarity
matrix, which gives wrong numbers when an image has multiple captions or
the lists aren't 1-to-1 aligned.
"""

import torch

from vl_jepa.utils.metrics import compute_retrieval_metrics


def test_legacy_diagonal_mode_still_works():
    """Without ids: behave like the old 1-to-1 diagonal logic."""
    torch.manual_seed(0)
    N, D = 50, 32
    embeds = torch.randn(N, D)
    metrics = compute_retrieval_metrics(embeds, embeds, topk=(1, 5))
    # Identical embeddings -> perfect retrieval.
    assert metrics['i2t_recall@1'] == 100.0
    assert metrics['t2i_recall@1'] == 100.0


def test_multi_caption_recall():
    """
    5 images, 5 captions each. The j-th caption of image i is the image's
    own embedding plus small noise, so the diagonal-only check would
    declare *every off-diagonal pair* incorrect even when the retrieved
    caption is one of the 5 valid positives.
    """
    torch.manual_seed(0)
    n_images = 5
    captions_per_image = 5
    D = 32

    image_embeds = torch.randn(n_images, D)
    image_ids = torch.arange(n_images)

    text_embeds = []
    text_image_ids = []
    for img_idx in range(n_images):
        for _ in range(captions_per_image):
            # Each caption is the image embedding plus tiny noise, so the
            # whole group of 5 captions retrieves the same image.
            text_embeds.append(image_embeds[img_idx] + 0.01 * torch.randn(D))
            text_image_ids.append(img_idx)
    text_embeds = torch.stack(text_embeds)
    text_image_ids = torch.tensor(text_image_ids)

    metrics = compute_retrieval_metrics(
        image_embeds, text_embeds,
        topk=(1, 5),
        image_ids=image_ids,
        text_image_ids=text_image_ids,
    )

    # All 5 captions of an image are close to it. So:
    #   i2t R@1: image -> any of its 5 captions in top-1. Should be ~100%.
    #   t2i R@1: caption -> its source image in top-1 of 5 candidate images.
    #   Should be ~100% because the caption embedding sits next to its image.
    assert metrics['i2t_recall@1'] == 100.0, metrics
    assert metrics['t2i_recall@1'] == 100.0, metrics
    # Mean recall must average all 4 stats.
    assert 99.0 <= metrics['mean_recall'] <= 100.0, metrics


def test_duplicate_image_gallery_depresses_t2i_recall_at_k():
    """
    Lock-in for the validate() dedupe step. The dataset emits one sample per
    (image, caption) pair, so 5 captions per image produce 5 identical image
    embeddings. If the dedupe is skipped, those 5 ties fill 5 top-K slots in
    t2i and t2i_recall@5 collapses to ~t2i_recall@1.

    This test reproduces that pathology with a fixed similarity layout where
    a single wrong image is the second-closest to every text query. With
    duplicates left in, that wrong image's 5 copies sit in t2i@5 and the
    correct (third-closest) image never makes it into top-5. After dedupe,
    each image contributes one row and the correct image lands in top-5.
    """
    import numpy as np

    D = 16
    n_unique = 8
    captions_per_image = 5
    unique_ids = list(range(100, 100 + n_unique))
    base = torch.eye(n_unique, D)  # well-separated unit vectors

    # Duplicated gallery: captions_per_image copies of each image.
    gallery_rows = []
    all_image_ids = []
    for i, iid in enumerate(unique_ids):
        for _ in range(captions_per_image):
            gallery_rows.append(base[i].clone())
            all_image_ids.append(iid)
    image_embeds_dup = torch.stack(gallery_rows)  # [40, D]

    # One text per image. Each text is most similar to a WRONG image and
    # second-most to its CORRECT one. After dedup, the correct image ranks
    # 2nd in t2i, so it lands in top-5. Without dedup, the 5 duplicates of
    # the wrong image fill all 5 top slots and the correct image's copies
    # sit at rank 6-10, outside top-5.
    text_embeds = torch.zeros(n_unique, D)
    text_image_ids_list = []
    for i in range(n_unique):
        wrong = (i + 1) % n_unique
        text_embeds[i] = 0.9 * base[wrong] + 0.4 * base[i]
        text_image_ids_list.append(unique_ids[i])
    text_image_ids = torch.tensor(text_image_ids_list)

    image_ids_dup = torch.tensor(all_image_ids)

    # Pathological path: leave duplicates in. t2i@5 should be ~0.
    bad = compute_retrieval_metrics(
        image_embeds_dup, text_embeds,
        topk=(1, 5),
        image_ids=image_ids_dup,
        text_image_ids=text_image_ids,
    )
    assert bad['t2i_recall@5'] < 50.0, (
        f"Expected duplicate-gallery t2i_recall@5 to be depressed, got {bad}"
    )

    # Apply the validate() dedupe step.
    ids_np = np.asarray(all_image_ids)
    uniq, first_idx = np.unique(ids_np, return_index=True)
    image_embeds_dedup = image_embeds_dup[torch.from_numpy(first_idx).long()]
    image_ids_dedup = torch.from_numpy(uniq).long()

    good = compute_retrieval_metrics(
        image_embeds_dedup, text_embeds,
        topk=(1, 5),
        image_ids=image_ids_dedup,
        text_image_ids=text_image_ids,
    )
    # After dedup, the correct image is the 2nd-closest unique image, so it
    # is solidly inside top-5.
    assert good['t2i_recall@5'] == 100.0, good
    # And the dedupe must improve substantially over the broken path.
    assert good['t2i_recall@5'] - bad['t2i_recall@5'] >= 50.0, (good, bad)
