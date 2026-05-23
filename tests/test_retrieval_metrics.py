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
