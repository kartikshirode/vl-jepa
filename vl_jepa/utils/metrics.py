"""
Evaluation metrics for image-text retrieval
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


def compute_retrieval_metrics(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    topk: Tuple[int, ...] = (1, 5, 10),
    image_ids: torch.Tensor = None,
    text_image_ids: torch.Tensor = None,
) -> Dict[str, float]:
    """
    Image-text retrieval R@K with proper multi-caption handling.

    COCO has 5 captions per image, so the diagonal-only "correct"
    assumption produces wrong numbers. When `text_image_ids` (length
    `len(text_embeds)`, integer ids) and `image_ids` (length
    `len(image_embeds)`, integer ids) are provided, a text query is
    considered correct if the top-K images include the image whose id
    matches text_image_ids[q]; an image query is correct if the top-K
    texts include any text whose text_image_ids matches image_ids[q].

    When both id tensors are None, the legacy 1-to-1 diagonal behavior
    is used. Pass ids on COCO; skip them only on toy / synthetic data.
    """
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    sim_matrix = image_embeds @ text_embeds.t()  # [N_img, N_txt]
    N_img, N_txt = sim_matrix.shape

    metrics: Dict[str, float] = {}

    if image_ids is None or text_image_ids is None:
        if N_img != N_txt:
            raise ValueError(
                "When image_ids / text_image_ids aren't provided, "
                "image_embeds and text_embeds must be the same length."
            )
        # Legacy 1-to-1 diagonal mode.
        for k in topk:
            _, idx = sim_matrix.topk(k, dim=1)
            correct = torch.arange(N_img, device=sim_matrix.device).unsqueeze(1)
            metrics[f'i2t_recall@{k}'] = (idx == correct).any(dim=1).float().mean().item() * 100.0
        for k in topk:
            _, idx = sim_matrix.t().topk(k, dim=1)
            correct = torch.arange(N_txt, device=sim_matrix.device).unsqueeze(1)
            metrics[f't2i_recall@{k}'] = (idx == correct).any(dim=1).float().mean().item() * 100.0
    else:
        image_ids = image_ids.to(sim_matrix.device)
        text_image_ids = text_image_ids.to(sim_matrix.device)

        # i2t: for each image, are any of its captions in top-K texts?
        for k in topk:
            _, top_txt = sim_matrix.topk(k, dim=1)  # [N_img, k]
            retrieved_owners = text_image_ids[top_txt]  # [N_img, k]
            hits = (retrieved_owners == image_ids.unsqueeze(1)).any(dim=1)
            metrics[f'i2t_recall@{k}'] = hits.float().mean().item() * 100.0

        # t2i: for each text, is its source image in top-K images?
        sim_t = sim_matrix.t()  # [N_txt, N_img]
        for k in topk:
            _, top_img = sim_t.topk(k, dim=1)  # [N_txt, k]
            retrieved_ids = image_ids[top_img]  # [N_txt, k]
            hits = (retrieved_ids == text_image_ids.unsqueeze(1)).any(dim=1)
            metrics[f't2i_recall@{k}'] = hits.float().mean().item() * 100.0

    metrics['mean_recall'] = float(np.mean([v for k, v in metrics.items() if 'recall@' in k]))
    return metrics


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> Dict[str, float]:
    """
    Compute top-k accuracy.
    
    Args:
        predictions: Prediction logits [N, C]
        targets: Ground truth labels [N]
        topk: Top-K values
        
    Returns:
        Dictionary with accuracy metrics
    """
    maxk = max(topk)
    batch_size = targets.size(0)
    
    # Get top-k predictions
    _, pred_topk = predictions.topk(maxk, dim=1, largest=True, sorted=True)
    pred_topk = pred_topk.t()  # [maxk, N]
    
    # Check correctness
    correct = pred_topk.eq(targets.view(1, -1).expand_as(pred_topk))
    
    metrics = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size).item()
        metrics[f'acc@{k}'] = acc
    
    return metrics


class AverageMeter:
    """
    Computes and stores the average and current value.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    print("Testing retrieval metrics...")
    
    # Create dummy embeddings
    N = 100
    D = 256
    
    image_embeds = torch.randn(N, D)
    text_embeds = torch.randn(N, D)
    
    # Compute metrics
    metrics = compute_retrieval_metrics(image_embeds, text_embeds)
    
    print("Retrieval metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2f}")
    
    # Test accuracy
    predictions = torch.randn(N, 10)
    targets = torch.randint(0, 10, (N,))
    
    acc_metrics = compute_accuracy(predictions, targets)
    
    print("\nAccuracy metrics:")
    for k, v in acc_metrics.items():
        print(f"  {k}: {v:.2f}")
    
    # Test average meter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"\nAverage meter: avg={meter.avg:.2f}, sum={meter.sum:.2f}, count={meter.count}")
    
    print("\nMetrics test passed!")
