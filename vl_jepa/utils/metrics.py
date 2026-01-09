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
) -> Dict[str, float]:
    """
    Compute image-text retrieval metrics.
    
    Args:
        image_embeds: Image embeddings [N, D]
        text_embeds: Text embeddings [N, D]
        topk: Top-K values to compute recall at
        
    Returns:
        Dictionary with retrieval metrics
    """
    # Normalize embeddings
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    
    # Compute similarity matrix
    sim_matrix = torch.matmul(image_embeds, text_embeds.t())  # [N, N]
    
    N = sim_matrix.shape[0]
    
    metrics = {}
    
    # Image to text retrieval
    for k in topk:
        # Get top-k text indices for each image
        _, topk_indices = sim_matrix.topk(k, dim=1)  # [N, k]
        
        # Check if correct text is in top-k
        correct = torch.arange(N, device=sim_matrix.device).unsqueeze(1)  # [N, 1]
        recall = (topk_indices == correct).any(dim=1).float().mean().item()
        
        metrics[f'i2t_recall@{k}'] = recall * 100.0
    
    # Text to image retrieval
    for k in topk:
        # Get top-k image indices for each text
        _, topk_indices = sim_matrix.t().topk(k, dim=1)  # [N, k]
        
        # Check if correct image is in top-k
        correct = torch.arange(N, device=sim_matrix.device).unsqueeze(1)  # [N, 1]
        recall = (topk_indices == correct).any(dim=1).float().mean().item()
        
        metrics[f't2i_recall@{k}'] = recall * 100.0
    
    # Mean recall
    mean_recall = np.mean([v for k, v in metrics.items() if 'recall@' in k])
    metrics['mean_recall'] = mean_recall
    
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
