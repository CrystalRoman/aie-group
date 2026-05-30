from __future__ import annotations

import torch


def threshold_logits(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return (probs > threshold).float()


def dice_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7) -> float:
    preds = threshold_logits(logits, threshold=threshold)
    targets = targets.float()

    dims = (1, 2, 3)
    intersection = (preds * targets).sum(dim=dims)
    union = preds.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2 * intersection + eps) / (union + eps)
    return float(dice.mean().item())


def iou_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7) -> float:
    preds = threshold_logits(logits, threshold=threshold)
    targets = targets.float()

    dims = (1, 2, 3)
    intersection = (preds * targets).sum(dim=dims)
    union = preds.sum(dim=dims) + targets.sum(dim=dims) - intersection
    iou = (intersection + eps) / (union + eps)
    return float(iou.mean().item())
