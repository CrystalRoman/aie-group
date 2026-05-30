from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)
    loss = 1 - (2 * intersection + eps) / (denom + eps)
    return loss.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * dice_loss(logits, targets)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.8, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        weights = self.alpha * ((1 - pt) ** self.gamma)
        return (weights * bce).mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, eps: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        tp = (probs * targets).sum(dim=dims)
        fp = ((1 - targets) * probs).sum(dim=dims)
        fn = (targets * (1 - probs)).sum(dim=dims)
        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return (1 - tversky).mean()


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, gamma: float = 1.33):
        super().__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.tversky(logits, targets)
        return loss ** self.gamma


class ComboFocalDiceLoss(nn.Module):
    def __init__(self, alpha: float = 0.8, gamma: float = 2.0):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.focal(logits, targets) + dice_loss(logits, targets)


def build_loss(loss_name: str) -> Callable:
    name = loss_name.lower()
    if name in {"bce_dice", "combo_loss"}:
        return BCEDiceLoss()
    if name == "focal":
        return FocalLoss()
    if name == "tversky":
        return TverskyLoss()
    if name in {"focal_tversky", "focal-tversky"}:
        return FocalTverskyLoss()
    if name in {"combo_focal_dice", "focal_dice"}:
        return ComboFocalDiceLoss()
    if name in {"bce", "bce_with_logits"}:
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"Unknown loss_name='{loss_name}'.")
