import torch

from src.training.metrics import dice_score, iou_score


def test_metrics_perfect_prediction():
    logits = torch.full((1, 1, 4, 4), 10.0)
    targets = torch.ones((1, 1, 4, 4), dtype=torch.float32)

    dice = dice_score(logits, targets, threshold=0.5)
    iou = iou_score(logits, targets, threshold=0.5)

    assert abs(dice - 1.0) < 1e-6
    assert abs(iou - 1.0) < 1e-6


def test_metrics_empty_prediction_against_positive_mask():
    logits = torch.full((1, 1, 4, 4), -10.0)
    targets = torch.ones((1, 1, 4, 4), dtype=torch.float32)

    dice = dice_score(logits, targets, threshold=0.5)
    iou = iou_score(logits, targets, threshold=0.5)

    assert 0.0 <= dice < 0.01
    assert 0.0 <= iou < 0.01
