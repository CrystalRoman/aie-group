from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm

from .metrics import dice_score, iou_score


LOGGER = logging.getLogger(__name__)


def _get_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]['lr'])


def train_one_epoch(
    model: torch.nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
    threshold: float = 0.5,
    mixed_precision: bool = True,
    epoch: Optional[int] = None,
    num_epochs: Optional[int] = None,
) -> Dict[str, float]:
    model.train()
    scaler = GradScaler(enabled=mixed_precision and device.type == 'cuda')

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    num_batches = 0

    desc = 'train'
    if epoch is not None and num_epochs is not None:
        desc = f'train {epoch}/{num_epochs}'

    for images, masks in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=mixed_precision and device.type == 'cuda'):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item())
        running_dice += dice_score(logits.detach(), masks, threshold=threshold)
        running_iou += iou_score(logits.detach(), masks, threshold=threshold)
        num_batches += 1

    return {
        'loss': running_loss / max(1, num_batches),
        'dice': running_dice / max(1, num_batches),
        'iou': running_iou / max(1, num_batches),
    }


@torch.no_grad()
def evaluate_one_epoch(
    model: torch.nn.Module,
    loader: Iterable,
    criterion,
    device: torch.device,
    threshold: float = 0.5,
    mixed_precision: bool = True,
    epoch: Optional[int] = None,
    num_epochs: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    num_batches = 0

    desc = 'eval'
    if epoch is not None and num_epochs is not None:
        desc = f'eval {epoch}/{num_epochs}'

    for images, masks in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast(enabled=mixed_precision and device.type == 'cuda'):
            logits = model(images)
            loss = criterion(logits, masks)

        running_loss += float(loss.item())
        running_dice += dice_score(logits, masks, threshold=threshold)
        running_iou += iou_score(logits, masks, threshold=threshold)
        num_batches += 1

    return {
        'loss': running_loss / max(1, num_batches),
        'dice': running_dice / max(1, num_batches),
        'iou': running_iou / max(1, num_batches),
    }


def fit(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
    num_epochs: int,
    threshold: float = 0.5,
    mixed_precision: bool = True,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scheduler_mode: str = 'max',
    checkpoint_path: Optional[str | Path] = None,
    early_stopping_patience: Optional[int] = None,
) -> List[Dict[str, float]]:
    history: List[Dict[str, float]] = []
    best_dice = float('-inf')
    epochs_without_improvement = 0

    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        'fit() started | device=%s | epochs=%s | mixed_precision=%s | checkpoint=%s',
        device,
        num_epochs,
        mixed_precision,
        checkpoint_path,
    )

    for epoch in range(1, num_epochs + 1):
        current_lr = _get_lr(optimizer)
        LOGGER.info('Epoch %s/%s started | lr=%.8f', epoch, num_epochs, current_lr)

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            threshold=threshold,
            mixed_precision=mixed_precision,
            epoch=epoch,
            num_epochs=num_epochs,
        )
        LOGGER.info(
            'Epoch %s/%s train finished | train_loss=%.4f train_dice=%.4f train_iou=%.4f',
            epoch,
            num_epochs,
            train_metrics['loss'],
            train_metrics['dice'],
            train_metrics['iou'],
        )

        val_metrics = evaluate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=threshold,
            mixed_precision=mixed_precision,
            epoch=epoch,
            num_epochs=num_epochs,
        )

        row = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_dice': train_metrics['dice'],
            'train_iou': train_metrics['iou'],
            'val_loss': val_metrics['loss'],
            'val_dice': val_metrics['dice'],
            'val_iou': val_metrics['iou'],
            'lr': current_lr,
        }
        history.append(row)

        LOGGER.info(
            'Epoch %s/%s finished | train_loss=%.4f train_dice=%.4f train_iou=%.4f | '
            'val_loss=%.4f val_dice=%.4f val_iou=%.4f',
            epoch,
            num_epochs,
            row['train_loss'],
            row['train_dice'],
            row['train_iou'],
            row['val_loss'],
            row['val_dice'],
            row['val_iou'],
        )

        if scheduler is not None:
            if scheduler_mode == 'max':
                scheduler.step(row['val_dice'])
                LOGGER.info('Scheduler step on val_dice=%.4f', row['val_dice'])
            else:
                scheduler.step(row['val_loss'])
                LOGGER.info('Scheduler step on val_loss=%.4f', row['val_loss'])

        if row['val_dice'] > best_dice:
            previous_best = best_dice
            best_dice = row['val_dice']
            epochs_without_improvement = 0
            LOGGER.info(
                'New best validation Dice: %.4f (previous best: %s)',
                best_dice,
                f'{previous_best:.4f}' if previous_best != float('-inf') else '-inf',
            )
            if checkpoint_path is not None:
                torch.save(model.state_dict(), checkpoint_path)
                LOGGER.info('Checkpoint saved to %s', checkpoint_path)
        else:
            epochs_without_improvement += 1
            LOGGER.info(
                'No improvement this epoch. epochs_without_improvement=%s',
                epochs_without_improvement,
            )

        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            LOGGER.info('Early stopping triggered after %s epochs without improvement.', epochs_without_improvement)
            break

    LOGGER.info('fit() finished | best_val_dice=%.4f | epochs_completed=%s', best_dice, len(history))
    return history


@torch.no_grad()
def search_best_threshold(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    thresholds: Iterable[float] = (0.3, 0.4, 0.5, 0.6, 0.7),
) -> Dict[str, float]:
    model.eval()
    best_threshold = None
    best_dice = float('-inf')
    all_scores: list[dict[str, float]] = []

    LOGGER.info('Threshold search started...')
    for threshold in thresholds:
        scores = []
        for images, masks in tqdm(loader, desc=f'threshold={float(threshold):.2f}', leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            scores.append(dice_score(logits, masks, threshold=float(threshold)))
        mean_dice = float(sum(scores) / max(1, len(scores)))
        all_scores.append({'threshold': float(threshold), 'dice': mean_dice})
        LOGGER.info('Threshold %.2f -> val_dice=%.4f', float(threshold), mean_dice)
        if mean_dice > best_dice:
            best_dice = mean_dice
            best_threshold = float(threshold)

    LOGGER.info(
        'Threshold search finished | best_threshold=%.2f | best_dice=%.4f',
        float(best_threshold),
        float(best_dice),
    )
    return {
        'best_threshold': float(best_threshold),
        'best_dice': float(best_dice),
        'scores': all_scores,
    }
