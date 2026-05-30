from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def tensor_to_numpy_image(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu().float()
    if image.ndim == 3 and image.shape[0] == 1:
        image = image.squeeze(0)
    return image.numpy()


def tensor_to_numpy_mask(mask: torch.Tensor) -> np.ndarray:
    mask = mask.detach().cpu().float()
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    return mask.numpy()


def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    image = image.astype(np.float32)
    if image.max() > 1:
        image = image / 255.0

    rgb = np.stack([image, image, image], axis=-1)
    red_overlay = rgb.copy()
    red_overlay[..., 0] = np.maximum(red_overlay[..., 0], mask.astype(np.float32))
    return np.clip((1 - alpha) * rgb + alpha * red_overlay, 0, 1)


def save_prediction_figure(
    image: torch.Tensor,
    target_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    save_path: str | Path,
    title: Optional[str] = None,
) -> None:
    image_np = tensor_to_numpy_image(image)
    target_np = tensor_to_numpy_mask(target_mask)
    pred_np = tensor_to_numpy_mask(pred_mask)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image_np, cmap="gray")
    axes[0].set_title("Image")
    axes[1].imshow(target_np, cmap="gray")
    axes[1].set_title("GT mask")
    axes[2].imshow(pred_np, cmap="gray")
    axes[2].set_title("Pred mask")
    axes[3].imshow(overlay_mask_on_image(image_np, pred_np), cmap="gray")
    axes[3].set_title("Overlay")

    if title:
        fig.suptitle(title)
    for ax in axes:
        ax.axis("off")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
