
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.data.dataset import PneumothoraxDataset
from src.data.split import stratified_train_val_test_split
from src.data.transforms import build_eval_transform, build_train_transform
from src.data.utils import build_metadata_dataframe, filter_by_mask_coverage, summarize_dataframe
from src.models.factory import create_model
from src.training.engine import fit, search_best_threshold
from src.training.losses import build_loss
from src.utils.config import load_yaml
from src.utils.io import ensure_dir, save_history_csv, save_json
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Universal training script for pneumothorax segmentation.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    return parser.parse_args()


def _split_train_val(df: pd.DataFrame, val_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < val_size < 1.0:
        raise ValueError(f"val_size must be in (0, 1), got {val_size}")
    stratify = None
    if "has_pneumo" in df.columns and df["has_pneumo"].nunique() > 1:
        stratify = df["has_pneumo"]
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def _setup_logger(logs_dir: Path) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(logs_dir / "train.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _filter_positive_only(df: pd.DataFrame, enabled: bool, logger: logging.Logger, name: str) -> pd.DataFrame:
    if not enabled:
        return df.reset_index(drop=True)
    if "has_pneumo" not in df.columns:
        logger.warning("positive_only=True requested for %s, but 'has_pneumo' column is missing. Skipping filter.", name)
        return df.reset_index(drop=True)
    before = len(df)
    df = df[df["has_pneumo"] == 1].reset_index(drop=True)
    logger.info("%s positive_only filter applied: %d -> %d rows", name, before, len(df))
    return df


def _plot_history(history: list[dict], figures_dir: Path, model_name: str) -> None:
    if not history:
        return
    history_df = pd.DataFrame(history)
    if history_df.empty:
        return
    epochs = history_df["epoch"]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history_df["train_loss"], label="train_loss")
    plt.plot(epochs, history_df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss curves - {model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / f"loss_curve_{model_name}.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history_df["train_dice"], label="train_dice")
    plt.plot(epochs, history_df["val_dice"], label="val_dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title(f"Dice curves - {model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / f"dice_curve_{model_name}.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history_df["train_iou"], label="train_iou")
    plt.plot(epochs, history_df["val_iou"], label="val_iou")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title(f"IoU curves - {model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / f"iou_curve_{model_name}.png", dpi=150)
    plt.close()


def _plot_split_sizes(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, figures_dir: Path, model_name: str) -> None:
    labels = ["train", "val", "test"]
    sizes = [len(train_df), len(val_df), len(test_df)]
    plt.figure(figsize=(7, 5))
    plt.bar(labels, sizes)
    plt.ylabel("Number of samples")
    plt.title(f"Split sizes - {model_name}")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / f"split_sizes_{model_name}.png", dpi=150)
    plt.close()


def _plot_class_balance(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, figures_dir: Path, model_name: str) -> None:
    def counts(df: pd.DataFrame) -> tuple[int, int]:
        if "has_pneumo" not in df.columns:
            return 0, len(df)
        pos = int((df["has_pneumo"] == 1).sum())
        neg = int((df["has_pneumo"] == 0).sum())
        return pos, neg

    train_pos, train_neg = counts(train_df)
    val_pos, val_neg = counts(val_df)
    test_pos, test_neg = counts(test_df)

    splits = ["train", "val", "test"]
    positives = [train_pos, val_pos, test_pos]
    negatives = [train_neg, val_neg, test_neg]

    plt.figure(figsize=(8, 5))
    plt.bar(splits, negatives, label="negative")
    plt.bar(splits, positives, bottom=negatives, label="positive")
    plt.ylabel("Number of samples")
    plt.title(f"Class balance by split - {model_name}")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / f"class_balance_{model_name}.png", dpi=150)
    plt.close()


def _plot_threshold_search(threshold_info: dict, figures_dir: Path, model_name: str, logger: logging.Logger) -> None:
    scores = threshold_info.get("scores")
    if not scores:
        logger.warning("Threshold score details are unavailable, threshold_search plot will be skipped.")
        return
    scores_df = pd.DataFrame(scores)
    if scores_df.empty or "threshold" not in scores_df.columns or "dice" not in scores_df.columns:
        logger.warning("Threshold score details are malformed, threshold_search plot will be skipped.")
        return
    plt.figure(figsize=(8, 5))
    plt.plot(scores_df["threshold"], scores_df["dice"], marker="o")
    plt.xlabel("Threshold")
    plt.ylabel("Validation Dice")
    plt.title(f"Threshold search - {model_name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / f"threshold_search_{model_name}.png", dpi=150)
    plt.close()


def _plot_mask_coverage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, figures_dir: Path, model_name: str, logger: logging.Logger) -> None:
    if not all("mask_coverage" in df.columns for df in [train_df, val_df, test_df]):
        logger.info("mask_coverage column is missing. Coverage plot will be skipped.")
        return

    plt.figure(figsize=(9, 5))
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        values = df["mask_coverage"].dropna().values
        if len(values) == 0:
            continue
        plt.hist(values, bins=30, alpha=0.45, label=name)
    plt.xlabel("Mask coverage")
    plt.ylabel("Count")
    plt.title(f"Mask coverage by split - {model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / f"mask_coverage_{model_name}.png", dpi=150)
    plt.close()


def _per_sample_binary_metrics(
    logits: torch.Tensor,
    masks: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> list[dict]:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    masks = masks.float()

    dims = (1, 2, 3)
    intersection = (preds * masks).sum(dim=dims)
    pred_sum = preds.sum(dim=dims)
    target_sum = masks.sum(dim=dims)
    dice = (2 * intersection + eps) / (pred_sum + target_sum + eps)
    iou = (intersection + eps) / (pred_sum + target_sum - intersection + eps)
    positives = (target_sum > 0).detach().cpu().tolist()

    items = []
    for d, j, p in zip(dice.detach().cpu().tolist(), iou.detach().cpu().tolist(), positives):
        items.append({"dice": float(d), "iou": float(j), "is_positive": bool(p)})
    return items


@torch.no_grad()
def _evaluate_loader_subset_metrics(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    threshold: float,
) -> dict:
    model.eval()
    items: list[dict] = []
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        items.extend(_per_sample_binary_metrics(logits, masks, threshold=threshold))

    def summarize(rows: list[dict]) -> dict:
        if not rows:
            return {"count": 0, "dice": None, "iou": None}
        return {
            "count": len(rows),
            "dice": float(sum(x["dice"] for x in rows) / len(rows)),
            "iou": float(sum(x["iou"] for x in rows) / len(rows)),
        }

    positives = [x for x in items if x["is_positive"]]
    negatives = [x for x in items if not x["is_positive"]]
    return {
        "all": summarize(items),
        "positive_only": summarize(positives),
        "negative_only": summarize(negatives),
    }


@torch.no_grad()
def _save_prediction_examples(
    model: torch.nn.Module,
    dataset: PneumothoraxDataset,
    device: torch.device,
    threshold: float,
    save_path: Path,
    num_examples: int = 6,
    positive_only: bool = True,
) -> None:
    model.eval()
    indices = list(range(len(dataset)))
    if positive_only and hasattr(dataset, "df") and "has_pneumo" in dataset.df.columns:
        positive_indices = dataset.df.index[dataset.df["has_pneumo"] == 1].tolist()
        indices = [i for i in range(len(dataset)) if dataset.df.iloc[i]["has_pneumo"] == 1]
        if not indices:
            indices = list(range(len(dataset)))

    indices = indices[:num_examples]
    if not indices:
        return

    fig, axes = plt.subplots(len(indices), 3, figsize=(12, 4 * len(indices)))
    if len(indices) == 1:
        axes = [axes]

    for row_idx, idx in enumerate(indices):
        image, mask = dataset[idx]
        image_batch = image.unsqueeze(0).to(device)
        logits = model(image_batch)
        pred = (torch.sigmoid(logits) > threshold).float().cpu().squeeze().numpy()

        image_np = image.squeeze().cpu().numpy()
        mask_np = mask.squeeze().cpu().numpy()

        axes[row_idx][0].imshow(image_np, cmap="gray")
        axes[row_idx][0].set_title(f"Image #{row_idx + 1}")
        axes[row_idx][1].imshow(mask_np, cmap="gray")
        axes[row_idx][1].set_title("Ground truth mask")
        axes[row_idx][2].imshow(pred, cmap="gray")
        axes[row_idx][2].set_title(f"Prediction (thr={threshold:.2f})")

        for ax in axes[row_idx]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    paths = config["paths"]
    data_cfg = config["data"]
    training_cfg = config["training"]

    artifacts_dir = ensure_dir(paths.get("artifacts_dir", "artifacts"))
    models_dir = ensure_dir(artifacts_dir / "models")
    metrics_dir = ensure_dir(artifacts_dir / "metrics")
    figures_dir = ensure_dir(artifacts_dir / "figures")
    logs_dir = ensure_dir(artifacts_dir / "logs")
    logger = _setup_logger(logs_dir)

    logger.info("STEP 1/11: Config loaded from %s", args.config)

    seed = int(config.get("seed", 42))
    set_seed(seed)
    logger.info("Random seed set to %s", seed)

    train_csv_path = paths.get("train_csv_path", paths.get("csv_path"))
    test_csv_path = paths.get("test_csv_path")
    positive_only = bool(data_cfg.get("positive_only", False))
    min_mask_coverage = float(data_cfg.get("min_mask_coverage", 0.0))

    logger.info("STEP 2/11: Paths and training config parsed")
    logger.info("train_csv_path=%s", train_csv_path)
    logger.info("test_csv_path=%s", test_csv_path)
    logger.info("images_dir=%s", paths["images_dir"])
    logger.info("masks_dir=%s", paths["masks_dir"])
    logger.info("artifacts_dir=%s", paths.get("artifacts_dir", "artifacts"))
    logger.info("model_name=%s | requested_device=%s | epochs=%s | batch_size=%s",
                training_cfg["model_name"],
                training_cfg.get("device", "cuda"),
                training_cfg.get("num_epochs", 20),
                data_cfg.get("batch_size", 8))
    logger.info("positive_only=%s | min_mask_coverage=%.6f | loss_name=%s",
                positive_only,
                min_mask_coverage,
                training_cfg.get("loss_name", "bce_dice"))

    logger.info("STEP 3/11: Building train/val dataframe...")
    train_val_df = build_metadata_dataframe(
        csv_path=train_csv_path,
        images_dir=paths["images_dir"],
        masks_dir=paths["masks_dir"],
        drop_missing_positive_masks=bool(data_cfg.get("drop_missing_positive_masks", True)),
    )
    train_val_df = filter_by_mask_coverage(train_val_df, min_coverage=min_mask_coverage)
    train_val_df = _filter_positive_only(train_val_df, positive_only, logger, "train_val")
    logger.info("Train/val dataframe built successfully. Rows after filtering: %d", len(train_val_df))
    logger.info("Train/val dataset summary: %s", summarize_dataframe(train_val_df))

    if test_csv_path:
        logger.info("STEP 4/11: Building external test dataframe...")
        test_df = build_metadata_dataframe(
            csv_path=test_csv_path,
            images_dir=paths["images_dir"],
            masks_dir=paths["masks_dir"],
            drop_missing_positive_masks=bool(data_cfg.get("drop_missing_positive_masks", True)),
        )
        test_df = filter_by_mask_coverage(test_df, min_coverage=min_mask_coverage)
        test_df = _filter_positive_only(test_df, positive_only, logger, "test")
        logger.info("External test dataframe built successfully. Rows after filtering: %d", len(test_df))
        logger.info("External test summary: %s", summarize_dataframe(test_df))

        logger.info("STEP 5/11: Splitting train_csv into train/val only...")
        train_df, val_df = _split_train_val(
            train_val_df,
            val_size=float(data_cfg.get("val_size", 0.15)),
            random_state=seed,
        )
    else:
        logger.info("STEP 4/11: No external test_csv_path provided, using internal split...")
        train_df, val_df, test_df = stratified_train_val_test_split(
            train_val_df,
            val_size=float(data_cfg.get("val_size", 0.15)),
            test_size=float(data_cfg.get("test_size", 0.15)),
            random_state=seed,
        )

    logger.info("Split created: train=%d | val=%d | test=%d", len(train_df), len(val_df), len(test_df))

    image_size = int(data_cfg.get("image_size", 256))
    batch_size = int(data_cfg.get("batch_size", 8))
    num_workers = int(data_cfg.get("num_workers", 0))

    logger.info("STEP 6/11: Creating Dataset objects...")
    train_dataset = PneumothoraxDataset(train_df, transform=build_train_transform(image_size))
    val_dataset = PneumothoraxDataset(val_df, transform=build_eval_transform(image_size))
    test_dataset = PneumothoraxDataset(test_df, transform=build_eval_transform(image_size))
    logger.info("Datasets ready: train=%d | val=%d | test=%d",
                len(train_dataset), len(val_dataset), len(test_dataset))

    logger.info("STEP 7/11: Creating model...")
    model = create_model(
        training_cfg["model_name"],
        in_channels=1,
        out_channels=1,
        base_channels=int(training_cfg.get("base_channels", 32)),
        encoder_name=training_cfg.get("encoder_name", "swin_tiny_patch4_window7_224"),
        pretrained=bool(training_cfg.get("pretrained", True)),
        img_size=image_size,
    )

    device_name = training_cfg.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA was requested but is not available. Falling back to CPU.")
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    logger.info("STEP 8/11: Creating DataLoaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    logger.info("DataLoaders ready. train_batches=%d | val_batches=%d | test_batches=%d",
                len(train_loader), len(val_loader), len(test_loader))

    model = model.to(device)
    logger.info("Model created and moved to device: %s", device)

    criterion = build_loss(training_cfg.get("loss_name", "bce_dice"))
    optimizer = AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-5)),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(training_cfg.get("scheduler_factor", 0.5)),
        patience=int(training_cfg.get("scheduler_patience", 2)),
    )

    checkpoint_path = models_dir / f"best_{training_cfg['model_name']}.pt"

    logger.info("STEP 9/11: Starting training loop...")
    logger.info("Training params: loss=%s | lr=%s | mixed_precision=%s",
                training_cfg.get("loss_name", "bce_dice"),
                training_cfg.get("learning_rate", 1e-4),
                training_cfg.get("mixed_precision", True))

    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=int(training_cfg.get("num_epochs", 20)),
        threshold=float(training_cfg.get("threshold", 0.5)),
        mixed_precision=bool(training_cfg.get("mixed_precision", True)),
        scheduler=scheduler,
        scheduler_mode="max",
        checkpoint_path=checkpoint_path,
        early_stopping_patience=training_cfg.get("early_stopping_patience"),
    )

    history_path = metrics_dir / f"history_{training_cfg['model_name']}.csv"
    save_history_csv(history, history_path)
    logger.info("Training loop finished. History saved to %s", history_path)

    logger.info("Loading best checkpoint from %s for final evaluation...", checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    threshold_info = search_best_threshold(
        model=model,
        loader=val_loader,
        device=device,
        thresholds=training_cfg.get("threshold_candidates", [0.3, 0.4, 0.5, 0.6, 0.7]),
    )
    best_threshold = float(threshold_info["best_threshold"])
    logger.info("Best threshold selected on validation: %.4f", best_threshold)

    logger.info("STEP 10/11: Evaluating best checkpoint on val/test with selected threshold...")
    val_eval = _evaluate_loader_subset_metrics(model, val_loader, device, best_threshold)
    test_eval = _evaluate_loader_subset_metrics(model, test_loader, device, best_threshold)
    logger.info("Validation metrics: %s", val_eval)
    logger.info("Test metrics: %s", test_eval)

    examples_dataset_name = str(data_cfg.get("prediction_examples_split", "test")).lower()
    examples_dataset = test_dataset if examples_dataset_name == "test" else val_dataset
    _save_prediction_examples(
        model=model,
        dataset=examples_dataset,
        device=device,
        threshold=best_threshold,
        save_path=figures_dir / f"prediction_examples_{training_cfg['model_name']}.png",
        num_examples=int(data_cfg.get("num_prediction_examples", 6)),
        positive_only=bool(data_cfg.get("prediction_examples_positive_only", True)),
    )

    save_json(
        {
            "train_val_summary": summarize_dataframe(train_val_df),
            "test_summary": summarize_dataframe(test_df),
            "best_model_path": str(checkpoint_path),
            "threshold_search": threshold_info,
            "selected_threshold": best_threshold,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
            "used_external_test_csv": bool(test_csv_path),
            "train_csv_path": str(train_csv_path),
            "test_csv_path": str(test_csv_path) if test_csv_path else None,
            "positive_only": positive_only,
            "min_mask_coverage": min_mask_coverage,
            "loss_name": training_cfg.get("loss_name", "bce_dice"),
            "val_metrics": val_eval,
            "test_metrics": test_eval,
        },
        metrics_dir / f"train_summary_{training_cfg['model_name']}.json",
    )

    logger.info("STEP 11/11: Saving split CSV files and figures...")
    train_df.to_csv(metrics_dir / "train_split.csv", index=False)
    val_df.to_csv(metrics_dir / "val_split.csv", index=False)
    test_df.to_csv(metrics_dir / "test_split.csv", index=False)

    _plot_history(history, figures_dir, training_cfg["model_name"])
    _plot_split_sizes(train_df, val_df, test_df, figures_dir, training_cfg["model_name"])
    _plot_class_balance(train_df, val_df, test_df, figures_dir, training_cfg["model_name"])
    _plot_threshold_search(threshold_info, figures_dir, training_cfg["model_name"], logger)
    _plot_mask_coverage(train_df, val_df, test_df, figures_dir, training_cfg["model_name"], logger)

    logger.info("Training finished. Best model saved to %s", checkpoint_path)
    logger.info("Figures saved to %s", figures_dir)
    logger.info("Logs saved to %s", logs_dir)


if __name__ == "__main__":
    main()
