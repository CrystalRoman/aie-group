from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import PneumothoraxDataset
from src.data.transforms import build_eval_transform
from src.models.factory import create_model
from src.training.engine import evaluate_one_epoch, search_best_threshold
from src.training.losses import build_loss
from src.utils.config import load_yaml
from src.utils.io import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained segmentation model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--split_csv", type=str, default=None, help="Optional CSV with a prepared split.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional path to model weights.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger = logging.getLogger("evaluate")

    paths = config["paths"]
    data_cfg = config["data"]
    training_cfg = config["training"]

    artifacts_dir = ensure_dir(paths.get("artifacts_dir", "artifacts"))
    metrics_dir = ensure_dir(artifacts_dir / "metrics")

    split_csv = args.split_csv or str(metrics_dir / "test_split.csv")
    if not Path(split_csv).exists():
        raise FileNotFoundError(
            f"Split CSV not found: {split_csv}. Run training first or pass --split_csv explicitly."
        )

    import pandas as pd
    test_df = pd.read_csv(split_csv)

    image_size = int(data_cfg.get("image_size", 256))
    batch_size = int(data_cfg.get("batch_size", 8))
    num_workers = int(data_cfg.get("num_workers", 0))

    dataset = PneumothoraxDataset(test_df, transform=build_eval_transform(image_size))

    model = create_model(
        training_cfg["model_name"],
        in_channels=1,
        out_channels=1,
        base_channels=int(training_cfg.get("base_channels", 32)),
        encoder_name=training_cfg.get("encoder_name", "swin_tiny_patch4_window7_224"),
        pretrained=False,
        img_size=image_size,
    )
    device_name = training_cfg.get("device", "cuda")
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    model = model.to(device)

    checkpoint_path = args.checkpoint or str(Path(paths.get("artifacts_dir", "artifacts")) / "models" / f"best_{training_cfg['model_name']}.pt")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    criterion = build_loss(training_cfg.get("loss_name", "bce_dice"))
    metrics = evaluate_one_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        threshold=float(training_cfg.get("threshold", 0.5)),
        mixed_precision=bool(training_cfg.get("mixed_precision", True)),
    )
    threshold_info = search_best_threshold(
        model=model,
        loader=loader,
        device=device,
        thresholds=training_cfg.get("threshold_candidates", [0.3, 0.4, 0.5, 0.6, 0.7]),
    )

    result = {
        "checkpoint_path": checkpoint_path,
        "num_test_samples": len(dataset),
        "metrics": metrics,
        "threshold_search": threshold_info,
    }
    out_path = metrics_dir / f"test_metrics_{training_cfg['model_name']}.json"
    save_json(result, out_path)
    logger.info("Evaluation results saved to %s", out_path)
    logger.info("Metrics: %s", result)


if __name__ == "__main__":
    main()
