from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.transforms import build_eval_transform
from src.models.factory import create_model
from src.utils.config import load_yaml
from src.utils.io import ensure_dir
from src.utils.visualization import save_prediction_figure


def load_model(config_path: str):
    config = load_yaml(config_path)
    training_cfg = config["training"]
    data_cfg = config["data"]
    paths = config["paths"]

    model = create_model(
        training_cfg["model_name"],
        in_channels=1,
        out_channels=1,
        base_channels=int(training_cfg.get("base_channels", 32)),
        encoder_name=training_cfg.get("encoder_name", "swin_tiny_patch4_window7_224"),
        pretrained=False,
        img_size=int(data_cfg.get("image_size", 256)),
    )

    device_name = training_cfg.get("device", "cuda")
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(paths.get("artifacts_dir", "artifacts")) / "models" / f"best_{training_cfg['model_name']}.pt"
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model, device, config


@torch.no_grad()
def predict_single_image(config_path: str, image_path: str, save_path: str | None = None, threshold: float | None = None):
    model, device, config = load_model(config_path)
    image_size = int(config["data"].get("image_size", 256))
    transform = build_eval_transform(image_size)

    image = Image.open(image_path).convert("L")
    dummy_mask = Image.fromarray(np.zeros((image.height, image.width), dtype=np.uint8), mode="L")
    image_tensor, _ = transform(image, dummy_mask)
    logits = model(image_tensor.unsqueeze(0).to(device))
    threshold = float(threshold if threshold is not None else config["training"].get("threshold", 0.5))
    pred_mask = (torch.sigmoid(logits) > threshold).float().cpu().squeeze(0)

    if save_path:
        ensure_dir(Path(save_path).parent)
        save_prediction_figure(image_tensor.cpu(), pred_mask.cpu(), pred_mask.cpu(), save_path, title=Path(image_path).name)

    return pred_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict pneumothorax mask for a single image.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to image.")
    parser.add_argument("--save_path", type=str, default="artifacts/figures/single_prediction.png")
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_mask = predict_single_image(
        config_path=args.config,
        image_path=args.image_path,
        save_path=args.save_path,
        threshold=args.threshold,
    )
    print(f"Prediction saved. Positive pixels: {int(pred_mask.sum().item())}")


if __name__ == "__main__":
    main()
