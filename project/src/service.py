from __future__ import annotations

import io
import json
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src.data.transforms import build_eval_transform
from src.models.factory import create_model
from src.utils.config import load_yaml
from src.utils.io import ensure_dir


LOGGER = logging.getLogger("service")


@dataclass
class LoadedService:
    config_path: Path
    training_config_path: Path
    training_config: dict
    model: torch.nn.Module
    device: torch.device
    threshold: float
    image_size: int
    model_name: str
    checkpoint_path: Path
    predictions_dir: Path


SERVICE_CONFIG_ENV = "SERVICE_CONFIG_PATH"
DEFAULT_SERVICE_CONFIG_PATH = Path("configs/service.yaml")
DEFAULT_PREDICTIONS_DIR = Path("artifacts/service_predictions")
FALLBACK_THRESHOLD = 0.5


def _setup_logging() -> None:
    if LOGGER.handlers:
        return
    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _service_config_path() -> Path:
    return Path(os.getenv(SERVICE_CONFIG_ENV, str(DEFAULT_SERVICE_CONFIG_PATH)))


def _resolve_training_config_path_from_service() -> Path:
    service_config_path = _service_config_path()
    if not service_config_path.exists():
        return Path("configs/training_transformer_unet.yaml")

    service_cfg = load_yaml(service_config_path)
    return Path(service_cfg.get("training_config_path", "configs/training_transformer_unet.yaml"))


def _resolve_threshold(training_config: dict, artifacts_dir: Path, model_name: str) -> float:
    training_cfg = training_config.get("training", {})
    default_threshold = float(training_cfg.get("threshold", FALLBACK_THRESHOLD))

    summary_path = artifacts_dir / "metrics" / f"train_summary_{model_name}.json"
    summary = _load_json(summary_path)
    if not summary:
        return default_threshold

    if "selected_threshold" in summary:
        try:
            return float(summary["selected_threshold"])
        except Exception:
            pass

    threshold_search = summary.get("threshold_search", {})
    if isinstance(threshold_search, dict) and "best_threshold" in threshold_search:
        try:
            return float(threshold_search["best_threshold"])
        except Exception:
            pass

    return default_threshold


def _resolve_form_default_threshold() -> float:
    """Threshold shown by default in Swagger UI."""
    try:
        training_config_path = _resolve_training_config_path_from_service()
        if not training_config_path.exists():
            return FALLBACK_THRESHOLD

        training_config = load_yaml(training_config_path)
        training_cfg = training_config.get("training", {})
        model_name = str(training_cfg.get("model_name", "unet"))
        artifacts_dir = Path(training_config.get("paths", {}).get("artifacts_dir", "artifacts"))
        return float(_resolve_threshold(training_config, artifacts_dir, model_name))
    except Exception:
        return FALLBACK_THRESHOLD


DEFAULT_FORM_THRESHOLD = _resolve_form_default_threshold()

app = FastAPI(
    title="Pneumothorax Segmentation API",
    description=(
        "Минимальный FastAPI-сервис для инференса модели сегментации пневмоторакса. "
        "Для интерактивной проверки откройте /docs."
    ),
    version="1.1.0",
)

STATE: Optional[LoadedService] = None
PREDICTIONS_DIR = ensure_dir(DEFAULT_PREDICTIONS_DIR)
app.mount("/predictions", StaticFiles(directory=str(PREDICTIONS_DIR)), name="predictions")


def _resolve_device(requested_device: str) -> torch.device:
    requested_device = str(requested_device).lower()
    if requested_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_model(training_config: dict, device: torch.device) -> tuple[torch.nn.Module, Path, float, int, str, Path]:
    paths = training_config["paths"]
    data_cfg = training_config["data"]
    training_cfg = training_config["training"]

    model_name = str(training_cfg["model_name"])
    image_size = int(data_cfg.get("image_size", 256))
    artifacts_dir = Path(paths.get("artifacts_dir", "artifacts"))
    checkpoint_path = artifacts_dir / "models" / f"best_{model_name}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Не найден чекпоинт модели: {checkpoint_path}. Сначала обучите модель или укажите корректный artifacts_dir."
        )

    model = create_model(
        model_name,
        in_channels=1,
        out_channels=1,
        base_channels=int(training_cfg.get("base_channels", 32)),
        encoder_name=training_cfg.get("encoder_name", "swin_tiny_patch4_window7_224"),
        pretrained=bool(training_cfg.get("pretrained", True)),
        img_size=image_size,
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    threshold = _resolve_threshold(training_config, artifacts_dir, model_name)
    return model, checkpoint_path, threshold, image_size, model_name, artifacts_dir


def load_service() -> LoadedService:
    _setup_logging()
    service_config_path = _service_config_path()
    if not service_config_path.exists():
        raise FileNotFoundError(
            f"Не найден конфиг сервиса: {service_config_path}. Создайте его или задайте {SERVICE_CONFIG_ENV}."
        )

    service_cfg = load_yaml(service_config_path)
    training_config_path = Path(service_cfg.get("training_config_path", "configs/training_transformer_unet.yaml"))
    if not training_config_path.exists():
        raise FileNotFoundError(
            f"Не найден training config: {training_config_path}. Укажите правильный training_config_path в service.yaml."
        )

    training_config = load_yaml(training_config_path)
    requested_device = str(service_cfg.get("device", training_config.get("training", {}).get("device", "cuda")))
    device = _resolve_device(requested_device)

    model, checkpoint_path, threshold, image_size, model_name, artifacts_dir = _build_model(training_config, device)

    predictions_dir = Path(service_cfg.get("predictions_dir", str(DEFAULT_PREDICTIONS_DIR)))
    predictions_dir = ensure_dir(predictions_dir)

    LOGGER.info(
        "Service loaded | model=%s | device=%s | checkpoint=%s | threshold=%.3f | image_size=%s",
        model_name,
        device,
        checkpoint_path,
        threshold,
        image_size,
    )

    return LoadedService(
        config_path=service_config_path,
        training_config_path=training_config_path,
        training_config=training_config,
        model=model,
        device=device,
        threshold=threshold,
        image_size=image_size,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        predictions_dir=predictions_dir,
    )


def _ensure_state() -> LoadedService:
    global STATE
    if STATE is None:
        STATE = load_service()
    return STATE


def _preprocess_image(file_bytes: bytes, image_size: int) -> tuple[Image.Image, torch.Tensor]:
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("L")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось открыть изображение: {exc}") from exc

    dummy_mask = Image.new("L", image.size, 0)
    transform = build_eval_transform(image_size)
    image_tensor, _ = transform(image, dummy_mask)
    return image, image_tensor.unsqueeze(0)


def _save_mask_and_overlay(
    original: Image.Image,
    mask_binary: torch.Tensor,
    predictions_dir: Path,
    stem: str,
) -> tuple[str, str, int, float]:
    import numpy as np

    mask_arr = (mask_binary.squeeze().cpu().numpy() * 255).astype("uint8")
    mask_img = Image.fromarray(mask_arr, mode="L")
    mask_img = mask_img.resize(original.size, resample=Image.NEAREST)

    original_rgb = original.convert("RGB")
    original_arr = np.array(original_rgb, dtype=np.uint8)
    mask_bool = np.array(mask_img, dtype=np.uint8) > 0
    overlay_arr = original_arr.copy()
    overlay_arr[mask_bool] = [255, 0, 0]
    overlay_img = Image.fromarray(overlay_arr, mode="RGB")

    mask_path = predictions_dir / f"{stem}_mask.png"
    overlay_path = predictions_dir / f"{stem}_overlay.png"
    mask_img.save(mask_path)
    overlay_img.save(overlay_path)

    positive_pixels = int(mask_bool.sum())
    coverage = float(positive_pixels / max(1, mask_bool.size))
    return str(mask_path), str(overlay_path), positive_pixels, coverage


@app.on_event("startup")
def on_startup() -> None:
    global STATE
    STATE = load_service()


@app.get("/health")
def health() -> dict:
    state = _ensure_state()
    return {
        "status": "ok",
        "model_name": state.model_name,
        "device": str(state.device),
        "threshold": state.threshold,
        "training_config_path": str(state.training_config_path),
        "checkpoint_path": str(state.checkpoint_path),
    }


@app.post("/reload")
def reload_service() -> dict:
    global STATE, DEFAULT_FORM_THRESHOLD
    STATE = load_service()
    DEFAULT_FORM_THRESHOLD = STATE.threshold
    return {
        "status": "reloaded",
        "model_name": STATE.model_name,
        "threshold": STATE.threshold,
        "checkpoint_path": str(STATE.checkpoint_path),
    }


@app.post("/predict")
def predict(
    file: UploadFile = File(..., description="PNG/JPG снимок грудной клетки"),
    threshold: float = Form(
        DEFAULT_FORM_THRESHOLD,
        ge=0.05,
        le=0.95,
        description=(
            "Порог бинаризации. По умолчанию подставляется лучший найденный threshold "
            "из артефактов обучения для выбранной модели."
        ),
    ),
) -> JSONResponse:
    state = _ensure_state()
    file_bytes = file.file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Файл пустой.")

    original_image, image_tensor = _preprocess_image(file_bytes, state.image_size)
    image_tensor = image_tensor.to(state.device)
    used_threshold = float(threshold)

    with torch.no_grad():
        logits = state.model(image_tensor)
        probs = torch.sigmoid(logits)
        mask_binary = (probs > used_threshold).float()

    stem = uuid.uuid4().hex[:12]
    mask_path, overlay_path, positive_pixels, coverage = _save_mask_and_overlay(
        original=original_image,
        mask_binary=mask_binary,
        predictions_dir=state.predictions_dir,
        stem=stem,
    )

    return JSONResponse(
        {
            "status": "ok",
            "model_name": state.model_name,
            "threshold": used_threshold,
            "input_filename": file.filename,
            "image_size": list(original_image.size),
            "positive_pixels": positive_pixels,
            "mask_coverage": coverage,
            "mask_path": mask_path,
            "overlay_path": overlay_path,
            "mask_url": f"/predictions/{Path(mask_path).name}",
            "overlay_url": f"/predictions/{Path(overlay_path).name}",
        }
    )
