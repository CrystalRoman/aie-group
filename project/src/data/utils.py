from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from PIL import Image


IMAGE_COL_CANDIDATES = ("image_path", "new_filename", "filename", "ImageId", "image_id", "id")
LABEL_COL_CANDIDATES = ("has_pneumo", "label", "target", "class")
MASK_COL_CANDIDATES = ("mask_path", "mask_filename", "mask_file")


def _find_first_existing_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    columns_set = set(columns)
    for candidate in candidates:
        if candidate in columns_set:
            return candidate
    return None


def _ensure_suffix(filename: str, suffix: str = ".png") -> str:
    filename = str(filename)
    if Path(filename).suffix:
        return filename
    return f"{filename}{suffix}"


def build_metadata_dataframe(
    csv_path: str | Path,
    images_dir: str | Path,
    masks_dir: str | Path,
    image_suffix: str = ".png",
    mask_suffix: str = ".png",
    drop_missing_images: bool = True,
    drop_missing_positive_masks: bool = True,
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV is empty.")

    image_col = _find_first_existing_column(df.columns, IMAGE_COL_CANDIDATES)
    if image_col is None:
        raise ValueError(
            f"Could not infer image column. Expected one of: {IMAGE_COL_CANDIDATES}. Got: {list(df.columns)}"
        )

    label_col = _find_first_existing_column(df.columns, LABEL_COL_CANDIDATES)
    if label_col is None:
        raise ValueError(
            f"Could not infer label column. Expected one of: {LABEL_COL_CANDIDATES}. Got: {list(df.columns)}"
        )

    mask_col = _find_first_existing_column(df.columns, MASK_COL_CANDIDATES)

    df = df.copy()
    df["image_name"] = df[image_col].astype(str).map(lambda x: _ensure_suffix(x, image_suffix))
    df["image_path"] = df["image_name"].map(lambda x: str(images_dir / x))

    if mask_col is not None:
        df["mask_name"] = df[mask_col].astype(str).map(lambda x: _ensure_suffix(x, mask_suffix))
    else:
        df["mask_name"] = df["image_name"]

    df["has_pneumo"] = df[label_col].astype(int)
    df["mask_path"] = df["mask_name"].map(lambda x: str(masks_dir / x))

    if drop_missing_images:
        df = df[df["image_path"].map(lambda x: Path(x).exists())].reset_index(drop=True)

    if drop_missing_positive_masks:
        positive_mask_exists = df["mask_path"].map(lambda x: Path(x).exists())
        keep_mask = (~df["has_pneumo"].astype(bool)) | positive_mask_exists
        df = df[keep_mask].reset_index(drop=True)

    return df[["image_name", "image_path", "mask_path", "has_pneumo"]].copy()


def compute_mask_coverage(mask_path: str | Path) -> float:
    mask_path = Path(mask_path)
    if not mask_path.exists():
        return 0.0

    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask) > 0
    return float(mask_np.mean())


def filter_by_mask_coverage(
    df: pd.DataFrame,
    min_coverage: float = 0.0,
    keep_negatives: bool = True,
) -> pd.DataFrame:
    if min_coverage <= 0:
        return df.copy().reset_index(drop=True)

    coverages = []
    for _, row in df.iterrows():
        if int(row["has_pneumo"]) == 0 and keep_negatives:
            coverages.append(1.0)
        else:
            coverages.append(compute_mask_coverage(row["mask_path"]))

    out = df.copy()
    out["mask_coverage"] = coverages
    out = out[out["mask_coverage"] >= min_coverage].reset_index(drop=True)
    return out


def summarize_dataframe(df: pd.DataFrame) -> dict:
    return {
        "num_samples": int(len(df)),
        "num_positive": int(df["has_pneumo"].sum()),
        "num_negative": int((1 - df["has_pneumo"]).sum()),
        "positive_ratio": float(df["has_pneumo"].mean()) if len(df) else 0.0,
    }
