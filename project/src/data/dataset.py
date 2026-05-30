from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class PneumothoraxDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None, return_meta: bool = False):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.transform = transform
        self.return_meta = return_meta

        required_columns = {"image_path", "mask_path", "has_pneumo"}
        missing = required_columns - set(self.dataframe.columns)
        if missing:
            raise ValueError(f"Dataset dataframe is missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.dataframe)

    def _load_image(self, path: str | Path) -> Image.Image:
        return Image.open(path).convert("L")

    def _load_mask(self, path: str | Path, size_hw: tuple[int, int]) -> Image.Image:
        path = Path(path)
        if path.exists():
            return Image.open(path).convert("L")
        width, height = size_hw[1], size_hw[0]
        return Image.fromarray(np.zeros((height, width), dtype=np.uint8), mode="L")

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        image = self._load_image(row["image_path"])
        mask = self._load_mask(row["mask_path"], size_hw=(image.height, image.width))

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        if isinstance(image, Image.Image):
            raise TypeError("Transform must convert image to torch.Tensor.")
        if isinstance(mask, Image.Image):
            raise TypeError("Transform must convert mask to torch.Tensor.")

        mask = mask.float()
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if image.ndim == 2:
            image = image.unsqueeze(0)

        if self.return_meta:
            meta = {
                "image_path": row["image_path"],
                "mask_path": row["mask_path"],
                "has_pneumo": int(row["has_pneumo"]),
            }
            return image, mask, meta
        return image, mask
