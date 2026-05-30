from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from src.data.dataset import PneumothoraxDataset
from src.data.transforms import build_eval_transform


def _make_image(path: Path, value: int) -> None:
    img = Image.new("L", (32, 32), color=value)
    img.save(path)


def test_dataset_returns_tensors(tmp_path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    image_path = images_dir / "sample.png"
    mask_path = masks_dir / "sample.png"

    _make_image(image_path, 128)
    _make_image(mask_path, 255)

    df = pd.DataFrame(
        [
            {
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "has_pneumo": 1,
            }
        ]
    )

    dataset = PneumothoraxDataset(df, transform=build_eval_transform(256))
    sample = dataset[0]

    assert isinstance(sample, (tuple, list))
    assert len(sample) >= 2

    image_tensor, mask_tensor = sample[0], sample[1]
    assert isinstance(image_tensor, torch.Tensor)
    assert isinstance(mask_tensor, torch.Tensor)
    assert image_tensor.ndim == 3
    assert mask_tensor.ndim == 3
    assert image_tensor.shape[-2:] == (256, 256)
    assert mask_tensor.shape[-2:] == (256, 256)
