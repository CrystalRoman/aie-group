from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageOps


class PairedTransform:
    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple:
        raise NotImplementedError


@dataclass
class ResizePair(PairedTransform):
    size: int

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple:
        image = image.resize((self.size, self.size), resample=Image.BILINEAR)
        mask = mask.resize((self.size, self.size), resample=Image.NEAREST)
        return image, mask


@dataclass
class RandomHorizontalFlipPair(PairedTransform):
    p: float = 0.5

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple:
        if random.random() < self.p:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)
        return image, mask


@dataclass
class RandomVerticalFlipPair(PairedTransform):
    p: float = 0.1

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple:
        if random.random() < self.p:
            image = ImageOps.flip(image)
            mask = ImageOps.flip(mask)
        return image, mask


@dataclass
class RandomRotatePair(PairedTransform):
    degrees: tuple = (-7, 7)

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple:
        angle = random.uniform(*self.degrees)
        image = image.rotate(angle, resample=Image.BILINEAR)
        mask = mask.rotate(angle, resample=Image.NEAREST)
        return image, mask


class ToTensorAndNormalizePair(PairedTransform):
    def __init__(self, mean: float = 0.5, std: float = 0.5):
        self.mean = mean
        self.std = std

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple:
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        mask_np = (np.asarray(mask, dtype=np.float32) > 0).astype(np.float32)

        image_t = torch.from_numpy(image_np).unsqueeze(0)
        image_t = (image_t - self.mean) / self.std

        mask_t = torch.from_numpy(mask_np).unsqueeze(0)
        return image_t, mask_t


class ComposePair(PairedTransform):
    def __init__(self, transforms: list[PairedTransform]):
        self.transforms = transforms

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple:
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask


def build_train_transform(image_size: int = 256) -> Callable:
    return ComposePair(
        [
            ResizePair(image_size),
            RandomHorizontalFlipPair(0.5),
            RandomVerticalFlipPair(0.1),
            RandomRotatePair((-7, 7)),
            ToTensorAndNormalizePair(mean=0.5, std=0.5),
        ]
    )


def build_eval_transform(image_size: int = 256) -> Callable:
    return ComposePair([ResizePair(image_size), ToTensorAndNormalizePair(mean=0.5, std=0.5)])
