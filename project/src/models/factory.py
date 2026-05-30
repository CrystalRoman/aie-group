from __future__ import annotations

import inspect
from typing import Any, Dict

from torch import nn

from .attention_unet import AttentionUNet
from .transformer_unet import TransformerUNet
from .unet import UNet


MODEL_REGISTRY = {
    "unet": UNet,
    "attention_unet": AttentionUNet,
    "transformer_unet": TransformerUNet,
    "unet_transformer": TransformerUNet,
    "swin_unet": TransformerUNet,
}


def create_model(model_name: str, **kwargs: Dict[str, Any]) -> nn.Module:
    name = model_name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name='{model_name}'.")
    cls = MODEL_REGISTRY[name]
    signature = inspect.signature(cls.__init__)
    allowed_kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}
    return cls(**allowed_kwargs)
