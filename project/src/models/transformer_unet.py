from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import ConvBlock

try:
    import timm
except ImportError:  # pragma: no cover - optional dependency
    timm = None


def _to_nchw(feature: torch.Tensor) -> torch.Tensor:
    """Convert timm feature maps to NCHW.

    Some transformer backbones (for example Swin in timm) return NHWC feature maps,
    while most CNN backbones return NCHW. The decoder below expects NCHW.
    """
    if feature.ndim != 4:
        raise ValueError(f"Expected 4D feature map, got shape {tuple(feature.shape)}.")

    # NHWC case, e.g. [B, H, W, C] = [4, 8, 8, 768]
    if feature.shape[-1] > feature.shape[1] and feature.shape[-1] > feature.shape[2]:
        return feature.permute(0, 3, 1, 2).contiguous()

    # Already NCHW
    return feature.contiguous()


class TransformerUNet(nn.Module):
    def __init__(
        self,
        encoder_name: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        out_channels: int = 1,
        img_size: int = 256,
    ):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for TransformerUNet. Install it or choose another model.")

        self.input_conv = nn.Conv2d(1, 3, kernel_size=1)
        self.backbone = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            img_size=(img_size, img_size),
            strict_img_size=False,
        )
        channels = self.backbone.feature_info.channels()
        if len(channels) < 4:
            raise ValueError(f"Backbone returned insufficient feature levels: {channels}")

        c1, c2, c3, c4 = channels[:4]

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c3 + c3, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c2 + c2, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c1 + c1, c1)

        self.up0 = nn.ConvTranspose2d(c1, 64, kernel_size=2, stride=2)
        self.dec0 = ConvBlock(64, 64)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_hw = x.shape[-2:]
        x = self.input_conv(x)
        features = self.backbone(x)[:4]
        features = [_to_nchw(feature) for feature in features]
        x1, x2, x3, x4 = features

        d3 = self.up3(x4)
        if d3.shape[-2:] != x3.shape[-2:]:
            d3 = F.interpolate(d3, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, x3], dim=1))

        d2 = self.up2(d3)
        if d2.shape[-2:] != x2.shape[-2:]:
            d2 = F.interpolate(d2, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))

        d1 = self.up1(d2)
        if d1.shape[-2:] != x1.shape[-2:]:
            d1 = F.interpolate(d1, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, x1], dim=1))

        d0 = self.up0(d1)
        d0 = self.dec0(d0)

        out = self.final(d0)
        out = F.interpolate(out, size=input_hw, mode="bilinear", align_corners=False)
        return out
