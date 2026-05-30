from __future__ import annotations

import torch
from torch import nn

from .blocks import ConvBlock


class AttentionGate(nn.Module):
    def __init__(self, g_channels: int, x_channels: int, inter_channels: int):
        super().__init__()
        self.w_g = nn.Sequential(nn.Conv2d(g_channels, inter_channels, kernel_size=1, bias=False), nn.BatchNorm2d(inter_channels))
        self.w_x = nn.Sequential(nn.Conv2d(x_channels, inter_channels, kernel_size=1, bias=False), nn.BatchNorm2d(inter_channels))
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        attn = self.relu(self.w_g(g) + self.w_x(x))
        attn = self.psi(attn)
        return x * attn


class AttentionUNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32):
        super().__init__()
        c1, c2, c3, c4, c5 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16

        self.pool = nn.MaxPool2d(2)
        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)
        self.bottleneck = ConvBlock(c4, c5)

        self.up4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.att4 = AttentionGate(c4, c4, c3)
        self.dec4 = ConvBlock(c4 + c4, c4)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.att3 = AttentionGate(c3, c3, c2)
        self.dec3 = ConvBlock(c3 + c3, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(c2, c2, c1)
        self.dec2 = ConvBlock(c2 + c2, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.att1 = AttentionGate(c1, c1, max(1, c1 // 2))
        self.dec1 = ConvBlock(c1 + c1, c1)

        self.final = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        e4 = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        e3 = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        e2 = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        e1 = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final(d1)
