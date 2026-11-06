"""Model wrappers: the network sits between classical stages.

HybridISPNet  in: packed WB'd RGGB (4) [+ noise sigma map (1)]  -> linear camera RGB (full res)
              = bilinear demosaic of the input + learned residual (joint denoise + demosaic)
E2EISPNet     in: packed RAW (4, no WB / metadata) -> sRGB directly (learns everything)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .naf_unet import NAFUNet
from .unet_student import StudentUNet


def bilinear_base(packed: torch.Tensor) -> torch.Tensor:
    """Cheap demosaic of packed RGGB (B,4,h,w) -> (B,3,2h,2w)."""
    rgb = torch.stack([packed[:, 0], 0.5 * (packed[:, 1] + packed[:, 2]), packed[:, 3]], 1)
    return F.interpolate(rgb, scale_factor=2, mode="bilinear", align_corners=False)


class HybridISPNet(nn.Module):
    def __init__(self, body: nn.Module, use_sigma: bool = True):
        super().__init__()
        self.body, self.use_sigma = body, use_sigma
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, packed: torch.Tensor, sigma: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.cat([packed, sigma * 10.0], 1) if self.use_sigma else packed
        return bilinear_base(packed) + self.shuffle(self.body(x))


class E2EISPNet(nn.Module):
    def __init__(self, body: nn.Module):
        super().__init__()
        self.body = body
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, packed: torch.Tensor, sigma=None) -> torch.Tensor:
        return self.shuffle(self.body(packed * 4.0))  # raw is dark; scale for a better-conditioned input


def build(name: str) -> nn.Module:
    if name == "teacher":
        return HybridISPNet(NAFUNet(5, 12, width=32), use_sigma=True)
    if name == "nosigma":
        return HybridISPNet(NAFUNet(4, 12, width=32), use_sigma=False)
    if name == "e2e":
        return E2EISPNet(NAFUNet(4, 12, width=32))
    if name in ("student", "student_kd", "student_scratch"):
        return HybridISPNet(StudentUNet(5, 12, width=16), use_sigma=True)
    raise ValueError(name)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
