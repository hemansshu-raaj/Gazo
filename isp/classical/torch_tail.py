"""Differentiable copy of the classical tail (CCM -> sRGB gamma -> tone curve) for training through it."""
import torch

from .tone_curve_data import TONE_CURVE

_TC = torch.tensor(TONE_CURVE, dtype=torch.float32)


def srgb_gamma(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp_min(0.0031308).pow(1 / 2.4) - 0.055)


def tone_curve(x: torch.Tensor) -> torch.Tensor:
    tc = _TC.to(x.device, x.dtype)
    n = tc.numel() - 1
    pos = x.clamp(0, 1) * n
    i0 = pos.floor().clamp(max=n - 1).long()
    w = pos - i0.to(x.dtype)
    return tc[i0] * (1 - w) + tc[i0 + 1] * w


def apply_ccm(rgb: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """rgb: (B,3,H,W), m: (B,3,3) camera->sRGB."""
    return torch.einsum("bij,bjhw->bihw", m, rgb).clamp(0.0, 1.0)


def tail(rgb_lin_cam: torch.Tensor, cam2srgb: torch.Tensor) -> torch.Tensor:
    return tone_curve(srgb_gamma(apply_ccm(rgb_lin_cam.clamp(0, 1), cam2srgb)))
