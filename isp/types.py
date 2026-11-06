"""Shared data contract used by every pipeline stage."""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RawImage:
    """A normalised Bayer image plus the metadata the ISP needs.

    bayer      float32 HxW in [0, 1], black level removed, CFA always RGGB (H, W even).
    wb_gains   (3,) R, G, B gains = 1 / AsShotNeutral (None -> estimate with gray-world).
    cam2srgb   (3, 3) white-balanced camera RGB -> linear sRGB (None -> identity).
    nlf        (a, b) noise model: variance = a * x + b, in normalised raw units (None -> estimate).
    orientation  EXIF orientation code (1 = normal).
    """
    bayer: np.ndarray
    wb_gains: Optional[np.ndarray] = None
    cam2srgb: Optional[np.ndarray] = None
    nlf: Optional[tuple] = None
    orientation: int = 1
    meta: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.bayer.shape


class ISPInputError(ValueError):
    """Raised for unreadable / unsupported inputs; message is shown to the user."""
