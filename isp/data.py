"""Dataset helpers: scene-wise split, scene loading, deterministic evaluation crops."""
import hashlib
import json
import os

import numpy as np

from .classical.bayer import rggb_offset
from .io import _build, scene_tags

# Split by *physical scene* (SIDD scene number), so test content is never seen in training.
TEST_SCENES = {"003", "007"}
VAL_SCENES = {"008"}


def split_of(scene_name: str) -> str:
    num = scene_name.split("_")[1]
    return "test" if num in TEST_SCENES else "val" if num in VAL_SCENES else "train"


def make_splits(scene_names) -> dict:
    out = {"train": [], "val": [], "test": []}
    for s in sorted(scene_names):
        out[split_of(s)].append(s)
    return out


def load_splits(path: str = os.path.join(os.path.dirname(__file__), "..", "splits", "splits.json")) -> dict:
    with open(path) as f:
        return json.load(f)


def eval_boxes(scene_name: str, h: int, w: int, n: int = 32, size: int = 512) -> list:
    """Seeded crop boxes (y, x) on even coordinates, identical for every method."""
    seed = int(hashlib.md5(scene_name.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    ys = rng.integers(0, (h - size) // 2, n) * 2
    xs = rng.integers(0, (w - size) // 2, n) * 2
    return list(zip(ys.tolist(), xs.tolist()))


class SceneArrays:
    """Memory-mapped, RGGB-aligned arrays of one preprocessed scene (for fast random crops)."""

    def __init__(self, scene_dir: str):
        with open(os.path.join(scene_dir, "meta.json")) as f:
            self.meta = json.load(f)
        letters = [[self.meta["color_desc"][i] for i in row] for row in self.meta["raw_pattern"]]
        dy, dx = rggb_offset(letters)
        self.dir = scene_dir
        H, W = self.meta["shape"]
        ref = _build(np.zeros((2 + dy, 2 + dx), np.float32), letters, scene_tags(self.meta),
                     self.meta.get("camera_whitebalance"), self.meta.get("rgb_xyz_matrix"))  # metadata only
        self.wb, self.cam2srgb, self.nlf = ref.wb_gains, ref.cam2srgb, ref.nlf
        self.h, self.w = (H - dy) // 2 * 2, (W - dx) // 2 * 2
        self.dy, self.dx = dy, dx
        self.white = float(self.meta["white_level"])
        self.black = np.asarray(self.meta["black_level_per_channel"], np.float32)
        pat = np.asarray(self.meta["raw_pattern"])
        # black level per RGGB position after the crop
        self.black4 = np.array([self.black[pat[(dy + i) % 2, (dx + j) % 2]] for i in (0, 1) for j in (0, 1)],
                               np.float32)
        self.noisy = [np.load(os.path.join(scene_dir, f"noisy_{k}.npy"), mmap_mode="r") for k in range(4)
                      if os.path.exists(os.path.join(scene_dir, f"noisy_{k}.npy"))]
        p = os.path.join(scene_dir, "gt_srgb.npy")
        self.gt_srgb = np.load(p, mmap_mode="r") if os.path.exists(p) else None

    def noisy_crop(self, k: int, y: int, x: int, size: int) -> np.ndarray:
        """Normalised RGGB crop (y, x even, in cropped coordinates)."""
        m = np.asarray(self.noisy[k][self.dy + y: self.dy + y + size, self.dx + x: self.dx + x + size],
                       dtype=np.float32)
        b = np.empty_like(m)
        b[0::2, 0::2], b[0::2, 1::2], b[1::2, 0::2], b[1::2, 1::2] = self.black4
        return np.clip((m - b) / (self.white - b), 0, None)

    def noisy_region(self, k: int, y: int, x: int, h: int, w: int) -> np.ndarray:
        m = np.asarray(self.noisy[k][self.dy + y: self.dy + y + h, self.dx + x: self.dx + x + w], dtype=np.float32)
        b = np.empty_like(m)
        b[0::2, 0::2], b[0::2, 1::2], b[1::2, 0::2], b[1::2, 1::2] = self.black4
        return np.clip((m - b) / (self.white - b), 0, None)

    def gt_raw_region(self, y: int, x: int, h: int, w: int) -> np.ndarray:
        """Clean (150-frame mean) normalised RAW, RGGB-aligned."""
        if not hasattr(self, "_gt_raw"):
            self._gt_raw = np.load(os.path.join(self.dir, "gt.npy"), mmap_mode="r")
        return np.asarray(self._gt_raw[self.dy + y: self.dy + y + h, self.dx + x: self.dx + x + w], dtype=np.float32)

    def gt_crop(self, y: int, x: int, size: int) -> np.ndarray:
        return np.asarray(self.gt_srgb[y: y + size, x: x + size], dtype=np.float32) / 65535.0
