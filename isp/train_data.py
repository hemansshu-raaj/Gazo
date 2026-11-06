"""Training / validation samples from preprocessed SIDD scenes.

Each sample: one random noisy frame crop (RGGB), white-balanced and packed, its noise sigma map,
the camera->sRGB matrix, and the matching ground-truth sRGB crop.
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .classical.bayer import pack
from .data import SceneArrays, eval_boxes
from .noise import noise_sigma_map


def make_sample(noisy: np.ndarray, gt: np.ndarray, sc: SceneArrays, nlf, rng=None, noise_aug: float = 0.0):
    """noisy: normalised RGGB crop (S,S); gt: sRGB crop (S,S,3)."""
    a, b = nlf
    if rng is not None and noise_aug > 0 and rng.random() < noise_aug:
        # Simulate a higher ISO / darker capture: add Poisson-Gaussian noise, update the noise model.
        k = rng.uniform(0.5, 4.0)
        extra = np.sqrt(np.maximum(k * a * noisy + k * b, 0)).astype(np.float32)
        noisy = noisy + extra * rng.standard_normal(noisy.shape).astype(np.float32)
        a, b = a * (1 + k), b * (1 + k)
    raw_packed = pack(np.clip(noisy, 0, 1))
    g4 = np.array([sc.wb[0], sc.wb[1], sc.wb[1], sc.wb[2]], np.float32)
    wb_packed = np.clip(raw_packed * g4, 0, 1)
    sigma = noise_sigma_map(wb_packed, (max(a, 1e-8), max(b, 1e-10)), sc.wb)
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x))
    return {"wb": t(wb_packed.transpose(2, 0, 1)), "raw": t(raw_packed.transpose(2, 0, 1)),
            "sigma": t(sigma[None]), "ccm": t(sc.cam2srgb.astype(np.float32)), "gt": t(gt.transpose(2, 0, 1))}


class SIDDTrain(Dataset):
    def __init__(self, root, scenes, crop=512, length=100000, noise_aug=0.3, seed=0):
        self.scenes = [SceneArrays(os.path.join(root, s)) for s in scenes]
        self.crop, self.length, self.noise_aug, self.seed = crop, length, noise_aug, seed

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i, os.getpid()))
        sc = self.scenes[rng.integers(len(self.scenes))]
        S = self.crop
        y = int(rng.integers(0, (sc.h - S - 2) // 2)) * 2
        x = int(rng.integers(0, (sc.w - S - 2) // 2)) * 2
        noisy = sc.noisy_crop(int(rng.integers(len(sc.noisy))), y, x, S + 2)
        gt = sc.gt_crop(y, x, S + 2)
        # Bayer-preserving flips: flip, then shift by one pixel so the crop is RGGB again.
        if rng.random() < 0.5:
            noisy, gt = noisy[:, ::-1], gt[:, ::-1]
            noisy, gt = noisy[:, 1:], gt[:, 1:]
        if rng.random() < 0.5:
            noisy, gt = noisy[::-1], gt[::-1]
            noisy, gt = noisy[1:], gt[1:]
        noisy, gt = noisy[:S, :S], gt[:S, :S]
        return make_sample(np.ascontiguousarray(noisy), np.ascontiguousarray(gt), sc, sc.nlf or (1e-4, 1e-6),
                           rng, self.noise_aug)


class SIDDEval(Dataset):
    """Fixed crops (noisy frame 0) for validation."""

    def __init__(self, root, scenes, n=8, crop=512):
        self.items = []
        self.scenes = {s: SceneArrays(os.path.join(root, s)) for s in scenes}
        for s, sc in self.scenes.items():
            for y, x in eval_boxes(s, sc.h, sc.w, n=n, size=crop):
                self.items.append((s, y, x))
        self.crop = crop

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        s, y, x = self.items[i]
        sc = self.scenes[s]
        return make_sample(sc.noisy_crop(0, y, x, self.crop), sc.gt_crop(y, x, self.crop), sc, sc.nlf or (1e-4, 1e-6))
