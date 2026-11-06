"""Image-quality metrics (inputs: float RGB in [0,1], HxWx3)."""
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity

_lpips_model = None


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 100.0 if mse == 0 else 10 * np.log10(1.0 / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    return float(structural_similarity(a, b, channel_axis=-1, data_range=1.0))


def delta_e(a: np.ndarray, b: np.ndarray) -> float:
    """Mean CIEDE2000 colour difference (sRGB -> Lab, D65)."""
    return float(np.mean(deltaE_ciede2000(rgb2lab(np.clip(a, 0, 1)), rgb2lab(np.clip(b, 0, 1)))))


def lpips(a: np.ndarray, b: np.ndarray, device: str | None = None) -> float:
    import torch
    import lpips as _lp

    global _lpips_model
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if _lpips_model is None:
        _lpips_model = _lp.LPIPS(net="alex", verbose=False).to(dev).eval()
    to_t = lambda x: torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].float().to(dev) * 2 - 1
    with torch.no_grad():
        return float(_lpips_model.to(dev)(to_t(a), to_t(b)).item())


def all_metrics(pred: np.ndarray, gt: np.ndarray, with_lpips: bool = True) -> dict:
    pred = pred.astype(np.float32) / 255.0 if pred.dtype == np.uint8 else pred
    gt = gt.astype(np.float32) / 255.0 if gt.dtype == np.uint8 else gt
    m = {"psnr": psnr(pred, gt), "ssim": ssim(pred, gt), "delta_e": delta_e(pred, gt)}
    if with_lpips:
        m["lpips"] = lpips(pred, gt)
    return m
