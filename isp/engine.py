"""Run an ISP network on full-size images by overlapping tiles, with a PyTorch or ONNX Runtime backend.

Tiles are 256x256 at packed (half) resolution = 512x512 output, the same size the networks were
trained on (keeps NAFNet's global channel attention consistent between training and inference).
"""
import numpy as np

TILE = 256
OVERLAP = 32


class TorchEngine:
    def __init__(self, ckpt_or_model, name: str | None = None, device: str | None = None, half: bool = False):
        import torch

        from .models import build

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(ckpt_or_model, str):
            ck = torch.load(ckpt_or_model, map_location="cpu", weights_only=False)
            model = build(name or ck["config"]["model"])
            model.load_state_dict(ck["model"])
        else:
            model = ckpt_or_model
        self.model = model.to(self.device).eval()
        self.dtype = torch.float16 if half else torch.float32
        if half:
            self.model = self.model.half()
        self.in_ch = 4 if getattr(model, "use_sigma", False) is False else 5
        self.kind = "e2e" if model.__class__.__name__ == "E2EISPNet" else "hybrid"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        t = self.torch.from_numpy(x).to(self.device, self.dtype)
        with self.torch.no_grad():
            if self.in_ch == 5:
                y = self.model(t[:, :4], t[:, 4:5])
            else:
                y = self.model(t[:, :4])
        return y.float().cpu().numpy()


class OnnxEngine:
    def __init__(self, path: str, threads: int = 0):
        import onnxruntime as ort

        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0]
        self.in_ch = self.inp.shape[1]
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.kind = meta.get("kind", "hybrid")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        outs = [self.sess.run(None, {self.inp.name: x[i:i + 1].astype(np.float32)})[0] for i in range(len(x))]
        return np.concatenate(outs, 0)


def _window(n: int, overlap: int) -> np.ndarray:
    w = np.ones(n, np.float32)
    ramp = (np.arange(overlap, dtype=np.float32) + 0.5) / overlap
    w[:overlap], w[-overlap:] = ramp, ramp[::-1]
    return w


def run_tiled(engine, inp: np.ndarray, tile: int = TILE, overlap: int = OVERLAP, batch: int = 4) -> np.ndarray:
    """inp: (h, w, C) packed input -> (2h, 2w, 3) output, blended across tile overlaps."""
    h, w, c = inp.shape
    ph, pw = max(tile - h, 0), max(tile - w, 0)
    x = np.pad(inp, ((0, ph), (0, pw), (0, 0)), mode="reflect") if (ph or pw) else inp
    H, W = x.shape[:2]
    step = tile - overlap
    ys = list(range(0, max(H - tile, 0) + 1, step)) or [0]
    xs = list(range(0, max(W - tile, 0) + 1, step)) or [0]
    if ys[-1] + tile < H:
        ys.append(H - tile)
    if xs[-1] + tile < W:
        xs.append(W - tile)
    out = np.zeros((2 * H, 2 * W, 3), np.float32)
    acc = np.zeros((2 * H, 2 * W, 1), np.float32)
    wy = _window(2 * tile, 2 * overlap)
    win = (wy[:, None] * wy[None, :])[..., None]
    coords = [(y, xx) for y in ys for xx in xs]
    for i in range(0, len(coords), batch):
        cs = coords[i:i + batch]
        tiles = np.stack([x[y:y + tile, xx:xx + tile].transpose(2, 0, 1) for y, xx in cs]).astype(np.float32)
        pred = engine(np.ascontiguousarray(tiles))
        for (y, xx), p in zip(cs, pred):
            out[2 * y:2 * y + 2 * tile, 2 * xx:2 * xx + 2 * tile] += p.transpose(1, 2, 0) * win
            acc[2 * y:2 * y + 2 * tile, 2 * xx:2 * xx + 2 * tile] += win
    out /= np.maximum(acc, 1e-8)
    return out[: 2 * h, : 2 * w]
