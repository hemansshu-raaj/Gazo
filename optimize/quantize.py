"""Static post-training quantisation with ONNX Runtime (QDQ, per-channel INT8 weights).

Default = W8A16: INT8 convolution weights, 16-bit activations. Full W8A8 (8-bit activations) loses ~3.4 dB on
this RAW-domain network: dark pixels produce tiny activations next to bright ones, and a single 8-bit range per
tensor cannot represent both (shadows posterise). W8A16 keeps the 3.6x size reduction with no measurable loss.

Calibration data: real packed RAW tiles (+ sigma maps) from validation scenes and unseen DNGs.
  python -m optimize.quantize weights/student_fp32.onnx weights/student_int8.onnx --calib data/scenes
"""
import argparse
import glob
import os

import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from isp.classical import wb
from isp.classical.bayer import pack
from isp.engine import TILE
from isp.io import load_any
from isp.noise import estimate_nlf, noise_sigma_map


def calibration_tiles(sources, n: int = 64, in_ch: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    tiles = []
    per = max(1, n // max(len(sources), 1))
    for src in sources:
        raw = load_any(src)
        g = raw.wb_gains if raw.wb_gains is not None else wb.gray_world(raw.bayer)
        packed = pack(np.clip(wb.apply_wb(raw.bayer, g, clip=False), 0, 1))
        sig = noise_sigma_map(packed, raw.nlf or estimate_nlf(raw.bayer), g)
        inp = np.concatenate([packed, sig[..., None]], -1) if in_ch == 5 else pack(raw.bayer)
        h, w = inp.shape[:2]
        for _ in range(per):
            y, x = rng.integers(0, h - TILE), rng.integers(0, w - TILE)
            tiles.append(inp[y:y + TILE, x:x + TILE].transpose(2, 0, 1)[None].astype(np.float32))
    return tiles[:n]


class _Reader(CalibrationDataReader):
    def __init__(self, tiles):
        self.it = iter([{"input": t} for t in tiles])

    def get_next(self):
        return next(self.it, None)


def quantize(src: str, dst: str, calib_sources, exclude_io_convs: bool = False, n: int = 24, conv_only: bool = True,
             act16: bool = True):
    """conv_only: quantise only the convolutions (where the weights and compute are). The RAW input,
    the bilinear base image and the residual add stay float, because 8-bit RAW input destroys shadow detail
    (dark pixels occupy only a few of the 256 levels)."""
    m = onnx.load(src)
    in_ch = m.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    pre = dst.replace(".onnx", "_pre.onnx")
    quant_pre_process(src, pre, skip_symbolic_shape=True)
    nodes = [n_.name for n_ in onnx.load(pre).graph.node if n_.op_type == "Conv"]
    exclude = [nodes[0], nodes[-1]] if exclude_io_convs and nodes else []
    quantize_static(pre, dst, _Reader(calibration_tiles(calib_sources, n, in_ch)), quant_format=QuantFormat.QDQ,
                    per_channel=True, weight_type=QuantType.QInt8,
                    activation_type=QuantType.QUInt16 if act16 else QuantType.QUInt8,
                    nodes_to_exclude=exclude, op_types_to_quantize=["Conv"] if conv_only else None,
                    extra_options={"ActivationSymmetric": False})
    os.remove(pre)
    q = onnx.load(dst)
    del q.metadata_props[:]
    for p in m.metadata_props:
        q.metadata_props.add().CopyFrom(p)
    q.metadata_props.add().key = "precision"
    q.metadata_props[-1].value = "int8 weights, " + ("16-bit" if act16 else "8-bit") + " activations"
    onnx.save(q, dst)
    print(f"INT8 -> {dst} ({os.path.getsize(dst)/1e6:.2f} MB, excluded {exclude})")
    return dst


def default_sources(scene_root: str):
    import json
    val = json.load(open("splits/splits.json"))["val"]
    scenes = [os.path.join(scene_root, s) for s in val if os.path.isdir(os.path.join(scene_root, s))]
    scenes = scenes or sorted(os.path.dirname(p) for p in glob.glob(os.path.join(scene_root, "*", "meta.json")))[:4]
    return scenes + sorted(glob.glob("data/unseen_dng/*.dng"))[:2]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--calib", default="data/scenes")
    ap.add_argument("--exclude-io", action="store_true", help="keep first/last conv in float")
    ap.add_argument("--all-ops", action="store_true", help="quantise every op (not only Conv)")
    ap.add_argument("--a8", action="store_true", help="8-bit activations (W8A8) instead of 16-bit")
    a = ap.parse_args()
    quantize(a.src, a.dst, default_sources(a.calib), a.exclude_io, conv_only=not a.all_ops, act16=not a.a8)
