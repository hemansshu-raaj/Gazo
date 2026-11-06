"""Export a trained checkpoint to ONNX (static 256x256 packed tile, matching training crops).

  python -m optimize.export_onnx runs/ckpt/teacher.pt weights/teacher_fp32.onnx [--fp16]
"""
import argparse
import os

import onnx
import torch

from isp.engine import TILE
from isp.models import build


class _Wrap(torch.nn.Module):
    """Single-tensor input (packed [+ sigma]) for ONNX."""

    def __init__(self, m, in_ch):
        super().__init__()
        self.m, self.in_ch = m, in_ch

    def forward(self, x):
        return self.m(x[:, :4], x[:, 4:5]) if self.in_ch == 5 else self.m(x[:, :4])


def export(ckpt: str, out: str, fp16: bool = False) -> str:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    name = ck["config"]["model"]
    model = build(name)
    model.load_state_dict(ck["model"])
    model.eval()
    in_ch = 5 if getattr(model, "use_sigma", False) else 4
    w = _Wrap(model, in_ch).eval()
    x = torch.rand(1, in_ch, TILE, TILE)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.onnx.export(w, (x,), out, input_names=["input"], output_names=["output"], opset_version=17,
                      dynamo=False)
    m = onnx.load(out)
    kind = "e2e" if name == "e2e" else "hybrid"
    for k, v in {"kind": kind, "model": name, "run": ck.get("run", ""), "step": str(ck.get("step", ""))}.items():
        p = m.metadata_props.add(); p.key, p.value = k, v
    if fp16:
        from onnxconverter_common import float16
        m = float16.convert_float_to_float16(m, keep_io_types=True)
    onnx.save(m, out)
    print(f"exported {name} -> {out} ({os.path.getsize(out)/1e6:.2f} MB)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("out"); ap.add_argument("--fp16", action="store_true")
    a = ap.parse_args()
    export(a.ckpt, a.out, a.fp16)
