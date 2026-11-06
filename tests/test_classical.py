import numpy as np
import pytest
import torch

from isp.classical import bayer, ccm, tone, torch_tail, wb
from isp.noise import estimate_nlf
from isp.types import ISPInputError, RawImage

rng = np.random.default_rng(0)


def test_pack_unpack_roundtrip():
    x = rng.random((8, 12), dtype=np.float32)
    assert np.array_equal(bayer.unpack(bayer.pack(x)), x)


@pytest.mark.parametrize("letters,expect", [
    ([["R", "G"], ["G", "B"]], (0, 0)), ([["G", "R"], ["B", "G"]], (0, 1)),
    ([["G", "B"], ["R", "G"]], (1, 0)), ([["B", "G"], ["G", "R"]], (1, 1))])
def test_rggb_offsets(letters, expect):
    assert bayer.rggb_offset(letters) == expect


def test_torch_tail_matches_numpy():
    rgb = rng.random((16, 16, 3), dtype=np.float32) ** 2
    m = ccm.cam2srgb_from_color_matrix(np.array([[0.8, -0.2, -0.05], [-0.4, 1.2, 0.2], [-0.05, 0.15, 0.6]]))
    ref = tone.tone_curve(tone.srgb_gamma(ccm.apply_ccm(rgb, m)))
    t = torch_tail.tail(torch.from_numpy(rgb).permute(2, 0, 1)[None], torch.from_numpy(m)[None])
    assert np.abs(t[0].permute(1, 2, 0).numpy() - ref).max() < 1e-4
    lut = tone.display_transform(ccm.apply_ccm(rgb, m))
    assert np.abs(lut - ref).max() < 2e-3


def test_ccm_keeps_neutral():
    m = ccm.cam2srgb_from_color_matrix(np.array([[0.9, -0.3, -0.1], [-0.4, 1.3, 0.1], [-0.1, 0.2, 0.7]]))
    assert np.allclose(m @ np.ones(3), 1.0, atol=1e-5)


def test_gray_world_recovers_cast():
    flat = np.full((64, 64), 0.4, np.float32)
    cast = wb.apply_wb(flat, np.array([0.5, 1.0, 0.8], np.float32))
    g = wb.gray_world(cast)
    assert np.allclose(g, [2.0, 1.0, 1.25], rtol=1e-3)


def test_nlf_estimate_close_to_truth():
    a, b = 4e-4, 2e-6
    clean = np.tile(np.linspace(0.02, 0.8, 512, dtype=np.float32), (512, 1))
    noisy = clean + rng.normal(size=clean.shape).astype(np.float32) * np.sqrt(a * clean + b)
    ea, eb = estimate_nlf(noisy)
    assert abs(ea - a) / a < 0.25


def test_bad_inputs(tmp_path):
    from isp.io import load_any
    p = tmp_path / "x.txt"
    p.write_text("hello")
    with pytest.raises(ISPInputError):
        load_any(str(p))
    d = tmp_path / "fake.dng"
    d.write_bytes(b"not a raw file" * 100)
    with pytest.raises(ISPInputError):
        load_any(str(d))


def test_traditional_runs_on_synthetic():
    from isp.pipelines import traditional
    raw = RawImage(bayer=rng.random((64, 64), dtype=np.float32) * 0.5)
    out, stages, info = traditional.run(raw, keep_stages=True)
    assert out.shape == (64, 64, 3) and out.min() >= 0 and out.max() <= 1
    assert info["wb_source"] == "gray-world"
