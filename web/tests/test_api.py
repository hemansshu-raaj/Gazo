"""Tests for the website backend.  Run from the repo root:  .venv/bin/python -m pytest -q web/tests"""
import io
import json
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from isp.io import load_any
from isp.pipelines import hybrid, learned, traditional
from web.api import service, trace
from web.api.main import app

EX = "sidd_normal_nexus6"
client = TestClient(app)


def _raw():
    raw = load_any(service.example_path(EX))
    service.crop(raw, 250_000)  # small and fast
    return raw


# ---- the stage-by-stage traces must give exactly what the real pipelines give
def test_classic_matches_traditional_pipeline():
    ref, _, _ = traditional.run(_raw())
    out, stages = trace.trace_classic(_raw())
    np.testing.assert_allclose(out, ref, atol=1e-6)
    assert [s["id"] for s in stages][:2] == ["photosites", "bayer"]


def test_hybrid_matches_hybrid_pipeline():
    eng = service.engine("student_int8.onnx")
    ref, _, _ = hybrid.run(_raw(), eng)
    out, _ = trace.trace_hybrid(_raw(), eng)
    np.testing.assert_allclose(out, ref, atol=1e-6)


def test_learned_matches_learned_pipeline():
    eng = service.engine("e2e_fp32.onnx")
    ref, _, _ = learned.run(_raw(), eng)
    out, _ = trace.trace_learned(_raw(), eng)
    np.testing.assert_allclose(out, ref, atol=1e-6)


# ---- stages.json must describe every stage the backend produces, in the same order
@pytest.mark.parametrize("method", ["classic", "learned", "hybrid"])
def test_stages_json_matches_trace(method):
    content = json.load(open(os.path.join(os.path.dirname(__file__), "..", "site", "content", "stages.json")))
    fn_name, model = service.METHODS[method]
    fn = getattr(trace, fn_name)
    _, stages = fn(_raw(), service.engine(model)) if model else fn(_raw())
    ids = [s["id"] for s in stages]
    order = content["methods"][method]["order"]
    assert [i for i in order if i in ids] == ids      # every stage listed, in the same order
    assert all(i in content["stages"] for i in ids)


# ---- HTTP API
def test_examples_listed():
    r = client.get("/api/examples")
    assert r.status_code == 200
    ids = [e["id"] for e in r.json()]
    assert EX in ids
    assert next(e for e in r.json() if e["id"] == EX)["has_ground_truth"] is True


def test_process_example_returns_every_stage():
    r = client.post("/api/process", json={"method": "classic", "example": EX})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["final"]["src"].startswith("data:image/jpeg;base64,") and body["final"]["w"] > 0
    assert len(body["stages"]) == 8
    assert body["stages"][0]["zoom"].startswith("data:image/png;base64,")


def test_upload_then_process():
    with open(service.example_path(EX), "rb") as f:
        r = client.post("/api/upload", files={"file": ("mine.npz", f, "application/octet-stream")})
    assert r.status_code == 200, r.text
    uid = r.json()["upload_id"]
    r = client.post("/api/process", json={"method": "hybrid", "upload_id": uid})
    assert r.status_code == 200, r.text
    assert r.json()["stages"][-1]["id"] == "gamma_tone"


def test_jpeg_upload_is_refused_politely():
    r = client.post("/api/upload", files={"file": ("holiday.jpg", io.BytesIO(b"\xff\xd8\xff"), "image/jpeg")})
    assert r.status_code == 415
    assert "RAW" in r.json()["detail"]


def test_bad_method_and_unknown_upload():
    assert client.post("/api/process", json={"method": "magic", "example": EX}).status_code == 400
    assert client.post("/api/process", json={"method": "classic", "upload_id": "../../etc"}).status_code == 404


def test_compare_has_metrics_for_sidd_example():
    r = client.get(f"/api/compare/{EX}")
    assert r.status_code == 200
    body = r.json()
    assert [x["method"] for x in body["results"]] == ["classic", "learned", "hybrid"]
    assert body["ground_truth"] is not None
    hyb = next(x for x in body["results"] if x["method"] == "hybrid")["metrics"]["psnr"]
    cls = next(x for x in body["results"] if x["method"] == "classic")["metrics"]["psnr"]
    assert hyb > cls  # the whole point of the project


def test_site_is_served():
    assert client.get("/").status_code == 200
    assert client.get("/content/stages.json").status_code == 200


# ---- Advanced screen settings
@pytest.mark.parametrize("ae,sh", [(True, True), (False, False)])
def test_traces_match_pipelines_with_settings(ae, sh):
    cfg = {"auto_exposure": ae, "sharpen": sh}
    ref, _, _ = traditional.run(_raw(), cfg)
    out, st = trace.trace_classic(_raw(), cfg)
    np.testing.assert_allclose(out, ref, atol=1e-6)
    assert ("sharpen" in [s["id"] for s in st]) == sh
    eng = service.engine("student_int8.onnx")
    ref, _, _ = hybrid.run(_raw(), eng, cfg)
    out, st = trace.trace_hybrid(_raw(), eng, cfg)
    np.testing.assert_allclose(out, ref, atol=1e-6)
    assert ("sharpen" in [s["id"] for s in st]) == sh


def test_process_with_options():
    r = client.post("/api/process", json={"method": "hybrid", "example": EX,
                                          "options": {"model": "teacher", "auto_exposure": True, "sharpen": True}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["options"] == {"model": "teacher", "auto_exposure": True, "sharpen": True, "full_res": False}
    assert body["stages"][-1]["id"] == "sharpen"
    bad = client.post("/api/process", json={"method": "classic", "example": EX, "options": {"model": "huge"}}).json()
    assert bad["options"]["model"] == "student"       # unknown values fall back to defaults
