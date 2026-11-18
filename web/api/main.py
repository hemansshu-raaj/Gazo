"""FastAPI app for the "Take a Pic" website: serves site/ and a small JSON API.

Run locally (from the repo root):
    .venv/bin/uvicorn web.api.main:app --reload --port 8000      ->  http://localhost:8000

API
    GET  /api/health                      -> {"ok": true}
    GET  /api/examples                    -> bundled examples [{id, title, file, has_ground_truth}]
    GET  /api/examples/{id}/thumb.jpg     -> small preview image of an example
    POST /api/upload   (multipart "file") -> {"upload_id"} after checking it is a readable RAW file
    POST /api/process  {"method": "classic"|"learned"|"hybrid",
                        "upload_id": "..." | "example": "...",
                        "options": {"model": "student"|"teacher", "auto_exposure": bool,
                                    "sharpen": "auto"|true|false, "full_res": bool}}   (options optional)
                                          -> {final, stages: [{id, image, zoom, info}], time_ms, ...}
                                             (images are {src: data URL, w, h}; zoom is a 256x256 data URL)
    GET  /api/compare/{example_id}        -> all three methods on a bundled example (+ metrics)
"""
import os
import shutil
import tempfile
import time
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from isp.io import load_any
from isp.types import ISPInputError

from . import service

SITE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "site")
UPLOAD_DIR = os.environ.get("ISP_UPLOAD_DIR", os.path.join(tempfile.gettempdir(), "take-a-pic-uploads"))
MAX_UPLOAD_BYTES = 80 * 1024 * 1024
UPLOAD_TTL_S = 2 * 60 * 60
RAW_EXTS = {".dng", ".npz", ".png", ".tif", ".tiff", ".nef", ".cr2", ".cr3", ".arw", ".orf", ".rw2", ".raf"}
PHOTO_EXTS = {".jpg", ".jpeg", ".heic", ".heif", ".webp", ".gif", ".bmp", ".avif"}
NEEDS_RAW = ("This is a finished photo, not a RAW file. Our ISP starts from the sensor's raw data, "
             "so it needs a .dng. Try one of the example files instead.")

os.makedirs(UPLOAD_DIR, exist_ok=True)
app = FastAPI(title="Take a Pic: Hybrid ISP", docs_url="/api/docs", openapi_url="/api/openapi.json")
app.add_middleware(GZipMiddleware, minimum_size=1024)   # three.js and the phone model shrink ~4x


# ---------------------------------------------------------------- uploads
def _upload_path(upload_id: str) -> str | None:
    if not upload_id or not all(c in "0123456789abcdef" for c in upload_id):
        return None
    for name in os.listdir(UPLOAD_DIR):
        if name.startswith(upload_id + "."):
            return os.path.join(UPLOAD_DIR, name)
    return None


def _clean_old_uploads():
    now = time.time()
    for name in os.listdir(UPLOAD_DIR):
        p = os.path.join(UPLOAD_DIR, name)
        if now - os.path.getmtime(p) > UPLOAD_TTL_S:
            os.remove(p)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext in PHOTO_EXTS:
        raise HTTPException(415, NEEDS_RAW)
    if ext not in RAW_EXTS:
        raise HTTPException(415, f"'{ext or 'this file'}' is not a RAW format we can read. Use a .dng file.")
    _clean_old_uploads()
    upload_id = uuid.uuid4().hex
    path = os.path.join(UPLOAD_DIR, upload_id + ext)
    with open(path, "wb") as out:
        shutil.copyfileobj(file.file, out, length=1 << 20)
    if os.path.getsize(path) > MAX_UPLOAD_BYTES:
        os.remove(path)
        raise HTTPException(413, "That file is larger than 80 MB.")
    try:  # read it once now, so a broken or ordinary-photo file fails here and not after the transition
        raw = await run_in_threadpool(load_any, path)
    except ISPInputError as e:
        os.remove(path)
        msg = str(e)
        raise HTTPException(415, NEEDS_RAW if ext in (".png", ".tif", ".tiff") and "16-bit" in msg else msg)
    except Exception as e:  # noqa: BLE001 - never show a stack trace
        os.remove(path)
        raise HTTPException(415, f"Could not read this file ({type(e).__name__}).")
    h, w = raw.shape
    return {"upload_id": upload_id, "name": file.filename, "width": w, "height": h}


# ---------------------------------------------------------------- processing
class ProcessRequest(BaseModel):
    method: str
    upload_id: str | None = None
    example: str | None = None
    options: dict | None = None    # Advanced screen: {model, auto_exposure, sharpen, full_res}


@app.post("/api/process")
async def process(req: ProcessRequest):
    if req.method not in service.METHODS:
        raise HTTPException(400, f"Unknown method '{req.method}'. Use classic, learned or hybrid.")
    if req.example:
        path = service.example_path(req.example)
        if not path:
            raise HTTPException(404, "Unknown example.")
    else:
        path = _upload_path(req.upload_id or "")
        if not path:
            raise HTTPException(404, "Upload not found or expired. Drop the file on the phone again.")
    try:
        result = await run_in_threadpool(service.run_method, path, req.method, req.options)
    except ISPInputError as e:
        raise HTTPException(415, str(e))
    result["source"] = {"example": req.example} if req.example else {"upload_id": req.upload_id}
    return result


@app.get("/api/compare/{example_id}")
async def compare(example_id: str):
    if not service.example_path(example_id):
        raise HTTPException(404, "Comparisons are only available for the bundled examples.")
    return await run_in_threadpool(service.compare_example, example_id)


# ---------------------------------------------------------------- examples + health
@app.get("/api/examples")
async def examples():
    return service.list_examples()


@app.get("/api/examples/{example_id}/thumb.jpg")
async def example_thumb(example_id: str):
    if not service.example_path(example_id):
        raise HTTPException(404, "Unknown example.")
    jpg = await run_in_threadpool(service.example_thumbnail, example_id)
    return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/health")
async def health():
    return {"ok": True}


@app.middleware("http")
async def revalidate_site_files(request, call_next):
    """Browsers must re-check site files on every load, so edits show up without a hard refresh."""
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-cache")
    return response


# the website itself (mounted last so /api/* wins)
app.mount("/", StaticFiles(directory=SITE_DIR, html=True), name="site")
