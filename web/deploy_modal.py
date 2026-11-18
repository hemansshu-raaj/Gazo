"""Host the "Take a Pic" website on Modal (CPU only, scales to zero when nobody is using it).

    .venv/bin/modal deploy web/deploy_modal.py        # from the repo root  (or: make deploy)

URL: https://<modal-workspace>--take-a-pic.modal.run
Set KEEP_WARM=1 when deploying to keep one container always on (no cold start, e.g. during judging).
"""
import os

import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEEP_WARM = int(os.environ.get("KEEP_WARM", "0"))

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgomp1")
    .pip_install_from_requirements(os.path.join(ROOT, "web", "requirements.txt"))
    .env({"ISP_THREADS": "4", "ISP_UPLOAD_DIR": "/tmp/take-a-pic-uploads", "PYTHONPATH": "/root"})
    .add_local_dir(os.path.join(ROOT, "isp"), "/root/isp", ignore=["**/__pycache__/**"])
    .add_local_dir(os.path.join(ROOT, "web"), "/root/web",
                   ignore=["**/__pycache__/**", "blender/*.png", "tests/**", "tools/**"])
    .add_local_dir(os.path.join(ROOT, "weights"), "/root/weights")
    .add_local_dir(os.path.join(ROOT, "examples"), "/root/examples")
)

app = modal.App("take-a-pic", image=image)


# One container: uploads are kept in its /tmp between "upload" and "process", so every request must
# reach the same container. 4 CPUs answer a 4 MP photo in a few seconds.
@app.function(cpu=4, memory=4096, timeout=600, max_containers=1, min_containers=KEEP_WARM,
              scaledown_window=15 * 60)
@modal.concurrent(max_inputs=20)
@modal.asgi_app(label="take-a-pic")
def web():
    from web.api.main import app as fastapi_app

    return fastapi_app
