"""Host the Gradio app on Modal (CPU, scales to zero when idle).

  modal deploy modal_app/serve.py          # from the repo root
Set KEEP_WARM=1 at deploy time to keep one container always on (e.g. during judging).
"""
import os

import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEEP_WARM = int(os.environ.get("KEEP_WARM", "0"))

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgomp1")
    .pip_install_from_requirements(os.path.join(ROOT, "requirements-app.txt"))
    .pip_install("fastapi[standard]")
    .env({"GRADIO_TEMP_DIR": "/tmp/gradio", "ISP_THREADS": "4"})
    .add_local_python_source("isp")
    .add_local_file(os.path.join(ROOT, "app.py"), "/root/app.py")
    .add_local_dir(os.path.join(ROOT, "weights"), "/root/weights")
    .add_local_dir(os.path.join(ROOT, "examples"), "/root/examples")
)

app = modal.App("hybrid-isp-app", image=image)


@app.function(cpu=4, memory=4096, timeout=600, max_containers=1, min_containers=KEEP_WARM,
              scaledown_window=15 * 60)
@modal.concurrent(max_inputs=50)
@modal.asgi_app(label="hybrid-isp")
def ui():
    import gradio as gr
    from fastapi import FastAPI

    import app as isp_app

    isp_app.demo.queue(max_size=16)
    return gr.mount_gradio_app(FastAPI(), isp_app.demo, path="/")
