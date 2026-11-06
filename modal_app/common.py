"""Shared Modal objects: app, container image and storage volumes."""
import modal

APP_NAME = "isp-p2"
DATA_DIR = "/data"   # SIDD scenes after preprocessing
CKPT_DIR = "/ckpt"   # checkpoints, logs, eval outputs

data_vol = modal.Volume.from_name("sidd-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("isp-ckpt", create_if_missing=True)

# CPU image for data preparation.
data_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy<2.3", "rawpy", "exifread", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("common")
)

# GPU image for training / evaluation; ships the local `isp` package.
train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy<2.3", "scipy", "torch", "torchvision", "rawpy", "exifread",
        "scikit-image", "colour-demosaicing", "opencv-python-headless", "lpips",
        "onnx", "onnxruntime", "onnxscript", "psutil", "matplotlib",
    )
    .add_local_python_source("isp", "common")
)
