"""Train ISP networks on Modal GPUs.

  PYTHONPATH=. modal run modal_app/train.py --run teacher --minutes 5 --steps 300     # smoke test
  PYTHONPATH=. modal run --detach modal_app/train.py --run teacher --minutes 150
Checkpoints / logs: volume isp-ckpt at /ckpt/<run>/{best,last}.pt, log.csv
"""
import json
import os

import modal

from common import APP_NAME, CKPT_DIR, DATA_DIR, ckpt_vol, data_vol, train_image

app = modal.App(APP_NAME + "-train")


@app.function(image=train_image, gpu="A10", cpu=6, memory=16384, timeout=4 * 3600,
              volumes={DATA_DIR: data_vol, CKPT_DIR: ckpt_vol})
def train_run(run: str, minutes: float, steps: int = 0, out_name: str = "", splits: dict | None = None):
    import torch

    from isp.train_data import SIDDEval, SIDDTrain
    from isp.train_loop import train

    torch.backends.cudnn.benchmark = True
    root = os.path.join(DATA_DIR, "scenes")
    tr = SIDDTrain(root, splits["train"], crop=512, noise_aug=0.3)
    va = SIDDEval(root, splits["val"], n=6, crop=512)
    out = os.path.join(CKPT_DIR, out_name or run)
    teacher = os.path.join(CKPT_DIR, "teacher", "best.pt")
    res = train(run, tr, va, out, device="cuda", max_minutes=minutes, steps=steps or None,
                teacher_ckpt=teacher, commit=ckpt_vol.commit, log_every=200, val_every=2500, workers=5)
    print("RESULT", json.dumps(res))
    return res


@app.local_entrypoint()
def main(run: str = "teacher", minutes: float = 150.0, steps: int = 0, out_name: str = "", gpu: str = "A10"):
    with open("splits/splits.json") as f:
        splits = json.load(f)
    fn = train_run if gpu == "A10" else train_run.with_options(gpu=gpu)
    print(fn.remote(run, minutes, steps, out_name, splits))
