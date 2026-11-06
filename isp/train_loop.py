"""Training loop shared by all runs (local smoke test and Modal)."""
import csv
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .classical.torch_tail import tail
from .models import build, n_params

CONFIGS = {
    #            model        input  KD teacher   lr     steps
    "teacher": dict(model="teacher", kd=None, lr=1e-3, steps=28000, batch=16),
    "nosigma": dict(model="nosigma", kd=None, lr=1e-3, steps=14000, batch=16),
    "e2e": dict(model="e2e", kd=None, lr=1e-3, steps=22000, batch=16),
    "student_kd": dict(model="student", kd="teacher", lr=1e-3, steps=30000, batch=16),
    "student_scratch": dict(model="student", kd=None, lr=1e-3, steps=30000, batch=16),
}


def charbonnier(a, b, eps=1e-3):
    return torch.sqrt((a - b) ** 2 + eps ** 2).mean()


def predict_srgb(model, batch, kind):
    """Network + classical tail. kind='e2e' outputs sRGB directly."""
    if kind == "e2e":
        return model(batch["raw"]).clamp(0, 1)
    return tail(model(batch["wb"], batch["sigma"]), batch["ccm"])


def psnr_t(a, b):
    mse = F.mse_loss(a.clamp(0, 1), b, reduction="none").flatten(1).mean(1)
    return (10 * torch.log10(1 / mse.clamp_min(1e-10))).mean().item()


def evaluate(model, loader, kind, device):
    model.eval()
    vals = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        for b in loader:
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            vals.append(psnr_t(predict_srgb(model, b, kind).float(), b["gt"]))
    model.train()
    return sum(vals) / len(vals)


def train(run: str, train_ds, val_ds, out_dir: str, device="cuda", max_minutes=150.0, steps=None,
          teacher_ckpt=None, commit=lambda: None, log_every=100, val_every=2000, workers=8):
    cfg = dict(CONFIGS[run])
    if steps:
        cfg["steps"] = steps
    kind = "e2e" if cfg["model"] == "e2e" else "hybrid"
    os.makedirs(out_dir, exist_ok=True)
    model = build(cfg["model"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4, betas=(0.9, 0.9))
    sched = lambda s: 1e-3 + (1 - 1e-3) * 0.5 * (1 + math.cos(math.pi * min(s, cfg["steps"]) / cfg["steps"]))
    teacher = None
    if cfg["kd"]:
        teacher = build(cfg["kd"]).to(device).eval()
        teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device)["model"])
        for p in teacher.parameters():
            p.requires_grad_(False)
    step, best = 0, -1.0
    last = os.path.join(out_dir, "last.pt")
    if os.path.exists(last):  # resume after preemption / timeout
        ck = torch.load(last, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        step, best = ck["step"], ck["best"]
        print(f"resumed from step {step} (best {best:.2f} dB)")
    print(f"run={run} params={n_params(model)/1e6:.2f}M steps={cfg['steps']}")
    dl = DataLoader(train_ds, batch_size=cfg["batch"], num_workers=workers, pin_memory=True, drop_last=True,
                    persistent_workers=workers > 0)
    vl = DataLoader(val_ds, batch_size=8, num_workers=min(workers, 4))
    logf = open(os.path.join(out_dir, "log.csv"), "a", newline="")
    log = csv.writer(logf)
    t0, t_save, it = time.time(), time.time(), iter(dl)

    def save(path):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best,
                    "run": run, "config": cfg}, path)

    while step < cfg["steps"]:
        try:
            b = next(it)
        except StopIteration:
            it = iter(dl); b = next(it)
        b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        for g in opt.param_groups:
            g["lr"] = cfg["lr"] * sched(step)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            pred = predict_srgb(model, b, kind).float()
            loss = charbonnier(pred, b["gt"])
            if teacher is not None:
                with torch.no_grad():
                    tpred = predict_srgb(teacher, b, "hybrid").float()
                loss = loss + 0.5 * charbonnier(pred, tpred)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        step += 1
        if step % log_every == 0:
            el = time.time() - t0
            print(f"step {step} loss {loss.item():.4f} lr {opt.param_groups[0]['lr']:.2e} {el/60:.1f}min", flush=True)
            log.writerow([step, "train_loss", loss.item(), el]); logf.flush()
        done = (time.time() - t0) / 60 > max_minutes or step >= cfg["steps"]
        if step % val_every == 0 or done:
            v = evaluate(model, vl, kind, device)
            print(f"step {step} VAL psnr {v:.2f} dB (best {max(best, v):.2f})", flush=True)
            log.writerow([step, "val_psnr", v, time.time() - t0]); logf.flush()
            if v > best:
                best = v
                save(os.path.join(out_dir, "best.pt"))
            save(last); commit(); t_save = time.time()
        elif time.time() - t_save > 600:
            save(last); commit(); t_save = time.time()
        if done:
            break
    logf.close()
    save(last); commit()
    return {"run": run, "step": step, "best_val_psnr": best}
