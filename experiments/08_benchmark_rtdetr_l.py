#!/usr/bin/env python
"""Stage 7 (variant) — RT-DETR-L benchmark run.

Same protocol as 07_benchmark_yolo11l.py; adds a retried pre-download of the
RT-DETR-L pretrained weights before training.
"""
import glob
import json
import os
import shutil
import subprocess
import sys

CONFIG = {
    "tag": "rtdetr-l",
    "weights": "rtdetr-l.pt",
    "imgsz": 640,
    "batch": 16,
    "epochs": 150,
}


def log(*a):
    print(*a, flush=True)


log("CONFIG:", CONFIG)

# ---------------------------------------------------------------- src bootstrap
def load_src_library():
    dst = "/tmp/lib/src"
    os.makedirs(dst, exist_ok=True)
    roots = [p for p in glob.glob("/kaggle/input/**/uwd-src*", recursive=True) if os.path.isdir(p)]
    roots.sort(key=len)
    src_root = roots[0] if roots else "/kaggle/input"
    for root, _, files in os.walk(src_root):
        for f in files:
            src = os.path.join(root, f)
            if f.endswith(".py"):
                shutil.copy(src, os.path.join(dst, f))
            elif f.endswith(".zip"):
                shutil.unpack_archive(src, "/tmp/lib/unzipped")
    for root, _, files in os.walk("/tmp/lib/unzipped"):
        for f in files:
            if f.endswith(".py"):
                shutil.copy(os.path.join(root, f), dst)
    sys.path.insert(0, dst)


load_src_library()
from uwd_utils import find_under_input, relocate_data_yaml, save_metrics  # noqa: E402

DATA = relocate_data_yaml(find_under_input("data_fixed.yaml"))

# pre-download weights with retries
import time as _t
import pathlib as _p

_w = "/tmp/rtdetr-l.pt"
if not _p.Path(_w).exists():
    for _u in ("https://github.com/ultralytics/assets/releases/download/v8.3.0/rtdetr-l.pt",):
        for _i in range(5):
            d = subprocess.run([sys.executable, "-c",
                                f"import urllib.request; urllib.request.urlretrieve({_u!r}, {_w!r})"],
                               capture_output=True, text=True)
            if d.returncode == 0 and _p.Path(_w).exists() and _p.Path(_w).stat().st_size > 1e6:
                break
            log(f"rtdetr-l download retry {_i}: {(d.stderr or '')[-160:]}")
            _t.sleep(20)
assert _p.Path(_w).exists() and _p.Path(_w).stat().st_size > 1e6, "rtdetr-l.pt download failed"

# dummy-font pre-place so ultralytics/yolov5 never fetch fonts mid-run
_font = _p.Path.home() / ".config" / "Ultralytics"
_font.mkdir(parents=True, exist_ok=True)
(_font / "Arial.ttf").write_bytes(b"")
assert DATA, "data_fixed.yaml not found under /kaggle/input"
WORK = "/kaggle/working"
log(f"data: {DATA}")

# ---------------------------------------------------------------- env
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"],
                   capture_output=True, text=True)
log(f"pip install ultralytics rc={r.returncode}" + ("" if r.returncode == 0 else " " + r.stderr[-500:]))
import ultralytics  # noqa: E402

log(f"ultralytics {ultralytics.__version__}")
from ultralytics import YOLO  # noqa: E402

# ---------------------------------------------------------------- train
model = YOLO("/tmp/rtdetr-l.pt")
model.train(
    data=DATA,
    imgsz=CONFIG["imgsz"],
    batch=CONFIG["batch"],
    epochs=CONFIG["epochs"],
    patience=60,
    cache="ram",
    device=0,
    project=f"{WORK}/train",
    name=CONFIG["tag"],
    exist_ok=True,
    verbose=False,
    plots=False,  # metric plots can crash on the font fetch; explicit val calls below cover metrics
)
best = f"{WORK}/train/{CONFIG['tag']}/weights/best.pt"
assert os.path.exists(best), f"best.pt missing at {best}"


# ---------------------------------------------------------------- eval helpers
def per_class_rows(m):
    names = m.names
    rows = []
    idx = m.box.ap_class_index
    for i, c in enumerate(idx):
        rows.append({
            "cls": names[int(c)],
            "P": round(float(m.box.p[i]), 4),
            "R": round(float(m.box.r[i]), 4),
            "mAP50": round(float(m.box.ap50[i]), 4),
            "mAP5095": round(float(m.box.ap[i]), 4),
        })
    return rows


def eval_split(split):
    out = {}
    for tta in (False, True):
        m = YOLO(best).val(
            data=DATA, split=split, imgsz=CONFIG["imgsz"], batch=CONFIG["batch"],
            device=0, augment=tta, verbose=False, plots=False,
        )
        key = f"{split}_tta" if tta else split
        out[key] = {
            "P": round(float(m.box.mp), 4), "R": round(float(m.box.mr), 4),
            "mAP50": round(float(m.box.map50), 4), "mAP5095": round(float(m.box.map), 4),
            "per_class": per_class_rows(m),
        }
        log(f"EVAL {key}: mAP50={out[key]['mAP50']:.4f} mAP5095={out[key]['mAP5095']:.4f}")
    return out


metrics = {"config": CONFIG, "ultralytics": ultralytics.__version__,
           "val": eval_split("val"), "test": eval_split("test")}
save_metrics(metrics, f"p4_metrics_{CONFIG['tag']}.json")
log(f"P4 {CONFIG['tag']} COMPLETE")
