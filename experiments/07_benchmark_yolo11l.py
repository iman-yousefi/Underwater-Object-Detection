#!/usr/bin/env python
"""Stage 7 — one Ultralytics-model benchmark run (model set in CONFIG).

Trains YOLO11-L (here) or RT-DETR-L (see 08) on the official split — same data
and epoch budget as the YOLOv5 runs — then evaluates on val and test, with and
without TTA.
"""
import glob
import json
import os
import shutil
import subprocess
import sys

CONFIG = {
    "tag": "yolo11l",
    "weights": "yolo11l.pt",
    "imgsz": 640,
    "batch": 32,
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
model = YOLO(CONFIG["weights"])
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
