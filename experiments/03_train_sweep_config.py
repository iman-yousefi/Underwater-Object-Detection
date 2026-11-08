#!/usr/bin/env python
"""Stage 3 — one YOLOv5 training run (model / image size set in CONFIG).

Run once per configuration and compare validation mAP@0.5 across runs to pick
the champion. Used here with: yolov5{m,l,x} @640 (batch 32/16/12) and
yolov5l @1024 (batch 6) — larger batches OOM on a 16 GB T4.
"""
import glob
import json
import os
import shutil
import subprocess
import sys

# Sweep: edit CONFIG per run — used: yolov5m/l/x @640 (batch 32/16/12) and yolov5l @1024 (batch 6).
CONFIG = {
    "tag": "m-640",
    "weights": "yolov5m.pt",
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
from uwd_utils import find_under_input, relocate_data_yaml, parse_results_csv, parse_val_table, save_metrics  # noqa: E402

# ---------------------------------------------------------------- p0 outputs
stats_path = find_under_input("p0_stats.json")
assert stats_path, "p0_stats.json not found under /kaggle/input"
with open(stats_path) as f:
    P0 = json.load(f)
PIN = P0.get("yolov5_clone", {}).get("commit")
DATA = relocate_data_yaml(find_under_input("data_fixed.yaml"))
assert DATA, "data_fixed.yaml not found under /kaggle/input"
log(f"p0: pin={PIN}  data={DATA}")

# ---------------------------------------------------------------- yolov5 setup
subprocess.run(["rm", "-rf", "/tmp/yolov5"], check=False)
r = subprocess.run(["git", "clone", "-q", "https://github.com/ultralytics/yolov5", "/tmp/yolov5"],
                   capture_output=True, text=True)
assert r.returncode == 0, f"yolov5 clone failed: {r.stderr[-500:]}"
if PIN:
    c = subprocess.run(["git", "-C", "/tmp/yolov5", "checkout", "-q", PIN], capture_output=True, text=True)
    log(f"checkout {PIN}: {'ok' if c.returncode == 0 else 'staying on HEAD'}")
head = subprocess.run(["git", "-C", "/tmp/yolov5", "rev-parse", "--short", "HEAD"],
                      capture_output=True, text=True).stdout.strip()
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "/tmp/yolov5/requirements.txt"],
                   capture_output=True, text=True)
log(f"yolov5 @ {head}, pip rc={r.returncode}")

# ---------------------------------------------------------------- train
WORK = "/kaggle/working"
tag = CONFIG["tag"]
CMD = [
    sys.executable, "/tmp/yolov5/train.py",
    "--img", str(CONFIG["imgsz"]),
    "--batch", str(CONFIG["batch"]),
    "--epochs", str(CONFIG["epochs"]),
    "--patience", "60",
    "--data", DATA,
    "--weights", CONFIG["weights"],
    "--project", f"{WORK}/train",
    "--name", tag,
    "--cache", "ram",
    "--device", "0",
    "--exist-ok",
]
log("TRAIN CMD:", " ".join(CMD))
tr = subprocess.run(CMD, capture_output=True, text=True)
log("---- train.py stdout tail ----")
log(tr.stdout[-4000:])
if tr.returncode != 0:
    log("---- train.py stderr tail ----")
    log(tr.stderr[-4000:])
    raise SystemExit(f"train.py failed rc={tr.returncode}")

run_dir = f"{WORK}/train/{tag}"
metrics = {"config": CONFIG, "yolov5_commit": head, "run_dir": run_dir,
           "final_epoch": parse_results_csv(f"{run_dir}/results.csv")}

# ---------------------------------------------------------------- val + test
for task in ("val", "test"):
    CMD = [
        sys.executable, "/tmp/yolov5/val.py",
        "--weights", f"{run_dir}/weights/best.pt",
        "--data", DATA,
        "--task", task,
        "--img", str(CONFIG["imgsz"]),
        "--batch", str(CONFIG["batch"]),
        "--project", f"{WORK}/val",
        "--name", tag,
        "--device", "0",
        "--exist-ok",
    ]
    log("VAL CMD:", " ".join(CMD))
    vr = subprocess.run(CMD, capture_output=True, text=True)
    log(f"---- val.py ({task}) stdout ----")
    log(vr.stdout[-3000:])
    if vr.returncode != 0:
        log(vr.stderr[-3000:])
        raise SystemExit(f"val.py {task} failed rc={vr.returncode}")
    metrics[task] = parse_val_table(vr.stdout)

save_metrics(metrics, f"p2_metrics_{tag}.json")
log(f"P2 {tag} COMPLETE")
