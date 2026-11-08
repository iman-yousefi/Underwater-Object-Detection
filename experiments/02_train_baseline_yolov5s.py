#!/usr/bin/env python
"""Stage 2 — YOLOv5s @ 640 baseline on the official train/val split.

Pipeline:
  1. load shared utils (shipped as a small dataset, see README)
  2. read p0 outputs (data_fixed.yaml, pinned yolov5 commit)
  3. clone yolov5 @ pinned commit, pip install requirements
  4. train yolov5s @640, 100 epochs (early-stop patience 50), cache images
  5. val.py on val split and test split with best.pt
  6. metrics.json to /kaggle/working (per-class tables + final-epoch curve)
"""
import glob
import json
import os
import shutil
import subprocess
import sys


def log(*a):
    print(*a, flush=True)


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
from uwd_utils import find_under_input, parse_results_csv, parse_val_table, save_metrics  # noqa: E402

# ---------------------------------------------------------------- p0 outputs
stats_path = find_under_input("p0_stats.json")
assert stats_path, "p0_stats.json not found under /kaggle/input (is uwd-p0-smoke attached?)"
with open(stats_path) as f:
    P0 = json.load(f)
PIN = P0.get("yolov5_clone", {}).get("commit")
log(f"p0: pinned yolov5 commit = {PIN}, gpus = {P0.get('gpus')}")

data_yaml_path = find_under_input("data_fixed.yaml")
assert data_yaml_path, "data_fixed.yaml not found under /kaggle/input"
DATA = data_yaml_path
log(f"data yaml: {DATA}")

# ---------------------------------------------------------------- yolov5 setup
subprocess.run(["rm", "-rf", "/tmp/yolov5"], check=False)
r = subprocess.run(["git", "clone", "-q", "https://github.com/ultralytics/yolov5", "/tmp/yolov5"],
                   capture_output=True, text=True)
assert r.returncode == 0, f"yolov5 clone failed: {r.stderr[-500:]}"
if PIN:
    c = subprocess.run(["git", "-C", "/tmp/yolov5", "checkout", "-q", PIN], capture_output=True, text=True)
    log(f"checkout {PIN}: {'ok' if c.returncode == 0 else 'FAILED (staying on HEAD): ' + c.stderr[-200:]}")
head = subprocess.run(["git", "-C", "/tmp/yolov5", "rev-parse", "--short", "HEAD"],
                      capture_output=True, text=True).stdout.strip()
log(f"yolov5 @ {head}")

r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "/tmp/yolov5/requirements.txt"],
                   capture_output=True, text=True)
log(f"pip install requirements rc={r.returncode}" + ("" if r.returncode == 0 else r.stderr[-500:]))

# ---------------------------------------------------------------- train
WORK = "/kaggle/working"
os.makedirs(f"{WORK}/train", exist_ok=True)
CMD = [
    sys.executable, "/tmp/yolov5/train.py",
    "--img", "640",
    "--batch", "32",
    "--epochs", "100",
    "--patience", "50",
    "--data", DATA,
    "--weights", "yolov5s.pt",
    "--project", f"{WORK}/train",
    "--name", "p1s640",
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

run_dir = f"{WORK}/train/p1s640"
metrics = {"yolov5_commit": head, "run_dir": run_dir,
           "final_epoch": parse_results_csv(f"{run_dir}/results.csv")}

# ---------------------------------------------------------------- val + test
for task in ("val", "test"):
    CMD = [
        sys.executable, "/tmp/yolov5/val.py",
        "--weights", f"{run_dir}/weights/best.pt",
        "--data", DATA,
        "--task", task,
        "--img", "640",
        "--batch", "32",
        "--project", f"{WORK}/val",
        "--name", f"p1{task}",
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

save_metrics(metrics, "p1_metrics.json")
log("P1 BASELINE COMPLETE")
