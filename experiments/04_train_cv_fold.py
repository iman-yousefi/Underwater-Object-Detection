#!/usr/bin/env python
"""Stage 4 — train the champion config on one stratified CV fold.

build_folds() is deterministic (sorted basenames + fixed seed), so every fold
run computes identical fold membership. Each run:
  1. builds 5 stratified folds over the official train+val images
  2. trains the champion config on 4 folds, monitors the held-out fold
  3. evaluates: held-out fold (no TTA) + official TEST with TTA (--augment)
     and --save-json -> test_preds.json for the WBF ensemble kernel
Outputs: weights, test_preds.json, p3cv_metrics_<fold>.json
"""
import glob
import json
import os
import shutil
import subprocess
import sys

# CV: one run per fold — edit CONFIG['fold'] (0..4). Config below is the champion (yolov5x @640).
CONFIG = {
    "tag": "f0",
    "fold": 0,
    "weights": "yolov5x.pt",
    "imgsz": 640,
    "batch": 12,
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
from uwd_utils import (build_folds, find_under_input, relocate_data_yaml, relocate_splits,
                       parse_results_csv, parse_val_table, save_metrics)  # noqa: E402

# ---------------------------------------------------------------- p0 outputs
stats_path = find_under_input("p0_stats.json")
assert stats_path, "p0_stats.json not found"
with open(stats_path) as f:
    P0 = json.load(f)
PIN = P0.get("yolov5_clone", {}).get("commit")
NAMES = P0.get("class_names")
SP = relocate_splits()
train_dir = SP["train"]
val_dir = SP["val"]
DATA = relocate_data_yaml(find_under_input("data_fixed.yaml"))
log(f"pin={PIN}  train={train_dir}  val={val_dir}")

# ---------------------------------------------------------------- folds
folds = build_folds(
    [os.path.join(train_dir, "images"), os.path.join(val_dir, "images")],
    k=5, seed=42, out_dir="/kaggle/working/folds",
)
fd = folds[CONFIG["fold"]]
log(f"fold {CONFIG['fold']}: train={len(fd['train'])} val={len(fd['val'])}")

import yaml  # noqa: E402

fold_yaml = "/kaggle/working/fold_data.yaml"
with open(fold_yaml, "w") as f:
    yaml.safe_dump({
        "train": "/kaggle/working/folds/fold%d_train.txt" % CONFIG["fold"],
        "val": "/kaggle/working/folds/fold%d_val.txt" % CONFIG["fold"],
        "nc": len(NAMES), "names": NAMES,
    }, f, sort_keys=False)

# ---------------------------------------------------------------- yolov5 setup
subprocess.run(["rm", "-rf", "/tmp/yolov5"], check=False)
r = subprocess.run(["git", "clone", "-q", "https://github.com/ultralytics/yolov5", "/tmp/yolov5"],
                   capture_output=True, text=True)
assert r.returncode == 0, f"yolov5 clone failed: {r.stderr[-500:]}"
if PIN:
    subprocess.run(["git", "-C", "/tmp/yolov5", "checkout", "-q", PIN], capture_output=True)
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
    "--data", fold_yaml,
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
log(tr.stdout[-3000:])
if tr.returncode != 0:
    log("---- train.py stderr tail ----")
    log(tr.stderr[-4000:])
    raise SystemExit(f"train.py failed rc={tr.returncode}")

run_dir = f"{WORK}/train/{tag}"
metrics = {"config": CONFIG, "yolov5_commit": head,
           "final_epoch": parse_results_csv(f"{run_dir}/results.csv")}

# ---------------------------------------------------------------- evals
# 1) held-out fold (no TTA) via fold yaml
vr = subprocess.run([
    sys.executable, "/tmp/yolov5/val.py",
    "--weights", f"{run_dir}/weights/best.pt",
    "--data", fold_yaml, "--task", "val",
    "--img", str(CONFIG["imgsz"]), "--batch", str(CONFIG["batch"]),
    "--project", f"{WORK}/val", "--name", f"{tag}-foldval",
    "--device", "0", "--exist-ok",
], capture_output=True, text=True)
log("---- fold val ----")
log(vr.stdout[-2500:])
if vr.returncode != 0:
    log(vr.stderr[-3000:])
    raise SystemExit(f"fold val failed rc={vr.returncode}")
metrics["fold_val"] = parse_val_table(vr.stdout)

# 2) official test with TTA + save-json (for the ensemble kernel)
vr = subprocess.run([
    sys.executable, "/tmp/yolov5/val.py",
    "--weights", f"{run_dir}/weights/best.pt",
    "--data", DATA, "--task", "test",
    "--img", str(CONFIG["imgsz"]), "--batch", str(CONFIG["batch"]),
    "--augment", "--save-json",
    "--project", f"{WORK}/val", "--name", f"{tag}-test",
    "--device", "0", "--exist-ok",
], capture_output=True, text=True)
log("---- test (TTA) ----")
log(vr.stdout[-2500:])
if vr.returncode != 0:
    log(vr.stderr[-3000:])
    raise SystemExit(f"test val failed rc={vr.returncode}")
metrics["test_tta"] = parse_val_table(vr.stdout)

# yolov5 writes predictions under the val run dir; name varies by version
# (predictions.json / best_predictions.json) — search generically.
preds = glob.glob(f"{WORK}/**/*predictions*.json", recursive=True)
assert preds, "no *predictions*.json found after test val"
shutil.copy(preds[0], f"{WORK}/test_preds.json")
log(f"test_preds.json <- {preds[0]} ({os.path.getsize(f'{WORK}/test_preds.json')} bytes)")

save_metrics(metrics, f"p3cv_metrics_{tag}.json")
log(f"P3 CV {tag} COMPLETE")
