#!/usr/bin/env python
"""Stage 6 — retrain the champion config on train+val (all 575 labeled images),
then evaluate on the untouched official test split, with and without TTA.

Produces the headline single-model numbers and the best.pt used by the demo
notebook. The during-training val overlaps the training set and is used only
for checkpoint selection.
"""
import glob
import json
import os
import shutil
import subprocess
import sys

CONFIG = {
    "tag": "final",
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
from uwd_utils import (find_under_input, relocate_data_yaml, relocate_splits,
                       parse_results_csv, parse_val_table, save_metrics)  # noqa: E402

# ---------------------------------------------------------------- p0 outputs
stats_path = find_under_input("p0_stats.json")
with open(stats_path) as f:
    P0 = json.load(f)
PIN = P0.get("yolov5_clone", {}).get("commit")
NAMES = P0.get("class_names")
SP = relocate_splits()
train_dir = SP["train"]
val_dir = SP["val"]
data_yaml_path = relocate_data_yaml(find_under_input("data_fixed.yaml"))
log(f"pin={PIN}")

# ---------------------------------------------------------------- train list (train+val)
def list_images(d):
    return sorted(p for p in glob.glob(os.path.join(d, "**", "*"), recursive=True)
                  if p.lower().endswith((".jpg", ".jpeg", ".png")))


all_train = list_images(os.path.join(train_dir, "images")) + list_images(os.path.join(val_dir, "images"))
os.makedirs("/kaggle/working/folds", exist_ok=True)
with open("/kaggle/working/folds/all_train.txt", "w") as f:
    f.write("\n".join(all_train))
log(f"all_train: {len(all_train)} images")

import yaml  # noqa: E402

final_yaml = "/kaggle/working/final_data.yaml"
with open(final_yaml, "w") as f:
    yaml.safe_dump({
        "train": "/kaggle/working/folds/all_train.txt",
        "val": os.path.join(val_dir, "images"),
        "nc": len(NAMES), "names": NAMES,
    }, f, sort_keys=False)

# ---------------------------------------------------------------- yolov5 setup
import time

subprocess.run(["rm", "-rf", "/tmp/yolov5"], check=False)
r = None
for attempt in range(4):
    r = subprocess.run(["git", "clone", "-q", "https://github.com/ultralytics/yolov5", "/tmp/yolov5"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        break
    log(f"clone retry {attempt}: {r.stderr[-200:]}")
    time.sleep(20)
assert r.returncode == 0, f"yolov5 clone failed: {r.stderr[-500:]}"
if PIN:
    subprocess.run(["git", "-C", "/tmp/yolov5", "checkout", "-q", PIN], capture_output=True)
head = subprocess.run(["git", "-C", "/tmp/yolov5", "rev-parse", "--short", "HEAD"],
                      capture_output=True, text=True).stdout.strip()
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "/tmp/yolov5/requirements.txt"],
                   capture_output=True, text=True)
log(f"yolov5 @ {head}, pip rc={r.returncode}")

# pre-download pretrained weights with retries (train.py does a single
# unguarded attempt that can fail)
W = f"/tmp/{CONFIG['weights']}"
if not os.path.exists(W):
    url = f"https://github.com/ultralytics/yolov5/releases/download/v7.0/{CONFIG['weights']}"
    for attempt in range(6):
        d = subprocess.run([sys.executable, "-c",
                            f"import urllib.request; urllib.request.urlretrieve({url!r}, {W!r})"],
                           capture_output=True, text=True)
        if d.returncode == 0 and os.path.exists(W) and os.path.getsize(W) > 1e6:
            break
        log(f"weights download retry {attempt}: {(d.stderr or '')[-200:]}")
        time.sleep(30)
assert os.path.exists(W) and os.path.getsize(W) > 1e6, f"weights download failed: {W}"
log(f"weights ready: {W} ({os.path.getsize(W) // 1048576} MB)")

# pre-place Arial.ttf so train.py's check_font() never downloads mid-run
import pathlib

font_dir = pathlib.Path.home() / ".config" / "Ultralytics"
font_dir.mkdir(parents=True, exist_ok=True)
FONT = font_dir / "Arial.ttf"
ok_font = FONT.exists() and FONT.stat().st_size > 1e5
for furl in ("https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf",
             "https://github.com/ultralytics/yolov5/releases/download/v1.0/Arial.ttf"):
    if ok_font:
        break
    for attempt in range(3):
        d = subprocess.run([sys.executable, "-c",
                            f"import urllib.request; urllib.request.urlretrieve({furl!r}, {str(FONT)!r})"],
                           capture_output=True, text=True)
        if d.returncode == 0 and FONT.exists() and FONT.stat().st_size > 1e5:
            ok_font = True
            break
        log(f"font retry {furl.rsplit('/', 1)[-1]} attempt {attempt}: {(d.stderr or '')[-160:]}")
        time.sleep(15)
if not ok_font:
    # dummy file: check_font() skips its download; PIL falls back to its
    # default font when plotting
    FONT.write_bytes(b"")
    log("font: using dummy (plot labels fall back to PIL default)")
log(f"font ready: {ok_font}")

# ---------------------------------------------------------------- train
WORK = "/kaggle/working"
CMD = [
    sys.executable, "/tmp/yolov5/train.py",
    "--img", str(CONFIG["imgsz"]),
    "--batch", str(CONFIG["batch"]),
    "--epochs", str(CONFIG["epochs"]),
    "--patience", "60",
    "--data", final_yaml,
    "--weights", W,
    "--project", f"{WORK}/train",
    "--name", CONFIG["tag"],
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

run_dir = f"{WORK}/train/{CONFIG['tag']}"
metrics = {"config": CONFIG, "yolov5_commit": head,
           "final_epoch": parse_results_csv(f"{run_dir}/results.csv")}

# ---------------------------------------------------------------- test eval (no-TTA + TTA)
for tta in (False, True):
    cmd = [
        sys.executable, "/tmp/yolov5/val.py",
        "--weights", f"{run_dir}/weights/best.pt",
        "--data", data_yaml_path, "--task", "test",
        "--img", str(CONFIG["imgsz"]), "--batch", str(CONFIG["batch"]),
        "--project", f"{WORK}/val", "--name", f"final-test{'-tta' if tta else ''}",
        "--device", "0", "--exist-ok",
    ]
    if tta:
        cmd.append("--augment")
    log("VAL CMD:", " ".join(cmd))
    vr = subprocess.run(cmd, capture_output=True, text=True)
    log(f"---- test ({'TTA' if tta else 'no TTA'}) ----")
    log(vr.stdout[-2500:])
    if vr.returncode != 0:
        log(vr.stderr[-3000:])
        raise SystemExit(f"test val failed rc={vr.returncode}")
    metrics["test_tta" if tta else "test"] = parse_val_table(vr.stdout)

save_metrics(metrics, "p3_final_metrics.json")
log("P3 FINAL COMPLETE")
