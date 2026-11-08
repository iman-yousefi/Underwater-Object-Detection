#!/usr/bin/env python
"""Stage 1 — environment probe + dataset audit (one short GPU run).

Answers, in one short GPU run:
  1. Is the GPU attached and usable? (torch/CUDA, matmul test)
  2. What is preinstalled? (ultralytics, PIL, yaml versions)
  3. Is kernel internet ON? (github, pypi reachability)
  4. Can we `git clone ultralytics/yolov5`? (pinned commit recorded)
  5. Dataset audit: per-split image/label counts, per-class instance counts,
     box size statistics, sample image resolutions, data.yaml contents.

Outputs (to /kaggle/working):
  p0_stats.json    — everything above, machine-readable
  data_fixed.yaml  — training-ready data.yaml with absolute /kaggle paths
"""
import glob
import json
import os
import shutil
import subprocess
import urllib.request
from collections import Counter

IMG_EXT = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def log(*a):
    print(*a, flush=True)


R = {}

# ---------------- 1. Hardware / preinstalled stack ----------------
import torch  # noqa: E402

R["torch"] = torch.__version__
R["cuda_available"] = bool(torch.cuda.is_available())
if R["cuda_available"]:
    R["gpu_count"] = torch.cuda.device_count()
    R["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    x = torch.randn(1024, 1024, device="cuda")
    y = (x @ x).sum().item()
    R["cuda_matmul_ok"] = bool(y == y)  # NaN check — op really ran on device
    log(f"GPUs: {R['gpus']} | CUDA {torch.version.cuda} | matmul OK")
else:
    log("WARNING: no CUDA available in this kernel session")

try:
    import ultralytics

    R["ultralytics"] = ultralytics.__version__
except Exception:
    R["ultralytics"] = None
log(f"torch={R['torch']}  cuda={R['cuda_available']}  ultralytics={R['ultralytics']}")

# ---------------- 2. Internet probe ----------------
def reachable(url, timeout=10):
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except Exception:
        return False


R["net_github"] = reachable("https://github.com/ultralytics/yolov5")
R["net_pypi"] = reachable("https://pypi.org/simple/")
log(f"internet: github={R['net_github']} pypi={R['net_pypi']}")

# ---------------- 3. yolov5 clone test ----------------
if R["net_github"]:
    shutil.rmtree("/tmp/yolov5", ignore_errors=True)
    p = subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/ultralytics/yolov5", "/tmp/yolov5"],
        capture_output=True, text=True, timeout=600,
    )
    if p.returncode == 0:
        c = subprocess.run(
            ["git", "-C", "/tmp/yolov5", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
        )
        R["yolov5_clone"] = {"ok": True, "commit": c.stdout.strip()}
        log(f"yolov5 clone OK, commit {R['yolov5_clone']['commit']}")
    else:
        R["yolov5_clone"] = {"ok": False, "err": p.stderr[-300:]}
        log(f"yolov5 clone FAILED: {p.stderr[-300:]}")
else:
    R["yolov5_clone"] = {"ok": False, "err": "no internet"}
    log("yolov5 clone skipped (no internet)")

# ---------------- 4. Locate the dataset ----------------
import yaml  # noqa: E402

inputs = sorted(os.listdir("/kaggle/input"))
log(f"/kaggle/input: {inputs}")
R["kaggle_input"] = inputs
# Kaggle mounts CLI-attached datasets under /kaggle/input/datasets/<owner>/<slug>/
# (older layout: /kaggle/input/<slug>) — walk recursively for the aquarium dir.
cands = []
for root, dirs, _ in os.walk("/kaggle/input"):
    for d in dirs:
        if "aquarium" in d.lower():
            cands.append(os.path.join(root, d))
assert cands, f"aquarium dataset dir not found under {inputs}"
DS = sorted(cands)[0]
R["dataset_root"] = DS
R["dataset_top_level"] = sorted(os.listdir(DS))
log(f"dataset root: {DS}")
log(f"top level: {R['dataset_top_level']}")

# ---------------- 5. Audit splits ----------------
def find_images(d):
    return sorted(
        p for p in glob.glob(os.path.join(d, "**", "*"), recursive=True)
        if p.endswith(IMG_EXT)
    )


# The dataset may nest splits under a subdirectory (e.g. aquarium_pretrain/train/...)
# — search recursively for dirs named train/valid/val/test that contain images.
SPLIT_ALIASES = {"train": "train", "valid": "val", "val": "val", "test": "test"}
found = {}
for root, dirs, _ in os.walk(DS):
    name = os.path.basename(root).lower()
    if name in SPLIT_ALIASES:
        img_dir = os.path.join(root, "images")
        d = img_dir if os.path.isdir(img_dir) else root
        imgs = find_images(d)
        if imgs:
            key = SPLIT_ALIASES[name]
            depth = root.count(os.sep)
            if key not in found or depth < found[key][0]:
                found[key] = (depth, root, imgs)
splits = {k: {"dir": v[1], "images": v[2]} for k, v in found.items()}
R["split_dirs"] = {k: v["dir"] for k, v in splits.items()}
assert "train" in splits, f"no train split found; top level: {R['dataset_top_level']}"
log(f"splits found: { {k: v['dir'] for k, v in splits.items()} }")

# class names from data.yaml if present (recursive search)
names = None
ypaths = sorted(glob.glob(os.path.join(DS, "**", "*.y*ml"), recursive=True),
                key=lambda p: (p.count(os.sep), p))
for ypath in ypaths:
    with open(ypath) as f:
        y = yaml.safe_load(f)
    if isinstance(y, dict) and y.get("names"):
        names = y["names"]
        log(f"data.yaml ({ypath}): {y}")
        break
if isinstance(names, dict):
    names = [names[k] for k in sorted(names)]
R["class_names"] = names

split_stats = {}
per_class = Counter()
box_w, box_h = [], []
n_empty_labels = 0
n_missing_labels = 0
img_sizes = []

for key, s in splits.items():
    img_dir = os.path.join(s["dir"], "images")
    lbl_dir = os.path.join(s["dir"], "labels")
    n_inst = 0
    cls_c = Counter()
    w_list, h_list = [], []
    missing = empty = 0
    for ip in s["images"]:
        stem = os.path.splitext(os.path.basename(ip))[0]
        if os.path.isdir(lbl_dir):
            lp = os.path.join(lbl_dir, stem + ".txt")
        else:
            lp = os.path.join(os.path.dirname(ip).replace("images", "labels"), stem + ".txt")
        if not os.path.exists(lp):
            missing += 1
            continue
        with open(lp) as f:
            lines = [l.split() for l in f.read().splitlines() if l.strip()]
        if not lines:
            empty += 1
        for parts in lines:
            c = int(float(parts[0]))
            w, h = float(parts[3]), float(parts[4])
            cls_c[c] += 1
            w_list.append(w)
            h_list.append(h)
    split_stats[key] = {
        "dir": s["dir"],
        "n_images": len(s["images"]),
        "n_label_files_missing": missing,
        "n_empty_label_files": empty,
        "n_instances": sum(cls_c.values()),
        "class_counts": {str(k): v for k, v in sorted(cls_c.items())},
    }
    per_class.update(cls_c)
    box_w.extend(w_list)
    box_h.extend(h_list)
    n_missing_labels += missing
    n_empty_labels += empty
    log(f"{key}: {len(s['images'])} imgs, {sum(cls_c.values())} instances, "
        f"missing labels={missing}, empty labels={empty}, classes={dict(cls_c)}")

# sample image resolutions (up to 30 per split)
from PIL import Image  # noqa: E402

for key, s in splits.items():
    sizes = Counter()
    for ip in s["images"][:30]:
        with Image.open(ip) as im:
            sizes[f"{im.width}x{im.height}"] += 1
    split_stats[key]["image_sizes_sample"] = dict(sizes)

def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None

R["splits"] = split_stats
R["totals"] = {
    "n_images": sum(v["n_images"] for v in split_stats.values()),
    "n_instances": sum(v["n_instances"] for v in split_stats.values()),
    "n_classes": len(per_class),
    "per_class_instances": {names[c] if names and c < len(names) else str(c): n
                            for c, n in sorted(per_class.items())},
    "missing_labels": n_missing_labels,
    "empty_labels": n_empty_labels,
    "box_w_norm": {"mean": sum(box_w) / len(box_w), "p10": q(box_w, 0.1),
                   "p50": q(box_w, 0.5), "p90": q(box_w, 0.9)},
    "box_h_norm": {"mean": sum(box_h) / len(box_h), "p10": q(box_h, 0.1),
                   "p50": q(box_h, 0.5), "p90": q(box_h, 0.9)},
}
log("TOTALS: " + json.dumps(R["totals"], indent=1))

# ---------------- 6. Write training-ready data.yaml ----------------
# Absolute split paths (no `path:` key) — yolov5/ultralytics resolve them as-is,
# immune to any dataset nesting.
data_out = {
    "train": os.path.join(splits["train"]["dir"], "images"),
    "val": os.path.join(splits["val"]["dir"], "images"),
    "nc": len(per_class),
    "names": names if names else [str(c) for c in sorted(per_class)],
}
if "test" in splits:
    data_out["test"] = os.path.join(splits["test"]["dir"], "images")
with open("/kaggle/working/data_fixed.yaml", "w") as f:
    yaml.safe_dump(data_out, f, sort_keys=False)
log(f"data_fixed.yaml written: {data_out}")

# ---------------- 7. Persist + finish ----------------
with open("/kaggle/working/p0_stats.json", "w") as f:
    json.dump(R, f, indent=1)
log("P0 SMOKE COMPLETE — stats saved to /kaggle/working/p0_stats.json")
