#!/usr/bin/env python
"""Stage 5 — fuse the 5 CV fold models' test predictions (WBF) and evaluate.

CPU-only. Inputs: the 5 fold runs' test_preds.json (COCO-style, TTA predictions).
Builds the official test COCO ground truth from the dataset labels, fuses
predictions per image with Weighted Boxes Fusion, and scores members + ensemble
with pycocotools COCOeval.
"""
import collections
import glob
import json
import os
import shutil
import subprocess
import sys


def log(*a):
    print(*a, flush=True)


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
from uwd_utils import find_under_input, relocate_splits, save_metrics  # noqa: E402

WORK = "/kaggle/working"

r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ensemble-boxes", "pycocotools"],
                   capture_output=True, text=True)
log(f"pip rc={r.returncode}" + ("" if r.returncode == 0 else " " + r.stderr[-500:]))

from ensemble_boxes import weighted_boxes_fusion  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402
from PIL import Image  # noqa: E402

# ---------------------------------------------------------------- inputs
stats_path = find_under_input("p0_stats.json")
with open(stats_path) as f:
    P0 = json.load(f)
NAMES = P0["class_names"]
test_dir = relocate_splits()["test"]
img_dir = os.path.join(test_dir, "images")
lbl_dir = os.path.join(test_dir, "labels")
imgs = sorted(p for p in glob.glob(os.path.join(img_dir, "*")) if p.lower().endswith((".jpg", ".jpeg", ".png")))
log(f"test images: {len(imgs)}")

pred_paths = sorted(set(glob.glob("/kaggle/input/**/test_preds.json", recursive=True)))
log(f"found {len(pred_paths)} test_preds.json files:")
for p in pred_paths:
    log("  ", p)
assert len(pred_paths) == 5, f"expected 5 fold prediction files, got {len(pred_paths)}"
pred_sets = [json.load(open(p)) for p in pred_paths]

# ---------------------------------------------------------------- GT (COCO)
stem_to_id, id_to_size = {}, {}
images, annotations = [], []
ann_id = 1
for i, ip in enumerate(imgs):
    stem = os.path.splitext(os.path.basename(ip))[0]
    stem_to_id[stem] = i
    with Image.open(ip) as im:
        W, H = im.size
    id_to_size[i] = (W, H)
    images.append({"id": i, "file_name": os.path.basename(ip), "width": W, "height": H})
    lp = os.path.join(lbl_dir, stem + ".txt")
    if os.path.exists(lp):
        with open(lp) as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                c, cx, cy, w, h = (float(x) for x in parts[:5])
                annotations.append({
                    "id": ann_id, "image_id": i, "category_id": int(c),
                    "bbox": [(cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H],
                    "area": w * W * h * H, "iscrowd": 0,
                })
                ann_id += 1
gt = {"images": images, "annotations": annotations,
      "categories": [{"id": i, "name": n} for i, n in enumerate(NAMES)]}
with open(f"{WORK}/test_gt.json", "w") as f:
    json.dump(gt, f)
log(f"GT: {len(annotations)} annotations over {len(images)} images")


def coco_eval(dt_list):
    """COCOeval for a COCO-format detection list (image_id already int)."""
    gt_coco = COCO()
    gt_coco.dataset = gt
    gt_coco.createIndex()
    dt_coco = gt_coco.loadRes(dt_list)
    e = COCOeval(gt_coco, dt_coco, "bbox")
    e.evaluate()
    e.accumulate()
    e.summarize()
    s = e.stats
    return {"mAP5095": round(float(s[0]), 4), "mAP50": round(float(s[1]), 4),
            "AP75": round(float(s[2]), 4), "mAR100": round(float(s[8]), 4)}


# ---------------------------------------------------------------- member evals
member_metrics = {}
for idx, (pp, preds) in enumerate(zip(pred_paths, pred_sets)):
    mapped = [{**d, "image_id": stem_to_id[str(d["image_id"])]} for d in preds]
    tag = f"fold{idx}"
    member_metrics[tag] = coco_eval(mapped)
    log(f"MEMBER {tag}: {member_metrics[tag]}")

# ---------------------------------------------------------------- WBF fusion
# ensemble-boxes expects, per image: a list of per-MODEL arrays
# (boxes (n,4) normalized, scores (n,), labels (n,))
import numpy as np

by_img = collections.defaultdict(lambda: [[] for _ in range(len(pred_sets))])
for m, preds in enumerate(pred_sets):
    for d in preds:
        i = stem_to_id[str(d["image_id"])]
        W, H = id_to_size[i]
        x, y, w, h = d["bbox"]
        by_img[i][m].append([x / W, y / H, (x + w) / W, (y + h) / H, d["score"], float(d["category_id"])])

merged = []
for i, models in by_img.items():
    W, H = id_to_size[i]
    boxes_list, scores_list, labels_list = [], [], []
    for dets in models:
        if dets:
            a = np.array(dets, dtype=float)
            a[:, :4] = np.clip(a[:, :4], 0.0, 1.0)  # clip BOX coords only — labels must survive
            boxes_list.append(a[:, :4])
            scores_list.append(a[:, 4])
            labels_list.append(a[:, 5])
        else:
            boxes_list.append(np.zeros((0, 4)))
            scores_list.append(np.zeros(0))
            labels_list.append(np.zeros(0))
    fb, fs, fl = weighted_boxes_fusion(boxes_list, scores_list, labels_list,
                                       iou_thr=0.55, skip_box_thr=0.0)
    for (x1, y1, x2, y2), s, l in zip(fb, fs, fl):
        merged.append({
            "image_id": i, "category_id": int(l), "score": float(s),
            "bbox": [x1 * W, y1 * H, (x2 - x1) * W, (y2 - y1) * H],
        })
log(f"merged predictions: {len(merged)} "
    f"(members total {sum(len(p) for p in pred_sets)})")

ensemble_metrics = coco_eval(merged)
log(f"ENSEMBLE (WBF of 5 folds, TTA): {ensemble_metrics}")

out = {"members": member_metrics, "ensemble": ensemble_metrics,
       "n_folds": len(pred_paths), "wbf_iou_thr": 0.55}
with open(f"{WORK}/test_preds_merged.json", "w") as f:
    json.dump(merged, f)
save_metrics(out, "ensemble_metrics.json")
log("P3 ENSEMBLE COMPLETE")
