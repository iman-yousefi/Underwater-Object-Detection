"""Shared helpers for the underwater detection pipeline.

Loaded inside kernels via load_src_library() (see below). Provides:
  parse_val_table()    parse yolov5 val.py per-class metric tables
  parse_results_csv()  parse yolov5 results.csv (final-epoch metrics)
  save_metrics()       write a metrics JSON to /kaggle/working
  load_src_library()   bootstrap helper (mounts the uwd-src dataset)
"""
import csv
import glob
import json
import os
import shutil
import sys


# ---------------------------------------------------------------- src bootstrap
def load_src_library():
    """Copy the flat .py files of the uwd-src dataset into /tmp/lib/src and import it.

    Kaggle flattens CLI dataset uploads (zip mode), so walk /kaggle/input,
    unzip any archives found, and collect *.py into a temp dir.
    """
    dst = "/tmp/lib/src"
    os.makedirs(dst, exist_ok=True)
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            src = os.path.join(root, f)
            if f.endswith(".py") and f != os.path.basename(__file__):
                shutil.copy(src, os.path.join(dst, f))
            elif f.endswith(".zip"):
                shutil.unpack_archive(src, "/tmp/lib/unzipped")
                for r2, _, f2 in os.walk("/tmp/lib/unzipped"):
                    for g in f2:
                        if g.endswith(".py"):
                            shutil.copy(os.path.join(r2, g), dst)
    sys.path.insert(0, dst)
    return dst


def find_under_input(substring):
    """Recursively find a file or dir under /kaggle/input whose name contains substring.

    Handles both mount layouts: /kaggle/input/<slug> and
    /kaggle/input/datasets/<owner>/<slug> (and kernel outputs under /kaggle/input).
    Returns the shortest match path, or None.
    """
    hits = []
    for root, dirs, files in os.walk("/kaggle/input"):
        for name in dirs + files:
            if substring.lower() in name.lower():
                hits.append(os.path.join(root, name))
    return sorted(hits, key=len)[0] if hits else None


# ---------------------------------------------------------------- parsers
def parse_val_table(text):
    """Parse yolov5 val.py output table.

    Returns list of rows: {cls, images, instances, P, R, mAP50, mAP5095}.
    The first row is always the 'all' summary; subsequent rows are per-class.
    """
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 7:
            continue
        try:
            images, instances = int(parts[1]), int(parts[2])
            p, r, m50, m5095 = (float(x) for x in parts[3:7])
        except ValueError:
            continue
        rows.append({
            "cls": parts[0], "images": images, "instances": instances,
            "P": p, "R": r, "mAP50": m50, "mAP5095": m5095,
        })
    return rows


def parse_results_csv(path):
    """Parse yolov5 results.csv → final-epoch metrics dict (epoch/mAP columns)."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    last = rows[-1]
    out = {}
    for k, v in last.items():
        k = k.strip()
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v.strip()
    return out


def save_metrics(metrics, name="metrics.json"):
    """Write a metrics dict to /kaggle/working/<name> and print it."""
    out = os.path.join("/kaggle/working", name)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=1)
    print(f"METRICS {name}: {json.dumps(metrics)[:2000]}", flush=True)
    return out


# ---------------------------------------------------------------- session relocation
def relocate_splits():
    """Locate train/val/test split dirs under the CURRENT session's dataset mount.

    Kaggle's mount layout drifts between sessions (/kaggle/input/<slug> vs
    /kaggle/input/datasets/<owner>/<slug> vs /kaggle/input/notebooks/...), so any
    absolute path recorded by an earlier kernel may be invalid here. This walks
    /kaggle/input for dirs named train/valid/val/test that contain an images/
    subdir and returns {"train": ..., "val": ..., "test"(optional): ...}.
    """
    found = {}
    for root, dirs, _ in os.walk("/kaggle/input"):
        name = os.path.basename(root).lower()
        if name in ("train", "valid", "val", "test"):
            key = {"valid": "val"}.get(name, name)
            if os.path.isdir(os.path.join(root, "images")) and key not in found:
                found[key] = root
    return found


def relocate_data_yaml(src_yaml, out="/kaggle/working/data.yaml"):
    """Rewrite a data yaml's split paths against the current session's mounts."""
    import yaml

    with open(src_yaml) as f:
        d = yaml.safe_load(f)
    sp = relocate_splits()
    for k in ("train", "val", "test"):
        if k in sp:
            d[k] = os.path.join(sp[k], "images")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        yaml.safe_dump(d, f, sort_keys=False)
    print(f"RELOCATED data yaml -> {out}: {d}", flush=True)
    return out


# ---------------------------------------------------------------- folds
IMG_EXT = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def find_images(d):
    """All image files under directory d (recursive), sorted."""
    return sorted(
        p for p in glob.glob(os.path.join(d, "**", "*"), recursive=True)
        if p.endswith(IMG_EXT)
    )


def build_folds(image_dirs, k=5, seed=42, out_dir="/kaggle/working/folds"):
    """Deterministic stratified k-fold over train+val images.

    All fold kernels call this with identical inputs (sorted basenames, fixed
    seed), so every kernel computes the same fold membership regardless of the
    session's mount layout (only basenames are used for ordering/identity).
    Stratification: images are dealt round-robin into the fold currently
    containing the fewest instances of each of the image's classes (rarest
    class first), then leftovers to the smallest fold. Writes
    fold<k>_train.txt / fold<k>_val.txt (yolov5 txt-file format) and returns
    {fold: {"train": [...], "val": [...]}}.
    """
    import random

    def label_path(img):
        lp = img.replace("/images/", "/labels/")
        return os.path.splitext(lp)[0] + ".txt"

    imgs = []
    for d in image_dirs:
        imgs.extend(find_images(d))
    imgs = sorted(set(imgs), key=lambda p: os.path.basename(p))

    cls_count = {}
    img_classes = {}
    for p in imgs:
        cs = set()
        lp = label_path(p)
        if os.path.exists(lp):
            with open(lp) as f:
                for line in f:
                    if line.strip():
                        cs.add(int(float(line.split()[0])))
        img_classes[p] = cs
        for c in cs:
            cls_count[c] = cls_count.get(c, 0) + 1

    order = list(imgs)
    random.Random(seed).shuffle(order)
    # rarest class first: deal images that contain it first
    cls_order = sorted(cls_count, key=lambda c: cls_count[c])
    fold_imgs = {i: [] for i in range(k)}
    fold_cls = {i: {} for i in range(k)}
    dealt = set()
    for c in cls_order:
        for p in order:
            if p in dealt or c not in img_classes[p]:
                continue
            tgt = min(fold_imgs, key=lambda i: (fold_cls[i].get(c, 0), len(fold_imgs[i]), i))
            fold_imgs[tgt].append(p)
            fold_cls[tgt][c] = fold_cls[tgt].get(c, 0) + 1
            for cc in img_classes[p]:
                fold_cls[tgt][cc] = fold_cls[tgt].get(cc, 0) + 1
            dealt.add(p)
    for p in order:
        if p not in dealt:
            tgt = min(fold_imgs, key=lambda i: (len(fold_imgs[i]), i))
            fold_imgs[tgt].append(p)
            dealt.add(p)
            for cc in img_classes[p]:
                fold_cls[tgt][cc] = fold_cls[tgt].get(cc, 0) + 1

    os.makedirs(out_dir, exist_ok=True)
    folds = {}
    for i in range(k):
        val_i = sorted(fold_imgs[i])
        train_i = sorted(p for j in range(k) for p in fold_imgs[j] if j != i)
        folds[i] = {"train": train_i, "val": val_i}
        for name, lst in ((f"fold{i}_train.txt", train_i), (f"fold{i}_val.txt", val_i)):
            with open(os.path.join(out_dir, name), "w") as f:
                f.write("\n".join(lst))
    # fold summary for the log
    summary = {}
    for i in range(k):
        summary[f"fold{i}"] = {
            "n_train": len(folds[i]["train"]), "n_val": len(folds[i]["val"]),
            "val_class_counts": {str(c): sum(1 for p in folds[i]["val"] if c in img_classes[p])
                                 for c in sorted(cls_count)},
        }
    print("FOLD SUMMARY: " + json.dumps(summary), flush=True)
    return folds
