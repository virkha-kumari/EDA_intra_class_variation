"""
USB (Use-Seen-Before) — Missed-Label Detection

Finds images that probably contain an unlabelled instance of a class, without running a
detector. Relies on a fixed-camera dataset (CCTV): a class recurs in the same image regions.

Per requested class:
 1. Collect every GT bbox of the class (normalized YOLO coords) across all splits.
 2. Greedy-dedup boxes at IoU < --region_iou  ->  seen_<cls>_regions.
 3. Pad each region by --pad (fraction of image W/H per side).
 4. Crop each region from EVERY image in the dataset. A candidate is skipped when it
    overlaps (IoU >= --region_iou) a GT box of the same class in that same image.
 5. Embed candidates with Script 2's DINOv2 function (identical pooling to the reference
    embeddings) — crops are written to a temp folder, embedded, then deleted.
 6. Reference pool from Script 2's cached <cls>/embeddings_dinov2.npy, re-clustered in-process
    with Script 3's DBSCAN (no montages): all noise points + per cluster: centroid
    + --n_farthest points by cosine distance from that centroid.
 7. max cosine sim(candidate, pool) > per-class threshold  ->  flagged.
 8. Output: <output_dir>/missing_labels/<cls>/<image>.jpg — full image, flagged regions in
    red with sim score, existing GT boxes of the class in green.

Usage:
    python "postannotation_scripts/5. usb_missed_labels.py" \
        --imgs_path path/to/images --label_path path/to/labels \
        --classes_txt path/to/classes.txt \
        --cropped_dir postann_pretrain_results/cropped_imgs_by_class_X \
        --output_dir postann_pretrain_results/clustering_results_X \
        --usb_classes screw holder --sim_thresholds 0.8 0.85 \
        --auto_tune --auto_tune_percentile 90 --min_samples 3
"""

import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.preprocessing import normalize
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

IMG_EXTS = (".png", ".jpg", ".jpeg")
_SCRIPTS_DIR = Path(__file__).resolve().parent


def _load_module(filename, name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _detect_splits(imgs_dir, label_dir):
    splits = []
    for split in ['train', 'val', 'test']:
        si, sl = os.path.join(imgs_dir, split), os.path.join(label_dir, split)
        if os.path.isdir(si) and os.path.isdir(sl):
            splits.append((split, si, sl))
    return splits if splits else [('', imgs_dir, label_dir)]


def _iou(a, b):
    """IoU of two (x1, y1, x2, y2) boxes — any consistent unit."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _yolo_to_xyxy_norm(cx, cy, w, h):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _read_gt_boxes(txt_path, class_id):
    """Normalized xyxy boxes of class_id in one YOLO label file. Missing file -> []."""
    boxes = []
    if not os.path.isfile(txt_path):
        return boxes
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5 or parts[0] != class_id:
                continue
            try:
                boxes.append(_yolo_to_xyxy_norm(*map(float, parts[1:5])))
            except ValueError:
                continue
    return boxes


def build_dataset_index(imgs_dir, label_dir):
    """[(split, img_path, txt_path)] for every image on disk. txt_path may not exist."""
    index = []
    for split, si, sl in _detect_splits(imgs_dir, label_dir):
        for f in sorted(os.listdir(si)):
            if not f.lower().endswith(IMG_EXTS):
                continue
            base = os.path.splitext(f)[0]
            index.append((split, os.path.join(si, f), os.path.join(sl, base + ".txt")))
    return index


def seen_regions(all_boxes, region_iou):
    regions = []
    for box in all_boxes:
        if all(_iou(box, r) < region_iou for r in regions):
            regions.append(box)
    return regions


def build_reference_pool(embeddings, labels, n_farthest):
    """Noise points + per cluster: centroid + n_farthest points by cosine distance."""
    emb = normalize(embeddings, norm='l2')
    pool = [emb[labels == -1]]
    for cid in sorted(set(labels) - {-1}):
        members = emb[labels == cid]
        centroid = normalize(members.mean(axis=0, keepdims=True), norm='l2')
        far_idx = np.argsort(members @ centroid[0])[:n_farthest]
        pool.append(centroid)
        pool.append(members[far_idx])
    return np.vstack(pool)


def main():
    p = argparse.ArgumentParser(description="USB (Use-Seen-Before) missed-label detection")
    p.add_argument("--imgs_path", required=True)
    p.add_argument("--label_path", required=True)
    p.add_argument("--classes_txt", required=True)
    p.add_argument("--cropped_dir", required=True, help="Script 1/2 output root with <cls>/embeddings .npy")
    p.add_argument("--output_dir", required=True, help="Clustering results dir — missing_labels/ is created inside")
    p.add_argument("--usb_classes", nargs="+", required=True, help="Class names to check")
    p.add_argument("--sim_thresholds", nargs="+", type=float, required=True,
                   help="Cosine sim threshold per class (same order as --usb_classes)")
    p.add_argument("--n_farthest", type=int, default=4, help="Farthest points per cluster added to reference pool")
    p.add_argument("--region_iou", type=float, default=0.8, help="IoU for region dedup AND GT suppression")
    p.add_argument("--pad", type=float, default=0.02, help="Region padding, fraction of image W/H per side")
    p.add_argument("--save_suffix", type=str, default="embeddings_dinov2.npy")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--min_samples", type=int, default=3)
    p.add_argument("--auto_tune", action="store_true")
    p.add_argument("--auto_tune_percentile", type=int, default=90)
    p.add_argument("--eps", type=float, default=0.15)
    args = p.parse_args()

    if len(args.sim_thresholds) != len(args.usb_classes):
        raise ValueError("--sim_thresholds must have one value per --usb_classes entry")

    with open(args.classes_txt, 'r') as f:
        class_names = [l.strip() for l in f if l.strip()]
    name_to_id = {n: str(i) for i, n in enumerate(class_names)}
    unknown = [c for c in args.usb_classes if c not in name_to_id]
    if unknown:
        raise ValueError(f"Classes not in classes.txt: {unknown}")

    embed_mod = _load_module("2. save_dinov2_embeddings_per_class.py", "usb_embed")
    cluster_mod = _load_module("3. clustering_of_classes_embeddings.py", "usb_cluster")

    print("\n📇  Indexing dataset...")
    index = build_dataset_index(args.imgs_path, args.label_path)
    split_of = {img_path: split for split, img_path, _ in index}
    print(f"    {len(index)} images across {len(_detect_splits(args.imgs_path, args.label_path))} split(s)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "facebook/dinov2-base"
    print(f"Loading {model_name} on {device}...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    out_root = Path(args.output_dir) / "missing_labels"
    temp_root = out_root / "_temp_crops"
    summary = []

    for cls_name, thr in zip(args.usb_classes, args.sim_thresholds):
        cls_id = name_to_id[cls_name]
        print(f"\n{'='*60}\nUSB: {cls_name} (id {cls_id}, sim threshold {thr})\n{'='*60}")

        emb_path = Path(args.cropped_dir) / cls_name / args.save_suffix
        if not emb_path.exists():
            print(f"  ⚠️  No embeddings at {emb_path} — skipping class")
            continue

        # --- GT boxes per image (needed for both region collection and suppression) ---
        gt_per_image = {img_path: _read_gt_boxes(txt_path, cls_id) for _, img_path, txt_path in index}
        all_boxes = [b for boxes in gt_per_image.values() for b in boxes]
        if not all_boxes:
            print(f"  ⚠️  No GT boxes found for {cls_name} — skipping class")
            continue
        regions = seen_regions(all_boxes, args.region_iou)
        print(f"  {len(all_boxes)} GT boxes → {len(regions)} unique seen regions (IoU < {args.region_iou})")
        print(f"  Up to {len(regions) * len(index)} candidate crops before suppression")

        # --- Reference pool from cached embeddings (no montages, no re-embedding) ---
        ref_emb = np.load(str(emb_path))
        eps = cluster_mod.auto_tune_eps(ref_emb, args.min_samples, args.auto_tune_percentile) if args.auto_tune else args.eps
        labels, _, _ = cluster_mod.cluster_single_class(ref_emb, cls_name, eps, args.min_samples)
        pool = build_reference_pool(ref_emb, labels, args.n_farthest)
        print(f"  Reference pool: {len(pool)} embeddings (from {len(ref_emb)} crops)")

        # --- Crop candidates from every image ---
        cls_temp = temp_root / cls_name
        if cls_temp.exists():
            shutil.rmtree(cls_temp)
        cls_temp.mkdir(parents=True)
        candidates = []  # (crop_path, img_path, (x1,y1,x2,y2) pixels)

        for split, img_path, _ in tqdm(index, desc=f"  Cropping regions [{cls_name}]", unit="img"):
            gt_boxes = gt_per_image[img_path]
            todo = [r for r in regions if all(_iou(r, g) < args.region_iou for g in gt_boxes)]
            if not todo:
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            H, W = img.shape[:2]
            base = os.path.splitext(os.path.basename(img_path))[0]
            prefix = f"{split}_" if split else ""
            for ridx, (x1, y1, x2, y2) in enumerate(todo):
                px1 = max(0, int((x1 - args.pad) * W)); py1 = max(0, int((y1 - args.pad) * H))
                px2 = min(W, int((x2 + args.pad) * W)); py2 = min(H, int((y2 + args.pad) * H))
                crop = img[py1:py2, px1:px2]
                if crop.size == 0 or min(crop.shape[:2]) < 3:
                    continue
                crop_path = cls_temp / f"{prefix}{base}_r{ridx}.jpg"
                cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                candidates.append((crop_path, img_path, (px1, py1, px2, py2)))

        print(f"  {len(candidates)} candidate crops after GT suppression")
        if not candidates:
            shutil.rmtree(cls_temp, ignore_errors=True)
            summary.append((cls_name, 0, 0))
            continue

        # --- Embed candidates with Script 2's exact function ---
        cand_emb, valid_paths = embed_mod.compute_embeddings_for_paths(
            [c[0] for c in candidates], processor, model, device, batch_size=args.batch)
        valid_set = {Path(v) for v in valid_paths}
        candidates = [c for c in candidates if c[0] in valid_set]
        sims = (normalize(cand_emb, norm='l2') @ pool.T).max(axis=1)
        shutil.rmtree(cls_temp, ignore_errors=True)

        # --- Draw flagged regions per image ---
        flagged = {}
        for (crop_path, img_path, box), sim in zip(candidates, sims):
            if sim > thr:
                flagged.setdefault(img_path, []).append((box, float(sim)))

        cls_out = out_root / cls_name
        cls_out.mkdir(parents=True, exist_ok=True)
        for img_path, hits in flagged.items():
            img = cv2.imread(img_path)
            H, W = img.shape[:2]
            for (x1, y1, x2, y2) in gt_per_image[img_path]:
                cv2.rectangle(img, (int(x1 * W), int(y1 * H)), (int(x2 * W), int(y2 * H)), (0, 200, 0), 2)
            for (px1, py1, px2, py2), sim in hits:
                cv2.rectangle(img, (px1, py1), (px2, py2), (0, 0, 255), 3)
                cv2.putText(img, f"{cls_name} {sim:.2f}", (px1, max(15, py1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            split = split_of[img_path]
            out_name = (f"{split}_" if split else "") + os.path.basename(img_path)
            cv2.imwrite(str(cls_out / Path(out_name).with_suffix(".jpg")), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

        n_hits = sum(len(h) for h in flagged.values())
        print(f"  🚩 {len(flagged)} image(s) flagged ({n_hits} region hits) → {cls_out}")
        summary.append((cls_name, len(flagged), n_hits))

        del cand_emb, sims
        if device == "cuda":
            torch.cuda.empty_cache()

    shutil.rmtree(temp_root, ignore_errors=True)

    print(f"\n{'='*60}\nUSB SUMMARY\n{'='*60}")
    for cls_name, n_imgs, n_hits in summary:
        print(f"  {cls_name:<30} {n_imgs:>6} images flagged  {n_hits:>6} region hits")
    print(f"  Output: {out_root}")


if __name__ == "__main__":
    main()
