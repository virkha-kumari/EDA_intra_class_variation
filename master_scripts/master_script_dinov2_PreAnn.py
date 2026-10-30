#!/usr/bin/env python3
"""
Pre-annotation frame quality assessment pipeline.
Analyzes raw frames BEFORE annotation using DINOv2 embeddings with [CLS || avg_patches]
concatenation pooling, PCA whitening, PaCMAP visualization, and HDBSCAN clustering.

Why DINOv2 over SigLIP:
    SigLIP's global pooler_output compresses the entire cluttered factory scene into one
    768d vector, letting shirt color / background dominate cluster assignments.
    DINOv2's [CLS || avg_patches] produces 1536d: CLS captures global identity while
    the 256 patch tokens (each 14x14px region) preserve localized spatial structure.
    Benchmark on project frames: DINOv2 separation=+0.543 vs SigLIP separation=+0.290.

Embedding Strategy (DINOv2-base, 224px):
- Output: [CLS token (768d) concatenated with mean of patch tokens (768d)] = 1536d
- Frames WITH persons: scene embedding (1536d) weighted 0.7 + lightweight pose features (20d) weighted 0.3
- Frames WITHOUT persons: scene embeddings (1536d) + zero-padded pose features (20d)
- Weights are applied to CONCATENATED dimensions (not blended scalars)
- Total dimensionality: 1556d (1536 scene + 20 pose)
- Pose features: limb angles + relative distances only (NO absolute position/scale to avoid encoding worker identity)

Pipeline changes vs SigLIP script:
1. DINOv2 [CLS || avg_patches] pooling (1536d) instead of SigLIP pooler_output (768d)
2. PCA whitening to 128d before HDBSCAN (decorrelates dominant axes like shirt color)
3. PaCMAP for 2D visualization (better global+local structure than UMAP)
4. HDBSCAN cluster_selection_method='eom' (merges sub-clusters into meaningful groups, reduces fragmentation)

Tuning Guide:
    EMBEDDING WEIGHTS (in compute_multiview_embeddings):
    - More scene-driven clustering => Increase scene_emb weight (0.9 → 0.95+)
    - More pose-driven clustering => Increase pose_features weight (0.1 → 0.15+)

    CLUSTERING GRANULARITY (in main):
    - Want more fine-grained activity clusters => Increase n_components PCA dims (128 → 256)
    - Min frames per activity => Adjust min_cluster_size (10 → higher for stricter)
    - Cluster strictness => Adjust min_samples (3 → lower for looser, higher for stricter)

    QUALITY THRESHOLDS (in main):
    - Blur detection sensitivity => anisotropy_threshold (3.6 → lower=stricter)

    PDF REPORT OPTIONS:
    - Samples shown per activity => activity_num_samples
    - Grid layout => grid_cols_activities, grid_cols_blurry
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
import shutil

# Ensure project root is on sys.path so `utils/` package is importable
# when this script is run from master_scripts/ subdirectory
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from transformers import AutoImageProcessor, AutoModel
import torch
import numpy as np
import gc
from PIL import Image as PILImage, UnidentifiedImageError
from tqdm import tqdm
import cv2
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
import pacmap
from sklearn.decomposition import PCA
import random
import questionary
from utils.preann_pdf_generate import (
    create_activity_montage,
    create_blurry_images_montage,
    generate_pdf_report,
)
from ultralytics import YOLO
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
matplotlib.use('Agg')  # Non-interactive backend


def scan_frames(root: Path, exts=(".png", ".jpg", ".jpeg")):
    """
    Scans root directory for image files.
    """
    image_paths = []

    for p in sorted(root.rglob('*')):
        if p.is_file() and p.suffix.lower() in exts:
            image_paths.append(p)

    return image_paths


def load_image_safe(path: Path):
    """Load image safely, return None if corrupted."""
    try:
        img = PILImage.open(path).convert("RGB")
        return img
    except (UnidentifiedImageError, OSError) as e:
        print(f"Warning: could not open image {path}: {e}", file=sys.stderr)
        return None


def detect_persons_and_extract_poses(image_paths, pose_model, device, conf_threshold, yolo_batch_size):
    """
    Detect persons in frames and extract pose keypoints using BATCHED inference.
    Returns:
        frame_data: List of dicts with keys:
            - 'path': Path to frame
            - 'persons': List of person detections, each with:
                - 'pose_features': Flattened normalized pose features
            - 'has_persons': Boolean indicating if frame contains persons
            - 'image_shape': (H, W) tuple
    """
    frame_data = []

    # Process in batches
    for batch_start in tqdm(range(0, len(image_paths), yolo_batch_size), desc="Detecting persons (batched)", unit="batch"):
        batch_paths = image_paths[batch_start:batch_start + yolo_batch_size]

        # Load batch of images
        batch_images = []
        batch_paths_valid = []
        batch_image_shapes = []

        for img_path in batch_paths:
            img_cv = cv2.imread(str(img_path))
            if img_cv is not None:
                batch_images.append(img_cv)
                batch_paths_valid.append(img_path)
                batch_image_shapes.append(img_cv.shape[:2])
            else:
                # Store None for corrupted images
                frame_data.append({
                    'path': img_path,
                    'persons': [],
                    'has_persons': False,
                    'image_shape': None
                })

        if len(batch_images) == 0:
            continue

        # Run YOLO pose detection on BATCH
        results_batch = pose_model(batch_images, conf=conf_threshold, verbose=False, device=device)

        # Process each result
        for img_shape, img_path, results in zip(batch_image_shapes, batch_paths_valid, results_batch):
            persons = []
            if results.keypoints is not None and len(results.keypoints) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                keypoints = results.keypoints.data.cpu().numpy()  # (N, 17, 3)

                for bbox, kpts in zip(boxes, keypoints):
                    h, w = img_shape
                    kpts_norm = kpts.copy()
                    kpts_norm[:, 0] /= w
                    kpts_norm[:, 1] /= h

                    pose_features = extract_pose_features(kpts_norm)

                    # Only store pose_features (bbox, keypoints, keypoints_norm are unused downstream)
                    persons.append({
                        'pose_features': pose_features
                    })

            frame_data.append({
                'path': img_path,
                'persons': persons,
                'has_persons': len(persons) > 0,
                'image_shape': img_shape  # (H, W)
            })

        del batch_images, results_batch
        if device == "cuda":
            torch.cuda.empty_cache()

    return frame_data


def extract_pose_features(keypoints_norm):
    """Extract action-invariant pose features: limb angles and relative distances only.

    Intentionally excludes absolute position (center_x/y) and body scale (x_range/y_range)
    to avoid encoding worker identity, camera distance, or position in frame.
    Only encodes WHAT the person is doing (joint angles + relative limb lengths).
    Output: 20-dim vector (10 limb pairs × 2 features each).
    """
    features = []

    limb_pairs = [
        (5, 7), (7, 9),     # left arm (shoulder-elbow, elbow-wrist)
        (6, 8), (8, 10),    # right arm
        (11, 13), (13, 15), # left leg (hip-knee, knee-ankle)
        (12, 14), (14, 16), # right leg
        (5, 11), (6, 12),   # torso (shoulder-hip)
    ]

    for kp1_idx, kp2_idx in limb_pairs:
        kp1, kp2 = keypoints_norm[kp1_idx], keypoints_norm[kp2_idx]
        if kp1[2] > 0.3 and kp2[2] > 0.3:
            dx = kp2[0] - kp1[0]
            dy = kp2[1] - kp1[1]
            angle = np.arctan2(dy, dx)
            distance = np.sqrt(dx**2 + dy**2)
            features.extend([angle, distance])
        else:
            features.extend([0.0, 0.0])

    return np.array(features, dtype=np.float32)


# Whole frame → 224x224 (no center crop). The processor default (short edge 256 → center-crop 224)
# drops the frame borders. Bump PREANN_PREPROC_TAG whenever preprocessing/pooling changes.
DINO_PREPROC = {"do_center_crop": False, "size": {"height": 224, "width": 224}}
PREANN_PREPROC_TAG = "facebook/dinov2-base|square224-nocrop|cls+patchmean|pose20"


def _frames_fingerprint(frame_data):
    """Hash of frame paths + sizes + mtimes + preprocessing tag — cache is valid only for the same frames."""
    h = hashlib.sha1(PREANN_PREPROC_TAG.encode())
    for f in frame_data:
        p = Path(f['path'])
        try:
            st = p.stat()
            h.update(f"{p}|{st.st_size}|{int(st.st_mtime)}".encode())
        except OSError:
            h.update(f"{p}|missing".encode())
    return h.hexdigest()


def compute_multiview_embeddings(frame_data, processor, model, device, batch_size, cache_dir=None):
    """
    Compute unweighted embeddings with optional pose integration:
    - If person detected: scene [CLS||patch-mean] (1536) + 20 pose features
    - If no person: scene + zero pose features (for consistent dimensions)
    Scene/pose weights are applied later, per block, in cluster_activities_with_pca_hdbscan()
    (weighting here would be undone by standardisation/whitening).
    Returns (embeddings, valid_frame_indices) — indices into frame_data of frames that loaded.
    """
    # Determine embedding mode FIRST (before checking cache)
    frames_with_persons = sum(1 for f in frame_data if f['has_persons'])
    frames_without_persons = len(frame_data) - frames_with_persons

    # If too few persons detected, use scene-only for ALL frames (avoids UMAP clustering issues)
    min_persons_for_pose = 3
    use_scene_only = frames_with_persons < min_persons_for_pose
    expected_dim = 1536 if use_scene_only else 1536 + 20  # Scene-only: 1536, Scene+pose: 1556

    # Check cache: same frames (paths/sizes/mtimes), same preprocessing, same dimension
    fingerprint = _frames_fingerprint(frame_data)
    if cache_dir is not None:
        cache_file = cache_dir / "temp_multiview_emb.npy"
        cache_indices_file = cache_dir / "temp_multiview_emb_indices.npy"
        cache_meta_file = cache_dir / "temp_multiview_emb_meta.json"

        if cache_file.exists() and cache_indices_file.exists() and cache_meta_file.exists():
            embeddings = np.load(cache_file)
            valid_frame_indices = np.load(cache_indices_file)
            try:
                cached_fp = json.loads(cache_meta_file.read_text(encoding="utf-8")).get("fingerprint")
            except ValueError:
                cached_fp = None

            if (cached_fp == fingerprint and embeddings.shape[1] == expected_dim
                    and len(embeddings) == len(valid_frame_indices)):
                print(f"\n✅ Found cached embeddings: {cache_file}")
                print(f"   Loaded {len(embeddings)} embeddings from cache (same frames + preprocessing)")
                return embeddings, valid_frame_indices.tolist()
            print(f"\n⚠️  Cached embeddings don't match these frames/preprocessing/mode — recomputing")

    # Indices (into frame_data) of frames that actually load — embeddings stay aligned with them
    valid_frame_indices = []

    if use_scene_only:
        print(f"\n⚠️  Only {frames_with_persons} frames with persons detected (minimum {min_persons_for_pose} needed)")
        print(f"   Switching to SCENE-ONLY embeddings for all {len(frame_data)} frames")
        print(f"   (This ensures sufficient samples for clustering)")
        pose_features_dim = 0  # No pose features in scene-only mode
    else:
        print(f"\nProcessing {len(frame_data)} frames:")
        print(f"  - {frames_with_persons} frames with persons (DINOv2 [CLS||avg_patches] 1536d + pose 20d)")
        print(f"  - {frames_without_persons} frames without persons (DINOv2 [CLS||avg_patches] 1536d only)")
        # Determine pose_features dimension ONCE upfront to avoid mismatch across batches
        # 10 limb pairs × 2 features (angle + distance) = 20
        pose_features_dim = 20

    # Pre-allocate zero arrays for frames without persons (reuse across all frames)
    zero_pose_features = np.zeros(pose_features_dim, dtype=np.float32) if pose_features_dim > 0 else np.array([], dtype=np.float32)

    multiview_embeddings = []
    model.eval()

    with torch.no_grad():
        progress_bar = tqdm(total=len(frame_data), desc="Computing embeddings", unit="frame")

        for batch_start in range(0, len(frame_data), batch_size):
            batch_frames = frame_data[batch_start:batch_start + batch_size]

            # LOAD IMAGES FRESH (avoid holding ALL images in RAM)
            batch_images_pil = []
            batch_has_persons = []
            batch_persons_data = []

            for frame_info in batch_frames:
                img_path = frame_info['path']
                img_cv = cv2.imread(str(img_path))

                if img_cv is None:
                    batch_images_pil.append(None)
                    batch_has_persons.append(False)
                    batch_persons_data.append([])
                    continue

                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                pil_img = PILImage.fromarray(img_rgb)
                batch_images_pil.append(pil_img)
                batch_has_persons.append(frame_info['has_persons'])
                batch_persons_data.append(frame_info['persons'])

            # Prepare scene images (all frames)
            batch_scene_imgs = []
            valid_indices_in_batch = []
            for idx, pil_img in enumerate(batch_images_pil):
                if pil_img is not None:
                    batch_scene_imgs.append(pil_img)
                    valid_indices_in_batch.append(idx)
                    valid_frame_indices.append(batch_start + idx)
                else:
                    print(f"\n⚠️  Unreadable frame skipped: {batch_frames[idx]['path']}")

            # Compute scene embeddings (all frames)
            scene_embs = None
            if len(batch_scene_imgs) > 0:
                inputs_scene = processor(images=batch_scene_imgs, return_tensors="pt", **DINO_PREPROC)
                inputs_scene = {k: v.to(device) for k, v in inputs_scene.items()}

                if device == "cuda":
                    with torch.amp.autocast('cuda'):
                        outputs_scene = model(**inputs_scene)
                        hs = outputs_scene.last_hidden_state  # [B, 1+num_patches, 768]
                        cls_tok = hs[:, 0]                    # [B, 768]
                        patch_mean = hs[:, 1:].mean(dim=1)    # [B, 768]
                        scene_embs = torch.cat([cls_tok, patch_mean], dim=1).cpu().numpy()  # [B, 1536]
                else:
                    outputs_scene = model(**inputs_scene)
                    hs = outputs_scene.last_hidden_state  # [B, 1+num_patches, 768]
                    cls_tok = hs[:, 0]                    # [B, 768]
                    patch_mean = hs[:, 1:].mean(dim=1)    # [B, 768]
                    scene_embs = torch.cat([cls_tok, patch_mean], dim=1).cpu().numpy()  # [B, 1536]

                del outputs_scene, inputs_scene


            # Use ONLY lightweight geometric pose features (20-dim) instead of full DINOv2 skeleton embedding (768-dim)

            # Aggregate pose features per frame (for frames with persons) - SKIP in scene-only mode
            frame_pose_features_aggregates = {}

            if not use_scene_only:
                for valid_idx, batch_idx in enumerate(valid_indices_in_batch):
                    if batch_has_persons[batch_idx]:
                        persons = batch_persons_data[batch_idx]
                        # Average pose features across all persons in frame
                        pose_features_list = [person['pose_features'] for person in persons]
                        frame_pose_features_aggregates[valid_idx] = np.mean(pose_features_list, axis=0)

            # Combine scene embeddings + lightweight pose features
            for valid_idx, batch_idx in enumerate(valid_indices_in_batch):
                scene_emb = scene_embs[valid_idx]

                if use_scene_only:
                    # Scene-only mode: just use DINOv2 scene embeddings (1536-dim)
                    multiview_embeddings.append(scene_emb)
                elif batch_has_persons[batch_idx] and valid_idx in frame_pose_features_aggregates:
                    # Frame with person: scene + pose features (unweighted — weights applied at clustering)
                    combined_emb = np.empty(1536 + pose_features_dim, dtype=np.float32)
                    combined_emb[:1536] = scene_emb
                    combined_emb[1536:] = frame_pose_features_aggregates[valid_idx]
                    multiview_embeddings.append(combined_emb)
                else:
                    # Frame without person: scene only + zero-padded pose features
                    combined_emb = np.empty(1536 + pose_features_dim, dtype=np.float32)
                    combined_emb[:1536] = scene_emb
                    combined_emb[1536:] = zero_pose_features  # Reuse pre-allocated zeros
                    multiview_embeddings.append(combined_emb)

            # Delete batch tensors and numpy arrays (keep image_cv for quality metrics step)
            del batch_images_pil, batch_scene_imgs, scene_embs
            if device == "cuda":
                torch.cuda.empty_cache()

            progress_bar.update(len(batch_frames))

        progress_bar.close()

        if device == "cuda":
            torch.cuda.empty_cache()

    multiview_embeddings = np.array(multiview_embeddings, dtype=np.float32)

    if cache_dir is not None:
        cache_file = cache_dir / "temp_multiview_emb.npy"
        cache_indices_file = cache_dir / "temp_multiview_emb_indices.npy"

        print(f"\n💾 Saving embeddings to cache: {cache_file}")
        np.save(cache_file, multiview_embeddings)
        np.save(cache_indices_file, np.array(valid_frame_indices))
        (cache_dir / "temp_multiview_emb_meta.json").write_text(
            json.dumps({"fingerprint": fingerprint, "preproc": PREANN_PREPROC_TAG}), encoding="utf-8")
        print(f"   ✅ Cache saved ({len(multiview_embeddings)} embeddings)")

    return multiview_embeddings, valid_frame_indices


def compute_quality_metrics_batch(frame_data, anisotropy_threshold, cache_blurry_samples):
    """Compute quality metrics by loading images fresh (minimal RAM usage).

    Blurry frames = high anisotropy (motion blur, camera shake).
    """
    metrics = {
        'brightness': [],
        'contrast': [],
        'anisotropy': [],
        'blurry_image_cache': [],
        'corrupted_images': []
    }

    for frame_info in tqdm(frame_data, desc="Computing quality metrics", unit="img"):
        img_path = frame_info['path']

        # LOAD IMAGE FRESH
        img = cv2.imread(str(img_path))

        if img is None:
            # Skip corrupted images
            print(f"\n⚠️  Warning: Corrupted image: {img_path}")
            metrics['corrupted_images'].append(str(img_path))
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        brightness = np.mean(gray)
        contrast = np.std(gray)

        # Directional blur detection (anisotropy - detects motion blur, camera shake)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        energy_x = np.mean(sobelx ** 2)
        energy_y = np.mean(sobely ** 2)
        anisotropy = max(energy_x, energy_y) / (min(energy_x, energy_y) + 1e-6)

        metrics['brightness'].append(brightness)
        metrics['contrast'].append(contrast)
        metrics['anisotropy'].append(anisotropy)

        # Blurry = high anisotropy (motion blur, camera shake)
        is_blurry = (anisotropy > anisotropy_threshold)
        if is_blurry and len(metrics['blurry_image_cache']) < cache_blurry_samples:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(img_rgb)
            metrics['blurry_image_cache'].append({
                'image': pil_img,
                'anisotropy': anisotropy,
                'brightness': brightness,
                'path': img_path
            })

        del img

    metrics['brightness'] = np.array(metrics['brightness'])
    metrics['contrast'] = np.array(metrics['contrast'])
    metrics['anisotropy'] = np.array(metrics['anisotropy'])

    return metrics


def categorize_lighting(brightness_values):
    """Categorize brightness: 0=Dark (<80), 1=Medium (80-175), 2=Bright (>175)."""
    lighting_labels = np.zeros(len(brightness_values), dtype=int)
    lighting_labels[brightness_values < 80] = 0
    lighting_labels[(brightness_values >= 80) & (brightness_values < 175)] = 1
    lighting_labels[brightness_values >= 175] = 2
    return lighting_labels


def cluster_activities_with_pca_hdbscan(embeddings, n_pca_components, min_cluster_size, min_samples,
                                        cluster_selection_epsilon=0.0, scene_dim=1536,
                                        scene_weight=0.7, pose_weight=0.3):
    """
    Cluster frames by activity using per-block PCA whitening + HDBSCAN.
    Pipeline:
    1. Check minimum samples
    2. Scene block (first scene_dim dims): standardize → PCA whiten to n_pca_components dims
    3. Pose block (remaining dims, if any): standardize
    4. Scale each block so its share of total variance = its weight (scene_weight / pose_weight).
       Weights are applied AFTER standardization/whitening — applying them before would be undone.
    5. L2-normalize (maps to unit hypersphere for Euclidean ≈ cosine distance)
    6. HDBSCAN 'eom' clustering (merges sub-clusters into meaningful groups, reduces fragmentation)
       cluster_selection_epsilon: clusters closer than this distance are merged (0.0=off)
    """
    min_samples_required = 3
    if embeddings.shape[0] < min_samples_required:
        print(f"  ⚠️  WARNING: Only {embeddings.shape[0]} samples")
        print(f"      Requires at least {min_samples_required} samples for clustering")
        print(f"      Skipping clustering - all frames assigned to single group")
        return np.zeros(embeddings.shape[0], dtype=int), embeddings, {"insufficient_samples": True}

    # 1-2. Scene block: standardize + PCA whitening (decorrelates + equalizes scene dimensions)
    scene_scaled = StandardScaler().fit_transform(embeddings[:, :scene_dim])
    actual_components = min(n_pca_components, scene_scaled.shape[0] - 1, scene_scaled.shape[1])
    pca = PCA(n_components=actual_components, whiten=True, random_state=42)
    scene_pcs = pca.fit_transform(scene_scaled)
    explained = pca.explained_variance_ratio_.sum() * 100

    print(f"  PCA: scene {scene_dim} dims → {actual_components} dims")
    print(f"       (whitened, explains {explained:.1f}% of scene variance)")

    # 3-4. Weight blocks by share of total variance (whitened PCs have unit variance each)
    blocks = [scene_pcs * np.sqrt(scene_weight / actual_components)]
    pose = embeddings[:, scene_dim:]
    if pose.shape[1] > 0 and pose_weight > 0:
        pose_scaled = StandardScaler().fit_transform(pose)
        pose_var = float(pose_scaled.var(axis=0).sum())  # constant columns stay 0
        if pose_var > 0:
            blocks.append(pose_scaled * np.sqrt(pose_weight / pose_var))
            print(f"  Blocks: scene weight {scene_weight}, pose weight {pose_weight} (share of total variance)")
    pca_embeddings = np.hstack(blocks)

    # 3. L2 normalize (Euclidean distance on unit hypersphere ≈ cosine distance)
    norms = np.linalg.norm(pca_embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    pca_embeddings_norm = pca_embeddings / norms

    # 4. HDBSCAN with 'eom' selection + optional epsilon merge to absorb nearby outliers
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',
        cluster_selection_epsilon=cluster_selection_epsilon
    )

    activity_labels = clusterer.fit_predict(pca_embeddings_norm)

    n_clusters = len(set(activity_labels)) - (1 if -1 in activity_labels else 0)
    n_noise = np.sum(activity_labels == -1)

    print(f"  HDBSCAN: Found {n_clusters} activity clusters, {n_noise} noise/outlier frames")

    cluster_quality = compute_cluster_quality_metrics(pca_embeddings_norm, activity_labels)

    return activity_labels, pca_embeddings_norm, cluster_quality


def compute_cluster_quality_metrics(embeddings, labels):
    """
    Compute cluster quality metrics (fast, no heavy computation).
    Metrics:
    - Silhouette Score: [-1, 1] - higher is better (measures cluster separation)
    - Calinski-Harabasz: [0, inf] - higher is better (ratio of between/within variance)
    - Davies-Bouldin: [0, inf] - LOWER is better (average similarity ratio)
    """
    # Filter out noise points for metrics calculation
    mask = labels != -1
    if np.sum(mask) < 2:
        # Not enough clustered points
        return {
            'silhouette': None,
            'calinski_harabasz': None,
            'davies_bouldin': None,
            'n_clusters': 0,
            'n_noise': len(labels),
            'n_clustered': 0
        }

    embeddings_clustered = embeddings[mask]
    labels_clustered = labels[mask]

    n_unique = len(set(labels_clustered))
    if n_unique < 2:
        # Only one cluster
        return {
            'silhouette': None,
            'calinski_harabasz': None,
            'davies_bouldin': None,
            'n_clusters': n_unique,
            'n_noise': np.sum(~mask),
            'n_clustered': np.sum(mask)
        }

    # Compute metrics
    try:
        # Silhouette: only sample if dataset is large (>10k points)
        n_clustered = len(embeddings_clustered)
        if n_clustered > 10000:
            silhouette = silhouette_score(embeddings_clustered, labels_clustered, sample_size=10000)
        else:
            # No sampling needed for small datasets
            silhouette = silhouette_score(embeddings_clustered, labels_clustered)

        calinski = calinski_harabasz_score(embeddings_clustered, labels_clustered)
        davies = davies_bouldin_score(embeddings_clustered, labels_clustered)

        print(f"  Quality Metrics:")
        print(f"    Silhouette Score: {silhouette:.3f} (higher is better, range [-1, 1])")
        print(f"    Calinski-Harabasz: {calinski:.1f} (higher is better)")
        print(f"    Davies-Bouldin: {davies:.3f} (LOWER is better)")

        return {
            'silhouette': float(silhouette),
            'calinski_harabasz': float(calinski),
            'davies_bouldin': float(davies),
            'n_clusters': n_unique,
            'n_noise': int(np.sum(~mask)),
            'n_clustered': int(np.sum(mask))
        }
    except Exception as e:
        print(f"  ⚠️  Could not compute quality metrics: {e}")
        return {
            'silhouette': None,
            'calinski_harabasz': None,
            'davies_bouldin': None,
            'n_clusters': n_unique,
            'n_noise': int(np.sum(~mask)),
            'n_clustered': int(np.sum(mask))
        }


def create_quality_chart(metrics, output_path, anisotropy_threshold):
    """Generate combined quality distribution charts."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Brightness histogram
    ax = axes[0, 0]
    ax.hist(metrics['brightness'], bins=30, color='#ffc107', edgecolor='black', alpha=0.7)
    ax.axvline(80, color='red', linestyle='--', linewidth=2, label='Low-light threshold')
    ax.axvline(175, color='green', linestyle='--', linewidth=2, label='Bright threshold')
    ax.set_xlabel('Brightness (0-255)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frame Count', fontsize=11, fontweight='bold')
    ax.set_title('Brightness Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Anisotropy histogram (motion blur detection)
    ax = axes[0, 1]
    ax.hist(metrics['anisotropy'], bins=30, color='#2196F3', edgecolor='black', alpha=0.7)
    ax.axvline(anisotropy_threshold, color='red', linestyle='--', linewidth=2,
               label=f'Blurry threshold ({anisotropy_threshold})')
    ax.set_xlabel('Anisotropy (Gradient Imbalance)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frame Count', fontsize=11, fontweight='bold')
    ax.set_title('Motion Blur Detection', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Lighting category pie chart
    ax = axes[1, 0]
    lighting_labels = categorize_lighting(metrics['brightness'])
    dark_count = np.sum(lighting_labels == 0)
    medium_count = np.sum(lighting_labels == 1)
    bright_count = np.sum(lighting_labels == 2)

    # Filter out zero-count slices to avoid label overlap
    all_counts = [dark_count, medium_count, bright_count]
    all_colors = ['#424242', '#FFA726', '#FFEB3B']
    all_labels = [f'Dark (<80)\n{dark_count} frames',
                  f'Medium (80-175)\n{medium_count} frames',
                  f'Bright (>175)\n{bright_count} frames']

    filtered_counts = [c for c in all_counts if c > 0]
    filtered_colors = [all_colors[i] for i, c in enumerate(all_counts) if c > 0]
    filtered_labels = [all_labels[i] for i, c in enumerate(all_counts) if c > 0]

    ax.pie(filtered_counts, labels=filtered_labels, colors=filtered_colors,
           autopct='%1.1f%%', startangle=90, pctdistance=0.85, labeldistance=1.2,
           textprops={'fontsize': 10, 'weight': 'bold'})
    ax.set_title('Lighting Distribution', fontsize=12, fontweight='bold')

    # Blur category pie chart (motion blur only)
    ax = axes[1, 1]
    blurry_mask = (metrics['anisotropy'] > anisotropy_threshold)
    blurry_count = np.sum(blurry_mask)
    sharp_count = len(metrics['anisotropy']) - blurry_count

    # Filter out zero-count slices to avoid label overlap
    all_blur_counts = [blurry_count, sharp_count]
    all_blur_colors = ['#f44336', '#4caf50']
    all_blur_labels = [f'Blurry (motion)\n{blurry_count} frames',
                       f'No motion blur\n{sharp_count} frames']

    filtered_blur_counts = [c for c in all_blur_counts if c > 0]
    filtered_blur_colors = [all_blur_colors[i] for i, c in enumerate(all_blur_counts) if c > 0]
    filtered_blur_labels = [all_blur_labels[i] for i, c in enumerate(all_blur_counts) if c > 0]

    ax.pie(filtered_blur_counts, labels=filtered_blur_labels, colors=filtered_blur_colors,
           autopct='%1.1f%%', startangle=90, pctdistance=0.85, labeldistance=1.2,
           textprops={'fontsize': 10, 'weight': 'bold'})
    ax.set_title('Motion Blur Quality', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    return output_path


def create_pacmap_visualization(embeddings, activity_labels, output_path):
    """Generate PaCMAP scatter plot colored by activity clusters (handles HDBSCAN noise).

    PaCMAP is used instead of UMAP because it better preserves both local and global structure,
    making cluster positions in 2D more interpretable (spatially related clusters stay nearby).
    """
    n_neighbors_viz = min(10, max(2, embeddings.shape[0] // 10))
    reducer = pacmap.PaCMAP(n_components=2, n_neighbors=n_neighbors_viz, random_state=42)
    X_2d = reducer.fit_transform(embeddings)

    unique_activities = np.unique(activity_labels)
    noise_mask = activity_labels == -1
    cluster_activities = [a for a in unique_activities if a != -1]
    n_clusters = len(cluster_activities)

    if n_clusters <= 15:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig, ax = plt.subplots(figsize=(14, 8))

    if noise_mask.any():
        ax.scatter(X_2d[noise_mask, 0], X_2d[noise_mask, 1],
                  c='lightgray', label='Noise/Outliers',
                  s=20, alpha=0.3, edgecolors='none')

    if n_clusters > 0:
        if n_clusters <= 20:
            cmap = matplotlib.colormaps.get_cmap('tab20')
        else:
            cmap = matplotlib.colormaps.get_cmap('hsv').resampled(n_clusters)

        for idx, activity_id in enumerate(cluster_activities):
            mask = activity_labels == activity_id
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                      c=[cmap(idx % 20 if n_clusters <= 20 else idx / n_clusters)],
                      label=f'Activity {activity_id}',
                      s=40, alpha=0.7, edgecolors='black', linewidth=0.5)

    ax.set_xlabel('PaCMAP Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('PaCMAP Dimension 2', fontsize=12, fontweight='bold')
    ax.set_title('Frame Diversity - Activity Clustering (PaCMAP)', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)

    if n_clusters <= 15:
        ax.legend(loc='best', fontsize=9, ncol=2)
    elif n_clusters <= 40:
        n_legend_cols = 2 if n_clusters <= 25 else 3
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, ncol=n_legend_cols)
    else:
        ax.text(0.98, 0.02, f'{n_clusters} clusters (legend omitted)',
                transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    return output_path


def create_coverage_heatmap(activity_labels, lighting_labels, output_path):
    """Generate heatmap showing Activity × Lighting coverage (excludes noise)."""
    unique_activities = np.unique(activity_labels)
    # Filter out noise (-1)
    cluster_activities = [a for a in unique_activities if a != -1]

    if len(cluster_activities) == 0:
        # No clusters found - create placeholder
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No activity clusters found', ha='center', va='center',
                fontsize=14, color='gray')
        ax.axis('off')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path

    lighting_names = ['Dark (<80)', 'Medium (80-175)', 'Bright (>175)']

    # Build matrix (exclude noise)
    heatmap_data = []
    for activity_id in cluster_activities:
        activity_mask = activity_labels == activity_id
        activity_lighting = lighting_labels[activity_mask]

        row = [
            np.sum(activity_lighting == 0),  # Dark
            np.sum(activity_lighting == 1),  # Medium
            np.sum(activity_lighting == 2)   # Bright
        ]
        heatmap_data.append(row)

    heatmap_data = np.array(heatmap_data)

    # Fixed row height for readability - generate multiple chunks if needed
    n_activities = len(cluster_activities)
    rows_per_page = 20  # Max rows per heatmap chunk

    if n_activities <= rows_per_page:
        # Single heatmap - fits on one page
        fig_height = max(4, n_activities * 0.4)
        fig, ax = plt.subplots(figsize=(8, fig_height))

        im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')

        ax.set_xticks(np.arange(len(lighting_names)))
        ax.set_yticks(np.arange(len(cluster_activities)))
        ax.set_xticklabels(lighting_names, fontsize=10, fontweight='bold')
        ax.set_yticklabels([f'Activity {i}' for i in cluster_activities], fontsize=10)

        for i in range(len(cluster_activities)):
            for j in range(len(lighting_names)):
                ax.text(j, i, int(heatmap_data[i, j]),
                       ha="center", va="center", color="black", fontsize=11, weight='bold')

        ax.set_title('Coverage Heatmap: Activity × Lighting Conditions', fontsize=13, fontweight='bold')
        ax.set_xlabel('Lighting Condition', fontsize=11, fontweight='bold')
        ax.set_ylabel('Activity Cluster', fontsize=11, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Frame Count', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        return output_path

    # Multiple chunks needed - split into separate images
    n_chunks = (n_activities + rows_per_page - 1) // rows_per_page
    output_paths = []

    for chunk_idx in range(n_chunks):
        start_row = chunk_idx * rows_per_page
        end_row = min(start_row + rows_per_page, n_activities)
        chunk_activities = cluster_activities[start_row:end_row]
        chunk_data = heatmap_data[start_row:end_row]

        fig_height = max(4, len(chunk_activities) * 0.4)
        fig, ax = plt.subplots(figsize=(8, fig_height))

        im = ax.imshow(chunk_data, cmap='YlOrRd', aspect='auto')

        ax.set_xticks(np.arange(len(lighting_names)))
        ax.set_yticks(np.arange(len(chunk_activities)))
        ax.set_xticklabels(lighting_names, fontsize=10, fontweight='bold')
        ax.set_yticklabels([f'Activity {i}' for i in chunk_activities], fontsize=10)

        for i in range(len(chunk_activities)):
            for j in range(len(lighting_names)):
                ax.text(j, i, int(chunk_data[i, j]),
                       ha="center", va="center", color="black", fontsize=11, weight='bold')

        title = f'Coverage Heatmap: Activity × Lighting (Part {chunk_idx + 1}/{n_chunks})'
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Lighting Condition', fontsize=11, fontweight='bold')
        ax.set_ylabel('Activity Cluster', fontsize=11, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Frame Count', fontsize=10, fontweight='bold')

        plt.tight_layout()

        chunk_path = output_path.parent / f"coverage_heatmap_part{chunk_idx + 1}.png"
        plt.savefig(chunk_path, dpi=100, bbox_inches='tight')
        plt.close()
        output_paths.append(chunk_path)

    # Return list of paths for multi-page rendering
    return output_paths


## ---------------------------------- MAIN PIPELINE -----------------------------------------
import os


def _setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("eda_preann")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def parse_args():
    p = argparse.ArgumentParser(description="Pre-annotation frame quality assessment pipeline")
    p.add_argument("--interactive", action="store_true",
                   help="Run with questionary prompts instead of CLI args")
    # Required in non-interactive mode
    p.add_argument("--frames_dir", type=str, default=None)
    # Model / embedding
    p.add_argument("--yolo_batch_size", type=int, default=16)
    p.add_argument("--yolo_conf", type=float, default=0.3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--anisotropy_threshold", type=float, default=3.6)
    # Clustering
    p.add_argument("--n_components", type=int, default=128)
    p.add_argument("--min_cluster_size", type=int, default=25)
    p.add_argument("--min_samples", type=int, default=3)
    # Embedding weights
    p.add_argument("--scene_weight", type=float, default=0.7)
    p.add_argument("--pose_weight", type=float, default=0.3)
    p.add_argument("--cluster_epsilon", type=float, default=0.3)
    # PDF / report
    p.add_argument("--cache_blurry_num_samples", type=int, default=24)
    p.add_argument("--activity_num_samples", type=int, default=20)
    p.add_argument("--outliers_num_samples", type=int, default=52)
    p.add_argument("--pdf_image_quality", type=int, default=70)
    p.add_argument("--grid_cols_activities", type=int, default=4)
    p.add_argument("--grid_cols_blurry", type=int, default=4)
    p.add_argument("--pdf_name", type=str, default=None)
    p.add_argument("--use_embedding_cache", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    if not args.interactive and args.frames_dir is None:
        p.error("Non-interactive mode requires: --frames_dir")
    return args


def ask_or_exit(prompt_result):
    """Exit cleanly if user cancels a questionary prompt (Ctrl+C)."""
    if prompt_result is None:
        print("\n   Cancelled by user. Exiting.")
        sys.exit(0)
    return prompt_result


def _ask(prompt_fn):
    """Wrap any questionary prompt with cancel + RuntimeError handling."""
    try:
        return ask_or_exit(prompt_fn.ask())
    except (KeyboardInterrupt, EOFError, RuntimeError):
        print("\n   Cancelled by user. Exiting.")
        sys.exit(0)


def _prompt_path(message, must_exist_dir=False, must_exist_file=False):
    """Prompt for a path with autocomplete and validation."""
    def _validate(p):
        p = p.strip().strip('"').strip("'")
        if not p:
            return "This field is required."
        if must_exist_dir and not os.path.isdir(p):
            return f"Directory does not exist: {p}"
        if must_exist_file and not os.path.isfile(p):
            return f"File does not exist: {p}"
        return True
    val = _ask(questionary.path(message, validate=_validate))
    return val.strip().strip('"').strip("'")


def _prompt_text(message, default=None):
    """Prompt for text with an optional default."""
    if default is not None:
        val = _ask(questionary.text(message, default=str(default)))
    else:
        val = _ask(questionary.text(message))
    val = val.strip().strip('"').strip("'")
    return val if val else (str(default) if default is not None else "")


def main():
    os.chdir(Path(__file__).resolve().parent.parent)

    args = parse_args()

    print("="*60)
    print("PRE-ANNOTATION PIPELINE — DINOv2 + PaCMAP + PCA")
    print("="*60)

    RESULTS_DIR = "preann_results"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.interactive:
        ## -----------------------------------------------##
        ##   REQUIRED INPUTS (prompted, no defaults)       ##
        ## -----------------------------------------------##
        print("\n📁  STEP 0: CONFIGURE PIPELINE INPUTS")
        print("="*60)

        frames_dir = _prompt_path("Path to frames folder:", must_exist_dir=True)

        print(f"\n📂  All results will be saved under: {os.path.abspath(RESULTS_DIR)}/")

        ## -----------------------------------------------##
        ##   OPTIONAL PARAMETERS                           ##
        ## -----------------------------------------------##

        change_defaults = _ask(questionary.confirm(
            "Change default parameters?", default=False))

        if change_defaults:
            yolo_batch_size      = int(_prompt_text("YOLO pose detection batch size", 16))
            yolo_conf_person_thr = float(_prompt_text("YOLO pose detection confidence threshold", 0.3))
            batch_size           = int(_prompt_text("DINOv2 embedding batch size", 64))
            anisotropy_threshold = float(_prompt_text("Motion blur anisotropy threshold (lower=stricter)", 3.6))
        else:
            yolo_batch_size      = 16
            yolo_conf_person_thr = 0.3
            batch_size           = 64
            anisotropy_threshold = 3.6

        # Clustering — always ask (these affect results the most)
        n_components      = int(_prompt_text("PCA output dimensions for clustering (128=balanced, 256=finer detail)", 128))
        min_cluster_size  = int(_prompt_text("HDBSCAN min cluster size (smallest group of frames that counts as an activity)", 25))
        min_samples       = int(_prompt_text("HDBSCAN min_samples (how strict a frame must match its neighbors to join a cluster; lower=more frames included, higher=only dense core frames)", 3))

        # Derive default PDF name from frames folder name
        _frames_folder_name = Path(frames_dir).name
        _default_pdf_name   = f"PreAnnotation_Quality_Report_{_frames_folder_name}.pdf"

        if change_defaults:
            scene_weight                 = float(_prompt_text("Scene embedding weight (0.7=scene-dominant, lower=more pose influence)", 0.7))
            pose_weight                  = float(_prompt_text("Pose features weight (0.3=more action influence, lower=less pose)", 0.3))
            cluster_selection_epsilon    = float(_prompt_text("Cluster merge epsilon (0.0=off, 0.3=absorb nearby outliers, higher=more merging)", 0.3))
            cache_blurry_num_samples     = int(_prompt_text("Blurry samples shown in PDF", 24))
            activity_num_samples         = int(_prompt_text("Samples shown per activity in PDF", 20))
            outliers_num_samples         = int(_prompt_text("Samples shown for outliers in PDF", 52))
            pdf_image_quality            = int(_prompt_text("PDF image JPEG quality (1-100)", 70))
            grid_cols_activities         = int(_prompt_text("Grid columns for activity montages in PDF", 4))
            grid_cols_blurry             = int(_prompt_text("Grid columns for blurry montage in PDF", 4))
            pdf_name                     = _prompt_text("Output PDF filename", _default_pdf_name)
        else:
            scene_weight                 = 0.7
            pose_weight                  = 0.3
            cluster_selection_epsilon    = 0.3
            cache_blurry_num_samples     = 24
            activity_num_samples         = 20
            outliers_num_samples         = 52
            pdf_image_quality            = 70
            grid_cols_activities         = 4
            grid_cols_blurry             = 4
            pdf_name                     = _default_pdf_name

        use_embedding_cache = _ask(questionary.confirm("Use embedding cache?", default=True))

    else:
        ## -----------------------------------------------##
        ##   NON-INTERACTIVE: resolve all values from args ##
        ## -----------------------------------------------##
        frames_dir           = args.frames_dir
        yolo_batch_size      = args.yolo_batch_size
        yolo_conf_person_thr = args.yolo_conf
        batch_size           = args.batch_size
        anisotropy_threshold = args.anisotropy_threshold
        n_components         = args.n_components
        min_cluster_size     = args.min_cluster_size
        min_samples          = args.min_samples
        scene_weight         = args.scene_weight
        pose_weight          = args.pose_weight
        cluster_selection_epsilon = args.cluster_epsilon
        cache_blurry_num_samples  = args.cache_blurry_num_samples
        activity_num_samples      = args.activity_num_samples
        outliers_num_samples      = args.outliers_num_samples
        pdf_image_quality         = args.pdf_image_quality
        grid_cols_activities      = args.grid_cols_activities
        grid_cols_blurry          = args.grid_cols_blurry
        use_embedding_cache       = False  # headless always recomputes — no stale cache

        _frames_folder_name = Path(frames_dir).name
        pdf_name = args.pdf_name if args.pdf_name else f"PreAnnotation_Quality_Report_{_frames_folder_name}.pdf"

        # Non-interactive: delete cache files so stale embeddings are never reused
        for _cache in ["temp_multiview_emb.npy", "temp_multiview_emb_indices.npy", "temp_multiview_emb_meta.json"]:
            _p = os.path.join(RESULTS_DIR, _cache)
            if os.path.isfile(_p):
                os.remove(_p)

        print(f"\n📂  All results will be saved under: {os.path.abspath(RESULTS_DIR)}/")

    output_dir = RESULTS_DIR

    logger = _setup_logger(os.path.join(RESULTS_DIR, "preann_run.log"))
    logger.info("Pipeline started")
    logger.info(f"frames_dir={frames_dir}  batch_size={batch_size}  n_components={n_components}")
    logger.info(f"min_cluster_size={min_cluster_size}  min_samples={min_samples}  anisotropy_threshold={anisotropy_threshold}")

    # Model Name for Embedding
    model_name = "facebook/dinov2-base"  # [CLS||avg_patches] = 1536d, strong fine-grained separation

    print("\n" + "="*60)

    # Set random seed for reproducibility (same frames = same report every time)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Setup
    frames_root = Path(frames_dir)
    if not frames_root.exists():
        raise RuntimeError(f"Frames directory does not exist: {frames_root}")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # GPU setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"Using GPU: {gpu_props.name}")
        print(f"GPU Memory: {gpu_props.total_memory/1e9:.1f}GB")
        torch.cuda.set_per_process_memory_fraction(0.95)
    else:
        print("Using CPU - GPU not available")

    # Load DINOv2 model from Hugging Face
    print(f"\nLoading DINOv2 model from Hugging Face: {model_name} ...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    # STEP 1: SCAN FRAMES
    print("\n" + "="*60)
    print("SCANNING FRAMES")
    print("="*60)
    image_paths = scan_frames(frames_root)

    if not image_paths:
        logger.error("No frames found in frames_dir — exiting")
        print("No frames found. Exiting.")
        return

    print(f"\nTotal frames found: {len(image_paths)}")

    # Load YOLOv11 Pose model
    print(f"\nLoading YOLO11 Pose model...")
    pose_model = YOLO('models/yolo11n-pose.pt')
    print("  ✅ YOLO11 Pose model loaded")

    # STEP 2: PERSON DETECTION & POSE EXTRACTION 
    print("\n" + "="*60)
    print("STEP 2: PERSON DETECTION & POSE EXTRACTION")
    print("="*60)
    print(f"✅ Using batch size {yolo_batch_size} for YOLO inference.")
    frame_data = detect_persons_and_extract_poses(image_paths, pose_model, device, conf_threshold=yolo_conf_person_thr, yolo_batch_size=yolo_batch_size)

    # Count frames with persons and max persons per frame
    frames_with_persons = sum(1 for f in frame_data if f['has_persons'])
    frames_without_persons = len(image_paths) - frames_with_persons
    max_persons_in_frame = max((len(f['persons']) for f in frame_data), default=0)

    print(f"\n  ✅ Found persons in {frames_with_persons}/{len(image_paths)} frames")
    print(f"  ℹ️  Max persons in single frame: {max_persons_in_frame}")
    if frames_without_persons > 0:
        print(f"  ℹ️  {frames_without_persons} frames without persons will use scene-only embeddings")

    # STEP 3: COMPUTE EMBEDDINGS (pose-aware when persons detected, scene-only otherwise)
    print("\n" + "="*60)
    print("STEP 3: COMPUTE DINOv2 EMBEDDINGS ([CLS||avg_patches], 1536d)")
    print("="*60)
    cache_dir = output_dir if use_embedding_cache else None
    embeddings, valid_frame_indices = compute_multiview_embeddings(frame_data, processor, model, device, batch_size, cache_dir=cache_dir)

    if len(embeddings) == 0:
        logger.error("No valid embeddings computed — all frames may be corrupted")
        print("No valid multiview embeddings. Exiting.")
        return

    print(f"  Embeddings shape: {embeddings.shape}")

    # Map valid_frame_indices to valid frame_data for quality metrics
    valid_frame_data = [frame_data[i] for i in valid_frame_indices]
    valid_paths = [f['path'] for f in valid_frame_data]  # Extract paths for later use

    # Recalculate person stats for VALID frames only (exclude corrupted/failed frames)
    frames_with_persons = sum(1 for f in valid_frame_data if f['has_persons'])
    max_persons_in_frame = max((len(f['persons']) for f in valid_frame_data), default=0)
    print(f"\n  ℹ️  Person stats (valid frames only): {frames_with_persons}/{len(valid_frame_data)} frames with persons, max {max_persons_in_frame} persons/frame")

    # STEP 4: COMPUTE QUALITY METRICS (reuses pre-loaded images from frame_data)
    print("\n" + "="*60)
    print("STEP 4: COMPUTE QUALITY METRICS")
    print("="*60)
    quality_metrics = compute_quality_metrics_batch(valid_frame_data, anisotropy_threshold=anisotropy_threshold, cache_blurry_samples=cache_blurry_num_samples)

    # Report corrupted images
    if len(quality_metrics['corrupted_images']) > 0:
        logger.warning(f"{len(quality_metrics['corrupted_images'])} corrupted/unreadable images skipped: {quality_metrics['corrupted_images'][:10]}")
        print(f"\n⚠️  Warning: {len(quality_metrics['corrupted_images'])} corrupted/unreadable images skipped")
        print(f"   These images will be excluded from quality metrics analysis")

    # STEP 5: UMAP + HDBSCAN CLUSTERING
    print("\n" + "="*60)
    print("STEP 5: PCA WHITENING + HDBSCAN CLUSTERING")
    print("="*60)
    activity_labels, pca_embeddings, cluster_quality = cluster_activities_with_pca_hdbscan(
        embeddings, n_pca_components=n_components, min_cluster_size=min_cluster_size, min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon, scene_weight=scene_weight, pose_weight=pose_weight)

    # Count activities (excluding noise)
    unique_activities = np.unique(activity_labels)
    n_activities = len([a for a in unique_activities if a != -1])
    n_noise = np.sum(activity_labels == -1)

    print(f"  ✅ Found {n_activities} distinct activity clusters")
    if n_noise > 0:
        print(f"  ℹ️  {n_noise} frames marked as noise/outliers (not enough similarity to form clusters)")

    # STEP 6: CATEGORIZE LIGHTING
    print("\n" + "="*60)
    print("STEP 6: CATEGORIZE LIGHTING")
    print("="*60)
    lighting_labels = categorize_lighting(quality_metrics['brightness'])

    dark_count = np.sum(lighting_labels == 0)
    medium_count = np.sum(lighting_labels == 1)
    bright_count = np.sum(lighting_labels == 2)

    print(f"  Dark frames (<80): {dark_count}")
    print(f"  Medium frames (80-175): {medium_count}")
    print(f"  Bright frames (>175): {bright_count}")

    # STEP 7: ANALYZE BLUR (motion blur via anisotropy)
    print("\n" + "="*60)
    print("STEP 7: ANALYZE BLUR (Motion Blur)")
    print("="*60)

    # Blurry = high anisotropy (motion blur, camera shake)
    blurry_mask = (quality_metrics['anisotropy'] > anisotropy_threshold)
    blurry_count = np.sum(blurry_mask)
    sharp_count = len(quality_metrics['anisotropy']) - blurry_count

    print(f"  Blurry frames (anisotropy >{anisotropy_threshold}): {blurry_count}")
    print(f"  Non-blurry frames: {sharp_count}")

    # STEP 8: SUMMARIZE ACTIVITIES
    print("\n" + "="*60)
    print("STEP 8: ACTIVITY SUMMARY")
    print("="*60)
    activity_names = {}
    unique_activities = np.unique(activity_labels)

    # Skip noise (-1)
    cluster_activities = [a for a in unique_activities if a != -1]

    for activity_id in cluster_activities:
        activity_mask = activity_labels == activity_id
        n_frames = np.sum(activity_mask)
        activity_names[activity_id] = f"Activity {activity_id}"
        print(f"  Activity {activity_id}: {n_frames} frames")

    # Add noise label
    if -1 in unique_activities:
        activity_names[-1] = "Noise/Outliers"
        print(f"  Noise/Outliers: {n_noise} frames")

    # STEP 9: COMPUTE QUALITY SCORE
    if len(quality_metrics['anisotropy']) == 0:
        # No valid frames for quality metrics
        quality_score = 0
        logger.error("No valid quality metrics — all images corrupted or failed to load")
        print("  ⚠️  No valid quality metrics - all images corrupted or failed to load")
    else:
        median_brightness = np.median(quality_metrics['brightness'])
        median_contrast = np.median(quality_metrics['contrast'])
        median_anisotropy = np.median(quality_metrics['anisotropy'])

        # Anisotropy score (lower is better - inverted)
        anisotropy_score = max(0, 10 - (median_anisotropy / 5) * 10)  # Lower anisotropy = less motion blur
        brightness_score = 10 - abs(median_brightness - 127) / 127 * 10
        contrast_score = min(10, median_contrast / 50 * 10)

        quality_score = (anisotropy_score + brightness_score + contrast_score) / 3

    # STEP 10: GENERATE RECOMMENDATIONS
    recommendations = []

    if len(valid_paths) == 0:
        recommendations.append("❌ CRITICAL: No valid frames found! All images are corrupted or unreadable.")
        dark_pct = 0
        blurry_pct = 0
        noise_pct = 0
    else:
        dark_pct = dark_count / len(valid_paths) * 100
        blurry_pct = blurry_count / len(valid_paths) * 100
        noise_pct = n_noise / len(valid_paths) * 100

    if dark_pct > 60:
        recommendations.append(f"⚠ {dark_pct:.0f}% frames are low-light. Consider adding well-lit frames or dropping extreme dark frames.")
    if blurry_pct > 40:
        recommendations.append(f"⚠ {blurry_pct:.0f}% frames are blurry (motion blur). Use higher shutter speed, stabilize camera, or reduce camera movement. Consider dropping these frames.")
    if n_activities < 3:
        recommendations.append(f"⚠ Only {n_activities} activity patterns detected. Capture more diverse work activities/scenarios.")
    if len(embeddings) < 50:
        recommendations.append(f"⚠ Only {len(embeddings)} valid frames. Minimum 100-200 recommended for robust YOLO training.")
    if noise_pct > 30:
        recommendations.append(f"⚠ {noise_pct:.0f}% frames marked as noise/outliers. Review noise montage - may indicate inconsistent capture conditions or rare activities.")
    elif noise_pct > 0:
        recommendations.append(f"ℹ️ {noise_pct:.1f}% frames are noise/outliers (rare activities or transitions). Review noise montage to verify quality.")

    if not recommendations:
        recommendations.append("✅ Dataset quality looks good! Proceed with annotation.")

    # STEP 11: GENERATE PDF REPORT
    print("\n" + "="*70)
    print("STEP 11: GENERATING PDF REPORT")
    print("="*70)

    # Create temp directory for charts
    temp_dir = Path("temp_preannotation_charts")
    temp_dir.mkdir(exist_ok=True)

    # Generate visualizations
    print("\nGenerating visualizations...")
    quality_chart_path = temp_dir / "quality_metrics.png"
    umap_chart_path = temp_dir / "umap_diversity.png"
    heatmap_chart_path = temp_dir / "coverage_heatmap.png"

    create_quality_chart(quality_metrics, quality_chart_path, anisotropy_threshold=anisotropy_threshold)
    plt.close('all') 
    gc.collect()

    create_pacmap_visualization(pca_embeddings, activity_labels, umap_chart_path)
    plt.close('all')
    gc.collect()

    create_coverage_heatmap(activity_labels, lighting_labels, heatmap_chart_path)
    plt.close('all')
    gc.collect()

    # Generate activity montages
    print("\nGenerating activity montages...")
    for activity_id in cluster_activities:
        activity_mask = activity_labels == activity_id
        activity_paths = [valid_paths[i] for i, mask_val in enumerate(activity_mask) if mask_val]

        montage = create_activity_montage(activity_paths, activity_id, max_samples=activity_num_samples, grid_cols=grid_cols_activities, target_height=250)
        montage_path = temp_dir / f"activity_{activity_id}_montage.png"
        montage.save(montage_path)
        del montage  # FREE MEMORY
        gc.collect()
        print(f"  ✅ Activity {activity_id} montage created")

    # Generate noise/outlier montage if present
    noise_paths = []  # Keep for post-PDF outlier folder prompt
    if -1 in unique_activities and n_noise > 0:
        noise_mask = activity_labels == -1
        noise_paths = [valid_paths[i] for i, mask_val in enumerate(noise_mask) if mask_val]

        # Show samples in montage (larger thumbnails for better visibility)
        montage = create_activity_montage(noise_paths, -1, max_samples=outliers_num_samples, grid_cols=grid_cols_activities, target_height=250)
        montage_path = temp_dir / f"activity_-1_montage.png"
        montage.save(montage_path)
        del montage  # FREE MEMORY
        gc.collect()
        print(f"  ✅ Noise/Outliers montage created ({n_noise} frames)")

    # Generate blurry images montage
    print("\nGenerating blurry images montage...")
    blurry_montage = create_blurry_images_montage(quality_metrics['blurry_image_cache'], grid_cols=grid_cols_blurry, target_height=250)
    if blurry_montage is not None:
        blurry_montage_path = temp_dir / "blurry_images_montage.png"
        blurry_montage.save(blurry_montage_path)
        del blurry_montage  # FREE MEMORY
        gc.collect()
        print("  ✅ Blurry images montage created")
    else:
        print("  ℹ️  No blurry images found - skipping montage")

    print("  ✅ All visualizations generated")

    # Prepare analysis data for PDF
    analysis_data = {
        'quality_metrics': quality_metrics,
        'activity_labels': activity_labels,
        'lighting_labels': lighting_labels,
        'activity_names': activity_names,
        'valid_paths': valid_paths,
        'anisotropy_threshold': anisotropy_threshold,
        'grid_cols_activities': grid_cols_activities,
        'grid_cols_blurry': grid_cols_blurry,
        'activity_num_samples': activity_num_samples,
        'outliers_num_samples': outliers_num_samples,
        'cache_blurry_num_samples': cache_blurry_num_samples,
        'recommendations': recommendations,
        'cluster_quality': cluster_quality,
        'config': {
            'model_name': model_name,
            'scene_weight': scene_weight,
            'pose_weight': pose_weight,
        },
        'summary_stats': {
            'total_frames': len(image_paths),
            'valid_frames': len(valid_paths),
            'n_activities': n_activities,
            'n_noise': n_noise,
            'frames_with_persons': frames_with_persons,
            'max_persons_in_frame': max_persons_in_frame,
            'dark_count': dark_count,
            'dark_pct': dark_pct,
            'medium_count': medium_count,
            'medium_pct': (medium_count / len(valid_paths) * 100) if len(valid_paths) > 0 else 0,
            'bright_count': bright_count,
            'bright_pct': (bright_count / len(valid_paths) * 100) if len(valid_paths) > 0 else 0,
            'blurry_count': blurry_count,
            'blurry_pct': blurry_pct,
            'sharp_count': sharp_count,
            'sharp_pct': (sharp_count / len(valid_paths) * 100) if len(valid_paths) > 0 else 0,
            'quality_score': quality_score
        }
    }

    # Generate PDF report
    pdf_path = output_dir / pdf_name
    generate_pdf_report(analysis_data, pdf_path, temp_dir, image_quality=pdf_image_quality)

    logger.info(f"Pipeline complete — PDF at {pdf_path}")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Output: {pdf_path}")
    print("="*60)

    # STEP 12: OPTIONAL — Save all outlier frames to a folder (interactive only)
    if args.interactive and noise_paths:
        print("\n" + "="*60)
        print("STEP 12: OUTLIER FRAMES (OPTIONAL SAVE)")
        print("="*60)
        print(f"  ℹ️  {n_noise} outlier frames detected.")
        print(f"  ⚠️  Warning: copying {n_noise} images could be a memory/disk burden for large datasets.")
        save_outliers = _ask(questionary.confirm(
            f"Save all {n_noise} outlier frames to a folder?", default=False))

        if save_outliers:
            outliers_folder_name = f"outliers_{_frames_folder_name}"
            outliers_folder = Path(output_dir) / outliers_folder_name
            if outliers_folder.exists():
                print(f"\n  ⚠️  WARNING: Folder '{outliers_folder}' already exists — it will be overwritten!")
                confirm_overwrite = _ask(questionary.confirm("Overwrite existing outliers folder?", default=False))
                if confirm_overwrite:
                    shutil.rmtree(outliers_folder)
                else:
                    print("  Skipping outlier folder save.")
                    save_outliers = False

        if save_outliers:
            outliers_folder.mkdir(parents=True, exist_ok=True)
            print(f"\n  Copying {n_noise} outlier frames to: {outliers_folder}")
            for src_path in tqdm(noise_paths, desc="Copying outliers", unit="file"):
                shutil.copy2(src_path, outliers_folder / src_path.name)
            print(f"  ✅ Outlier frames saved to: {outliers_folder}")

    # Clean up cache files (optional - user can keep them for faster re-runs)
    if use_embedding_cache:
        cache_file = output_dir / "temp_multiview_emb.npy"
        cache_indices_file = output_dir / "temp_multiview_emb_indices.npy"
        if cache_file.exists() or cache_indices_file.exists():
            print(f"\nℹ️  Embedding cache files saved in {output_dir}")
            print("   Delete temp_multiview_emb.* files to force recomputation on next run")

    # Clean up GPU memory
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
