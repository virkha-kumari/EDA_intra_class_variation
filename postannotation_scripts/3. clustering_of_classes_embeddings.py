"""
Clustering using: Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
    - unique approach to identifying clusters based on the density of data points
    - MinPts (Minimum Points), is a parameter that specifies min number of points required to form a dense region, which is consider a cluster.
    - Epsilon - key parameter, is max distance between two points for them to be considered as part of the same neighborhood.
    - Unlike some clustering algorithms, DBSCAN does not predict the cluster membership of new, unseen data points.
      Once the model is trained, it is applied to the existing dataset without the ability to generalize to new observations outside the training set.
"""

"""
Analyze intra-class variations using DBSCAN clustering on DINOv2 embeddings.

This script clusters embeddings WITHIN each class to discover sub-groups,
outliers, and understand class heterogeneity. Saves sample images from each cluster.

Usage:
    python '.\EDA_intra_class_variation_scripts\3. clustering_of_classes_embeddings.py' --auto_tune --save_montage
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter threading issues
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from pathlib import Path
import numpy as np
import argparse
import seaborn as sns
from PIL import Image
import shutil
import pandas as pd
import time

try:
    import cuml
    import torch
    if not torch.cuda.is_available():
        raise ImportError("cuML installed but no CUDA device available — falling back to CPU")
    from cuml.cluster import DBSCAN as _DBSCAN
    from cuml.neighbors import NearestNeighbors as _NearestNeighbors
    from cuml.manifold import UMAP as _UMAP
    GPU_BACKEND = True
    print(f"[backend] cuML {cuml.__version__} detected — GPU mode")
except ImportError as e:
    from sklearn.cluster import DBSCAN as _DBSCAN
    from sklearn.neighbors import NearestNeighbors as _NearestNeighbors
    from umap import UMAP as _UMAP
    GPU_BACKEND = False
    print(f"[backend] CPU mode — {e}")

CLASSES_PER_OVERVIEW_PAGE = 30
MAX_OUTLIERS_PER_CLASS = 30


def knee_index(sorted_vals):
    """Elbow of an ascending curve: point farthest below the chord from first to last value
    (Kneedle-style, on [0,1]-normalised axes). None if the curve is flat."""
    y = np.asarray(sorted_vals, dtype=float)
    if len(y) < 3 or y[-1] - y[0] <= 0:
        return None
    x = np.linspace(0.0, 1.0, len(y))
    y_norm = (y - y[0]) / (y[-1] - y[0])
    return int(np.argmax(x - y_norm))


def auto_tune_eps(embeddings, min_samples, percentile, method="knee"):
    """
        Automatically find optimal eps using k-nearest neighbors.
        What this does?
        - For every point in your data, find its `min_samples` closest neighbors
        - Calculates the distance to each neighbor using cosine distance
        - Returns a matrix: `distances[i, j]` = distance from point `i` to its `j-th` nearest neighbor

        Example: If `min_samples=3` and you have 200 images:
        ```
        Point 0: [0.05, 0.12, 0.18]  ← distances to 1st, 2nd, 3rd nearest neighbors
        Point 1: [0.08, 0.15, 0.22]
        Point 2: [0.03, 0.10, 0.14]
        ...
        Point 199: [0.06, 0.13, 0.19]
        ```

        kneighbors() on the fit data counts each point as its own 1st neighbour, so the last column
        is exactly DBSCAN's core-point test (min_samples incl. the point itself): a point is core
        iff its k-distance <= eps.
        - method="percentile": eps = p-th percentile → ~p% of points are core BY CONSTRUCTION, so the
          outlier rate is capped near (100-p)% and reflects the setting, not the data.
        - method="knee" (default): eps = elbow of the sorted k-distance curve (classic DBSCAN
          heuristic, Ester et al. 1996) → the outlier rate is driven by the data. Falls back to
          percentile if the curve is flat.
    """

    # Converts each 768D embedding vector to unit length to make cosine distance calculations work properly
    embeddings_norm = normalize(embeddings, norm='l2')

    # Find k-nearest neighbors
    # GPU path: cuML kNN only supports euclidean — but on L2-normalized vectors,
    # cosine_dist = euclidean² / 2, so we convert back after computing the percentile
    t0 = time.time()
    if GPU_BACKEND:
        neighbors = _NearestNeighbors(n_neighbors=min_samples, metric='euclidean')
    else:
        neighbors = _NearestNeighbors(n_neighbors=min_samples, metric='cosine')
    neighbors.fit(embeddings_norm)
    distances, _ = neighbors.kneighbors(embeddings_norm)
    print(f"  [time] kNN fit+query: {time.time()-t0:.2f}s")

    # Use the distance to the k-th nearest neighbor --- distance to the **farthest** of the k-nearest neighbors
    k_distances = np.sort(np.asarray(distances[:, -1]))

    knee = knee_index(k_distances) if method == "knee" else None
    if knee is not None:
        pct = k_distances[knee]
        print(f"  [eps] knee at {knee / max(len(k_distances) - 1, 1) * 100:.1f}th percentile of k-distances")
    else:
        if method == "knee":
            print("  [eps] flat k-distance curve — falling back to percentile")
        pct = np.percentile(k_distances, percentile)

    # Convert euclidean percentile back to cosine-domain so eps stays in cosine units throughout
    if GPU_BACKEND:
        optimal_eps = float(pct ** 2) / 2.0
    else:
        optimal_eps = float(pct)

    return optimal_eps

def cluster_single_class(embeddings, class_name, eps, min_samples):
    """Apply DBSCAN to a single class's embeddings."""
    # DBSCAN works better on normalized embeddings in high-D space
    embeddings_norm = normalize(embeddings, norm='l2')

    # GPU path: cuML DBSCAN is euclidean-only — convert eps from cosine to euclidean domain.
    # On unit vectors: euclidean² = 2 * cosine_dist, so eps_euc = sqrt(2 * eps_cos)
    t0 = time.time()
    if GPU_BACKEND:
        eps_euc = np.sqrt(2.0 * eps)
        dbscan = _DBSCAN(eps=eps_euc, min_samples=min_samples, metric='euclidean')
    else:
        dbscan = _DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    cluster_labels = dbscan.fit_predict(embeddings_norm)
    if GPU_BACKEND:
        cluster_labels = np.asarray(cluster_labels)
    print(f"  [time] DBSCAN fit_predict: {time.time()-t0:.2f}s")

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_outliers = int(np.sum(cluster_labels == -1))

    print(f"\n{class_name}:")
    print(f"  - Found {n_clusters} sub-clusters")
    print(f"  - Identified {n_outliers} outliers ({n_outliers/len(embeddings)*100:.1f}%)")

    return cluster_labels, n_clusters, n_outliers

def save_cluster_samples(image_files, cluster_labels, class_name, output_dir, max_samples):
    """Save sample images from each cluster and outliers."""
    cluster_dir = output_dir / f"{class_name}_samples"
    cluster_dir.mkdir(exist_ok=True)

    # Get unique clusters (excluding -1 which is outliers)
    unique_clusters = sorted(set(cluster_labels))

    print(f"  - Saving sample images to: {cluster_dir}")

    for cluster_id in unique_clusters:
        # Get indices for this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        # Determine folder name
        if cluster_id == -1:
            folder_name = "outliers"
        else:
            folder_name = f"cluster_{cluster_id}"

        cluster_folder = cluster_dir / folder_name
        cluster_folder.mkdir(exist_ok=True)

        # Save all outlier images else Sample random images from this cluster
        if folder_name == "outliers":
            n_samples = len(cluster_indices)
            sampled_indices = cluster_indices
        else:
            n_samples = min(max_samples, len(cluster_indices))
            sampled_indices = np.random.choice(cluster_indices, n_samples, replace=False)

        # Copy images
        for idx in sampled_indices:
            src_path = image_files[idx]
            dst_path = cluster_folder / src_path.name
            shutil.copy2(src_path, dst_path)

        print(f"    - {folder_name}: saved {n_samples}/{len(cluster_indices)} samples")

def create_cluster_montage(image_files, cluster_labels, class_name, output_path, max_per_cluster):
    """Create a montage showing sample images from each cluster.

    Splits into multiple part files when cluster count would exceed JPEG's
    65500px height limit (libjpeg hard constraint).
    """
    unique_clusters = sorted(set(cluster_labels))
    n_cols = max_per_cluster
    # At 150 DPI, each row is 3in * 150 = 450px. Stay under 65000px → max 144 rows.
    MAX_ROWS_PER_PART = max(1, 65000 // int(3 * 150))

    base_path = Path(output_path).with_suffix('.jpg')
    parts = [unique_clusters[i:i + MAX_ROWS_PER_PART]
             for i in range(0, len(unique_clusters), MAX_ROWS_PER_PART)]

    saved_paths = []
    for part_idx, cluster_batch in enumerate(parts):
        n_rows = len(cluster_batch)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for row_idx, cluster_id in enumerate(cluster_batch):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            n_samples = min(max_per_cluster, len(cluster_indices))
            sampled_indices = np.random.choice(cluster_indices, n_samples, replace=False)

            if cluster_id == -1:
                row_title = f"Outliers (n={len(cluster_indices)})"
                color = 'red'
            else:
                row_title = f"Cluster {cluster_id} (n={len(cluster_indices)})"
                color = 'blue'

            for col_idx in range(n_cols):
                ax = axes[row_idx, col_idx]
                if col_idx < len(sampled_indices):
                    img_path = image_files[sampled_indices[col_idx]]
                    img = Image.open(img_path).convert("RGB")
                    img.thumbnail((600, 600), Image.LANCZOS)
                    ax.imshow(np.asarray(img))
                    ax.set_title(img_path.name, fontsize=8)
                ax.axis('off')

            axes[row_idx, 0].text(-0.1, 0.5, row_title,
                                 transform=axes[row_idx, 0].transAxes,
                                 fontsize=12, fontweight='bold', color=color,
                                 rotation=90, va='center', ha='right')

        title_suffix = f" (part {part_idx + 1}/{len(parts)})" if len(parts) > 1 else ""
        plt.suptitle(f'Cluster Samples: {class_name}{title_suffix}', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if len(parts) == 1:
            jpg_path = base_path
        else:
            jpg_path = base_path.with_name(base_path.stem + f"_part{part_idx + 1}.jpg")

        plt.savefig(jpg_path, dpi=150, bbox_inches='tight', format='jpeg', pil_kwargs={'quality': 85})
        print(f"  - Saved montage: {jpg_path}")
        plt.close()
        saved_paths.append(jpg_path)

    return saved_paths

def visualize_intra_class(X_2d, cluster_labels, class_name, output_path):
    """Visualize clusters within a single class."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Plot 1: Full scatter plot
    outlier_mask = cluster_labels == -1

    # Plot outliers
    if np.any(outlier_mask):
        ax1.scatter(X_2d[outlier_mask, 0], X_2d[outlier_mask, 1],
                   c='red', marker='x', s=100, alpha=0.6,
                   label='Outliers', linewidths=2)

    # Plot clusters
    unique_clusters = sorted(set(cluster_labels[~outlier_mask]))
    colors = sns.color_palette('husl', n_colors=len(unique_clusters))

    for i, cluster in enumerate(unique_clusters):
        mask = cluster_labels == cluster
        ax1.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=[colors[i]], s=100, alpha=0.7,
                   label=f'Sub-cluster {cluster} (n={np.sum(mask)})')

    ax1.set_title(f'Intra-Class Variation: {class_name}')
    ax1.set_xlabel('UMAP Dimension 1')
    ax1.set_ylabel('UMAP Dimension 2')
    ax1.legend()

    # Plot 2: Cluster size distribution
    cluster_counts = [np.sum(cluster_labels == c) for c in unique_clusters]
    outlier_count = np.sum(outlier_mask)

    labels = [f'C{c}' for c in unique_clusters] + ['Outliers']
    counts = cluster_counts + [outlier_count]
    colors_bar = list(colors) + ['red']

    ax2.bar(labels, counts, color=colors_bar, alpha=0.7)
    ax2.set_title(f'Cluster Size Distribution')
    ax2.set_xlabel('Cluster ID')
    ax2.set_ylabel('Number of Samples')
    ax2.grid(axis='y', alpha=0.3)

    for i, (label, count) in enumerate(zip(labels, counts)):
        ax2.text(i, count + 2, str(count), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  - Saved visualization: {output_path}")
    plt.close()


def generate_centroid_overview(all_class_data, output_dir):
    """
    Build centroid overview plots — one UMAP fit on cluster centroids + sampled outliers.
    Splits into pages of CLASSES_PER_OVERVIEW_PAGE classes each.

    all_class_data: list of (class_name, embeddings, cluster_labels)
    """
    print("\n" + "="*60)
    print("GENERATING CENTROID OVERVIEW PLOTS")
    print("="*60)

    # --- Build small matrix: centroids + sampled outliers ---
    points = []       # embedding vectors
    point_labels = [] # class index
    point_types = []  # 'centroid' or 'outlier'
    point_cluster_ids = []  # cluster id (for centroid label)
    class_names = [d[0] for d in all_class_data]

    for cls_idx, (class_name, embeddings, cluster_labels) in enumerate(all_class_data):
        unique_clusters = sorted(set(cluster_labels))

        for cid in unique_clusters:
            mask = cluster_labels == cid
            cluster_embs = embeddings[mask]

            if cid == -1:
                # Sample up to MAX_OUTLIERS_PER_CLASS outliers
                n = min(MAX_OUTLIERS_PER_CLASS, len(cluster_embs))
                chosen = cluster_embs[np.random.choice(len(cluster_embs), n, replace=False)]
                for e in chosen:
                    points.append(e)
                    point_labels.append(cls_idx)
                    point_types.append('outlier')
                    point_cluster_ids.append(-1)
            else:
                centroid = cluster_embs.mean(axis=0)
                points.append(centroid)
                point_labels.append(cls_idx)
                point_types.append('centroid')
                point_cluster_ids.append(cid)

    if not points:
        print("  No data to plot.")
        return []

    points = np.array(points)
    point_labels = np.array(point_labels)
    point_types = np.array(point_types)
    point_cluster_ids = np.array(point_cluster_ids)

    print(f"  UMAP fit on {len(points)} points ({len(all_class_data)} classes)")

    t0 = time.time()
    reducer = _UMAP(n_components=2, random_state=42, init='random', min_dist=0.1)
    pts_2d = reducer.fit_transform(normalize(points, norm='l2'))
    if GPU_BACKEND:
        pts_2d = np.asarray(pts_2d)
    print(f"  [time] UMAP fit_transform (centroid overview): {time.time()-t0:.2f}s")

    # --- Split classes into pages ---
    n_classes = len(all_class_data)
    class_colors = sns.color_palette('husl', n_colors=n_classes)
    saved_paths = []

    page_starts = list(range(0, n_classes, CLASSES_PER_OVERVIEW_PAGE))
    for page_idx, start in enumerate(page_starts):
        end = min(start + CLASSES_PER_OVERVIEW_PAGE, n_classes)
        page_class_indices = list(range(start, end))

        fig, ax = plt.subplots(figsize=(14, 10))

        for cls_idx in page_class_indices:
            class_name = class_names[cls_idx]
            color = class_colors[cls_idx]

            cls_mask = point_labels == cls_idx

            # Outliers — x marker
            outlier_mask = cls_mask & (point_types == 'outlier')
            if np.any(outlier_mask):
                ax.scatter(pts_2d[outlier_mask, 0], pts_2d[outlier_mask, 1],
                           c=[color], marker='x', s=40, alpha=0.5, linewidths=1)

            # Centroids — circle marker + cluster id label
            centroid_mask = cls_mask & (point_types == 'centroid')
            if np.any(centroid_mask):
                ax.scatter(pts_2d[centroid_mask, 0], pts_2d[centroid_mask, 1],
                           c=[color], marker='o', s=120, alpha=0.9,
                           label=class_name, edgecolors='black', linewidths=0.5)
                for i in np.where(centroid_mask)[0]:
                    ax.annotate(str(point_cluster_ids[i]),
                                (pts_2d[i, 0], pts_2d[i, 1]),
                                fontsize=7, ha='center', va='bottom',
                                color='black', alpha=0.8)

        ax.set_title(f'Cluster Centroid Overview — Classes {start+1}–{end}', fontsize=14, fontweight='bold')
        ax.set_xlabel('UMAP Dimension 1')
        ax.set_ylabel('UMAP Dimension 2')
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8, framealpha=0.7)

        # Legend note
        ax.scatter([], [], c='gray', marker='o', s=80, label='● centroid')
        ax.scatter([], [], c='gray', marker='x', s=40, label='✕ outlier')

        plt.tight_layout()
        out_path = output_dir / f"centroid_overview_{page_idx + 1}.png"
        plt.savefig(out_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path}")
        saved_paths.append(out_path)

    return saved_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="cropped_imgs_by_class")
    parser.add_argument("--eps", type=float, default=0.15, help="DBSCAN eps for cosine distance (try 0.1-0.3)")
    parser.add_argument("--min_samples", type=int, default=3, help="Min samples to form cluster")
    parser.add_argument("--auto_tune", action="store_true", help="Automatically find optimal eps for each class")
    parser.add_argument("--auto_tune_percentile", type=int, default=90, help="Percentile for auto-tune (90=tight, 95=balanced, 98=loose)")
    parser.add_argument("--output_dir", type=str, default="clustering_results")
    parser.add_argument("--max_samples", type=int, default=5, help="Max sample images per cluster to save")
    parser.add_argument("--save_montage", action="store_true", help="Create image montage for each class")
    parser.add_argument("--save_class_scatter", action="store_true", help="Save per-class UMAP scatter plot (slow — runs UMAP per class)")
    parser.add_argument("--umap_min_dist", type=float, default=0.05, help="UMAP min_dist for per-class scatter (only used with --save_class_scatter)")
    parser.add_argument("--save_suffix", type=str, default="embeddings_dinov2.npy", help="Embedding filename to load (must match script 2 output)")
    parser.add_argument("--uniform_eps_threshold", type=float, default=0.10, help="If auto-tuned eps < this, consider class uniform")
    parser.add_argument("--uniform_downsample_target", type=int, default=5000, help="Target samples for uniform classes")
    parser.add_argument("--uniform_min_samples", type=int, default=10000, help="Only downsample if class has more than this")
    parser.add_argument("--eps_method", choices=["knee", "percentile"], default="knee",
                        help="Auto-tune rule: knee of k-distance curve (data-driven outlier rate) or "
                             "--auto_tune_percentile (outlier rate capped near 100-p%%)")
    parser.add_argument("--pca_components", type=int, default=128, help="PCA dims before clustering (0=disabled). Reduces 768d→Nd to fix curse of dimensionality at large N.")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    root = Path(args.root)
    all_class_data = []  # (class_name, embeddings_to_cluster, cluster_labels)
    stats_data = []

    print("="*60)
    print("INTRA-CLASS CLUSTERING ANALYSIS")
    print("="*60)

    # Process each class separately
    for cls_folder in sorted(root.iterdir()):
        if not cls_folder.is_dir():
            continue

        emb_path = cls_folder / args.save_suffix
        if not emb_path.exists():
            print(f"\n⚠️  Skipping {cls_folder.name} - no embeddings found ({args.save_suffix}).")
            continue

        # Load embeddings
        embeddings = np.load(str(emb_path))
        class_name = cls_folder.name

        # Load image mapping file to ensure correct alignment
        base_name = args.save_suffix.replace('.npy', '')
        mapping_path = cls_folder / f"{base_name}_image_list.txt"
        if mapping_path.exists():
            with open(mapping_path, 'r', encoding='utf-8') as f:
                image_names = [line.strip() for line in f.readlines()]
            image_files = [cls_folder / name for name in image_names]
            print(f"\n✓ Using saved image mapping from: {mapping_path.name}")
        else:
            image_files = sorted([f for f in cls_folder.iterdir() if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
            print(f"\n⚠️  No mapping file found - using sorted image list (may be misaligned)")

        if len(image_files) != len(embeddings):
            print(f"❌ ERROR: {class_name} has {len(image_files)} images but {len(embeddings)} embeddings")
            print(f"   CLASS '{class_name}' SKIPPING! Re-run script 2 to fix alignment.")
            continue

        print(f"\nProcessing: {class_name} ({embeddings.shape[0]} samples)")

        if embeddings.shape[0] < args.min_samples:
            print(f"  ⚠️  Skipping {class_name}: only {embeddings.shape[0]} samples (need at least {args.min_samples} for min_samples)")
            continue

        # Check if class is uniform and should be downsampled
        n_samples = len(embeddings)

        # PCA dimensionality reduction before clustering (fixes curse of dimensionality at large N)
        # Applied per-class so PCA captures each class's own variance structure.
        # Original 768d embeddings kept intact for centroid overview UMAP.
        n_components = args.pca_components
        if n_components > 0 and n_samples > n_components:
            t0 = time.time()
            pca = PCA(n_components=n_components, random_state=42)
            embeddings_for_clustering = normalize(
                pca.fit_transform(normalize(embeddings, norm='l2')), norm='l2'
            )
            print(f"  [PCA] 768d → {n_components}d ({pca.explained_variance_ratio_.sum()*100:.1f}% variance retained) in {time.time()-t0:.2f}s")
        else:
            embeddings_for_clustering = embeddings
            if n_components > 0:
                print(f"  [PCA] skipped — n_samples ({n_samples}) <= pca_components ({n_components})")

        # Auto-tune eps if requested
        if args.auto_tune:
            optimal_eps = auto_tune_eps(embeddings_for_clustering, args.min_samples,
                                        percentile=args.auto_tune_percentile, method=args.eps_method)
            print(f"  Auto-tuned eps: {optimal_eps:.4f} (method={args.eps_method})")
            eps_to_use = optimal_eps
        else:
            eps_to_use = args.eps

        is_uniform = (n_samples > args.uniform_min_samples and
                      eps_to_use < args.uniform_eps_threshold)

        if is_uniform:
            print(f"  ⚠️  Uniform class detected (eps={eps_to_use:.4f} < {args.uniform_eps_threshold})")
            print(f"  📊 Downsampling from {n_samples} to {args.uniform_downsample_target} samples (grid-based)")

            stride = max(1, n_samples // args.uniform_downsample_target)
            subsample_indices = np.arange(0, n_samples, stride)[:args.uniform_downsample_target]

            embeddings_to_cluster = embeddings_for_clustering[subsample_indices]
            image_files_to_cluster = [image_files[i] for i in subsample_indices]
            n_samples_to_cluster = len(embeddings_to_cluster)

            print(f"  ✓ Using {n_samples_to_cluster} samples for clustering (stride={stride})")

            # Subsample has 1/stride the density — re-tune eps on what DBSCAN will actually see,
            # otherwise the full-set eps is too tight and inflates noise
            if args.auto_tune:
                eps_to_use = auto_tune_eps(embeddings_to_cluster, args.min_samples,
                                           percentile=args.auto_tune_percentile, method=args.eps_method)
                print(f"  Re-tuned eps on subsample: {eps_to_use:.4f}")
        else:
            embeddings_to_cluster = embeddings_for_clustering
            image_files_to_cluster = image_files
            n_samples_to_cluster = n_samples

        # Cluster
        cluster_labels, n_clusters, n_outliers = cluster_single_class(
            embeddings_to_cluster, class_name, eps_to_use, args.min_samples
        )

        stats_data.append({
            'class': class_name,
            'n_samples': f"{n_samples_to_cluster}/{n_samples}" if is_uniform else n_samples,
            'n_clusters': n_clusters,
            'n_outliers': n_outliers,
            'outlier_rate': f"{n_outliers/n_samples_to_cluster*100:.1f}%",
            'eps_used': eps_to_use,
            'downsampled': 'yes' if is_uniform else 'no'
        })

        # Store for centroid overview — use original (non-PCA) embeddings so all classes share same 768d space
        embeddings_for_overview = embeddings[subsample_indices] if is_uniform else embeddings
        all_class_data.append((class_name, embeddings_for_overview, cluster_labels))

        # Per-class scatter — only if requested (runs UMAP per class, slow)
        if args.save_class_scatter:
            t0 = time.time()
            if n_samples_to_cluster > args.uniform_min_samples:
                print(f"  Large dataset - using optimized UMAP settings")
                reducer = _UMAP(
                    n_components=2, random_state=42, init='random',
                    n_epochs=200, n_neighbors=15,
                    min_dist=args.umap_min_dist, verbose=True
                )
            else:
                reducer = _UMAP(n_components=2, random_state=42, init='random', min_dist=args.umap_min_dist)

            X_2d = reducer.fit_transform(embeddings_to_cluster)
            if GPU_BACKEND:
                X_2d = np.asarray(X_2d)
            print(f"  [time] UMAP fit_transform (per-class scatter): {time.time()-t0:.2f}s")
            output_path = output_dir / f"{class_name}_clusters.png"
            visualize_intra_class(X_2d, cluster_labels, class_name, output_path)

        # Save sample images from each cluster
        save_cluster_samples(image_files_to_cluster, cluster_labels, class_name, output_dir, args.max_samples)

        # Create montage if requested
        if args.save_montage:
            montage_path = output_dir / f"{class_name}_montage.jpg"
            montage_class_name = f"{class_name} (Uniform - analyzed {n_samples_to_cluster}/{n_samples} samples)" if is_uniform else class_name
            create_cluster_montage(image_files_to_cluster, cluster_labels, montage_class_name, montage_path, args.max_samples)

    # Save statistics CSV
    if stats_data:
        df = pd.DataFrame(stats_data)
        print("\nSummary:")
        print(df.to_string(index=False))
        stats_path = output_dir / "cluster_statistics.csv"
        df.to_csv(stats_path, index=False)
        print(f"\nSaved statistics: {stats_path}")

    # Centroid overview — delete stale pages from prior run before writing new ones
    for stale in output_dir.glob("centroid_overview_*.png"):
        stale.unlink()
    if all_class_data:
        generate_centroid_overview(all_class_data, output_dir)

    print("\n" + "="*60)
    print("Analysis complete! Check the output directory for:")
    print("  - Centroid overview plot(s): centroid_overview_N.png")
    print("  - Sample images organized by cluster")
    if args.save_montage:
        print("  - Image montages for visual comparison")
    if args.save_class_scatter:
        print("  - Per-class scatter plots")
    print("  - Statistics CSV file")
    print("="*60)

if __name__ == "__main__":
    main()
