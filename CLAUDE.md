# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Intra-class variation analysis pipeline** for object detection datasets. Analyzes class heterogeneity using DINOv2 embeddings and clustering to discover sub-groups and outliers.

**Three pipeline modes:**
- **Pre-annotation**: Analyzes raw frames BEFORE annotation to assess quality/diversity (uses DINOv2 + PaCMAP)
- **Pre-training**: Analyzes annotated dataset AFTER annotation, before model training (uses DINOv2 embeddings + YOLO annotation .txt files)
- **Post-training**: Analyzes trained YOLOv8 model detections on video (uses DINOv2 embeddings, validates model quality)

## Core Principles
- **Keep it simple** - Exploratory EDA, not production code
- **No overcomplications** - Engineers need cluster patterns and outliers
- **Consistency first** - README must match actual script behavior exactly

## Scripts

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `master_scripts/master_script_dinov2_PreAnn.py` | **Pre-annotation**: Frame quality assessment with PDF report (uses DINOv2 + PaCMAP) | BEFORE annotation |
| `master_scripts/master_script_dinov2_PostAnn_PreTrain.py` | Automated pre-training pipeline (uses DINOv2) | AFTER annotation |
| `master_scripts/master_script_dinov2_PostTrain.py` | ⚠️ DEPRECATED, not maintained — post-training pipeline, reference only | — |
| `postannotation_scripts/1. ann_txt_files_crop_bbox.py` | Pre-training: Extract crops from YOLO annotations | Part of pretrain pipeline |
| `postannotation_scripts/1. yolo_model_crop_bbox_per_class.py` | Post-training: Extract crops from model detections | Part of posttrain pipeline |
| `postannotation_scripts/2. save_dinov2_embeddings_per_class.py` | Generate DINOv2 embeddings | Part of both pipelines |
| `postannotation_scripts/3. clustering_of_classes_embeddings.py` | DBSCAN clustering analysis | Part of both pipelines |
| `postannotation_scripts/4. generate_pdf.py` | Generate PDF reports combining dataset stats + clustering results | Part of both pipelines |
| `postannotation_scripts/5. usb_missed_labels.py` | USB (Use-Seen-Before): flag images with missed labels for a class | Optional Step 5 of pretrain pipeline |
| `utils/preann_pdf_generate.py` | PDF/montage generation for pre-annotation pipeline | Imported by DINOv2 PreAnn master script |
| `master_scripts/1. interactive_cluster_viewer.py` | Optional: Interactive Plotly HTML viewer | Optional addon |

## Running the Pipeline

**IMPORTANT**: Always run from **project root** directory (where `exec.bat` lives). Running from `master_scripts/` causes path errors.

### Automated (Recommended)
```bash
# Use the launcher — Pre-Annotation and Pre-Training available via menu:
./exec.bat

# Or run directly from project root:
python "master_scripts/master_script_dinov2_PreAnn.py"    # Pre-annotation
python "master_scripts/master_script_dinov2_PostAnn_PreTrain.py"  # Pre-training
python "master_scripts/master_script_dinov2_PostTrain.py"         # Post-training (not in exec.bat)
```

The pre-training master script asks a run mode right after `classes.txt`: **Full EDA** / **Full EDA + USB** / **USB only** (crops + embeddings + USB for selected classes, no clustering/PDF).

Both master scripts support two modes:
- **Interactive** (default): questionary prompts — path autocomplete, Yes/No confirms, "Change default parameters?" gate
- **Headless / non-interactive**: pass `--imgs_path`, `--label_path`, `--classes_txt` (PostAnn) or `--frames_dir` (PreAnn) without `--interactive` — all args resolve from CLI, no stdin. Used for programmatic/service invocation.

```bash
# Headless PostAnn example
python "master_scripts/master_script_dinov2_PostAnn_PreTrain.py" \
    --imgs_path /data/images --label_path /data/labels --classes_txt /data/classes.txt

# Headless PreAnn example
python "master_scripts/master_script_dinov2_PreAnn.py" \
    --frames_dir /data/frames
```

Headless behaviour: both dirs/outputs wiped fresh at start; crops deleted at end (PostAnn); embedding cache force-disabled (PreAnn); `input()` prompts skipped (RAM warning logs and continues); logs written to `eda_run.log` / `preann_run.log` co-located with results.

### Manual Step-by-Step

**Pre-annotation workflow** (raw frames → quality PDF):
```bash
python "master_scripts/master_script_dinov2_PreAnn.py"
```

**Pre-training workflow** (annotated images → clustering):
```bash
# Step 1: Pre-training - crop from annotations
python "postannotation_scripts/1. ann_txt_files_crop_bbox.py" --imgs_label_path "path/to/LabelledData" --classes 0 1 2 --class_ids_to_names 0 board 1 screw 2 holder --output_dir postann_pretrain_results/cropped_imgs_by_class

# Step 2: Generate DINOv2 embeddings
python "postannotation_scripts/2. save_dinov2_embeddings_per_class.py" --root ./postann_pretrain_results/cropped_imgs_by_class --batch 32

# Step 3: DBSCAN clustering analysis
python "postannotation_scripts/3. clustering_of_classes_embeddings.py" --root ./postann_pretrain_results/cropped_imgs_by_class --auto_tune --min_samples 3 --save_montage
```

**Post-training workflow** (video + model → clustering):
```bash
# Step 1: Post-training - crop from video using model
python "postannotation_scripts/1. yolo_model_crop_bbox_per_class.py" --model model.pt --video video.mp4 --classes board screw holder --num_frames 100 --frame_stride 3 --output posttrain_results/cropped_images

# Steps 2-3: Same as pre-training (use --root ./posttrain_results/cropped_images)
```

## Architecture

### Pipeline Flow

**Pre-annotation workflow**:
```
Input: Raw frames (.png, .jpg, .jpeg)
    |
    v
[master_scripts/master_script_dinov2_PreAnn.py]
    - Person detection (YOLOv11-pose)
    - DINOv2 [CLS||avg_patches] embeddings (1536d, adaptive: scene weight + pose weight when persons detected)
    - Per-block PCA whitening + weights → HDBSCAN clustering (PaCMAP = 2-D plot only)
    - Quality metrics (brightness, contrast, motion blur via gradient anisotropy)
    - Lighting categorization
    - Visual montages per activity
    |
    v
Output: preann_results/PreAnnotation_Quality_Report_{FolderName}.pdf (5 pages)
    - Executive summary with quality score (includes embedding model + scene/pose weights + outlier count)
    - Quality metrics dashboard
    - Activity diversity analysis with PaCMAP
    - Visual montages (N samples per activity)
    - Coverage gap heatmap
    - Optional: preann_results/outliers_{FolderName}/ (user prompted at end of run)
```

**Pre-training/Post-training workflow**:

Each master script defines its own `RESULTS_DIR` (`postann_pretrain_results/` for pre-training, `posttrain_results/` for post-training). All outputs go under that directory. Below uses `<results>/` as placeholder:

```
Input (annotated images OR video+model)
    |
    v
[Script 1] Crop bounding boxes per class --> <results>/cropped_imgs_by_class/<class>/*.jpg
                                             <results>/temp_ann_file.txt (statistics, pre-training only)
    |
    v
[Script 2] DINOv2 embeddings --> <results>/cropped_imgs_by_class/<class>/embeddings_dinov2.npy
                                 <results>/cropped_imgs_by_class/<class>/embeddings_dinov2_image_list.txt
    |
    v
[Script 3] DBSCAN clustering --> <results>/clustering_results/<class>_clusters.png
                                 <results>/clustering_results/<class>_montage.jpg
                                 <results>/clustering_results/<class>_samples/cluster_N/
                                 <results>/clustering_results/cluster_statistics.csv
    |
    v (optional)
[Script 4] PDF report generator --> <results>/clustering_results/PDF_REPORT_<name>.pdf (combines Script 1 + 3 results)
[Script 5] USB missed labels --> <results>/clustering_results/missing_labels/<class>/<image>.jpg
[Interactive viewer] --> <results>/clustering_results/<class>_interactive.html
```

### Key Data Structures
- **Embeddings**:
  - Pre-annotation: 1536-dimensional vectors from `facebook/dinov2-base` (CLS + avg_patches concatenated), with PCA whitening before HDBSCAN (PaCMAP only for the 2-D plot)
  - Pre/Post-training: 768-dimensional vectors from `facebook/dinov2-base`
- **Image mapping**: `embeddings_dinov2_image_list.txt` maintains alignment between embeddings and source images
- **YOLO format**: `class_id x_center y_center width height` (normalized coordinates)

### Script Interdependencies
- Script 1a/1b → Creates class folders with cropped images + `temp_ann_file.txt` statistics
- Script 2 → Reads class folders, creates `.npy` + `_image_list.txt`
- Script 3 → Reads embeddings + mapping file, creates visualizations + `cluster_statistics.csv`
- Script 4 → Reads `temp_ann_file.txt` (Script 1) + clustering results (Script 3), generates PDF
- Script 5 (USB) → Reads Script 2's `.npy` embeddings + original images/labels; imports Script 2's `compute_embeddings_for_paths` and Script 3's `auto_tune_eps`/`cluster_single_class` via `importlib` (filenames contain spaces). Independent of Script 3's on-disk output.
- Interactive viewer → (Optional) Reads embeddings, creates interactive HTML (independent of Scripts 3 & 4)

**Critical**:
- If Script 2 changes output format, Script 3 must be updated to match!
- Script 4 depends on Script 1's txt output and Script 3's montage PNG files

### Master Script Features
- Interactive questionary prompts (path autocomplete, Yes/No confirms, "Change defaults?" gate)
- Cancel handling: Ctrl+C exits cleanly at any prompt (handles prompt_toolkit RuntimeError on Windows)
- Memory management: GPU cache clearing, garbage collection between steps
- Cooling breaks: 10-second pauses to prevent laptop GPU thermal throttling
- Resource monitoring: Warns if RAM >90% or GPU >85%
- Error handling: Cleanup on failure or keyboard interrupt
- Launcher: `exec.bat` provides menu to select which pipeline to run

## Key Parameters

### Pre-Annotation Script (master_script_dinov2_PreAnn.py)
All parameters configured via interactive prompts. Clustering params always asked; niche params only shown if "Change default parameters?" = Yes.

**Always prompted (clustering):**
- `min_cluster_size`: Smallest frame group that counts as activity (default 25)
- `min_samples`: How strict frame must match neighbors to join cluster (default 3, lower=more frames included)
- `use_embedding_cache`: Cache embeddings to disk for faster re-runs (default True)

**Behind "Change defaults?" gate:**
- `batch_size`: DINOv2 inference batch size (default 64)
- `anisotropy_threshold`: Motion blur detection threshold (2.5 strict, 3.6 balanced, 5.0 lenient)
- `cache_blurry_num_samples`: Blurry sample images in PDF (default 24)
- `grid_cols_activities`/`grid_cols_blurry`: Grid columns for montages (default 4)

### Clustering (Script 3)
- `--eps`: DBSCAN epsilon (0.1 strict, 0.15 balanced, 0.2-0.3 lenient)
- `--min_samples`: Minimum points to form cluster (2-3 typical)
- `--auto_tune`: Recommended - eps per class from the k-distance curve
- `--eps_method`: `knee` (default, data-driven outlier rate) or `percentile` (outlier rate capped near 100-p% by construction)
- `--auto_tune_percentile`: only with `--eps_method percentile` (default: 90)
- `--umap_min_dist`: UMAP min_dist parameter (default: 0.05; 0.0=tight, 0.1=loose)
- `--save_suffix`: Embedding filename to load (default: `embeddings_dinov2.npy`, must match Script 2 output)
- `--max_samples`: Sample images per cluster (default: 5, outliers all saved) — master script overrides to 20
- `--pca_components`: PCA dims before DBSCAN (default: 128; 0=disabled). Reduces 768d→Nd before clustering — improves DBSCAN on large classes (curse of dimensionality). Applied per-class. — master script default is 0 (disabled); set 128 manually for large classes (10K+ samples)
- `--uniform_eps_threshold`: If auto-tuned eps < this value, class considered uniform (default: 0.10)
- `--uniform_downsample_target`: Target sample count when downsampling uniform classes (default: 5000) — master script overrides to 4000
- `--uniform_min_samples`: Only downsample if class has more than this many samples (default: 10000) — master script overrides to 12000

### USB Missed-Label Detection (Script 5)
- `--usb_classes` / `--sim_thresholds`: class names + one cosine-sim threshold each (default 0.8; master prompts per class)
- `--n_farthest`: farthest-from-centroid points per cluster added to reference pool (default 4)
- `--region_iou`: IoU for BOTH region dedup and same-image GT suppression (default 0.8)
- `--suppress_overlap`: also suppress a region when a same-class GT box covers this fraction of the smaller box (default 0.5)
- Master script args: `--usb_classes`, `--usb_sim_thresholds` (single value broadcasts to all classes), `--usb_n_farthest`, `--usb_only`

### Video Processing (Script 1 post-training)
- `--frame_stride`: Process every Nth frame (3 = 10fps for 30fps video)
- `--num_frames`: Target samples per class (uniform sampling)
- `--conf_thresh`: Detection confidence threshold (default 0.4)

## Known Quirks

### Pre-Annotation Script
1. **Embedding Model**: Uses DINOv2 (`facebook/dinov2-base`) with CLS token + avg_patches concatenated = 1536d
2. **Activity Clustering**: scene block standardize → PCA whiten, pose block standardize, blocks weighted by variance share → L2 → HDBSCAN. PaCMAP is visualisation only.
3. **Adaptive Embeddings**:
   - Frames WITH persons: scene (1536d) + pose features, weighted and concatenated
   - Frames WITHOUT persons: scene embeddings (1536d) + zero-padded pose features
   - Weights applied to CONCATENATED dimensions (not blended scalars)
4. **Visual Montages**: Creates configurable grid (default 4 columns) with variable sample counts per activity
5. **Lighting Thresholds**: Fixed thresholds (Dark <80, Medium 80-175, Bright >175) based on grayscale mean
6. **Quality Score**: Computed from median brightness/anisotropy/contrast, scale 0-10 (lower anisotropy = sharper)
7. **Motion Blur Detection**: Uses directional gradient anisotropy (Sobel X/Y energy ratio) instead of Laplacian variance
8. **Person Detection**: Uses YOLOv11-pose (conf=0.3 default) to extract pose features when persons visible

### Pre-Training/Post-Training Scripts
1. **Image-Embedding Alignment**: Script 2 creates mapping file with name derived from `--save_suffix` parameter (e.g., `embeddings_dinov2.npy` → `embeddings_dinov2_image_list.txt`). Script 3 must receive same `--save_suffix` value or alignment will fail. Mapping file ensures correct correspondence even when corrupted images are skipped.
2. **Sorted Order**: Both scripts use `sorted()` on file paths to maintain consistent ordering
3. **CSV Saving**: Script 3 has CSV saving enabled (not commented out)
4. **Outlier Sampling**: Script 3 saves ALL outlier images (not sampled), while regular clusters sampled up to `--max_samples`
5. **Montage rendering**: Script 3 uses matplotlib for montages (layout: `n_cols × n_rows` grid). Images pre-decoded and resized via PIL before passing to `imshow` for speed. Layout and visual output identical to original.
6. **PDF Image Splitting**: Script 4 splits tall montages at cluster row boundaries (never mid-cluster) by reading `cluster_statistics.csv` to determine row count
7. **PDF Image Compression**: Script 4 converts all PNG montages to JPEG quality 75 for ~80% file size reduction (configurable via `--pdf_quality`)
8. **PIL Decompression Bomb**: Script 4 disables PIL's `MAX_IMAGE_PIXELS` limit to handle large montages (25+ clusters)
9. **PDF Page 1 layout**: Dataset overview table → Pipeline config table → Annotation distribution bar chart (always) → Contrastive group chart (only if `--contrastive_groups_json` provided, shown after plain chart) → Per-class clustering summary table. Annotation issue files detail section rendered if any issues found.
10. **PDF Page 2+ — centroid overview**: Pages 2+ show `centroid_overview_N.png` plots (one per page). `--cross_class` flag removed; replaced by `--save_class_scatter` for per-class UMAP scatter (off by default).
11. **PDF args from master script**: Script 4 accepts `--imgs_path`, `--label_path`, `--classes_txt`, `--auto_tune`, `--auto_tune_percentile`, `--epsilon`, `--pdf_quality`, `--contrastive_groups_json` — all passed automatically by pre-training master script.
12. **Reuse crops/embeddings**: Master script pre-flight detects existing `.npy` embeddings in cropped folder — offers to skip Steps 1 & 2 and jump straight to clustering. Folder is `shutil.rmtree`-deleted before overwrite (not warned).
13. **Contrastive distribution chart**: When `--contrastive_groups_json` provided (JSON `{group_name: [class_a, class_b]}`), Script 4 renders second grouped horizontal bar chart **after** plain annotation chart (plain chart always shown). Only grouped classes appear — ungrouped not appended. One row per group, sub-bars per variant, palette `['#3a7fc1', '#d95f49', ...]`. Legend labels derived from last `_`-token of class name at each variant position (if all groups agree on token); falls back to `Variant N`. Master script prompts via `questionary.checkbox` — user ticks class names to form groups, auto-names as `"A vs B"` (editable). Backward compatible — no arg = plain chart only.
14. **PDF Insights & Recommendations page**: Script 4 appends final page after montages — **issue-centric layout** grouped by severity (🔴 Critical → 🟡 Warning → 🟢 OK → ℹ Info). Each row = one unique issue type; "Classes Affected" column lists all classes that triggered it. Max ~1 page regardless of class count. 13 rules evaluated: outlier rate, cluster count, all-noise, tight-core+noise, insufficient data, low sample count, systematic noise, eps spread, over-fragmentation, over-clustering on small class, homogeneous OK, zero outliers, large class downsampled. No class-imbalance row. `n_samples` parsed as both plain int and `"clustered/total"` string.
15. **Duplicate image basename detection**: Script 1 detects per-split basenames with multiple image formats (e.g. `foo.png` + `foo.jpg` in same folder). Count written to temp txt under `DUPLICATE IMAGE BASENAMES` section. Script 4 parses it and shows row in Dataset Overview table + file detail in Annotation Issue Files block.
16. **USB (Script 5) reuses Script 2/3 code, not their on-disk output**: Script 3 never persists per-sample cluster labels (only class-level `cluster_statistics.csv`), and for "uniform" downsampled classes its in-memory labels aren't even row-aligned with the full `.npy`. So USB re-runs DBSCAN in-process on Script 2's cached embeddings (cheap — pure array math, no GPU, no image I/O) and must NOT call Script 3's `save_cluster_samples`/`create_cluster_montage` (montage saving is the slow part on 50K+ sample classes; clustering itself is fast).
17. **USB region assumption**: `seen_<cls>_regions` are fixed pixel locations reused across every image — valid only for fixed-camera (CCTV) datasets. Candidates are cropped with Script 1's exact crop rule (no padding) so they match the reference crops; regions are suppressed by IoU >= region_iou OR containment >= suppress_overlap with same-class GT.
18. **USB target walk is unavoidable**: no script persists a full dataset image/label manifest (Script 1 holds per-split filename sets in memory, writes only capped/aggregate counts to `temp_ann_file.txt`), so USB does its own `_detect_splits` + `os.listdir` walk. Deliberate decision — not worth a Script 1 change.
19. **USB-only mode**: master script Step 0 asks run mode (Full EDA / Full EDA + USB / USB only). USB-only crops+embeds ONLY the selected classes (`class_ids` filtered), skips Step 3 clustering and PDF entirely. Headless equivalent: `--usb_only` (requires `--usb_classes`). Clustering params (`auto_tune`, `epsilon`, `min_pts`, `auto_tune_percentile`) are still prompted/passed in USB-only mode because USB's own reference-pool DBSCAN uses them.

## File Naming Patterns

### Pre-Annotation Script
- Output: `preann_results/PreAnnotation_Quality_Report_{FolderName}.pdf`
- Outliers: `preann_results/outliers_{FolderName}/` (user prompted at end of run; warns on overwrite)
- Temp files: `temp_preannotation_charts/` (auto-cleaned after PDF generation)
- Embedding cache files: `temp_multiview_emb_indices.npy` and `temp_multiview_emb.npy` in output_dir/ (DINOv2 embeddings, delete before next run if inputs changed!)

### Pre-Training/Post-Training Scripts (under `postann_pretrain_results/` or `posttrain_results/` respectively)
- Script 1a outputs: `{basename}_crop_{idx}.jpg` (JPEG quality 95) + `temp_ann_file.txt`
- Script 1b outputs: `frame_{frame_idx:06d}.png`
- Script 2 outputs: `{save_suffix}.npy` + `{save_suffix_without_.npy}_image_list.txt`
  - Default: `embeddings_dinov2.npy` + `embeddings_dinov2_image_list.txt`
  - Custom: `custom_emb.npy` + `custom_emb_image_list.txt`
- Script 3 outputs: `{class}_clusters.png` (if `--save_class_scatter`), `{class}_montage.jpg` (if `--save_montage`), `cluster_statistics.csv`, `centroid_overview_N.png`
- Script 4 outputs: `{pdf_name}.pdf` + `temp_pdf_charts/` directory with temp JPEG chunks (auto-cleaned)
- Script 5 (USB) outputs: `missing_labels/<class>/{split_}{image_name}.jpg` (full frame, flagged regions red + sim score, existing GT boxes green) inside the clustering results dir. Candidate crops staged in `missing_labels/_temp_crops/<class>/` and deleted after embedding.

### Pre-Training Master Script Output Naming (folder-name-derived)
Pre-training master script derives all output names from images folder name (`{FolderName}`) at prompt time:
- Cropped images: `postann_pretrain_results/cropped_imgs_by_class_{FolderName}/`
- Clustering results: `postann_pretrain_results/clustering_results_{FolderName}/`
- PDF report: `postann_pretrain_results/clustering_results_{FolderName}/PDF_REPORT_{FolderName}.pdf`
Always prompted with derived default shown — user can accept or override.

## Dependencies

- PyTorch >= 2.1 with CUDA
- ultralytics (YOLOv8/YOLOv11)
- transformers (DINOv2 for all pipelines)
- scikit-learn (DBSCAN, PCA)
- pacmap (pre-annotation 2-D visualisation)
- hdbscan (pre-annotation clustering)
- opencv-python, pillow, numpy, pandas, matplotlib, seaborn, psutil, tqdm
- reportlab (for PDF generation in Script 4)

---

## Guidelines for Claude Code

### Core Principles
- **Simplicity first** — exploratory EDA, not production code. Minimal changes, minimal impact.
- **No laziness** — find root causes, not workarounds. Verify assumptions against actual code before calling something bug.
- **Do only what's asked** — don't add features, refactor surrounding code, or introduce abstractions beyond task.

### Request Types
- **"Review" / "Check" / "Analyze"** — report only. List issues with file:line refs. No code changes until told.
- **"Fix" / "Implement" / "Change"** — make change. Don't ask permission on obvious fixes.
- **Ambiguous** — make call, state assumption, proceed.

### Writing Code
- Reuse existing patterns and utilities before writing new logic.
- Keep changes minimal — don't clean up surrounding code unless asked.
- No block comments, no excessive docstrings, no print statements (use logger if present).
- Memory matters — GPU is RTX 4060 8GB VRAM, 16GB RAM. Load-process-delete for large datasets (5K+ images). Never hold all images in RAM.
- Before hardcoding batch sizes or thread counts — ask about hardware/scale.
- Imports at top of file.

### Docs & Changelog
- **"Update docs"** = go through ALL THREE: `README.md`, `CLAUDE.md`, `docs/changes.md` + `docs/plans.md`. Not one.
- Every code change: update README section for that script + CLAUDE.md Known Quirks if behavior changed.
- When user says "ready to commit": update `docs/changes.md` (add bullet), remove completed items from `docs/plans.md`. Never leave ✅ DONE items in plans.md.
- Only update `README.md` and `CLAUDE.md` for code/behavior changes. New `.md` files only in `docs/` for standalone writeups (design decisions, ablations) — must be linked from README.

### Consistency
- Script names exact: e.g. `postannotation_scripts/1. ann_txt_files_crop_bbox.py`
- Argparse defaults in scripts must match parameter tables in README and CLAUDE.md.
- Output paths in docs must match actual `os.makedirs()` / file writes in code.

### Prohibited
- Complex logging frameworks, database persistence, web dashboards, async processing (unless asked)
- New documentation files outside `docs/` — integrate into existing README.md / CLAUDE.md only
- **Never delete output/results dirs** — only intermediate dirs (e.g. `cropped_bbox_dir`). Output dirs (`cluster_dir`, `preann_results/`) are the deliverable.

## File Paths
- Main directory: `<project_root>/`
- Master scripts: `<project_root>/master_scripts/`
- Core scripts (1, 2, 3) for Post-annotation: `<project_root>/postannotation_scripts/`
- Utility modules: `<project_root>/utils/` (e.g., `preann_pdf_generate.py`)
- README: `<project_root>/README.md`
- Docs (writeups, ablations, changelog): `<project_root>/docs/` (`changes.md`, `plans.md`, `embedding-ablation.md`, `usb-missed-labels-ablation.md`)

## DINOv2 preprocessing + cache invariants

- All scoring embeddings (Script 2, USB, PreAnn) resize the whole crop/frame to 224x224 with **no center crop** (`DINO_PREPROC`). The processor default (short edge 256 → center-crop 224) drops box ends / frame borders.
- Script 2 writes `<suffix>_meta.json` with `PREPROC_TAG`; `--use_cache`, USB, and the master "reuse" prompt only accept embeddings with the current tag. Bump the tag whenever preprocessing or pooling changes.
- PreAnn cache is keyed by a fingerprint (frame paths + sizes + mtimes + preprocessing tag), not by dimension.
- PreAnn scene/pose weights are applied per block after standardization/whitening (share of total variance), not before.
