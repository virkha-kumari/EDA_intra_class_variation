# EDA Intra-Class Variation Scripts

Personal pipeline I use to check dataset quality before/after annotation. Runs fully local — no API calls, no cloud, everything stays on your machine (needs internet once to download model weights from HuggingFace).

Three modes depending on where you are in the pipeline:
- **Pre-annotation** — check if your frames are worth annotating (diversity, blur, lighting)
- **Pre-training** — check your annotated dataset for intra-class variation and outliers
- **Post-training** — check if your trained YOLO model detections make sense

---

## Design Decisions

### Embedding strategy: `[CLS || avg_patches]` vs mean pooling

Two different pooling strategies are used intentionally across the pipelines.

**Pre-annotation** uses `[CLS || avg_patches]` → **1536d** vectors:
- Concatenates the CLS token (global scene summary) with the mean of all patch tokens (spatial detail)
- The two halves capture different things — CLS = "what is this scene overall", patches = "where are the distinctive regions"
- Full frames contain background, lighting, and context noise. Separating CLS from patch signals gives HDBSCAN more structure to work with across diverse scenes
- Benchmarked: 1536d `[CLS||avg_patches]` outperforms 768d SigLIP `pooler_output` on scene-level separation

**Pre/post-training** uses `mean(last_hidden_state)` → **768d** vectors:
- Averages all 257 tokens (CLS + 256 patches) into one vector
- Works well for crops because the crop IS the object — no background noise to suppress, no need to separate "scene identity" from "spatial detail"
- Simpler, half the memory, fast enough at crop scale

If you want to upgrade the crop pipeline to 1536d for consistency, change `outputs.last_hidden_state.mean(dim=1)` to explicit `CLS || patch_mean` in `postannotation_scripts/2. save_dinov2_embeddings_per_class.py`. Not done by default because 768d clustering is already effective on tight crops and the extra dims add noise without clear gain.

### Why DINOv2 `[CLS||avg_patches]` for pre-annotation (and not SigLIP or mean pooling)

This went through a few iterations.

**First attempt — DINOv2 mean pooling** (same as pre/post-training crops): didn't work for full frames. Averaging all 257 tokens into one vector is fine when the image is a tight crop of one object — there's not much else there. For full scene frames it erases the spatial variation that separates one activity from another; everything averages out.

**Second attempt — SigLIP**: measurably better. SigLIP's scene-level understanding pulled the clusters apart more cleanly. But it had a cluster purity problem — frames that clearly didn't belong kept appearing inside clusters. The model groups scenes by semantic category ("indoors", "outdoors") but isn't sensitive enough to texture and spatial layout to keep outlier frames out.

**What worked — DINOv2 with `[CLS||avg_patches]`**: back to DINOv2, different pooling. Concatenating the CLS token (global scene identity) with the mean of all patch tokens (spatial detail) gave tighter clusters with better purity. DINOv2's patch tokens are more discriminative at the fine-grained level — trained to be useful on their own, not just supporting a global label.

The `+0.253` silhouette improvement is vs SigLIP's pooled output. The purity improvement is harder to quantify but visible in the output montages.

For implementation details and how benchmarking was performed: `utils/test_embed_models.py`. Extended writeup with the full ablation narrative: [docs/embedding-ablation.md](docs/embedding-ablation.md).

---

### Dimensionality reduction: PaCMAP (pre-annotation) vs UMAP (pre-training)

**Pre-annotation** uses PaCMAP:
- Preserves global structure better than UMAP — distant clusters stay distant, not collapsed into one blob
- Critical for frame diversity analysis: you need to see if "outdoor morning" and "indoor evening" are genuinely separate regions, not artificially pulled together
- PaCMAP's loss balances near-pair attraction, mid-pair repulsion, and far-pair weak attraction — avoids UMAP's tendency to over-compress global distances

**Pre-training** uses UMAP:
- Better for local cluster separation within a single object class — tighter intra-cluster cohesion
- Crops of the same class already share global structure (same object type); what matters is local sub-group boundaries
- UMAP's parametric flexibility (`min_dist`, `n_neighbors`) is more tunable per-class via `--umap_min_dist`

Short version: PaCMAP for "are my scenes diverse?" (global), UMAP for "do my crops split into sub-groups?" (local).

---

## Quick Start

Always run from project root. The launcher shows a menu: option 1 = pre-annotation, option 2 = pre-training. Post-training isn't in the menu — run it directly.

```bat
exec.bat        # Windows
./exec.sh       # Linux/Mac
```

Or run directly:
```bash
python "master_scripts/master_script_dinov2_PreAnn.py"    # pre-annotation
python "master_scripts/master_script_dinov2_PostAnn_PreTrain.py"  # pre-training
python "master_scripts/master_script_dinov2_PostTrain.py"         # post-training (edit config vars at top of script)
```

Pre-annotation and pre-training support two modes: **interactive** (questionary prompts, default) and **headless** (CLI args, for programmatic/service use). Post-training requires editing config variables at the top of the script.

**Headless usage:**
```bash
# PostAnn/PreTrain — minimum required args
python "master_scripts/master_script_dinov2_PostAnn_PreTrain.py" \
    --imgs_path /data/images --label_path /data/labels --classes_txt /data/classes.txt

# PreAnn — minimum required args
python "master_scripts/master_script_dinov2_PreAnn.py" --frames_dir /data/frames
```
In headless mode: output dirs wiped fresh each run, embedding cache disabled, no stdin prompts.

---

## Pre-Annotation: Frame Quality Assessment

**Script**: `master_scripts/master_script_dinov2_PreAnn.py`

Use this when you have raw frames and want to know if they're worth annotating. Catches: redundant frames, motion blur, bad lighting, missing scenarios.

**What it does**:
- DINOv2 [CLS||avg_patches] embeddings (1536d) — adaptive: adds pose features when persons detected
- PCA whitening → PaCMAP dimensionality reduction → HDBSCAN clustering
- Motion blur detection via directional gradient anisotropy
- Generates a 5-page PDF report

**Output**: `preann_results/PreAnnotation_Quality_Report_{FolderName}.pdf`

**Prompts**:
1. Path to frames folder
2. "Change default parameters?" — No = sensible defaults, Yes = exposes everything
3. Clustering params (always asked):
   - `n_components`: PCA dims before HDBSCAN (default: 128)
   - `min_cluster_size`: Minimum frames per activity group (default: 25)
   - `min_samples`: HDBSCAN core point strictness (default: 3)
4. Use embedding cache? (Yes = faster re-runs)

**Key parameters**:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `n_components` | 128 | Higher = finer activity separation |
| `min_cluster_size` | 25 | Lower = more small groups detected |
| `min_samples` | 3 | Lower = more frames included in clusters |
| `anisotropy_threshold` | 3.6 | Motion blur cutoff (2.5 strict, 5.0 lenient) |
| `batch_size` | 64 | DINOv2 batch size |

**PDF structure**:
- Page 1: Executive summary — quality score, frame breakdown, recommendations
- Page 2: Quality metrics dashboard (brightness, blur histograms, lighting distribution)
- Page 3: PaCMAP scatter plot + activity breakdown table
- Pages 4–M: Activity montages (4 columns × up to 5 rows per activity)
- Page M+1: Coverage gap heatmap (activity × lighting)

**Interpreting results**:

Quality score (0–10): average of three sub-scores — sharpness (`max(0, 10 - median_anisotropy/5*10)`), brightness centering (`10 - |median_brightness - 127|/127*10`), contrast (`min(10, median_contrast/50*10)`). 7–10 = good, 5–7 = review needed, <5 = fix before annotating.

Activity count: 1–2 = low diversity (same angle/video), 3–6 = good, 7+ = excellent

> Note: Always validate montages visually. Clusters can form on viewpoint/lighting rather than semantic content.

**Output files**:
```
preann_results/
├── PreAnnotation_Quality_Report_{FolderName}.pdf
└── outliers_{FolderName}/   (optional — prompted at end)
```

---

## Pre-Training: Annotated Dataset Analysis

**Script**: `master_scripts/master_script_dinov2_PostAnn_PreTrain.py`

Use after annotation to understand intra-class variation — are your classes homogeneous or do they have distinct sub-groups? Finds outliers and annotation errors.

**What it does**: Crops bboxes from YOLO annotations → DINOv2 768d embeddings → DBSCAN clustering → PDF report

**Output**: `postann_pretrain_results/clustering_results_{FolderName}/PDF_REPORT_{FolderName}.pdf`

**Prompts**:
1. Images folder, labels folder, classes.txt
2. Output subfolder names (smart defaults from your dataset folder name)
3. "Change default parameters?" gate
4. Clustering flags (always asked): auto-tune eps, per-class UMAP scatter (off by default)
5. k-NN percentile (if auto-tune on) or manual epsilon (if off)
6. Generate PDF?

**Default values** (master script defaults — override the individual script's own defaults):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 64 | DINOv2 batch size |
| `auto_tune` | True | Auto-calculate DBSCAN epsilon per class |
| `auto_tune_percentile` | 90 | k-NN percentile (90=tight, 95=balanced, 98=loose) |
| `epsilon` | 0.15 | Manual eps fallback (only if auto-tune off) |
| `save_class_scatter` | False | Per-class UMAP scatter plots (slow — runs UMAP per class) |
| `min_pts` | 3 | Min points per cluster |
| `umap_min_dist` | 0.05 | UMAP min_dist |
| `max_cluster_samples` | 20 | Max sample images per cluster (overrides Script 3's own default of 5) |
| `pca_components` | 0 | PCA dims before clustering (0=disabled; reduces 768d→Nd before DBSCAN; set 128 for large classes 10K+ samples) |
| `uniform_downsample_target` | 4000 | Downsample target for uniform classes (overrides Script 3's default of 5000) |
| `uniform_min_samples` | 12000 | Trigger downsampling only if class exceeds this (overrides Script 3's default of 10000) |
| `use_embedding_cache` | True | Skip re-generation if embeddings exist |
| `num_workers` | 4 | Crop extraction threads (behind "Change defaults?" gate) |

**PDF prompts** (after basic setup, when PDF generation is on):
- "Group contrastive class pairs in distribution chart?" — if Yes, tick class names from a checkbox list to form named groups (e.g., "fit vs wrong_fit"). A second grouped horizontal bar chart is added **after** the plain annotation chart, showing only the grouped classes. Legend labels are auto-inferred from class name suffixes (falls back to "Variant N").

**PDF structure**:
- Page 1: Dataset overview table → Pipeline config table → Annotation distribution bar chart (always) → Contrastive group chart (only if groups provided) → Per-class clustering summary table (+ annotation issue files detail if any issues found)
- Pages 2–N: Centroid overview plots (`centroid_overview_N.png`)
- Remaining pages: Per-class montages (split at cluster row boundaries, never mid-cluster)
- Last page: Insights & Recommendations — issue-centric layout grouped by severity (🔴 Critical → 🟡 Warning → 🟢 OK → ℹ Info). Each row = one unique issue type; "Classes Affected" column lists all classes that triggered it. Max ~1 page regardless of class count. No class-imbalance row.

**Pipeline steps** (run automatically):
1. `postannotation_scripts/1. ann_txt_files_crop_bbox.py` — crop + validate dataset
2. `postannotation_scripts/2. save_dinov2_embeddings_per_class.py` — generate embeddings
3. `postannotation_scripts/3. clustering_of_classes_embeddings.py` — DBSCAN clustering
4. `postannotation_scripts/4. generate_pdf.py` — PDF report (optional)

---

## Post-Training: Model Detection Analysis

**Script**: `master_scripts/master_script_dinov2_PostTrain.py`

Use after training to check if model detections are sensible. Not in `./exec.bat` — run directly.

```bash
python "master_scripts/master_script_dinov2_PostTrain.py"
```

Same pipeline as pre-training but Step 1 runs YOLO inference on video instead of reading annotation files.

**Configuration variables** (at top of script):
```python
RESULTS_DIR = "posttrain_results"
model_path = "path/to/model.pt"
video_path = "path/to/video.mp4"
classes_space_separated = ["class0", "class1", "class2"]
per_class_num_frames = 1000
conf_thresh = 0.4
frame_stride_per_video = 3        # every 3rd frame = 10fps on 30fps video
```

**Frame stride guide** (30fps video):

| Stride | Effective FPS | Use when |
|--------|---------------|----------|
| 2 | 15 | High variation scenes |
| 3 | 10 | Recommended default |
| 5 | 6 | Slow-moving objects |

---

## Individual Scripts

### postannotation_scripts/1. ann_txt_files_crop_bbox.py

```bash
python "postannotation_scripts/1. ann_txt_files_crop_bbox.py" \
    --imgs_label_path "path/to/LabelledData" \
    --classes 0 1 2 \
    --class_ids_to_names 0 class0 1 class1 2 class2 \
    --output_dir cropped_imgs_by_class
```

Validates dataset before cropping — catches unexpected class IDs, empty labels, imbalance ratio >10x, and duplicate image basenames (same filename in multiple formats, e.g. `foo.png` + `foo.jpg` in the same folder). Output: `cropped_imgs_by_class/{class}/*.jpg`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_workers` | 4 | Crop extraction threads (4=safe NVMe laptop, 8=fast desktop) |

### postannotation_scripts/1. yolo_model_crop_bbox_per_class.py

```bash
python "postannotation_scripts/1. yolo_model_crop_bbox_per_class.py" \
    --model "path/to/model.pt" \
    --video "path/to/video.mp4" \
    --classes class0 class1 class2 \
    --frame_stride 3 \
    --num_frames 100 \
    --conf_thresh 0.4 \
    --output "cropped_images"
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_frames` | 100 | Target samples per class (uniform sampling) |
| `--frame_stride` | 3 | Process every Nth frame |
| `--conf_thresh` | 0.4 | Detection confidence threshold |

### postannotation_scripts/2. save_dinov2_embeddings_per_class.py

```bash
python "postannotation_scripts/2. save_dinov2_embeddings_per_class.py" \
    --root ./cropped_imgs_by_class --batch 32
```

Generates 768d DINOv2 embeddings per class. Saves `embeddings_dinov2.npy` + `embeddings_dinov2_image_list.txt` (image-embedding alignment) per class folder.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch` | 32 | Batch size |
| `--save_suffix` | `embeddings_dinov2.npy` | Output filename |
| `--use_cache` | False | Skip if embeddings already exist |

### postannotation_scripts/3. clustering_of_classes_embeddings.py

```bash
# Auto-tune (recommended)
python "postannotation_scripts/3. clustering_of_classes_embeddings.py" \
    --root ./cropped_imgs_by_class --auto_tune --min_samples 3 --save_montage

# Manual epsilon
python "postannotation_scripts/3. clustering_of_classes_embeddings.py" \
    --root ./cropped_imgs_by_class --eps 0.15 --min_samples 3
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--eps` | 0.15 | DBSCAN epsilon (0.1 strict, 0.3 lenient) |
| `--min_samples` | 3 | Min points to form cluster |
| `--auto_tune` | False | k-NN percentile auto-tune per class |
| `--auto_tune_percentile` | 90 | 90=tight, 95=balanced, 98=loose |
| `--umap_min_dist` | 0.05 | UMAP min_dist (0.0=tight, 0.1=loose) |
| `--max_samples` | 5 | Sample images per cluster (outliers: all saved) |
| `--save_class_scatter` | False | Per-class UMAP scatter plots (slow — runs UMAP per class) |
| `--save_montage` | False | Generate image grid montages |
| `--pca_components` | 128 | PCA dims before DBSCAN (0=disabled; reduces 768d→Nd, improves large-class clustering) |
| `--uniform_eps_threshold` | 0.10 | Below this eps = class treated as uniform |
| `--uniform_downsample_target` | 5000 | Downsample target for uniform classes |
| `--uniform_min_samples` | 10000 | Only downsample if class has more than this |

**GPU backend**: Script 3 auto-detects cuML at startup. If cuML is installed and a CUDA device is available, uses GPU-accelerated DBSCAN + kNN + UMAP. Falls back to sklearn/umap-learn (CPU) silently otherwise — no config needed.

**Installing cuML (WSL2 / Linux server):**

RAPIDS doesn't ship pip wheels. Requires mamba (conda's fast solver) — plain conda hangs forever on RAPIDS due to SAT solver limitations (500+ packages, 3 channels).

**Option A — miniforge (recommended for personal machines, no root needed):**
```bash
# Install miniforge to $HOME — ships mamba built-in, no root needed
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3
$HOME/miniforge3/bin/conda init bash
# reopen shell — mamba now available directly
```

**Option B — install mamba into existing anaconda/conda (company machines where you can't change base install):**
```bash
conda install -c conda-forge mamba -y   # one-time, into base env
```

**Create env + install (same for both options):**
```bash
# Create fresh env with cuML (CUDA 12.8)
mamba create -n new_eda_tool -c rapidsai -c conda-forge -c nvidia \
    cuml=25.* python=3.11 cuda-version=12.8 -y

# Activate
eval "$(mamba shell hook --shell bash)"   # if shell not initialized yet
mamba activate new_eda_tool              # or: conda activate new_eda_tool

# Install PyTorch — use pinned version, pip path only (conda pytorch channel has no cuda=12.8)
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# Install pipeline deps
pip install -r requirements.txt
```

**Verify:**
```bash
python -c "import cuml; from cuml.cluster import DBSCAN; print(cuml.__version__)"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

**Key lessons learned:**
- `mamba create` not `conda create` — conda solver deadlocks on RAPIDS
- miniforge = mamba built-in; plain anaconda = need `conda install mamba` first
- PyTorch must be pinned (eg., `torch==2.7.1`) — unpinned pulls incompatible nvidia-* pip packages that conflict with conda's CUDA libs — remember to ensure that torch version is compatible with cuda
- Do NOT `pip install nvidia-cudnn-cu12` separately — torch's pinned install handles it; wrong version = `libcudnn.so` import error
- conda pytorch channel has no `pytorch-cuda=12.8` yet — pip is the only path

Output:
```
clustering_results/
├── {class}_clusters.png   (only if --save_class_scatter)
├── {class}_montage.jpg
├── {class}_samples/
│   ├── cluster_0/
│   └── outliers/
├── centroid_overview_1.png
└── cluster_statistics.csv
```

Interpreting cluster counts: 1 = homogeneous class, 2–4 = normal variation, 5+ = high diversity (check montages before splitting classes — clusters can reflect viewpoint/lighting, not always semantic sub-classes).

Outlier rates: <5% = good, 5–10% = moderate variation, >20% = check data quality or loosen eps.

### postannotation_scripts/4. generate_pdf.py

Usually invoked automatically by the pre-training master script. Standalone use:

```bash
python "postannotation_scripts/4. generate_pdf.py" \
    --temp_txt_file postann_pretrain_results/cropped_imgs_by_class_MyDataset/temp_ann_file.txt \
    --clustering_dir postann_pretrain_results/clustering_results_MyDataset \
    --pdf_name "postann_pretrain_results/clustering_results_MyDataset/PDF_REPORT_MyDataset" \
    --imgs_path path/to/images \
    --label_path path/to/labels \
    --classes_txt path/to/classes.txt \
    --auto_tune --auto_tune_percentile 95
```

| `--contrastive_groups_json` | None | JSON string `{"Group Name": ["class_a", "class_b"]}` — adds a second grouped horizontal bar chart after the plain annotation chart, showing only grouped classes. Legend labels auto-inferred from class name suffixes. |

PDF structure: Page 1 = dataset overview table (includes Duplicate Image Basenames row) → pipeline config table → annotation distribution bar chart (always) → contrastive group chart (only if `--contrastive_groups_json` provided, grouped classes only) → per-class clustering summary table (+ annotation issue files detail if any issues found). Pages 2–N = centroid overview plots (`centroid_overview_N.png`). Remaining pages = per-class montages (split at cluster row boundaries, never mid-cluster). Last page = Insights & Recommendations — issue-centric, grouped by severity (🔴 Critical → 🟡 Warning → 🟢 OK → ℹ Info); each row is one issue type listing all affected classes. No class-imbalance row.

### master_scripts/1. interactive_cluster_viewer.py (optional)

Interactive Plotly HTML explorer. Run after Script 2, independent of Script 3.

```bash
python "master_scripts/1. interactive_cluster_viewer.py" \
    --root postann_pretrain_results/cropped_imgs_by_class
```

---

## Decision Guide

**Pre-annotation output → what to do:**

| Result | Action |
|--------|--------|
| Quality score < 5 | Fix data before annotating — re-shoot or filter frames |
| >30% blurry frames | Drop them or re-extract with higher shutter speed |
| >40% dark frames | Add well-lit frames or adjust camera settings |
| 1–2 activities | Low diversity — capture from more angles/scenarios |
| Activity montage shows mixed unrelated content | Decrease `min_cluster_size` (allow smaller tighter groups) or increase `n_components` (finer PCA detail) and re-run |

**Pre/post-training clustering output → what to do:**

| Result | Action |
|--------|--------|
| 0 clusters (all outliers) | eps too strict — increase `--eps` or use `--auto_tune` |
| 1 cluster | Class is visually homogeneous — nothing to split |
| 5+ clusters | Check montages first — if clusters are semantic, consider sub-classes; if viewpoint/lighting artifacts, ignore |
| >20% outliers | Check outlier folder — annotation errors or genuinely rare appearances |
| Class imbalance >10x flagged | Add more data for underrepresented class before training |

---

## Pipeline Summary

| Script | When to use |
|--------|-------------|
| `master_scripts/master_script_dinov2_PreAnn.py` | Before annotation — frame quality check |
| `master_scripts/master_script_dinov2_PostAnn_PreTrain.py` | After annotation — intra-class variation |
| `master_scripts/master_script_dinov2_PostTrain.py` | After training — model detection check |
| `postannotation_scripts/1. ann_txt_files_crop_bbox.py` | Manual: crop from YOLO annotations |
| `postannotation_scripts/1. yolo_model_crop_bbox_per_class.py` | Manual: crop from model detections |
| `postannotation_scripts/2. save_dinov2_embeddings_per_class.py` | Manual: generate embeddings |
| `postannotation_scripts/3. clustering_of_classes_embeddings.py` | Manual: DBSCAN clustering |
| `postannotation_scripts/4. generate_pdf.py` | Manual: generate PDF report |

---

## Dependencies

```bash
pip install -r requirements.txt
```

Core: PyTorch >= 2.1 with CUDA, transformers (DINOv2), ultralytics (YOLO), scikit-learn, pacmap, hdbscan, opencv-python, pillow, numpy, pandas, matplotlib, seaborn, reportlab, tqdm, questionary

---

## References

- DINOv2 clustering: https://medium.com/@EnginDenizTangut/%EF%B8%8F-image-clustering-with-dinov2-and-hdbscan-a-hands-on-guide-35c6e29036f2
- DBSCAN: https://medium.com/@sachinsoni600517/clustering-like-a-pro-a-beginners-guide-to-dbscan-6c8274c362c4
- Post-training pipeline inspiration: https://medium.com/@albertferrevidal/facings-product-identifier-using-yolov8-and-image-embeddings-d3ca34463022
