# EDA for Object-Detection Datasets: Diversity, Intra-Class Variation and Missed Labels

**Abstract.** Detection datasets fail silently: redundant or one-sided frames, classes that are really several visual sub-types, and objects that were never labelled. This project is a local, training-free toolkit that audits a dataset at two points in the labelling workflow using frozen self-supervised embeddings (DINOv2). **Before annotation** it clusters raw frames to measure scenario diversity and flags motion blur and low light. **After annotation** it clusters every labelled box per class to expose sub-types, outliers and likely labelling errors. It also includes **USB (Use-Seen-Before)**, a detector-free method that flags probable *missed* labels on fixed-camera data by reusing the dataset's own ground-truth box geometry as region proposals. Everything runs locally on one consumer GPU and produces a PDF report.

I built this as my own pipeline to address a real industrial annotation problem (fixed-camera assembly-line video). No customer data or names are included.

---

## 1. Problem and research questions

- **RQ1.** Which frozen embedding and pooling best separates *scenarios* in full, cluttered frames (pre-annotation), and why does the answer differ from tight object crops (post-annotation)?
- **RQ2.** Can density clustering of per-class crop embeddings surface sub-types and labelling errors without any labels beyond the boxes themselves?
- **RQ3.** On fixed-camera data, can missed labels be flagged without a detector, by reusing where the class has been annotated before?

## 2. Contributions

1. **A pooling-matched embedding design, measured.** Full frames use `[CLS ‖ mean(patch tokens)]` (1536d). On real industrial frames it separates activities clearly better than SigLIP (pair AUC 0.948 vs 0.917) and leaves the fewest frames unclustered (39% vs 43–52%). Object crops use the mean of all tokens (768d). The reproducible benchmark is [utils/test_embed_models.py](utils/test_embed_models.py) (§5).
2. **Per-class crop clustering** (DBSCAN, data-driven eps from the knee of the k-distance curve) with an automatic insights page covering outlier rate, cluster count, sample size and systematic noise.
3. **USB missed-label flagging**, which needs one model (DINOv2) and no detector. Regions come from deduplicated ground-truth boxes; each candidate is scored against a reference pool sampled for diversity (noise points + per-cluster centroid + farthest members). [docs/usb-missed-labels-ablation.md](docs/usb-missed-labels-ablation.md) gives the full reasoning.
4. **An engineering-honest pipeline**: headless and interactive modes, an optional cuML GPU backend with an exact cosine→euclidean eps conversion, and cache validation by content fingerprint and preprocessing tag.

---

## 3. Method

### 3.1 Pre-annotation: frame diversity and quality ([master_scripts/master_script_dinov2_PreAnn.py](master_scripts/master_script_dinov2_PreAnn.py))

```
frames ─► YOLO11-pose (person keypoints → 20 limb angle/length features, if ≥3 frames have persons)
       ─► DINOv2 [CLS ‖ mean(patch tokens)]  1536d  (whole frame → 224×224, no center crop)
       ─► scene block: standardize → PCA whiten (default 128 dims)
          pose block:  standardize
          blocks scaled so their share of total variance = scene_weight / pose_weight (default 0.7 / 0.3)
       ─► L2-normalize ─► HDBSCAN (eom)  →  "activity" clusters + noise frames
       ─► PaCMAP 2-D projection (visualisation only)
quality: brightness, contrast, motion blur = Sobel gradient-energy anisotropy (max/min of x,y energy)
```

- **Why `[CLS ‖ patch mean]`.** CLS summarises the whole frame. The mean of patch tokens aggregates *local* patch-level content: what is present across the frame, not where it is, since averaging discards position. Keeping them as separate halves stops the local signal from being folded into one averaged vector. This matters on cluttered full frames and much less on tight crops. Measured numbers: §5.
- **Weights.** They are applied *after* standardization and whitening, per block. Applying them earlier would be undone by those steps.
- **Motion blur.** Anisotropy is the ratio of x- to y-gradient energy, so directional (motion or shake) blur raises it. It is a motion-blur detector; it does not detect isotropic defocus.

### 3.2 Post-annotation: intra-class variation ([master_scripts/master_script_dinov2_PostAnn_PreTrain.py](master_scripts/master_script_dinov2_PostAnn_PreTrain.py))

```
Script 1  crop every labelled box (YOLO txt) + dataset health checks (bad/empty/degenerate crops,
          unexpected class ids, empty labels, duplicate basenames, imbalance)
Script 2  DINOv2 mean of all 257 tokens, 768d (whole crop → 224×224, no center crop) + preprocessing tag
Script 3  per class: (optional PCA) → DBSCAN (cosine; cuML euclidean with eps_euc = √(2·eps_cos))
          eps: knee of the sorted k-distance curve (default) or a fixed percentile
          large low-variance classes are downsampled and eps is re-tuned on the subsample
          UMAP 2-D plots for visualisation only
Script 4  PDF: dataset overview, config, distributions, centroid overviews, montages, insights page
Script 5  USB missed-label flagging (optional, §3.3)
```

**Eps rule.** scikit-learn's k-NN on the fitted data counts each point as its own first neighbour, so the k-distance equals DBSCAN's core-point test. Choosing eps at the *p*-th percentile therefore makes about *p*% of points core points by construction, which caps the outlier rate near (100−*p*)%. In a synthetic check with 33% scattered points, the 95th-percentile rule reported 1.1% outliers and the knee rule reported 33.3%. The knee rule is the default; percentile is kept as an option.

### 3.3 USB: missed-label flagging ([postannotation_scripts/5. usb_missed_labels.py](postannotation_scripts/5.%20usb_missed_labels.py))

1. Collect every GT box of the class across all splits. Greedy-dedup at IoU < 0.8 into `seen_<cls>_regions`, which are fixed pixel locations; this assumes a fixed camera.
2. Crop every region from every image using **Script 1's exact crop rule**, so candidates match the reference crops.
3. Skip a candidate if a same-class GT box in that image overlaps it (IoU ≥ 0.8, or intersection / smaller box ≥ 0.5). The second test is needed because kept regions can overlap each other.
4. Embed with Script 2's function and preprocessing tag. Score each candidate as the max cosine similarity against a reference pool built from the class's own embeddings: DBSCAN noise points, plus per cluster the centroid and its N farthest members.
5. Above the per-class threshold (default 0.8), flag the frame. The output shows flagged regions in red and existing GT in green.

**Scope.** USB ranks frames for human review; it does not claim boxes. It cannot find an object where the class has never been annotated, and it breaks if the camera moves.

---

## 4. Evaluation protocol

| What | How |
|---|---|
| Embedding choice (RQ1) | `utils/test_embed_models.py`: DINOv2 mean / CLS / CLS‖patch-mean vs SigLIP. (a) `--pairs SAME_DIR DIFF_DIR` gives a same-vs-different activity pair AUC. (b) `--frames DIR --groups prefix:N\|parent` clusters every variant with PreAnn's own function and reports noise %, silhouette, and ARI/NMI against the same reference groups |
| Clustering behaviour (RQ2) | Per-class outlier rate, cluster count and insights flags from Script 3/4. The knee eps makes the outlier rate data-driven |
| USB (RQ3) | Planned: label-dropout test (remove a known fraction of GT boxes, measure precision and recall of the flags) |

## 5. Results: embedding choice (RQ1)

Data: 4,175 frames from an industrial fixed-camera inspection dataset (not public), DINOv2-base and SigLIP-base, same preprocessing as the pipeline. Reproduce with the commands in §4 / §7.

### 5.1 Same vs different activity (pair AUC)

12 frames of one activity and 11 frames of different activities. **Pair AUC** is the probability that a random same-activity pair is more similar than a random different-activity pair. It is scale-free, so it can be compared across models.

| Embedding | Mean cos, same activity | Mean cos, different activities | Gap | **Pair AUC** |
|---|---|---|---|---|
| DINOv2 mean of all tokens (768d) | 0.946 | 0.620 | +0.326 | 0.964 |
| DINOv2 CLS (768d) | 0.910 | 0.449 | +0.461 | 0.941 |
| **DINOv2 [CLS ‖ patch-mean] (1536d)** | 0.919 | 0.489 | +0.430 | 0.948 |
| SigLIP pooled (768d) | 0.953 | 0.787 | +0.165 | 0.917 |

### 5.2 Full dataset, pipeline clustering (4,175 frames)

All variants are clustered with PreAnn's own function and the same parameters. As a reference label, "recording day" gives 5 groups.

| Embedding | Clusters | **Noise (unclustered)** | Silhouette (own clusters) | Silhouette vs day | ARI vs day | NMI vs day |
|---|---|---|---|---|---|---|
| DINOv2 mean | 44 | 43.2% | 0.107 | −0.054 | 0.176 | 0.366 |
| DINOv2 CLS | 50 | 44.4% | 0.119 | −0.086 | 0.127 | 0.304 |
| **DINOv2 [CLS ‖ patch-mean]** | 45 | **39.4%** | 0.096 | −0.079 | 0.114 | 0.325 |
| SigLIP pooled | 38 | 51.7% | 0.124 | 0.016 | 0.099 | 0.267 |

### 5.3 Findings

1. **DINOv2 separates activities clearly better than SigLIP.** Pair AUC is 0.94–0.96 vs 0.92, and the same/different gap is 2–3× larger. SigLIP scores even different activities as very similar (mean cos 0.787), which matches the impure SigLIP clusters seen in the montages.
2. **[CLS ‖ patch-mean] keeps different activities far apart and clusters the most data.** Different-activity pairs score cosine 0.489 against 0.620 for mean pooling, giving a same/different margin of +0.430, about 32% larger than mean pooling's +0.326 (CLS-only: +0.461). Mean pooling ranks pairs marginally better (AUC 0.964 vs 0.948) because it pulls *all* frames close together; AUC checks order only, not distance. It also leaves the fewest frames unclustered: 39.4% vs 43–44% for the other DINOv2 poolings and 51.7% for SigLIP, about 150–200 more frames grouped than the other DINOv2 variants and about 510 more than SigLIP, at comparable cluster quality. With 23 frames the DINOv2 poolings' AUCs are within noise of each other.
3. **Recording day is not a useful activity proxy.** DINOv2 variants give negative silhouette against day, so the same scenes recur across days. SigLIP separates days slightly (+0.016), most likely through global appearance such as lighting, which is a nuisance factor for activity grouping.
4. **Choice:** DINOv2 [CLS ‖ patch-mean] is the best overall trade-off. It keeps a large same/different margin (well above mean pooling and SigLIP), separates activities clearly better than SigLIP, and leaves the fewest frames unclustered. Ranking the DINOv2 poolings definitively needs a larger labelled activity set (one folder per activity, `--groups parent`).

## 6. Limitations

- Thresholds below are working defaults tuned on industrial fixed-camera data, not validated across domains.
- Silhouette on a variant's own clustering is not comparable across embeddings; use pair AUC or reference groups (§4).
- The activity test is small (23 frames, one dataset): enough to separate DINOv2 from SigLIP, not to rank the DINOv2 poolings against each other.
- Anisotropy detects motion blur only.
- USB assumes a fixed camera; region count grows if a class appears everywhere in the frame.
- `none`/absence-type classes have no visual prototype; neither clustering nor USB handles them meaningfully.

---

## 7. Usage

Always run from the project root.

```bat
exec.bat        # Windows menu: 1 = pre-annotation, 2 = pre-training
./exec.sh       # Linux/Mac
```

```bash
python "master_scripts/master_script_dinov2_PreAnn.py" --interactive            # pre-annotation (prompts)
python "master_scripts/master_script_dinov2_PostAnn_PreTrain.py" --interactive  # pre-training (prompts)

# headless (service / scripted use)
python "master_scripts/master_script_dinov2_PreAnn.py" --frames_dir /data/frames
python "master_scripts/master_script_dinov2_PostAnn_PreTrain.py" \
    --imgs_path /data/images --label_path /data/labels --classes_txt /data/classes.txt
```

Headless mode runs without prompts. PreAnn deletes its embedding cache first; PostAnn wipes its crop and cluster output folders at the start and deletes the crops at the end.

### Pre-annotation parameters

| Parameter | Default | Notes |
|---|---|---|
| `n_components` | 128 | Scene PCA dims before HDBSCAN |
| `min_cluster_size` | 25 | Minimum frames per activity |
| `min_samples` | 3 | HDBSCAN core strictness |
| `cluster_selection_epsilon` | 0.0 | Merge clusters closer than this distance |
| `scene_weight` / `pose_weight` | 0.7 / 0.3 | Share of total variance per block (pose only if ≥3 frames have persons) |
| `anisotropy_threshold` | 3.6 | Motion-blur cutoff |
| `batch_size` | 64 | DINOv2 batch size |

**Report thresholds (as implemented).** Dark (< 80 brightness) frames are flagged above 60% and suggested for dropping above 70%. Motion-blurred frames are flagged above 40%. Fewer than 3 activities is flagged. The overall verdict is "good" when the quality score is ≥ 7, there are ≥ 3 activities and blurred frames are < 30%. The quality score (0–10) is the mean of a motion-blur score `max(0, 10 − median_anisotropy/5·10)`, a brightness-centering score `10 − |median_brightness − 127|/127·10` and a contrast score `min(10, median_contrast/50·10)`.

The PDF contains: an executive summary, a quality dashboard, sample motion-blurred frames (if any), a PaCMAP scatter with the activity table, cluster-quality metrics, a coverage heatmap (activity × lighting) and activity montages. The page count varies with the number of activities.

### Pre-training parameters (master-script defaults)

| Parameter | Default | Notes |
|---|---|---|
| `auto_tune` | True | Auto eps per class |
| `eps_method` | knee | `knee` (data-driven) or `percentile` (outlier rate capped near 100−p%) |
| `auto_tune_percentile` | 90 | Used only with `eps_method=percentile` |
| `epsilon` | 0.15 | Manual eps when auto-tune is off |
| `min_pts` | 3 | DBSCAN min_samples |
| `pca_components` | 0 | PCA dims before DBSCAN (0 = off) |
| `max_cluster_samples` | 20 | Sample images saved per cluster |
| `uniform_downsample_target` / `uniform_min_samples` | 4000 / 12000 | Downsampling of large low-variance classes |
| `usb_classes` / `usb_sim_thresholds` / `usb_n_farthest` | — / 0.8 / 4 | USB (headless `--usb_classes a b --usb_sim_thresholds 0.8`) |

**Insights page thresholds (as implemented).**

| Flag | Rule |
|---|---|
| Outlier rate | Warning > 3%, critical > 8% |
| Cluster count | Warning > 5, critical > 15; 0 clusters = all noise (critical) |
| Sample size | Warning < 200 clustered samples, critical < 50 |
| Other flags | Systematic noise on downsampled classes; very spread embeddings (eps > 0.5); over-fragmentation at tight eps; homogeneous class (OK) |

### Scripts

| Script | Role |
|---|---|
| `postannotation_scripts/1. ann_txt_files_crop_bbox.py` | Crop boxes + dataset health checks |
| `postannotation_scripts/2. save_dinov2_embeddings_per_class.py` | 768d crop embeddings (+ preprocessing tag) |
| `postannotation_scripts/3. clustering_of_classes_embeddings.py` | Per-class DBSCAN, montages, centroid overviews (cuML auto-detected) |
| `postannotation_scripts/4. generate_pdf.py` | PDF report |
| `postannotation_scripts/5. usb_missed_labels.py` | USB missed-label flagging |
| `master_scripts/1. interactive_cluster_viewer.py` | Optional Plotly HTML explorer |
| `utils/test_embed_models.py` | Embedding benchmark (§4) |
| `utils/test_qwen.py` | Paused Qwen3-VL experiment |
| `master_scripts/master_script_dinov2_PostTrain.py`, `postannotation_scripts/1. yolo_model_crop_bbox_per_class.py` | ⚠️ **Deprecated, not maintained**, kept for reference only |

### Installing cuML (optional GPU clustering, WSL2/Linux)

```bash
mamba create -n eda_gpu -c rapidsai -c conda-forge -c nvidia cuml=25.* python=3.11 cuda-version=12.8 -y
mamba activate eda_gpu
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Use `mamba`, not `conda` (the conda solver stalls on RAPIDS). Pin torch; an unpinned torch pulls nvidia-* wheels that conflict with conda's CUDA libraries. Script 3 falls back to scikit-learn on CPU when cuML or CUDA is unavailable.

### Related work

Earlier, I tried missed-label detection by auto-annotating and diffing against the human labels, in a separate repo: [Auto_Annotation_YOLO_formatted](https://github.com/virkha-kumari/Auto_Annotation_YOLO_formatted). It covers few-shot visual prompting with YOLOe, SAM2/SAM3 and DINOv2, and its k-shot evaluation is in progress. Detector-based proposals struggled exactly on the small, visually ambiguous classes most likely to be missed by annotators, which motivated USB.

## References

- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023.
- Zhai et al., *Sigmoid Loss for Language Image Pre-Training (SigLIP)*, ICCV 2023.
- Ester et al., *A Density-Based Algorithm for Discovering Clusters (DBSCAN)*, KDD 1996.
- Campello, Moulavi, Sander, *Density-Based Clustering Based on Hierarchical Density Estimates (HDBSCAN)*, PAKDD 2013.
- Satopää et al., *Finding a "Kneedle" in a Haystack*, ICDCS Workshops 2011.
- McInnes, Healy, Melville, *UMAP*, 2018.
- Wang et al., *Understanding How Dimension Reduction Tools Work (PaCMAP)*, JMLR 2021.
