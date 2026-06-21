# Plans / Bucket List

Planned changes in rough priority order. Move items to `changes.md` once done and tested.

---

## Qwen PreAnn Workflow (Gradio App) — `preann_qwen_workflow/`

**Status:** Architecture designed, `utils/test_qwen.py` written to validate Qwen3-VL API. Build after test_qwen.py passes and is satisfactory.

**Concept:** Reinovates the current PreAnn script with a Qwen3-VL-based Gradio app for richer pre-annotation analysis.

### Architecture (5-module package)
- `embedder.py` — Qwen3-VL-Embedding model load + inference (image + text). EOS-token pooling. L2-normalize output. Separate `caption_frames()` loads generative model sequentially to avoid OOM.
- `clustering.py` — PCA whiten → HDBSCAN pipeline. Also `cluster_embeddings_raw()` (no PCA, for small per-cluster re-cluster). YOLO pose supplement: appends 20-dim pose features (10 limb pairs × angle+dist) to Qwen embeddings, then re-clusters that cluster only.
- `modes.py` — Mode A and Mode B orchestration:
  - **Mode A (Unsupervised + Guided):** embed all → HDBSCAN clusters → Qwen captions 3 centroid frames per cluster → cluster gets auto-label → user can rename + run YOLO pose re-cluster per cluster
  - **Mode B (Guided Activity Search):** perceptual-dedup sample 200 frames → Qwen captions → cluster captions (text embeddings + HDBSCAN) → return suggested activity list → user confirms/edits → embed all frames → cosine-match each frame to activity embeddings → frames below threshold → "Other/Unknown" (-2)
- `coverage.py` — per-activity frame counts, % coverage, status flags, recommendations
- `pdf_export.py` — PDF report export
- `app.py` — Gradio Blocks UI (dark theme), 5 tabs:
  1. **Setup & Run** — folder path, model selector (2B/4B/8B), mode radio, params (min_cluster_size, min_samples, PCA dims, eps, batch_size, match threshold, pdf quality, cache toggle), Run button, progress log; Mode B shows editable suggestions panel after Step 1
  2. **Cluster Browser** — dropdown + prev/next nav, stats, gallery (12 sample frames), PaCMAP scatter; Mode A shows rename + YOLO pose re-cluster controls
  3. **Frame Query** — text query → cosine search → gallery of top-K matching frames
  4. **Coverage Balance** — activity frame counts table + bar chart
  5. **Recommendations & Export** — text action plan + Export PDF button

### Key design decisions
- Sequential load: embedding model freed before generative model loads (8GB VRAM ceiling)
- `use_cache` toggle: embeddings cached to `preann_qwen_results/cache_{folder}/embeddings.npy`
- Model dropdown: auto-warns if selected model needs more VRAM than available
- YOLO pose supplement weight: 0.3 (pose features scaled before hstack with Qwen embs)
- Mode B match threshold: 0.25 cosine similarity default (frames below → "Other/Unknown")
- Tab visibility: tabs 2-5 hidden until pipeline completes

### Supported models
| Model | Dim | VRAM |
|---|---|---|
| Qwen/Qwen3-VL-Embedding-2B | 2048 | ~5.5GB |
| Qwen/Qwen3-VL-Embedding-4B | 2048 | ~9.0GB |
| Qwen/Qwen3-VL-Embedding-8B | 4096 | ~18.0GB |

### Next steps when test_qwen.py passes
1. Verify `caption_frames()` works with the generative model (test_qwen.py validates this path)
2. Restore `preann_qwen_workflow/` package from git history or rewrite from this spec
3. Entry point: `python -m preann_qwen_workflow` or `python app_launcher.py`
4. Add to `exec.bat` menu

---

## Dashboard instead of PDF
- can be viewed over a dashboard or something with images saved / persisted under assets and if that saved us the hassle of generating documents
- https://diffprog.slack.com/archives/C06Q7DXJDUJ/p1777549467871949

## Missing Labels Handling
- Detect and report frames that are missing annotation label files
- Surface this info clearly in the PDF report or terminal output
- Decide: hard error vs warning vs skip-and-count

## Integrate Interactive Cluster Viewer into Master Scripts
- Fold `interactive_cluster_viewer.py` as an optional step inside the master script prompts
- User sees: "Generate interactive HTML viewer? [Y/n]"
- Avoids needing to run it as a separate standalone script

---

## LOW PRIORITY — Architecture Robustness

## Embedding Cache Auto-Invalidation
- Current: manual delete required when input images change — silent footgun
- Option: hash input folder contents → auto-invalidate cache on change