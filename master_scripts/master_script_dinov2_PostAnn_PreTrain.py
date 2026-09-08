"""
Master script to run entire intra and inter class variation EDA pipeline.
Includes memory management and cooling breaks for laptop GPUs.

(with PDF) 34 classes, 5789 annotations: 15-16 mins (2 mins by pdf)

Usage modes:
  Interactive (human-friendly, questionary prompts):
      python master_script_dinov2_PostAnn_PreTrain.py --interactive

  Non-interactive (headless, programmatic):
      python master_script_dinov2_PostAnn_PreTrain.py \\
          --imgs_path /data/images --label_path /data/labels \\
          --classes_txt /data/classes.txt [--option value ...]
"""

import argparse
import logging
import subprocess
import sys
import gc
import time
import psutil
import os
import shutil
from pathlib import Path
import torch
import questionary

def clear_memory():
    """Aggressive memory cleanup."""
    print("\n🧹 Clearing memory...")
    
    # Python garbage collection
    gc.collect()
    
    # Clear PyTorch cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Get GPU memory stats
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"   GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    # Get system memory stats
    mem = psutil.virtual_memory()
    print(f"   RAM Usage: {mem.percent}% ({mem.used/1e9:.1f}GB / {mem.total/1e9:.1f}GB)")
    print("   ✅ Memory cleared\n")

def cooling_break(seconds):
    """Pause execution to let GPU cool down."""
    print(f"\n❄️  Cooling break: {seconds} seconds...")
    print("   This prevents thermal throttling on your laptop GPU")
    
    for remaining in range(seconds, 0, -5):
        print(f"   ⏳ {remaining} seconds remaining...", end='\r')
        time.sleep(5)
    
    print("\n   ✅ Cooling break complete\n")

def check_system_resources(interactive=True):
    """Check if system has enough resources to continue."""
    mem = psutil.virtual_memory()

    if mem.percent > 90:
        print("⚠️  WARNING: RAM usage is very high (>90%)")
        print("   Consider closing other applications")
        if interactive:
            response = input("   Continue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
        else:
            print("   Headless mode — continuing regardless")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        usage_percent = (allocated / total) * 100
        
        if usage_percent > 85:
            print("⚠️  WARNING: GPU memory usage is very high (>85%)")
            print("   The script will attempt to clear cache")
            clear_memory()

def run_step(script, args, cool_down_after=True, logger=None, interactive=True):
    """Run a pipeline step with memory management."""
    print(f"\n{'='*60}")
    print(f"Running: {script}")
    print(f"{'='*60}\n")

    if logger:
        logger.info(f"Step start: {script}")

    # Check resources before starting
    check_system_resources(interactive=interactive)

    # Run the script
    result = subprocess.run([sys.executable, script] + args)

    if result.returncode != 0:
        print(f"❌ Error in {script}")
        if logger:
            logger.error(f"Step FAILED: {script} exited with code {result.returncode}")
        clear_memory()  # Clean up even on failure
        sys.exit(1)

    print(f"\n✅ {script} completed successfully")
    if logger:
        logger.info(f"Step complete: {script}")

    # Post-execution cleanup
    if cool_down_after:
        clear_memory()
        cooling_break(10)  # 10 second break between steps

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


def _load_classes_txt(path):
    """Read class names from a classes.txt file (one name per line, skip blanks)."""
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def _setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("eda_postann")
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
    p = argparse.ArgumentParser(description="Intra-class variation EDA pipeline (PostAnnotation/PreTraining)")
    p.add_argument("--interactive", action="store_true",
                   help="Run with questionary prompts instead of CLI args")
    # Required in non-interactive mode
    p.add_argument("--imgs_path", type=str, default=None)
    p.add_argument("--label_path", type=str, default=None)
    p.add_argument("--classes_txt", type=str, default=None)
    # Output dirs — None means auto-derive from imgs_path folder name
    p.add_argument("--cropped_dir", type=str, default=None)
    p.add_argument("--cluster_dir", type=str, default=None)
    # Embedding
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--save_suffix", type=str, default="embeddings_dinov2.npy")
    p.add_argument("--num_workers", type=int, default=4)
    # Clustering
    p.add_argument("--auto_tune", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save_class_scatter", action="store_true", default=False)
    p.add_argument("--auto_tune_percentile", type=int, default=90)
    p.add_argument("--epsilon", type=float, default=0.15)
    p.add_argument("--min_pts", type=int, default=3)
    p.add_argument("--umap_min_dist", type=float, default=0.05)
    p.add_argument("--max_cluster_samples", type=int, default=20)
    p.add_argument("--pca_components", type=int, default=0)
    p.add_argument("--uniform_eps_threshold", type=float, default=0.1)
    p.add_argument("--uniform_downsample_target", type=int, default=4000)
    p.add_argument("--uniform_min_samples", type=int, default=12000)
    # PDF
    p.add_argument("--pdf", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pdf_name", type=str, default=None)
    p.add_argument("--pdf_quality", type=int, default=75)
    p.add_argument("--contrastive_groups_json", type=str, default=None)
    # USB missed-label detection (runs after PDF, before crop deletion)
    p.add_argument("--usb_classes", nargs="+", default=None,
                   help="Class names to check for missed labels (omit = skip USB)")
    p.add_argument("--usb_sim_thresholds", nargs="+", type=float, default=None,
                   help="Cosine sim threshold per USB class (single value = applied to all; default 0.8)")
    p.add_argument("--usb_n_farthest", type=int, default=4)
    p.add_argument("--usb_only", action="store_true",
                   help="Run only crops + embeddings + USB for --usb_classes (skips clustering and PDF)")
    args = p.parse_args()
    if not args.interactive:
        missing = [f for f, v in [("--imgs_path", args.imgs_path), ("--label_path", args.label_path), ("--classes_txt", args.classes_txt)] if v is None]
        if missing:
            p.error(f"Non-interactive mode requires: {', '.join(missing)}")
    if args.usb_only and not args.usb_classes:
        p.error("--usb_only requires --usb_classes")
    return args


def main():
    # Anchor CWD to project root — safe no-op if already correct, guards against running from wrong dir
    os.chdir(Path(__file__).resolve().parent.parent)

    args = parse_args()

    # GPU specs print
    print("="*60)
    print("INTRA-CLASS VARIATION EDA PIPELINE")
    print("="*60)

    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {gpu_props.name}")
        print(f"  VRAM: {gpu_props.total_memory/1e9:.1f}GB")
    else:
        print("  GPU: Not available or PyTorch not installed")

    print("="*60)

    if args.interactive:
        ## -----------------------------------------------##
        ##   REQUIRED INPUTS (prompted, no defaults)       ##
        ## -----------------------------------------------##
        print("\n📁  STEP 0: CONFIGURE PIPELINE INPUTS")
        print("="*60)

        imgs_path   = _prompt_path("Path to images folder:", must_exist_dir=True)
        label_path  = _prompt_path("Path to labels folder (can be same as images):", must_exist_dir=True)

        # Detect train/val/test splits
        _splits = [s for s in ['train', 'val']
                   if os.path.isdir(os.path.join(imgs_path, s)) and os.path.isdir(os.path.join(label_path, s))]
        if _splits:
            print(f"\n📂  Detected dataset splits: {', '.join(_splits)}")
            print(f"    Images and labels from all splits will be processed together")

        classes_txt = _prompt_path("Path to classes.txt file:", must_exist_file=True)

        class_names = _load_classes_txt(classes_txt)
        print(f"\n✅  Loaded {len(class_names)} classes from {classes_txt}")

        ## -----------------------------------------------##
        ##   RUN MODE + USB (Use-Seen-Before) inputs       ##
        ## -----------------------------------------------##
        _MODE_EDA, _MODE_EDA_USB, _MODE_USB = (
            "Full EDA pipeline (clustering + PDF)",
            "Full EDA pipeline + USB missed-label check",
            "USB missed-label check only (crops + embeddings + USB, no clustering/PDF)",
        )
        run_mode = _ask(questionary.select("What do you want to run?",
                                           choices=[_MODE_EDA, _MODE_EDA_USB, _MODE_USB]))
        usb_only = run_mode == _MODE_USB
        usb_classes, usb_sim_thresholds, usb_n_farthest = [], [], 4
        if run_mode != _MODE_EDA:
            while not usb_classes:
                usb_classes = _ask(questionary.checkbox(
                    "Tick classes you suspect have missed labels", choices=class_names))
                if not usb_classes:
                    if usb_only:
                        print("  ⚠️  USB-only mode needs at least one class.")
                    else:
                        print("  ℹ️  No classes ticked — USB skipped")
                        break
            for c in usb_classes:
                usb_sim_thresholds.append(float(_prompt_text(
                    f"Sim threshold for '{c}' (0.8=balanced, higher=stricter)", 0.8)))
            if usb_classes:
                usb_n_farthest = int(_prompt_text(
                    "Farthest points per cluster in reference pool", 4))
                print(f"  ✅ USB will check {len(usb_classes)} class(es)")

        ## -----------------------------------------------##
        ##   RESULTS DIRECTORY (fixed, not changeable)      ##
        ## -----------------------------------------------##
        RESULTS_DIR = "postann_pretrain_results"
        os.makedirs(RESULTS_DIR, exist_ok=True)
        print(f"\n📂  All results will be saved under: {os.path.abspath(RESULTS_DIR)}/")

        # Derive smart defaults from the images folder name
        _folder_name = os.path.basename(os.path.normpath(imgs_path))

        ## -----------------------------------------------##
        ##   OUTPUT NAMES — always asked, with defaults    ##
        ## -----------------------------------------------##
        _default_cropped   = f"cropped_imgs_by_class_{_folder_name}"
        _default_clust_dir = f"clustering_results_{_folder_name}"
        _default_pdf_name  = f"PDF_REPORT_{_folder_name}"

        cropped_bbox_dir   = os.path.join(RESULTS_DIR, _prompt_text("Subfolder for cropped images", _default_cropped))
        output_cluster_dir = os.path.join(RESULTS_DIR, _prompt_text("Subfolder for clustering results", _default_clust_dir))

        ## -----------------------------------------------##
        ##   OPTIONAL PARAMETERS                           ##
        ## -----------------------------------------------##

        change_defaults = _ask(questionary.confirm(
            "Change default parameters?", default=False))

        # Embedding Handling
        if change_defaults:
            batch_size  = int(_prompt_text("DINOv2 batch size (reduce if GPU OOM)", 64))
            save_suffix = _prompt_text("Embeddings filename", "embeddings_dinov2.npy")
            num_workers = int(_prompt_text("Crop extraction threads (4=safe for NVMe laptop, 8=fast NVMe desktop)", 4))
        else:
            batch_size  = 64
            save_suffix = "embeddings_dinov2.npy"
            num_workers = 4

        # Clustering flags (auto-tune/eps/min_pts also drive USB's reference-pool DBSCAN)
        auto_tune        = _ask(questionary.confirm("Enable auto-tune eps?", default=True))
        save_class_scatter = False if usb_only else _ask(questionary.confirm("Save per-class UMAP scatter plots? (slow — runs UMAP per class)", default=False))

        # Cluster Handling — conditionally prompt eps OR percentile
        if auto_tune:
            auto_tune_percentile = int(_prompt_text("Auto-tune k-NN percentile (90=tight, 95=balanced, 98=loose)", 90))
            epsilon = 0.15  # unused fallback
        else:
            epsilon = float(_prompt_text("DBSCAN epsilon (0.10=strict, 0.15=balanced, 0.20-0.30=lenient)", 0.15))
            auto_tune_percentile = 95  # unused fallback

        if change_defaults and usb_only:
            min_pts             = int(_prompt_text("Min points per cluster", 3))
            umap_min_dist, max_cluster_samples, pca_components = 0.05, 20, 0
            uniform_class_eps_threshold, uniform_class_downsample_target, uniform_class_min_samples = 0.1, 4000, 12000
        elif change_defaults:
            min_pts             = int(_prompt_text("Min points per cluster", 3))
            umap_min_dist       = float(_prompt_text("UMAP min_dist (lower=tighter packing)", 0.05))
            max_cluster_samples = int(_prompt_text("Max sample images saved per cluster", 20))
            # PCA reduces 768d embeddings → Nd before clustering to fix curse of dimensionality (esp. large classes 10K+ samples)
            pca_components      = int(_prompt_text("PCA dims before clustering (0=disabled, 128=balanced; reduces 768d→Nd)", 0))

            # Uniform class handling
            uniform_class_eps_threshold     = float(_prompt_text("Uniform class eps threshold", 0.1))
            uniform_class_downsample_target = int(_prompt_text("Uniform class downsample target", 4000))
            uniform_class_min_samples       = int(_prompt_text("Uniform class min size to trigger downsampling", 12000))
        else:
            min_pts             = 3
            umap_min_dist       = 0.05
            max_cluster_samples = 20
            pca_components      = 0
            uniform_class_eps_threshold     = 0.1
            uniform_class_downsample_target = 4000
            uniform_class_min_samples       = 12000

        # PDF handling
        temp_file    = os.path.join(cropped_bbox_dir, "temp_ann_file.txt")
        pdf_generate = False if usb_only else _ask(questionary.confirm("Generate PDF report?", default=True))
        pdf_name     = _prompt_text("Output PDF filename (no extension)", _default_pdf_name) if pdf_generate else _default_pdf_name
        pdf_quality  = int(_prompt_text("PDF image quality (1-95, 75=balanced, 90=high, 50=small file)", 75)) if change_defaults and pdf_generate else 75

        # Contrastive class grouping for distribution chart
        import json as _json
        contrastive_groups_json = None
        if pdf_generate:
            use_contrastive = _ask(questionary.confirm(
                "Group contrastive class pairs in distribution chart?", default=False))
            if use_contrastive:
                contrastive_groups = {}
                group_num = 1
                while True:
                    selected = _ask(questionary.checkbox(
                        f"Group {group_num}: tick classes to pair together (Enter alone = done)",
                        choices=class_names,
                    ))
                    if not selected:
                        break
                    default_name = " vs ".join(selected)
                    group_name = _prompt_text(f"Name for this group", default_name)
                    contrastive_groups[group_name] = selected
                    group_num += 1
                    another = _ask(questionary.confirm("Add another group?", default=False))
                    if not another:
                        break
                if contrastive_groups:
                    contrastive_groups_json = _json.dumps(contrastive_groups)
                    print(f"  ✅ {len(contrastive_groups)} group(s) configured")
                else:
                    print("  ℹ️  No groups — using plain chart")

        ## -----------------------------------------------##
        ##   PRE-FLIGHT: CHECK OUTPUT DIRS ARE EMPTY      ##
        ## -----------------------------------------------##
        skip_to_clustering = False

        print("\n" + "="*60)

        # --- Cropped images folder check ---
        if os.path.isdir(cropped_bbox_dir):
            # Check if embeddings already exist in any class subfolder
            existing_embs = list(Path(cropped_bbox_dir).rglob(save_suffix))
            if existing_embs:
                print(f"\n✅  Cropped images folder already exists WITH embeddings ({len(existing_embs)} class(es)):")
                print(f"   {os.path.abspath(cropped_bbox_dir)}")
                if pdf_generate:
                    print(f"   ⚠️  Note: reusing crops skips Step 1 — PDF dataset stats table will use cached temp_ann_file.txt if present.")
                reuse = _ask(questionary.confirm(
                    "   Reuse existing crops + embeddings? (No = overwrite from scratch)", default=True))
                if reuse:
                    skip_to_clustering = True
                    print("   ✅ Skipping Steps 1 & 2 — jumping straight to clustering.")
                else:
                    print(f"   🗑️  Deleting: {os.path.abspath(cropped_bbox_dir)}")
                    shutil.rmtree(os.path.abspath(cropped_bbox_dir))
                    print(f"   ✅ Deleted — will recreate from scratch.")
            else:
                print(f"\n⚠️  Cropped images folder already exists and is NOT empty (no embeddings found inside):")
                print(f"   {os.path.abspath(cropped_bbox_dir)}")
                overwrite = _ask(questionary.confirm(
                    "   Overwrite existing contents? (No = exit)", default=True))
                if not overwrite:
                    print("\n   Exiting. Please choose a different output folder or clear the existing one.")
                    sys.exit(0)
                print(f"   🗑️  Deleting: {os.path.abspath(cropped_bbox_dir)}")
                shutil.rmtree(os.path.abspath(cropped_bbox_dir))
                print(f"   ✅ Deleted — will recreate from scratch.")

        # --- Clustering results folder check (only when not reusing) ---
        if not skip_to_clustering and os.path.isdir(output_cluster_dir):
            print(f"\n⚠️  Clustering results folder already exists and is NOT empty:")
            print(f"   {os.path.abspath(output_cluster_dir)}")
            overwrite = _ask(questionary.confirm(
                "   Overwrite existing contents? (No = exit)", default=True))
            if not overwrite:
                print("\n   Exiting. Please choose a different output folder or clear the existing one.")
                sys.exit(0)
            print(f"   🗑️  Deleting: {os.path.abspath(output_cluster_dir)}")
            shutil.rmtree(os.path.abspath(output_cluster_dir))
            print(f"   ✅ Deleted — will recreate from scratch.")

        print("\n" + "="*60)

    else:
        ## -----------------------------------------------##
        ##   NON-INTERACTIVE: resolve all values from args ##
        ## -----------------------------------------------##
        imgs_path   = args.imgs_path
        label_path  = args.label_path
        classes_txt = args.classes_txt

        class_names = _load_classes_txt(classes_txt)

        RESULTS_DIR = "postann_pretrain_results"
        os.makedirs(RESULTS_DIR, exist_ok=True)

        _folder_name = os.path.basename(os.path.normpath(imgs_path))

        cropped_bbox_dir   = os.path.join(RESULTS_DIR, args.cropped_dir  if args.cropped_dir  else f"cropped_imgs_by_class_{_folder_name}")
        output_cluster_dir = os.path.join(RESULTS_DIR, args.cluster_dir  if args.cluster_dir  else f"clustering_results_{_folder_name}")

        batch_size              = args.batch_size
        save_suffix             = args.save_suffix
        num_workers             = args.num_workers
        auto_tune               = args.auto_tune
        save_class_scatter      = args.save_class_scatter
        auto_tune_percentile    = args.auto_tune_percentile
        epsilon                 = args.epsilon
        min_pts                 = args.min_pts
        umap_min_dist           = args.umap_min_dist
        max_cluster_samples     = args.max_cluster_samples
        pca_components          = args.pca_components
        uniform_class_eps_threshold     = args.uniform_eps_threshold
        uniform_class_downsample_target = args.uniform_downsample_target
        uniform_class_min_samples       = args.uniform_min_samples
        usb_only                = args.usb_only
        pdf_generate            = False if usb_only else args.pdf
        pdf_name                = args.pdf_name if args.pdf_name else f"PDF_REPORT_{_folder_name}"
        pdf_quality             = args.pdf_quality
        contrastive_groups_json = args.contrastive_groups_json

        usb_classes    = args.usb_classes or []
        usb_n_farthest = args.usb_n_farthest
        if usb_classes:
            thr = args.usb_sim_thresholds or [0.8]
            if len(thr) == 1:
                thr = thr * len(usb_classes)
            if len(thr) != len(usb_classes):
                sys.exit("--usb_sim_thresholds must be a single value or one per --usb_classes entry")
            usb_sim_thresholds = thr
        else:
            usb_sim_thresholds = []

        temp_file = os.path.join(cropped_bbox_dir, "temp_ann_file.txt")

        # Non-interactive: always start fresh — wipe both dirs
        for _d in [cropped_bbox_dir, output_cluster_dir]:
            if os.path.isdir(_d):
                shutil.rmtree(_d)

        skip_to_clustering = False

        print("\n" + "="*60)

    # Logger — co-located with clustering results so all logs land next to output
    os.makedirs(output_cluster_dir, exist_ok=True)
    logger = _setup_logger(os.path.join(output_cluster_dir, "eda_run.log"))
    logger.info("Pipeline started")
    logger.info(f"imgs_path={imgs_path}  label_path={label_path}  classes_txt={classes_txt}")
    logger.info(f"cropped_bbox_dir={cropped_bbox_dir}  output_cluster_dir={output_cluster_dir}")
    logger.info(f"batch_size={batch_size}  num_workers={num_workers}  auto_tune={auto_tune}  auto_tune_percentile={auto_tune_percentile}  epsilon={epsilon}")
    logger.info(f"pdf={pdf_generate}  pdf_quality={pdf_quality}  pca_components={pca_components}")
    logger.info(f"usb_only={usb_only}  usb_classes={usb_classes}  usb_sim_thresholds={usb_sim_thresholds}  usb_n_farthest={usb_n_farthest}")

    # Classes to ids and etc — USB-only mode crops/embeds just the USB classes
    class_ids = [str(i) for i, name in enumerate(class_names) if not usb_only or name in usb_classes]
    class_ids_to_names = []
    for i, name in enumerate(class_names):
        class_ids_to_names.extend([str(i), name])


    total_steps = 3 if usb_only else 3 + int(pdf_generate) + int(bool(usb_classes))

    if not skip_to_clustering:
        # Step 1: Crop YOLO bboxes -- Only send temp txt file if requested for PDF Generation
        print("\n" + "="*60)
        print(f"STEP 1/{total_steps}: EXTRACTING BOUNDING BOXES FROM LABELLED DATA")
        print("⚠️  This step is time-consuming")
        print("="*60)
        if pdf_generate:
            run_step("postannotation_scripts/1. ann_txt_files_crop_bbox.py", [
                "--imgs_path", imgs_path,
                "--label_path", label_path,
                "--classes"] + class_ids + [
                "--class_ids_to_names"] + class_ids_to_names + [
                "--output_dir", cropped_bbox_dir,
                "--output_txt_file", temp_file,
                "--num_workers", str(num_workers)
            ], logger=logger, interactive=args.interactive)
        else:
            run_step("postannotation_scripts/1. ann_txt_files_crop_bbox.py", [
                "--imgs_path", imgs_path,
                "--label_path", label_path,
                "--classes"] + class_ids + [
                "--class_ids_to_names"] + class_ids_to_names + [
                "--output_dir", cropped_bbox_dir,
                "--num_workers", str(num_workers)
            ], logger=logger, interactive=args.interactive)

        # Step 2: Embed the cropped YOLO bbox
        print("\n" + "="*60)
        print(f"STEP 2/{total_steps}: GENERATING DINOV2 EMBEDDINGS")
        print("="*60)
        print("⚠️  This step is GPU-intensive")

        embedding_args = [
            "--root", cropped_bbox_dir,
            "--batch", str(batch_size),
            "--save_suffix", save_suffix,
            "--use_cache",
        ]

        run_step("postannotation_scripts/2. save_dinov2_embeddings_per_class.py", embedding_args, cool_down_after=True, logger=logger, interactive=args.interactive)
    else:
        print(f"\n⏭️  STEP 1 & 2 SKIPPED — using existing crops + embeddings in: {cropped_bbox_dir}")


    # Step 3: Cluster using DBSCAN to visualize (skipped in USB-only — Script 5 clusters in-process)
    if usb_only:
        print("\n⏭️  STEP 3 (clustering) & PDF SKIPPED — USB-only mode")
    else:
        print("\n" + "="*60)
        print(f"STEP 3/{total_steps}: CLUSTERING ANALYSIS")
        print("="*60)
        cluster_args = [
            "--root", cropped_bbox_dir,
            "--eps", str(epsilon),
            "--min_samples", str(min_pts),
            "--output_dir", output_cluster_dir,
            "--max_samples", str(max_cluster_samples),
            "--save_suffix", save_suffix,
            "--auto_tune_percentile", str(auto_tune_percentile),
            "--umap_min_dist", str(umap_min_dist),
            "--save_montage",
            "--uniform_eps_threshold", str(uniform_class_eps_threshold),
            "--uniform_downsample_target", str(uniform_class_downsample_target),
            "--uniform_min_samples", str(uniform_class_min_samples),
            "--pca_components", str(pca_components)
        ]
        if auto_tune:
            cluster_args.append("--auto_tune")
        if save_class_scatter:
            cluster_args.append("--save_class_scatter")

        run_step("postannotation_scripts/3. clustering_of_classes_embeddings.py",
                 cluster_args, cool_down_after=False, logger=logger, interactive=args.interactive)

    # Step 4 (Optional): Generate PDF Report
    if pdf_generate:
        print("\n" + "="*60)
        print(f"STEP 4/{total_steps}: GENERATING PDF REPORT")
        print("="*60)
        pdf_args = [
            "--temp_txt_file", temp_file,
            "--clustering_dir", output_cluster_dir,
            "--pdf_name", os.path.join(output_cluster_dir, pdf_name),
            "--imgs_path", imgs_path,
            "--label_path", label_path,
            "--classes_txt", classes_txt,
            "--auto_tune_percentile", str(auto_tune_percentile),
            "--epsilon", str(epsilon),
            "--pdf_quality", str(pdf_quality),
        ]
        if auto_tune:
            pdf_args.append("--auto_tune")
        if contrastive_groups_json:
            pdf_args.extend(["--contrastive_groups_json", contrastive_groups_json])
        run_step("postannotation_scripts/4. generate_pdf.py", pdf_args, cool_down_after=False, logger=logger, interactive=args.interactive)

        print("\n" + "="*60)
        print(f"📄 PDF Report generated: {os.path.join(output_cluster_dir, pdf_name)}.pdf")
        print("="*60)

    # Step 5 (Optional): USB missed-label detection — needs crops dir (.npy) so runs before deletion
    if usb_classes:
        print("\n" + "="*60)
        print(f"STEP {total_steps}/{total_steps}: USB MISSED-LABEL DETECTION")
        print("="*60)
        usb_args = [
            "--imgs_path", imgs_path,
            "--label_path", label_path,
            "--classes_txt", classes_txt,
            "--cropped_dir", cropped_bbox_dir,
            "--output_dir", output_cluster_dir,
            "--usb_classes"] + usb_classes + [
            "--sim_thresholds"] + [str(t) for t in usb_sim_thresholds] + [
            "--n_farthest", str(usb_n_farthest),
            "--save_suffix", save_suffix,
            "--batch", str(batch_size),
            "--min_samples", str(min_pts),
            "--auto_tune_percentile", str(auto_tune_percentile),
            "--eps", str(epsilon),
        ]
        if auto_tune:
            usb_args.append("--auto_tune")
        run_step("postannotation_scripts/5. usb_missed_labels.py", usb_args, cool_down_after=False, logger=logger, interactive=args.interactive)
        print(f"\n🚩 USB results: {os.path.join(output_cluster_dir, 'missing_labels')}/")

    # Non-interactive: always delete intermediate crops after pipeline finishes
    if not args.interactive and os.path.isdir(cropped_bbox_dir):
        try:
            shutil.rmtree(cropped_bbox_dir)
            logger.info(f"Deleted intermediate crops: {cropped_bbox_dir}")
        except Exception as e:
            logger.error(f"Failed to delete crops dir {cropped_bbox_dir}: {e}")

    # Ask user whether to delete cropped images folder (interactive only)
    if args.interactive and os.path.isdir(cropped_bbox_dir):
        cropped_size_gb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, files in os.walk(cropped_bbox_dir)
            for f in files
        ) / 1e9
        print("\n" + "="*60)
        print(f"📁 Cropped images folder: {cropped_bbox_dir}/")
        print(f"   Size on disk: {cropped_size_gb:.2f} GB")
        print("⚠️  WARNING: Deleting this folder is PERMANENT — embeddings (.npy) are kept,")
        print("   but you will NOT be able to view the original cropped images again.")
        response = input("\n🗑️  Delete cropped images folder to free disk space? (y/n): ")
        if response.lower() == 'y':
            try:
                shutil.rmtree(cropped_bbox_dir)
                print(f"   ✅ Deleted: {cropped_bbox_dir}/")
            except Exception as e:
                print(f"   ❌ Error deleting folder: {e}")
        else:
            print(f"   📌 Keeping: {cropped_bbox_dir}/")

    # Final cleanup
    clear_memory()
    logger.info("Pipeline complete")

    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE!")
    print("="*60)
    print("\nOutput Locations:")
    print(f"  Cropped Images:     {cropped_bbox_dir}/")
    print(f"  Clustering Results: {output_cluster_dir}/")
    if pdf_generate:
        print(f"  PDF Report:         {os.path.join(output_cluster_dir, pdf_name)}.pdf")
    if usb_classes:
        print(f"  USB Missed Labels:  {os.path.join(output_cluster_dir, 'missing_labels')}/")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        clear_memory()
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        clear_memory()
        sys.exit(1)