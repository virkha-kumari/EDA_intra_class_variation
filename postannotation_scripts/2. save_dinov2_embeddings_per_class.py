#!/usr/bin/env python3
"""
Use: cropped_images folder
Embed each cropped image accordingly; DINOv2 (good for visual work esp used for zero-shot object detection so it has a knowledge for that). 

save_dinov2_embeddings_per_class.py

Walks a root folder of cropped images organized like:
  cropped_images/<class_name>/*.png

For each class it:
 - loads images in batches
 - encodes them with facebook/dinov2-base
 - resizes each crop to 224x224 with NO center crop (the whole box reaches the model —
   the processor default resizes short edge to 256 then center-crops, cutting box ends)
 - computes a 768-d vector per image (mean pooling of last_hidden_state)
 - saves embeddings to: cropped_images/<class_name>/embeddings_dinov2.npy
   + embeddings_dinov2_meta.json (preprocessing tag; --use_cache only reuses a matching tag)

Usage:
    python save_dinov2_embeddings_per_class.py --root ./cropped_imgs_by_class --batch 32
"""

import argparse
import json
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModel
import torch
import numpy as np
from tqdm import tqdm
import sys
import gc
import torch.cuda

# Whole crop → 224x224 (no center crop). Change PREPROC_TAG whenever preprocessing/pooling changes,
# so cached .npy files from older settings are never mixed with fresh ones (Script 5 checks it too).
DINO_PREPROC = {"do_center_crop": False, "size": {"height": 224, "width": 224}}
PREPROC_TAG = "facebook/dinov2-base|square224-nocrop|mean-all-tokens"


def meta_path_for(emb_path: Path) -> Path:
    return emb_path.with_name(emb_path.stem + "_meta.json")


def cached_tag(emb_path: Path):
    """Preprocessing tag stored next to an embeddings file, or None if missing/unreadable."""
    try:
        return json.loads(meta_path_for(emb_path).read_text(encoding="utf-8")).get("preproc")
    except (OSError, ValueError):
        return None


def find_class_folders(root: Path):
    return [p for p in sorted(root.iterdir()) if p.is_dir()]

def gather_image_paths(class_folder: Path, exts=(".png", ".jpg", ".jpeg")):
    files = []
    for p in sorted(class_folder.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    
    return files

def load_image_safe(path: Path):
    try:
        img = Image.open(path).convert("RGB")
        return img
    except (UnidentifiedImageError, OSError) as e:
        # corrupted or unreadable
        print(f"Warning: could not open image {path}: {e}", file=sys.stderr)
        return None

def compute_embeddings_for_paths(paths, processor, model, device, batch_size):
    embeddings = []
    valid_image_paths = []  # Track which images successfully loaded
    model.eval()
    
    with torch.no_grad():
        # Add progress bar for batches
        progress_bar = tqdm(range(0, len(paths), batch_size), desc="Computing embeddings", unit="batch")
        
        for i in progress_bar:
            # Clear GPU cache if memory is getting full
            if device == "cuda" and torch.cuda.memory_allocated() > 0.8 * torch.cuda.get_device_properties(0).total_memory:
                torch.cuda.empty_cache()
                gc.collect()
            
            batch_paths = paths[i:i+batch_size]
            imgs = []
            batch_valid_paths = []
            
            for p in batch_paths:
                img = load_image_safe(p)
                if img is None:
                    continue
                imgs.append(img)
                batch_valid_paths.append(p)

            if len(imgs) == 0:
                continue

            # Use mixed precision for CUDA operations
            if device == "cuda":
                with torch.amp.autocast('cuda'):
                    inputs = processor(images=imgs, return_tensors="pt", **DINO_PREPROC)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = model(**inputs)
                    emb_batch = outputs.last_hidden_state.mean(dim=1)
            else:
                inputs = processor(images=imgs, return_tensors="pt", **DINO_PREPROC)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                emb_batch = outputs.last_hidden_state.mean(dim=1)
            
            # Move to CPU and convert to numpy immediately to free GPU memory
            emb_batch = emb_batch.cpu()
            if device == "cuda":
                torch.cuda.synchronize()  # Ensure GPU operations are complete
            emb_batch = emb_batch.numpy()
            embeddings.append(emb_batch)
            valid_image_paths.extend(batch_valid_paths)

            # Clean up GPU memory
            del outputs, inputs
            if device == "cuda":
                torch.cuda.empty_cache()
            
            # Update progress bar description
            progress_bar.set_postfix({
                'batch_size': len(imgs),
                'total_processed': len(valid_image_paths),
                'gpu_mem': f'{torch.cuda.memory_allocated()/1e9:.1f}GB' if device == "cuda" else "N/A"
            })

    if len(embeddings) == 0:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32), []
    return np.vstack(embeddings).astype(np.float32), valid_image_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="cropped_imgs_by_class", help="Root folder with per-class subfolders")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for model inference")
    parser.add_argument("--save_suffix", type=str, default="embeddings_dinov2.npy", help="Filename used to save embeddings in each class folder")
    parser.add_argument("--use_cache", action="store_true", help="Skip embedding generation if file already exists (faster for re-runs)")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise RuntimeError(f"Root folder does not exist: {root}")

    # Setup device and display GPU info if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"Using GPU: {gpu_props.name}")
        print(f"GPU Memory: {gpu_props.total_memory/1e9:.1f}GB")
        print(f"CUDA Version: {torch.version.cuda}")
        # Set memory allocation strategy
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use up to 95% of GPU memory
    else:
        print("Using CPU - GPU not available")

    model_name = "facebook/dinov2-base"
    print(f"Loading processor and model: {model_name} ... (may download if first time run)")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    class_folders = find_class_folders(root)
    if not class_folders:
        print(f"No class subfolders found under {root}. Nothing to do.")
        return

    for cls_folder in class_folders:
        print(f"\n{'-'*80}")
        print(f"Processing class folder: {cls_folder.name}")
        image_paths = gather_image_paths(cls_folder)
        n_images = len(image_paths)
        print(f"Found {n_images} images")

        if n_images == 0:
            print("  Skipping (no images).")
            continue

        # compute embeddings
        emb_out_path = cls_folder / args.save_suffix
        mapping_path = cls_folder / f"{args.save_suffix.replace('.npy', '_image_list.txt')}"

        # Check if embeddings already exist and use_cache is enabled
        if args.use_cache and emb_out_path.exists() and mapping_path.exists():
            if cached_tag(emb_out_path) == PREPROC_TAG:
                print(f"✅ Using cached embeddings from: {emb_out_path.name}")
                continue
            print(f"⚠️  Cached {emb_out_path.name} was made with different/unknown preprocessing — regenerating")
        elif emb_out_path.exists():
            print(f"⚠️  Note: {emb_out_path.name} already exists. Overwriting previous embeddings!")

        embeddings, valid_paths = compute_embeddings_for_paths(image_paths, processor, model, device, batch_size=args.batch)
        
        # Sanity check
        if len(valid_paths) != n_images:
            print(f"  ⚠️  Warning: {n_images - len(valid_paths)} images failed to load")
            print(f"  Embeddings created for {len(valid_paths)}/{n_images} images")
        print(f"  Embeddings shape: {embeddings.shape}, dtype={embeddings.dtype}")

        # Save embeddings as N x 768 numpy file
        np.save(str(emb_out_path), embeddings)
        meta_path_for(emb_out_path).write_text(json.dumps({"preproc": PREPROC_TAG}), encoding="utf-8")
        print(f"  Saved embeddings to: {emb_out_path}")
        
        # Save mapping of valid image filenames to preserve alignment
        mapping_path = cls_folder / f"{args.save_suffix.replace('.npy', '_image_list.txt')}"
        with open(mapping_path, 'w', encoding='utf-8') as f:
            for p in valid_paths:
                f.write(f"{p.name}\n")
        print(f"  Saved image mapping to: {mapping_path}")

    print("\nDone. Per-class embeddings have been saved as <class>/embeddings_dinov2.npy")

if __name__ == "__main__":
    main()