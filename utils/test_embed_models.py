"""
Pre-annotation embedding benchmark — DINOv2 pooling variants vs SigLIP (+ optional V-JEPA 2.1, DINOv3).

Embeds the same frames with every variant, clusters each with the PreAnn pipeline's OWN clustering
function (cluster_activities_with_pca_hdbscan: standardize → PCA whiten → L2 → HDBSCAN, scene-only,
same params as PreAnn), and reports:

  Internal (each variant scored on its own clustering — NOT strictly comparable across variants,
  every variant clusters in its own space and picks its own noise points):
    n_clusters, noise %, silhouette_own (cosine on L2-normalised embeddings, noise excluded)

  External (comparable across variants — same reference groups for all, via --groups):
    silhouette_groups (cosine, raw embeddings vs reference groups), ARI / NMI of the clustering vs groups

Variants:
    dinov2_mean       mean of all 257 tokens (CLS + 256 patches)        768d
    dinov2_cls        CLS token only                                     768d
    dinov2_cls_patch  [CLS || mean(patch tokens)]  (PreAnn default)      1536d
    siglip_pooled     SigLIP base-patch16-224 pooler_output              768d
    vjepa_mean        V-JEPA 2.1 ViT-B/16 mean token  (--models vjepa, needs models/ checkpoint)
    dinov3_cls        DINOv3 ViT-B/16 CLS             (--models dinov3, gated on HF)
DINOv2 inputs use PreAnn's preprocessing (whole frame → 224x224, no center crop).

Usage:
    python utils/test_embed_models.py --frames path/to/frames [--groups parent | prefix:10 | none]
        [--models dinov2 siglip vjepa dinov3] [--out benchmark_preann.json]
        [--n_components 128 --min_cluster_size 25 --min_samples 3]

--groups parent    reference group = parent folder name (e.g. one folder per activity / scenario)
--groups prefix:N  reference group = first N filename characters (e.g. video id)

Same-vs-different activity test (no clustering):
    python utils/test_embed_models.py --pairs path/to/ACTIVITY_SAME path/to/ACTIVITY_DIFF [--out bench_pairs.json]
SAME = frames of one activity, DIFF = frames of different activities. Per variant: mean cosine of SAME
pairs, mean cosine of DIFF pairs, gap, and pair AUC = P(random SAME pair more similar than random DIFF
pair). AUC is scale-free, so it is comparable across models (raw cosine ranges differ per model).
"""

import argparse
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent.parent
VJEPA_CHECKPOINT = ROOT / "models" / "vjepa2_1_vitb_dist_vitG_384.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_EXTS = {".jpg", ".jpeg", ".png"}
SEED = 42

TRANSFORM_VJEPA = transforms.Compose([
    transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def load_preann():
    """PreAnn master module — reuse its exact DINOv2 preprocessing and clustering function."""
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("preann", ROOT / "master_scripts" / "master_script_dinov2_PreAnn.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _batches(paths, batch_size):
    for i in range(0, len(paths), batch_size):
        yield [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]


# ─── Embedders ────────────────────────────────────────────────────────────────
@torch.no_grad()
def embed_dinov2(paths, preproc, batch_size):
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE).eval()
    out = {"dinov2_mean": [], "dinov2_cls": [], "dinov2_cls_patch": []}
    for imgs in _batches(paths, batch_size):
        hs = model(**proc(images=imgs, return_tensors="pt", **preproc).to(DEVICE)).last_hidden_state.float()
        cls_tok, patch_mean = hs[:, 0], hs[:, 1:].mean(dim=1)
        out["dinov2_mean"].append(hs.mean(dim=1).cpu().numpy())
        out["dinov2_cls"].append(cls_tok.cpu().numpy())
        out["dinov2_cls_patch"].append(torch.cat([cls_tok, patch_mean], dim=1).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return {k: np.concatenate(v) for k, v in out.items()}


@torch.no_grad()
def embed_siglip(paths, batch_size):
    from transformers import AutoModel, AutoProcessor
    proc = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(DEVICE).eval()
    embs = [model.vision_model(pixel_values=proc(images=imgs, return_tensors="pt")["pixel_values"].to(DEVICE))
            .pooler_output.float().cpu().numpy() for imgs in _batches(paths, batch_size)]
    del model
    torch.cuda.empty_cache()
    return {"siglip_pooled": np.concatenate(embs)}


@torch.no_grad()
def embed_vjepa(paths, batch_size):
    if not VJEPA_CHECKPOINT.exists():
        print(f"  [SKIP] V-JEPA checkpoint not found: {VJEPA_CHECKPOINT}")
        return {}
    encoder, _ = torch.hub.load("facebookresearch/vjepa2", "vjepa2_1_vit_base_384", pretrained=False)
    ckpt = torch.load(VJEPA_CHECKPOINT, map_location="cpu", weights_only=False)
    encoder.load_state_dict({k.replace("module.", "").replace("backbone.", ""): v
                             for k, v in ckpt["ema_encoder"].items()}, strict=True)
    encoder.eval().to(DEVICE)
    embs = []
    for imgs in _batches(paths, batch_size):
        t = torch.stack([TRANSFORM_VJEPA(im) for im in imgs]).unsqueeze(2).to(DEVICE)  # [B, 3, 1, 384, 384]
        embs.append(encoder(t).mean(dim=1).float().cpu().numpy())
    del encoder
    torch.cuda.empty_cache()
    return {"vjepa_mean": np.concatenate(embs)}


@torch.no_grad()
def embed_dinov3(paths, batch_size):
    from transformers import AutoImageProcessor, AutoModel
    name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    proc = AutoImageProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(DEVICE).eval()
    embs = [model(**proc(images=imgs, return_tensors="pt").to(DEVICE)).last_hidden_state[:, 0].float().cpu().numpy()
            for imgs in _batches(paths, batch_size)]
    del model
    torch.cuda.empty_cache()
    return {"dinov3_cls": np.concatenate(embs)}


# ─── Evaluation ───────────────────────────────────────────────────────────────
def reference_groups(paths, mode):
    if mode == "none":
        return None
    if mode == "parent":
        return [p.parent.name for p in paths]
    if mode.startswith("prefix:"):
        n = int(mode.split(":", 1)[1])
        return [p.name[:n] for p in paths]
    raise ValueError(f"unknown --groups {mode}")


def evaluate(name, emb, labels, groups):
    mask = labels != -1
    n_clusters = len(set(labels[mask].tolist()))
    emb_n = normalize(emb)
    row = {
        "variant": name, "dim": int(emb.shape[1]), "n_clusters": n_clusters,
        "noise_pct": float((~mask).mean() * 100),
        "silhouette_own": (float(silhouette_score(emb_n[mask], labels[mask], metric="cosine", random_state=SEED))
                           if n_clusters >= 2 else None),
    }
    if groups is not None and len(set(groups)) >= 2:
        g = np.array(groups)
        row["silhouette_groups"] = float(silhouette_score(emb_n, g, metric="cosine", random_state=SEED))
        row["ari"] = float(adjusted_rand_score(g, labels))
        row["nmi"] = float(normalized_mutual_info_score(g, labels))
    return row


def pair_stats(emb, n_same):
    """First n_same rows = SAME frames, rest = DIFF frames."""
    sim = normalize(emb) @ normalize(emb).T
    s = np.array([sim[i, j] for i, j in itertools.combinations(range(n_same), 2)])
    d = np.array([sim[i, j] for i, j in itertools.combinations(range(n_same, len(emb)), 2)])
    auc = float((s[:, None] > d[None, :]).mean() + 0.5 * (s[:, None] == d[None, :]).mean())
    return {"same_cos": float(s.mean()), "diff_cos": float(d.mean()), "gap": float(s.mean() - d.mean()), "pair_auc": auc}


def list_images(folder):
    return sorted(q for q in Path(folder).rglob("*") if q.suffix.lower() in IMG_EXTS)


def embed_variants(paths, models, preproc, batch_size):
    variants = {}
    if "dinov2" in models:
        variants.update(embed_dinov2(paths, preproc, batch_size))
    if "siglip" in models:
        variants.update(embed_siglip(paths, batch_size))
    if "vjepa" in models:
        variants.update(embed_vjepa(paths, batch_size))
    if "dinov3" in models:
        variants.update(embed_dinov3(paths, batch_size))
    return variants


def run_pairs(args):
    same, diff = list_images(args.pairs[0]), list_images(args.pairs[1])
    if len(same) < 2 or len(diff) < 2:
        sys.exit("--pairs needs >= 2 images in each folder")
    print(f"{len(same)} SAME frames, {len(diff)} DIFF frames")
    variants = embed_variants(same + diff, args.models, load_preann().DINO_PREPROC, args.batch_size)
    rows = [{"variant": name, **pair_stats(emb, len(same))} for name, emb in variants.items()]
    print("\n| variant | mean cos SAME | mean cos DIFF | gap | pair AUC |\n|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['variant']} | {r['same_cos']:.3f} | {r['diff_cos']:.3f} | {r['gap']:+.3f} | {r['pair_auc']:.3f} |")
    print("\nCompare on pair AUC (scale-free); raw cosine ranges differ between models.")
    Path(args.out).write_text(json.dumps({"n_same": len(same), "n_diff": len(diff), "results": rows}, indent=2))
    print(f"[saved] {args.out}")


def main():
    p = argparse.ArgumentParser(description="Pre-annotation embedding benchmark")
    p.add_argument("--frames", help="Frames folder for the clustering benchmark")
    p.add_argument("--pairs", nargs=2, metavar=("SAME_DIR", "DIFF_DIR"),
                   help="Same-vs-different activity test instead of the clustering benchmark")
    p.add_argument("--groups", default="parent", help="parent | prefix:N | none")
    p.add_argument("--models", nargs="+", default=["dinov2", "siglip"], choices=["dinov2", "siglip", "vjepa", "dinov3"])
    p.add_argument("--n_components", type=int, default=128)
    p.add_argument("--min_cluster_size", type=int, default=25)
    p.add_argument("--min_samples", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--out", default="benchmark_preann.json")
    args = p.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if args.pairs:
        return run_pairs(args)
    if not args.frames:
        sys.exit("give --frames (clustering benchmark) or --pairs SAME_DIR DIFF_DIR")
    paths = list_images(args.frames)
    if not paths:
        sys.exit(f"no images under {args.frames}")
    groups = reference_groups(paths, args.groups)
    print(f"{len(paths)} frames" + (f", {len(set(groups))} reference groups ({args.groups})" if groups else ""))

    preann = load_preann()
    variants = embed_variants(paths, args.models, preann.DINO_PREPROC, args.batch_size)

    rows = []
    for name, emb in variants.items():
        print(f"\n[{name}] clustering ({emb.shape[1]}d)")
        # scene-only, exactly as PreAnn clusters (whole vector = scene block, no pose)
        labels, _, _ = preann.cluster_activities_with_pca_hdbscan(
            emb, n_pca_components=args.n_components, min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples, scene_dim=emb.shape[1], scene_weight=1.0, pose_weight=0.0)
        rows.append(evaluate(name, emb, np.asarray(labels), groups))

    cols = ["variant", "dim", "n_clusters", "noise_pct", "silhouette_own", "silhouette_groups", "ari", "nmi"]
    fmt = lambda v: "—" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
    print("\n| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    print("\nCompare variants on silhouette_groups / ARI / NMI (same reference groups for all); "
          "silhouette_own is scored on each variant's own clustering.")

    Path(args.out).write_text(json.dumps({"n_frames": len(paths), "params": vars(args), "results": rows}, indent=2))
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
