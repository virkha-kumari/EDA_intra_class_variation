"""
Qwen3-VL test: visual embeddings + captions + cross-modal retrieval.
Run from project root:  python utils/test_qwen.py

Phase 1: Visual-embed CHECK_DIR frames (retrieval pool).
Phase 2: Visual-embed + caption FRAMES_DIR frames (query set).
Phase 3: Text-embed each caption (same model, text-only) → retrieve top-1 from CHECK_DIR pool.
"""

import gc
import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoProcessor, AutoModel, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# ── CONFIG ────────────────────────────────────────────────────────────────────

FRAMES_DIR = Path(os.environ.get("EDA_TEST_FRAMES", "test_frames"))   # query frames
CHECK_DIR  = Path(os.environ.get("EDA_CHECK_FRAMES", "check_frames"))  # retrieval pool
MAX_FRAMES = 20
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# Auto-select models based on available VRAM
# ≥16GB (training server): 8B embedding + 8B-Instruct caption
# <16GB (8GB laptop):       2B embedding + 3B-Instruct caption (sequential, fits in 8GB)
_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
if _vram_gb >= 20:
    EMBED_MODEL   = "Qwen/Qwen3-VL-Embedding-8B"
    CAPTION_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
else:
    EMBED_MODEL   = "Qwen/Qwen3-VL-Embedding-2B"
    CAPTION_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

print(f"VRAM detected: {_vram_gb:.1f}GB → embed={EMBED_MODEL.split('/')[-1]}, caption={CAPTION_MODEL.split('/')[-1]}")

CAPTION_INSTRUCTION_SINGLE = (
    "Industrial assembly CCTV. Describe: worker present or not, posture, action on assembly, assembly completeness. "
    "Skip clothing and background. One sentence."
)

CAPTION_INSTRUCTION_DUAL = (
    "Two CCTV views stitched side by side. Describe: workers present or not, posture, action on assembly, assembly completeness. "
    "Skip clothing and background. One sentence."
)

# Pixel budget tuned for 640x480 and 1280x480 frames
VIDEO_MIN_PIXELS = 128 * 28 * 28   # 100,352 — covers 640x480
VIDEO_MAX_PIXELS = 800 * 28 * 28   # 627,200 — covers 1280x480


# ── LOAD FRAMES ───────────────────────────────────────────────────────────────

def load_frames(directory, limit):
    paths = sorted(directory.glob("*.jpg"))[:limit] + sorted(directory.glob("*.png"))[:limit]
    paths = sorted(set(paths))[:limit]
    return paths

frame_paths = load_frames(FRAMES_DIR, MAX_FRAMES)
check_paths = load_frames(CHECK_DIR, MAX_FRAMES)

if not frame_paths:
    raise FileNotFoundError(f"No frames found in {FRAMES_DIR.resolve()}")
if not check_paths:
    raise FileNotFoundError(f"No frames found in {CHECK_DIR.resolve()}")

print(f"Query frames  (FRAMES_DIR): {len(frame_paths)} in {FRAMES_DIR}")
print(f"Pool frames   (CHECK_DIR):  {len(check_paths)} in {CHECK_DIR}")

# ── PHASE 1: VISUAL EMBEDDINGS ────────────────────────────────────────────────

DEFAULT_INSTRUCTION = "Represent the user's input."
RETRIEVAL_INSTRUCTION = "Retrieve images relevant to the given description."


def _eos_pool(hidden, attn_mask):
    # flip mask → find last valid token position per sequence (batch-safe)
    last_idx = int(attn_mask[0].sum().item()) - 1 if attn_mask is not None else hidden.shape[1] - 1
    return hidden[0, last_idx, :]


def visual_embed(paths, model, processor):
    """Image-side embeddings — no instruction, images are documents not queries."""
    embs = []
    with torch.no_grad():
        for p in paths:
            img = Image.open(p).convert("RGB")
            messages = [
                {"role": "system", "content": [{"type": "text", "text": DEFAULT_INSTRUCTION}]},
                {"role": "user", "content": [{"type": "image", "image": img}]},
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, return_tensors="pt", padding=True,
                min_pixels=VIDEO_MIN_PIXELS, max_pixels=VIDEO_MAX_PIXELS,
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            outputs = model(**inputs)
            emb = _eos_pool(outputs.last_hidden_state, inputs.get("attention_mask"))
            embs.append(emb.cpu().float().numpy())

            del inputs, outputs
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    embs = np.array(embs, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.where(norms == 0, 1.0, norms)


def text_embed(texts, model, processor, instruction=RETRIEVAL_INSTRUCTION):
    """Text-side embeddings — instruction-prefixed for retrieval alignment."""
    embs = []
    with torch.no_grad():
        for t in texts:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": instruction}]},
                {"role": "user", "content": [{"type": "text", "text": t}]},
            ]
            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            inputs = processor(text=[text_input], return_tensors="pt", padding=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            outputs = model(**inputs)
            emb = _eos_pool(outputs.last_hidden_state, inputs.get("attention_mask"))
            embs.append(emb.cpu().float().numpy())

            del inputs, outputs
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    embs = np.array(embs, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.where(norms == 0, 1.0, norms)


print(f"\n[Phase 1] Loading embedding model: {EMBED_MODEL}")
embed_processor = AutoProcessor.from_pretrained(EMBED_MODEL)
embed_model = AutoModel.from_pretrained(
    EMBED_MODEL,
    dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
).to(DEVICE).eval()
print(f"  Model on device: {next(embed_model.parameters()).device}")

print(f"  Embedding CHECK_DIR pool ({len(check_paths)} frames)...")
check_embeddings = visual_embed(check_paths, embed_model, embed_processor)
print(f"  CHECK_DIR embeddings: {check_embeddings.shape}")

print(f"  Embedding FRAMES_DIR queries ({len(frame_paths)} frames)...")
embeddings = visual_embed(frame_paths, embed_model, embed_processor)
print(f"  FRAMES_DIR embeddings: {embeddings.shape}")

# Free embed model before loading caption model
del embed_model, embed_processor
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()
print("  Embed model freed.")

# ── PHASE 2: CAPTIONS ─────────────────────────────────────────────────────────

print(f"\n[Phase 2] Loading caption model: {CAPTION_MODEL}")
caption_processor = AutoProcessor.from_pretrained(CAPTION_MODEL)
caption_model = Qwen3VLForConditionalGeneration.from_pretrained(
    CAPTION_MODEL,
    dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    low_cpu_mem_usage=True,
).to(DEVICE).eval()
print(f"  Model on device: {next(caption_model.parameters()).device}")

captions = []

with torch.no_grad():
    for p in frame_paths:
        img = Image.open(p).convert("RGB")
        instruction = CAPTION_INSTRUCTION_DUAL if img.width > 640 else CAPTION_INSTRUCTION_SINGLE
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        text = caption_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = caption_processor(
            text=[text], images=image_inputs, return_tensors="pt",
            min_pixels=VIDEO_MIN_PIXELS, max_pixels=VIDEO_MAX_PIXELS,
        ).to(DEVICE)
        inputs.pop("token_type_ids", None)

        generated_ids = caption_model.generate(**inputs, max_new_tokens=128, do_sample=False)
        trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
        caption = caption_processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        captions.append(caption)

        del inputs, generated_ids
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

del caption_model, caption_processor
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()
print("  Caption model freed.")

# ── PHASE 3: TEXT-EMBED CAPTIONS → RETRIEVE FROM CHECK_DIR ───────────────────

print(f"\n[Phase 3] Loading embed model again for text embedding: {EMBED_MODEL}")
embed_processor = AutoProcessor.from_pretrained(EMBED_MODEL)
embed_model = AutoModel.from_pretrained(
    EMBED_MODEL,
    dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
).to(DEVICE).eval()
print(f"  Model on device: {next(embed_model.parameters()).device}")

print(f"  Text-embedding {len(captions)} captions...")
caption_embeddings = text_embed(captions, embed_model, embed_processor)
print(f"  Caption embeddings: {caption_embeddings.shape}")

del embed_model, embed_processor
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()
print("  Embed model freed.")

# ── RESULTS ───────────────────────────────────────────────────────────────────

print("\n=== Captions ===")
for p, cap in zip(frame_paths, captions):
    print(f"  {p.name}:\n    {cap}\n")

print("\n=== Cross-Modal Retrieval: caption → CHECK_DIR frame ===")
# caption_embeddings (N_query, D) vs check_embeddings (N_pool, D)
retrieval_sims = cosine_similarity(caption_embeddings, check_embeddings)  # (N_query, N_pool)

for i, (qpath, cap) in enumerate(zip(frame_paths, captions)):
    sims = retrieval_sims[i]
    top5_idx = np.argsort(sims)[::-1][:5]
    print(f"  QUERY  : {qpath.name}")
    print(f"  CAPTION: {cap}")
    for rank, idx in enumerate(top5_idx):
        print(f"  TOP-{rank+1} : {check_paths[idx].name}  (sim={sims[idx]:.3f})")
    print()

print("\n=== Visual Similarity Matrix (FRAMES_DIR internal) ===")
sim_matrix = cosine_similarity(embeddings)
print(np.round(sim_matrix, 3))

THRESHOLD_SAME    = 0.85
THRESHOLD_SIMILAR = 0.65

avg_sim = sim_matrix[np.triu_indices(len(frame_paths), k=1)].mean()
print(f"\nAverage pairwise visual similarity: {avg_sim:.3f}")
if avg_sim > 0.85:
    print("Overall: Single continuous activity")
elif avg_sim > 0.65:
    print("Overall: Related workflow steps")
else:
    print("Overall: Multiple different activities")