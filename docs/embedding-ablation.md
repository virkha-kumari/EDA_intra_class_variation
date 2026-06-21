# Finding the Right Embedding for Dataset Quality Checks

*An embedding ablation from the EDA intra-class variation pipeline focusing on raw frames where environment is cluttered, multiple objects present, and no direction of where to actually focus.*

## The problem

Before training a detection model on industrial assembly-line video, I needed a way to answer a simple question: **are these frames actually worth annotating?** A folder of raw video frames can look fine at a glance and still be a bad training set — redundant near-duplicates, all shot from one angle, missing lighting conditions, or just not diverse enough to teach a model anything generalizable.

The plan was to embed each frame with a frozen vision encoder, cluster the embeddings, and read the cluster structure as a proxy for dataset diversity: few, tight clusters mean low diversity; many well-separated clusters mean the dataset actually covers different scenarios. The open question was which encoder and which pooling strategy would give clusters that reflected *real* scene differences rather than incidental noise.

## Attempt 1 — DINOv2, mean-pooled

The first thing I tried was the same pooling strategy I already used successfully for cropped object embeddings (post-annotation/pre-training) in the pipeline: average all 257 DINOv2 tokens (1 CLS + 256 patches) into a single 768-dimensional vector.

This works well on a tight crop, because a crop *is* the object — there's not much else in the frame to average away. On a full scene frame, it failed for the opposite reason: averaging over the whole image erases exactly the spatial variation that distinguishes one activity from another. A frame of someone fastening a bolt and a frame of someone scanning a barcode ended up looking similar once everything got flattened into one mean vector. The clusters that formed didn't track activity at all.

## Attempt 2 — SigLIP, pooled output

Next I tried SigLIP's pooled output, reasoning that a model trained with image-text contrastive objectives might have stronger scene-level semantics baked into its global representation.

This was a real improvement. The clusters were measurably better separated — SigLIP's scene-level understanding pulled apart genuinely different activities more cleanly than DINOv2's averaged tokens had. But a new problem showed up: **cluster purity**. Frames that obviously didn't belong kept showing up inside clusters they had no business being in. SigLIP seemed to be grouping by broad semantic category — "this looks like an indoor scene," "this looks outdoors" — without being sensitive enough to the texture and spatial layout differences that actually separate one assembly activity from a visually similar one.

Better separation, worse purity. Not a clean win.

## Attempt 3 — DINOv2, with a different pooling strategy

The fix turned out to be a different way of using the *same* encoder I'd already ruled out, not a different encoder. Instead of mean-pooling all 257 tokens, I concatenated just two things:

- the **CLS token** — DINOv2's summary of the whole scene
- the **mean of all patch tokens** — averaged separately, not folded into the CLS

This gives a 1536-dimensional vector where the first half answers "what is this scene overall" and the second half answers "what's actually here, spatially." Keeping them concatenated rather than further averaged turned out to matter: DINOv2's patch tokens are trained to be useful on their own (not just as supporting context for a global label), so the spatial detail they carry doesn't survive a second round of averaging.

The result: tighter clusters *and* better purity than the SigLIP run. Quantitatively, this `[CLS||avg_patches]` approach beat SigLIP's pooled output by **+0.253 in silhouette score** — a substantial improvement in how cleanly the embedding space separates within-class from cross-class structure. The purity gain is harder to put a single number on, but it's visible directly in the cluster montages: far fewer frames sitting in the wrong group.

## Why this matters beyond one pipeline

The takeaway isn't "DINOv2 is better than SigLIP" — attempt 1 used DINOv2 and lost to SigLIP. The takeaway is that **pooling strategy and input scale need to match**. A tight object crop and a full scene frame aren't the same kind of input, and forcing them through the same pooling logic throws away exactly the signal you need. This pipeline now deliberately uses two different strategies depending on where it sits in the pipeline:

- **Full scene frames** (pre-annotation, diversity checks): `[CLS||avg_patches]`, 1536d — separating global identity from spatial detail gives clustering more structure to work with on noisy, varied scenes.
- **Object crops** (pre/post-training, intra-class variation): mean-pooled, 768d — the crop *is* the object, so the extra dimensions from keeping CLS and patches separate just add noise without a clear gain.

This kind of "it depends on what you're embedding" answer doesn't show up if you only ever benchmark one encoder on one task. It only shows up by trying the wrong thing, understanding *why* it was wrong, and testing whether the fix generalizes.

## What's next

The current open question is whether a similar pooling mismatch shows up with newer multimodal encoders — there's an in-progress workflow built on Qwen3-VL-Embedding that re-tests some of this with a model that natively does both image and text embedding, plus a captioning-based auto-labeling mode for clusters. If the same CLS/patch-separation logic turns out to matter there too, that's a stronger signal it's a general property of how scene-level vision-language models encode spatial information — not a quirk specific to DINOv2.