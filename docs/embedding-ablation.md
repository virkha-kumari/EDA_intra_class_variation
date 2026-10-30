# Finding the Right Embedding for Dataset Quality Checks

*An embedding ablation from the pre-annotation pipeline: raw frames where the scene is cluttered, several objects are present, and nothing tells the model where to focus.*

## The problem

Before training a detector on industrial fixed-camera video, I needed to answer a simple question: **are these frames worth annotating?** A folder of frames can look fine and still be a poor training set: near-duplicates, one viewpoint, missing lighting conditions, or too few distinct scenarios.

The plan: embed each frame with a frozen vision encoder, cluster the embeddings, and read the cluster structure as a proxy for diversity. The open question was which encoder and which pooling give clusters that track *real* scenario differences rather than incidental noise.

## Attempt 1 — DINOv2, mean of all tokens

The first try reused the pooling that already worked for object crops: average all 257 DINOv2 tokens (1 CLS + 256 patches) into one 768d vector.

On a tight crop this is fine, because the crop *is* the object. On a full frame the clusters did not follow the scenarios. Note what this vector actually is: with 256 patch tokens and 1 CLS token, the mean of all 257 is almost exactly the mean of the patch tokens. The global CLS summary contributes about 1/257 of it.

## Attempt 2 — SigLIP pooled output

SigLIP's pooled output, from an image-text contrastive model, looked better separated by eye. But clusters were less pure: frames that clearly belonged elsewhere showed up inside clusters, apparently because grouping followed broad scene semantics more than the finer differences between similar assembly scenarios.

## Attempt 3 — DINOv2, `[CLS ‖ mean(patch tokens)]`

Same encoder, different pooling: concatenate the **CLS token** (DINOv2's global summary of the frame) with the **mean of the patch tokens**, kept as separate halves of a 1536d vector.

What the patch half adds is **local, patch-level content**. DINOv2's patch tokens are trained to be informative on their own, so their average summarises *what is present across the frame's patches*: the objects, parts and textures. Averaging discards *where* they are; this is a bag of local features, not a spatial layout. The CLS half adds the global view that attempt 1 had diluted to 1/257 of the vector. Concatenating rather than averaging keeps both signals at full weight, so the clustering sees the global summary and the local content separately.

In my runs this gave visibly cleaner clusters than both the CLS-only and the SigLIP variants, with fewer misplaced frames in the montages.

## Numbers

Measured with [utils/test_embed_models.py](../utils/test_embed_models.py) on 4,175 industrial frames, using the pipeline's preprocessing (whole frame, no center crop).

**Same vs different activity.** 12 frames of one activity, 11 of different activities. Pair AUC is the probability that a same-activity pair is more similar than a different-activity pair; it is scale-free, so it compares across models.

| Variant | Same-activity cos | Different-activity cos | Gap | Pair AUC |
|---|---|---|---|---|
| DINOv2 mean (768d) | 0.946 | 0.620 | +0.326 | 0.964 |
| DINOv2 CLS (768d) | 0.910 | 0.449 | +0.461 | 0.941 |
| **DINOv2 CLS ‖ patch-mean (1536d)** | 0.919 | 0.489 | +0.430 | 0.948 |
| SigLIP pooled (768d) | 0.953 | 0.787 | +0.165 | 0.917 |

**Full set, clustered by the pipeline's own function.**

| Variant | Clusters | Noise (unclustered) | Silhouette (own clusters) |
|---|---|---|---|
| DINOv2 mean | 44 | 43.2% | 0.107 |
| DINOv2 CLS | 50 | 44.4% | 0.119 |
| **DINOv2 CLS ‖ patch-mean** | 45 | **39.4%** | 0.096 |
| SigLIP pooled | 38 | 51.7% | 0.124 |

What the numbers say:

- **SigLIP's purity problem is real and measurable.** It rates *different* activities at cosine 0.787, barely below same-activity pairs (0.953), so its same/different gap is 2–3× smaller than any DINOv2 variant's. That is the "frames that don't belong inside clusters" effect seen in the montages. It also leaves the most frames unclustered (51.7%).
- **CLS ‖ patch-mean keeps different activities far apart.** Different-activity cosine is 0.489 vs 0.620 for mean pooling, a margin of +0.430 vs +0.326 (about 32% larger; CLS-only +0.461). Mean pooling ranks pairs marginally better (AUC 0.964) only because it pulls every frame close together; AUC checks order, not distance.
- **CLS ‖ patch-mean clusters the most data:** 39.4% noise, about 510 more frames grouped than SigLIP and 150–200 more than the other DINOv2 poolings, at comparable cluster quality. Its activity separation beats CLS-only (0.948 vs 0.941).
- **Among DINOv2 poolings, separation is statistically tied on this small subset.** Mean pooling scores 0.964, but with 23 frames the differences are within noise. A larger labelled activity set would settle the ranking.
- Silhouette on each variant's *own* clustering is not a fair head-to-head. SigLIP's 0.124 is computed only on the 48% of frames it managed to cluster.

An earlier write-up quoted "+0.253 silhouette vs SigLIP". That number is withdrawn: it could not be reproduced from this repo's code, and silhouettes from different embedding spaces and clusterings are not directly comparable. The measurements above replace it.

## Takeaway

The lesson is not "DINOv2 beats SigLIP": attempt 1 used DINOv2 and lost. It is that **pooling has to match the input scale**. A crop and a full scene are different inputs:

- **Full frames** (pre-annotation): `[CLS ‖ patch-mean]`, 1536d. The global summary and the local content are kept separate.
- **Object crops** (post-annotation): mean of all tokens, 768d. The crop is the object, so the local content *is* the signal.

## What's next

The current open question is whether a similar pooling mismatch shows up with newer multimodal encoders — there's an in-progress workflow built on Qwen3-VL-Embedding that re-tests some of this with a model that natively does both image and text embedding, plus a captioning-based auto-labeling mode for clusters. If the same CLS/patch-separation logic turns out to matter there too, that's a stronger signal it's a general property of how scene-level vision-language models encode spatial information — not a quirk specific to DINOv2.