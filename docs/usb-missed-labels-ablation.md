# USB (Use-Seen-Before) — Finding Missed Labels Without a Detector

*How a failed auto-annotation effort turned into a much smaller, much dumber algorithm that actually works for the one job I needed.*

## The problem

The pre-training EDA pipeline in this repo answers "how heterogeneous is each class?" — crop every annotated bbox, embed with DINOv2, cluster with DBSCAN, read the outliers and sub-clusters. That works. But it only ever looks at **boxes that exist**. It is structurally blind to the most expensive annotation error there is: an object that is present in the frame and simply never got labelled.

Missed labels are worse than wrong labels. A mislabelled box gets flagged as an outlier in its own class cluster — the pipeline already catches those. A *missing* box produces no crop, no embedding, no cluster membership, no outlier. It shows up nowhere. It just quietly teaches the detector that this particular patch of pixels is background, and the model learns to suppress the very thing you wanted it to find.

So: given an annotated dataset, find the frames where a human annotator probably missed an instance of class X.

## Attempt 1 — throw a detector at it (`Auto_Annotation_YOLO_formatted`)

The obvious framing: missed-label detection is auto-annotation. If I can auto-annotate the dataset and diff those predictions against the human labels, whatever the model finds that the human didn't is a missed-label candidate. I already had a whole repo for this — [Auto_Annotation_YOLO_formatted](https://github.com/virkha-kumari/Auto_Annotation_YOLO_formatted) — built over ~3 months for exactly the adjacent problem: label 10–15 instances per class, auto-annotate thousands of frames.

That repo went through a long chain of proposal generators, and every one of them is documented there with the reason it failed:

- **SAM2 auto mask generation** — thousands of masks per image, zero useful ones on industrial scenes. Auto-mask works on natural images with strong texture-contrast at object boundaries. Large uniform-colour industrial objects give it nothing to latch onto.
- **SAM3 auto mask generation** — same architectural premise, same failure. Dropped.
- **OWLv2 image-guided detection** — tested at 640px, 1008px, 1280px, per-crop and batched. On a large uniform object covering ~60% of the frame, *every single proposal* was a tiny box at the image edge. Root cause is architectural: OWLv2 is a patch-based ViT that matches 16×16 tile-level texture and cannot compose a box spanning many tiles. Not a tuning problem.
- **YOLOe visual-prompt detection** — finally worked as a proposal generator. Multi-scale feature pyramid, genuine native few-shot via `get_vpe(refer_image)`. Confirmed well-localized proposals at conf ≥ 0.06.

So Pipeline A became YOLOe → SAM2 (masking) → DINOv2 (similarity scoring) → WBF → containment filter. Pipeline B later replaced the front half with SAM3 canvas-composite few-shot prompting (SAM3 has no cross-image exemplar API, so reference and target get pasted into one shared canvas — the [FSS-SAM3](https://github.com/WongKinYiu/FSS-SAM3) trick). Both pipelines work end-to-end. Neither is good enough.

**The numbers, on Construction-PPE:**

| | Pipeline A (50 img, 7 classes w/ GT) | Pipeline B (150 img, 11 classes) |
|---|---|---|
| mAP@.50 | 0.254 | 0.147 |
| Best class | Person, F1=0.690 | Person, F1=0.725 |
| Worst | goggles, prec=0.023 (43 preds, ~1 TP) | no_boots, prec=0.004 (540 preds, 53 GT) |

Person and vest are usable. Helmet at F1=0.46 is marginal. Gloves at F1=0.11, boots at 0.21, goggles at prec=0.023 are not — for those, **correcting the auto-annotations costs more than labelling from scratch**, and goggles is actively harmful (43 false positives to buy one true positive; misses can be skipped, FPs have to be actively deleted).

The failure isn't a threshold that needs tuning. Standard YOLO trained on the same dataset gets mAP@.50 > 0.85. YOLOe visual-prompt mode gets 0.254 on the same backbone. The gap is entirely in the classification head — a trained YOLO has learned per-class decision boundaries from thousands of examples; YOLOe's VPE is a single mean-pooled reference vector compared by cosine distance. For visually ambiguous classes (gloves vs bare hands, goggles vs glasses, boots vs shoes) that's a hard architectural ceiling.

And the cost: ~20 minutes for 50 targets × 10 classes on an RTX 4060 8GB, three models loaded and unloaded sequentially, YOLOe's Phase 1 scaling as `O(stems × ceil(targets/batch))`. Not practical past ~200–300 targets.

## The realisation — I was solving a much harder problem than I needed to

Every one of those approaches was trying to answer: **"where is this object, in this image, drawn from scratch?"** That's full localization from a cold start. It's genuinely hard, and it's what all the failures above have in common.

But that is not the question I actually needed answered. My question especially for EDA-integration is: **"does this image probably contain an unlabelled instance of class X?"**

That's a *flagging* problem, not a localization problem. I don't need a tight box — a human is going to open the frame and draw the real box anyway. I need a strong enough signal to say "look at this one." And crucially, I already have something none of the general-purpose detectors get to use: **a fully annotated dataset of exactly this class, from exactly this camera.**

The dataset is CCTV. Fixed camera, no pan, no zoom, no crop variance across frames. Which means the class doesn't appear in arbitrary places — it appears, over and over, in the same handful of image regions. A detector rediscovers that geometry from scratch on every single frame. That's the expensive part, and it's the part that keeps failing.

So don't rediscover it. **Reuse it.** Where the class has been seen before is where to look for it again.

## Attempt 2 — USB (Use-Seen-Before)

The core idea is to replace learned localization with *remembered* localization. Instead of a model proposing boxes, the existing ground-truth boxes propose the boxes. Then one embedding comparison decides whether each proposed region actually contains the object this time.

This is deliberately imitating **active learning**: the pipeline's job is not to annotate, it's to rank frames by "how likely is a human to find something here" and hand that queue to a human. Flagging is the entire deliverable.

### The algorithm

**1. User picks the suspect classes.** Explicit, not automatic. The annotator usually knows which classes are error-prone — small ones, ones added late in the labelling pass, ones with an ambiguous definition. No point scanning all 34 classes when three are suspect.

**2. Collect every ground-truth box for that class**, across all splits (train/val/test), as normalized YOLO coordinates.

**3. Greedy-dedup them into `seen_<cls>_regions`.** Walk the boxes; keep one only if its IoU with every already-kept region is below 0.8. What survives is the small set of distinct places this class has ever occupied. On a fixed camera, thousands of GT boxes typically collapse to a handful of regions.

**4. Pad each region by 2% of image width/height** per side. Not box-relative padding (which is what the auto-annotation repo's crop extractor uses) — image-relative, so every region gets the same absolute slack regardless of its size. A tight GT box that clipped a fingertip or an edge shouldn't produce a candidate crop that clips it too.

**5. Crop every region out of every image in the dataset.** All of them, labelled or not — a missed instance can perfectly well sit in a frame that has other classes correctly labelled.

**6. Suppress candidates that overlap existing ground truth.** If a candidate region has IoU ≥ 0.8 with a GT box *of the same class in that same image*, that instance is already labelled. Drop it. This is what keeps the output list to genuine misses instead of re-reporting the entire dataset back at you.

**7. Build the reference pool from the class's own embeddings.** This is where USB gets to reuse the EDA pipeline it lives inside. Script 2 has already embedded every GT crop of this class with DINOv2 (768d, mean-pooled). Re-running DBSCAN on those cached embeddings (cheap — pure array math, no GPU, no image I/O) gives cluster structure, and from it:
   - **every noise point** (DBSCAN label −1) — these are the genuinely unusual-looking instances, and a missed instance is disproportionately likely to be unusual-looking
   - **per cluster: the centroid** — the canonical appearance
   - **per cluster: the N points farthest from that centroid** by cosine distance (default 4, tunable) — the edges of each cluster

   Just using centroids would be the averaged-prototype mistake the auto-annotation repo already made and documented: averaging diverse crops "washes out the discriminative signal — the average lives in a region of embedding space that no individual crop actually occupies." Centroid + farthest points + noise covers typical *and* edge appearance without collapsing them together.

**8. Embed the candidate crops and score them.** Same DINOv2 function Script 2 uses — literally imported from it, not reimplemented, because the reference embeddings and the candidate embeddings must live in the same subspace or the cosine similarities are meaningless. (The auto-annotation repo learned this one the hard way too: *"refs and proposals for each class must use the same embedding method."*) Score each candidate as max cosine similarity against the reference pool — per-crop max, not against an average.

**9. Flag anything above the per-class threshold** (default 0.8, tunable per class because visually tight classes and visually variable classes need different cutoffs). Output is the full frame with the flagged region drawn in red with its similarity score, plus existing GT boxes of that class in green for context, written to `missing_labels/<class>/<image>.jpg`.

### What this buys

| | Detector approach | USB |
|---|---|---|
| Localization | learned, from scratch, per frame | remembered, from existing GT |
| Models loaded | YOLOe + SAM2 + DINOv2 (sequential, 8GB juggling) | DINOv2 only |
| Reference embeddings | computed fresh | reused from Script 2 cache |
| Clustering | — | reused math from Script 3, in-process, no montages |
| Output | boxes claimed to be correct | frames flagged for human review |
| Fails when | class is visually ambiguous | camera moves |

The honest framing: USB is **not a better detector**. It is a worse detector that solves a different, smaller problem. It cannot find an instance in a place the class has never been annotated before, and it will produce garbage the moment the camera moves. Those are real limitations and they're the price of skipping the localization problem entirely.

But for a fixed-camera dataset where a human just needs to know which frames to re-open, the trade is overwhelmingly worth it — and both of the things that made the detector approach fail (learned localization on ambiguous classes, and the cost of running three models over every frame) are simply not in the loop.

## Is any of this novel?

Not the individual pieces, and it would be dishonest to claim otherwise:

- **Prototype/exemplar matching** — scoring candidates by cosine similarity against a bank of reference embeddings is standard few-shot practice (ProtoNet-style), and the auto-annotation repo's own `build_proto_banks`/`score_dets_dinov2` does exactly this.
- **Template matching with learned features** — classic CV, just with DINOv2 embeddings where it used to be raw pixels. Location priors on fixed cameras are older than deep learning.
- **Active learning** — flagging high-uncertainty or high-confidence-unlabelled samples for human review is a well-established annotation workflow.

What is specific here is the combination and where it's aimed: using the dataset's **own ground-truth box geometry as the region proposer** to sidestep detection entirely, with a reference pool deliberately sampled for diversity (noise + centroid + farthest points) rather than averaged, applied to missed-label auditing rather than annotation. Searching the prior auto-annotation work turned up no version of this recipe — every approach there is model-driven localization.

It's a "jugaad". It works because it refuses to solve the hard problem.

## Where it lives

USB ships as `postannotation_scripts/5. usb_missed_labels.py`, wired into `master_scripts/master_script_dinov2_PostAnn_PreTrain.py` as an optional Step 5 — after the PDF report, before the intermediate crops get deleted (it needs those cached `.npy` embeddings). There's also a USB-only mode that skips clustering and PDF generation entirely and runs just crops → embeddings → USB for the selected classes.

## What's next

The main open question is threshold behaviour across class types. 0.8 is a starting default, not a validated number — visually tight classes and visually variable ones will almost certainly want different cutoffs, and the right way to find them is to look at the score distribution of flagged-vs-suppressed candidates per class rather than guessing. Worth measuring: precision of the flagged set against a hand-audited sample, since the only metric that matters here is what fraction of flagged frames actually contain something a human wants to draw.

The other open question is what happens on a class whose `seen_<cls>_regions` set doesn't collapse — if a class genuinely appears all over the frame, the region count stops being a handful and the candidate-crop count grows toward "every image × many regions," which puts the embedding cost back on the table. A cap or a smarter region-merge would be the fix, but it needs a real dataset that exhibits the problem first.
