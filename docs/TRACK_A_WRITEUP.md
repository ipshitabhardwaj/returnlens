# Track A - Evidence Engine: method and results (small-pool pilot)

Owner: Ipshita. Status: pipeline complete and run end to end; numbers below are a pilot on 100 products and are NOT final.

## 1. Problem
Given a return-claim photo, decide whether it is a reused or edited copy of an earlier photo (another customer's upload, a catalogue image, or the same customer's earlier claim), and report how much to trust that decision. Output follows `contracts/evidence.json`: `reuse_risk`, `evidence_uncertainty`, `cross_customer_reuse`, `catalogue_image`, `matches`. Track A never rejects a claim; it only supplies evidence to the router.

## 2. Method
1. **Pair construction.** Positives: a photo and an edited copy of it (1-3 of resize, rotate, crop, JPEG, brightness, blur, border). Negatives: (a) hard negatives - a different photo of the same product, (b) random photos of other products. Splits are by product group, never by row, and thresholds are chosen on val and locked before test.
2. **Hand-built evidence features.** pHash similarity; ORB keypoints with ratio test (0.75) and RANSAC homography inliers (count and ratio); image quality (blur, minimum side).
3. **Embedding.** DINOv2-small CLS embedding, cosine similarity as a feature. Optional supervised-contrastive fine-tune (last 4 blocks + 128-d projection).
4. **Calibrated verifier.** Gradient-boosted trees (LightGBM on Linux, HistGradientBoosting on Windows) with isotonic calibration, so the output is a probability.
5. **Evidence uncertainty.** Computed from the query photo alone (blur, size), so a unique but low-quality photo is not penalised as if it were a match.
6. **Evidence graph.** Claims are nodes, verified matches are edges (p >= 0.8); connected components expose reuse chains across customers.

## 3. Data
Amazon Reviews 2023 (Electronics): 234 images, 100 products, split 70/15/15 by product. 1,170 pairs total (702 same-source, 234 same-product-other-photo, 234 random-other). Test set: 175 pairs (105 / 35 / 35).
**All positives are synthetic edits produced by our own transform function, and all customers in the chain experiment are synthetic.** This is not real fraud data.

## 4. Results (test split, n = 175)
| Method | Precision | Recall | F1 | Hard-neg FPR |
|---|---|---|---|---|
| pHash alone (thr 0.625) | 0.944 | 0.810 | 0.872 | 0.114 |
| ORB good matches (thr 20) | 0.981 | 0.971 | 0.976 | 0.057 |
| ORB RANSAC inliers (thr 13) | 1.000 | 0.981 | 0.990 | 0.000 |
| Inlier ratio (thr 0.833) | 1.000 | 0.943 | 0.971 | 0.000 |
| **Calibrated verifier (full)** | 1.000 | 1.000 | 1.000 | 0.000 |

Calibration (full verifier): Brier 0.0005, ECE 0.0048. Only 3 reliability bins are populated (68 pairs near 0, 105 near 1, 2 in between).

Ablation (one feature group removed): drop pHash F1 1.0; drop image quality F1 1.0; drop ORB/RANSAC F1 0.990 (one missed positive, Brier 0.0075); drop embedding F1 0.991 with hard-negative FPR 0.029 (1 of 35).

Retrieval, Recall@K on 35 test images: pretrained DINOv2 R@1 = 1.000; fine-tuned R@1 = 0.971; R@5 = R@10 = 1.0 for both.

Chain experiment (15 synthetic chains): graph recall 0.978 vs pairwise 0.956, precision 1.0 for both.

## 5. What the results do and do not show
- pHash is clearly the weakest signal (hard-negative FPR 11%). Geometric verification (ORB + RANSAC) is the strongest single signal and separates this dataset almost perfectly: test hard negatives have at most 12 inliers, while positives have a median of 522.
- The verifier's main contribution here is calibrated probabilities, not a large accuracy gain over a single ORB threshold. The embedding feature is what removes the one remaining hard-negative false positive in the ablation.
- Perfect scores come from a near-separable pilot. With 105 positives and 70 negatives in the test set, a perfect result is still compatible with a true error rate of a few percent (the 95% upper bound for 0 errors in 70 negatives is about 5%, and about 10% for 0 errors in 35 hard negatives).
- Fine-tuning did not help: pretrained DINOv2 already reaches R@1 = 1.0 on a 35-image gallery (ceiling effect), and the fine-tuned model was trained on only 70 products and fits its training set (loss about 3e-4).
- The graph vs pairwise gap is about one pair across 15 chains and is not significant.

## 6. Limitations
Synthetic edits and synthetic customers; small pool (15 test products); ORB alone nearly solves the task; no real-world reuse data; no evaluation on adversarial edits designed to defeat ORB (heavy crops, perspective warps, re-photographing a screen).

## 7. Next steps
Re-run on a larger pool (>= 400 products) with harder edits and a larger distractor gallery; add an out-of-distribution check (MVTec AD or Copydays); decide whether fine-tuning is worth keeping once the test set is large enough to separate the two embeddings.

## 8. Reproducing
`results/` holds all tables and the packaged scorer (`results/artifacts`). Notebooks: a1_a3 (pool, pairs, baselines), a4 (embedding fine-tune, GPU), a5_a8 (verifier, graph, evaluation). See `results/README.md` for caveats.
