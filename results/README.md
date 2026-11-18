# Track A results (small-pool pilot)

Source: Colab runs of notebooks a1_a3, a4, a5_a8 on 2026-11-18.

**Data:** Amazon Reviews 2023 (Electronics), 234 images, 100 products (70 train / 15 val / 15 test, split by product).
**Test set:** 175 pairs = 105 same-source (edited copies), 35 same-product-other-photo (hard negatives), 35 random-other.
Edits are synthetic (our own transform function), so labels are *synthetic-edit gold*, not real fraud.

| File | What it is | Label |
|---|---|---|
| baselines_table.csv | pHash / ORB thresholds picked on val, evaluated on test | synthetic edits |
| ablation.csv | verifier with one feature group removed | synthetic edits |
| reliability.csv | calibration bins (only 3 populated) | synthetic edits |
| chain_experiment.csv | graph vs pairwise on 15 chains | SYNTHETIC customers |
| recall_at_k.csv | pretrained vs fine-tuned DINOv2, 35 test images | inconclusive (ceiling) |
| error_gallery.csv / .jpg | closest calls (rows marked false_positive/miss only if a real error) | synthetic edits |
| pairs*.csv, pool.json, splits.json | the pairs, pool and split used | - |
| artifacts/ | verifier.joblib + gallery.json for the /evidence endpoint | - |

## Caveats (read before quoting any number)
- Small pool: 15 test products, so confidence intervals are wide.
- ORB inliers alone reaches F1 0.99 (FPR 0); the full verifier reaches 1.0. The task is near-separable here.
- Graph vs pairwise differ by about one pair on 15 chains: not significant.
- Fine-tuning gave no gain (R@1 1.000 pretrained vs 0.971 fine-tuned, 35 queries).
- Not in git (kept on Drive): embedder.pt (88 MB), img_pool.zip.
- error_gallery.csv was produced before the labeling fix; re-run that cell after pulling to get hardest_negative / weakest_positive labels.
