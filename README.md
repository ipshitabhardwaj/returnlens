# ReturnLens (Track A: Evidence Engine, Ipshita)

Question answered: is this claim photo a reused or edited copy of another photo?

## Layout
- `src/evidence.py` transforms, pHash, ORB + RANSAC, quality features (step A2)
- `src/build_pairs.py` pair table with hard negatives (A1)
- `src/baselines.py` thresholds chosen on val, hard-negative FPR (A3)
- `src/embed_ft.py` DINOv2-small contrastive fine-tune (A4, GPU, syntax-checked only, smoke-test first)
- `src/graph_verifier.py` calibrated verifier, uncertainty, evidence graph (A5, A6)
- `src/api.py`, `src/router.py` shared stub API and router (integration)
- `contracts/` JSON shapes agreed with the team
- `notebooks/` Colab notebooks, `results/` tables and figures (label each real/weak/gold/synthetic)

## Run
    pip install -r requirements.txt
    pytest -q tests

Branch: `track-a`. Split train/val/test by product group, never by row. Lock thresholds on val before touching test.
