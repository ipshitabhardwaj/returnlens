# Track A status (Ipshita) - what exists, what is tested, what is left

Tested = executed on synthetic images in CI-style pytest (`pytest -q tests`, 6 tests). Nothing has been run on real photos or on a GPU.

| Step | Files | Status |
|---|---|---|
| A1 pool + pairs | `src/pool.py`, `src/build_pairs.py`, notebook `a1_a3` | code tested (synthetic, mocked network). Real Amazon / SOP downloads NOT run |
| A2 features | `src/evidence.py`, `a1_a3` eyeball cell | tested |
| A3 baselines | `src/baselines.py`, `a1_a3` | tested |
| A4 embedding fine-tune | `src/embed_ft.py`, `src/retrieval.py`, notebook `a4` | `retrieval.recall_at_k` tested; **torch training + DINOv2 loading never run** (smoke test first) |
| A5 verifier + calibration | `src/graph_verifier.py`, `src/evaluate.py`, `a5_a8` | tested |
| A6 evidence graph + chain experiment | `graph_verifier.py`, `evaluate.chain_experiment` | tested (synthetic customers) |
| A7 endpoint | `src/score_claim.py`, `src/api.py` | tested with FastAPI test client |
| A8 ablations, error gallery, demo | `evaluate.py`, `app_track_a.py`, `a5_a8` | tested (demo UI not launched) |

## Run order
1. `a1_a3_pool_pairs_baselines.ipynb` (CPU ok). Produces pairs.csv, pool.json, splits.json, img_pool.zip on Drive.
2. `a4_embedding_finetune.ipynb` (GPU). Smoke test, full run, Recall@K table.
3. `a5_a8_verifier_graph_eval.ipynb` (GPU for embeddings). Verifier, ablation, chain experiment, error gallery, scorer artifacts.
4. Locally: `RETURNLENS_ARTIFACTS=<artifacts folder> uvicorn src.api:app`  (run from `src/`), and `streamlit run app_track_a.py`.

## Still yours (cannot be generated)
- Real data download and a first real baselines table; look at the matches before trusting numbers.
- Write one-line reasons for each pair in the error gallery.
- Hand-label nothing here (gold set belongs to Track B), but Track A needs a few real "same-defect, different customer" examples if you want a stronger hard-negative test.
- Method + results writing (about 4 h), Evidence tab wiring into the shared Streamlit app.
- Optional: INRIA Copydays check, MVTec hard negatives (not wired into the notebook).

## Known caveats
- On synthetic images every metric is ~1.0; ignore it. Real photos decide.
- `emb_cos` is computed at pair-build time, so changing the embedder means rebuilding the pairs (the a5_a8 notebook does this).
- `evidence_uncertainty` in the scorer comes from the query photo only (`query_uncertainty`), not from match quality.
