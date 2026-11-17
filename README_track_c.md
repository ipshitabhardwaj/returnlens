# ReturnLens: Track C (Festival Forecaster + Router, Rajan)

Question answered: how many returns tomorrow, and how many claims can we physically inspect?

Branch: `track-c`. Platform: Kaggle Notebook. Output contract: `GET /forecast` returns `date, p10, p50, p90, inspection_capacity`.

## Files (src/)
- `retail_returns.py` (Listing C-1): daily sales and returns from Online Retail II (step C1)
- `forecast.py` (Listing C-2): festival and sale calendar, event features, LightGBM quantiles, wQL, conformal widening, rolling-origin CV (steps C2-C4)
- `nn_quantile.py` (Listing C-3): quantile GRU, syntax-checked only (step C6)
- `event_reasoner.py` (Listing C-4): LLM event reasoner and skip gate (step C7)
- `router.py`, `api.py`: shared router and stub API (from the integration listings, not edited here)

## Environment fingerprint (Kaggle, CPU session, 17 Nov 2026)
Python 3.13.18 | torch 2.11.0+cpu | CUDA available: False

| Package | Version |
|---|---|
| chronos-forecasting | 2.3.2 |
| lightgbm | 4.6.0 |
| holidays | 0.107 |
| pandas | 2.3.3 |
| transformers | 5.20.2 |
| torch | 2.11.0+cpu |
| torchvision | 0.26.0+cpu |
| torchaudio | 2.11.0+cpu |
| sentence-transformers | 5.7.0 |
| pytorch-lightning | 2.6.6 |
| torchmetrics | 1.9.0 |
| geopandas | 1.2.0 |

Sanity check: `holidays.India(years=2025)` lists Diwali on 2025-10-20; `years=2026` lists 2026-11-08.

## Notes
- This fingerprint is from a CPU session. Re-run the fingerprint cell on a GPU session before the GPU stretch items (Chronos, GRU, event reasoner) and update the torch line.
- Sale-event dates in `forecast.py` (SALES) are placeholders until verified from two sources.
- Every results table must be labelled real / weak / gold / synthetic.
