import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
import graph_verifier as gv


def test_falls_back_when_lightgbm_native_lib_crashes(monkeypatch):
    class Broken:
        def __init__(self, *a, **k): pass
        def get_params(self, deep=True): return {}
        def fit(self, *a, **k): raise OSError("exception: access violation reading 0x0")
    monkeypatch.setattr(gv.lgb, "LGBMClassifier", Broken)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f: rng.random(120) for f in gv.FEATS})
    df["y"] = (df.phash_sim > 0.5).astype(int)
    clf = gv.fit_verifier(df)
    assert clf.predict_proba(df[clf.feats_]).shape == (120, 2)
