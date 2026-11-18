import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, cv2
import pool, evaluate as ev, retrieval
from build_pairs import build_pairs
from graph_verifier import fit_verifier
from test_pool import photo


def _emb(img):
    v = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (16, 16)).astype(np.float32).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-9)


def test_pipeline_pieces(tmp_path):
    for p in range(12):
        for k in range(3):
            cv2.imwrite(str(tmp_path / f"{1000 + p}_{k}.JPG"), photo(p * 10 + k))
    P = pool.folder_pool(str(tmp_path), max_products=12)
    df = build_pairs(P, n_variants=2, emb_fn=_emb)
    assert {"emb_cos", "path_a", "path_b"} <= set(df.columns)
    tr, va, te = pool.split_by_group(df)
    ab = ev.ablation(tr, va, te)
    assert list(ab.variant)[0] == "full" and len(ab) == 5
    clf = fit_verifier(tr)
    p = ev.score_split(clf, te)
    assert 0 <= ev.ece(te.y, p) <= 1
    assert len(ev.error_gallery(te, p, k=2)) == 4
    assert retrieval.recall_at_k(np.eye(5), np.eye(5), range(5))["R@1"] == 1.0
    assert set(ev.chain_experiment(P, clf, emb_fn=_emb, n_chains=2).index) == {"graph", "pairwise"}
