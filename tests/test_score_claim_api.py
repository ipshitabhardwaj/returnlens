import sys, pathlib, random
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import cv2
import api, pool
from build_pairs import build_pairs
from graph_verifier import fit_verifier
from evidence import transform
from score_claim import EvidenceScorer, save_artifacts, load_scorer
from test_pool import photo
from fastapi.testclient import TestClient


def _make(tmp_path):
    for p in range(15):
        for k in range(3):
            cv2.imwrite(str(tmp_path / f"{1000 + p}_{k}.JPG"), photo(p * 10 + k))
    P = pool.folder_pool(str(tmp_path), max_products=15)
    clf = fit_verifier(build_pairs(P, n_variants=3))
    gallery = [dict(image_id=p["source_id"], path=p["path"], claim_id=f"G{i}", customer_id=f"cust{i}",
                    product_id=p["product_id"], is_catalogue=(i == 0)) for i, p in enumerate(P)]
    return P, clf, gallery


def test_scorer_flags_reuse_and_not_fresh_photo(tmp_path):
    random.seed(1)
    P, clf, gallery = _make(tmp_path)
    sc = EvidenceScorer(gallery, clf)
    reused = transform(cv2.imread(P[5]["path"]))[0]
    out = sc.score(reused, "NEW1", customer_id="someone_else")
    assert out["reuse_risk"] > 0.8 and out["cross_customer_reuse"] and out["matches"][0]["other_claim"] == "G5"
    fresh = sc.score(photo(9999), "NEW2", customer_id="x")
    assert fresh["reuse_risk"] < 0.5 and not fresh["cross_customer_reuse"]
    assert set(out) == {"claim_id", "reuse_risk", "evidence_uncertainty", "cross_customer_reuse", "catalogue_image", "matches"}


def test_api_uses_scorer_and_roundtrips_artifacts(tmp_path):
    P, clf, gallery = _make(tmp_path)
    sc = EvidenceScorer(gallery, clf)
    save_artifacts(sc, str(tmp_path / "art"))
    api.set_scorer(load_scorer(str(tmp_path / "art")))
    c = TestClient(api.app)
    q = tmp_path / "q.jpg"
    cv2.imwrite(str(q), transform(cv2.imread(P[3]["path"]))[0])
    j = c.post("/evidence", params={"claim_id": "N", "image_path": str(q), "customer_id": "z"}).json()
    assert j["reuse_risk"] > 0.8
    api.set_scorer(None)
    assert c.post("/evidence", params={"claim_id": "N"}).json()["reuse_risk"] == 0.05
