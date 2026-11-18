import sys, os, io, gzip, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, cv2, requests
import pool
from build_pairs import build_pairs


def photo(seed):
    r = np.random.default_rng(seed)
    img = cv2.resize((r.random((24, 32, 3)) * 255).astype(np.uint8), (480, 360), interpolation=cv2.INTER_CUBIC)
    for _ in range(20):
        cv2.circle(img, (int(r.integers(0, 480)), int(r.integers(0, 360))), int(r.integers(5, 30)),
                   [int(c) for c in r.integers(0, 255, 3)], -1)
    return img


def test_folder_pool_pairs_and_group_split(tmp_path):
    for p in range(12):
        for k in range(3):
            cv2.imwrite(str(tmp_path / f"{1000 + p}_{k}.JPG"), photo(p * 10 + k))
    pl = pool.folder_pool(str(tmp_path), max_products=12)
    df = build_pairs(pl, n_variants=2)
    assert set(df.kind) == {"same_source", "same_product_other_photo", "random_other"}
    tr, va, te = pool.split_by_group(df)
    assert not (set(tr.group) & set(va.group)) and not (set(va.group) & set(te.group)) and not (set(tr.group) & set(te.group))


def test_amazon_pool_with_mocked_network(tmp_path, monkeypatch):
    recs = [{"parent_asin": f"B{p}", "user_id": f"U{p}{u}", "images": [{"large_image_url": f"http://x/{p}_{u}.jpg"}]}
            for p in range(5) for u in range(3)]

    class R:
        status_code = 200
        def __init__(s, content=b"", raw=None): s.content, s.raw = content, raw
        def raise_for_status(s): pass
        def __enter__(s): return s
        def __exit__(s, *a): pass

    def fake_get(url, **kw):
        if url.endswith(".jsonl.gz"):
            return R(raw=io.BytesIO(gzip.compress("\n".join(json.dumps(r) for r in recs).encode())))
        p, u = os.path.basename(url)[:-4].split("_")
        return R(content=cv2.imencode(".jpg", photo(int(p) * 10 + int(u)))[1].tobytes())

    monkeypatch.setattr(requests, "get", fake_get)
    ap = pool.amazon_pool("Fake", str(tmp_path), max_products=5)
    assert len(ap) >= 10 and len({x["product_id"] for x in ap}) == 5


def test_find_review_url_falls_back_to_second_source(monkeypatch):
    class R:
        def __init__(s, code): s.status_code = code
        def __enter__(s): return s
        def __exit__(s, *a): pass
    monkeypatch.setattr(requests, "get", lambda url, **kw: R(404 if "ucsd" in url else 200))
    assert "huggingface.co" in pool.find_review_url("Electronics")
    monkeypatch.setattr(requests, "get", lambda url, **kw: R(404))
    import pytest
    with pytest.raises(RuntimeError):
        pool.find_review_url("Electronics")
