"""Track A - build the image pool: Amazon review photos (grouped by product) and Stanford Online Products."""

import gzip, io, os, glob, random, requests
import numpy as np, pandas as pd, cv2

REVIEW_URL = (
    "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/{cat}.jsonl.gz"
)


def stream_jsonl_gz(url, max_lines):
    """Yield the first max_lines JSON lines of a remote .jsonl.gz without downloading the whole file."""
    import json

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with gzip.GzipFile(fileobj=r.raw) as gz:
            for i, line in enumerate(io.TextIOWrapper(gz, encoding="utf-8")):
                if i >= max_lines:
                    break
                yield json.loads(line)


def _save_image(url, path, max_side=512, timeout=15):
    """Download one image, shrink to max_side, save as JPEG. Returns True on success."""
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        if img is None or min(img.shape[:2]) < 120:
            return False
        s = max_side / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return True
    except Exception:
        return False


def amazon_pool(cat, out_dir, max_lines=300_000, max_products=400, per_product=(2, 4), seed=0):
    """Pool of review photos. Products with >=2 photos from DIFFERENT reviews give natural hard negatives.

    Returns list of dict(source_id, product_id, path). Dead URLs are skipped silently.
    """
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    cand = {}  # parent_asin -> list of (review_key, url)
    for k, rec in enumerate(stream_jsonl_gz(REVIEW_URL.format(cat=cat), max_lines)):
        imgs = rec.get("images") or []
        if not imgs or not rec.get("parent_asin"):
            continue
        url = imgs[0].get("large_image_url") or imgs[0].get("medium_image_url")
        if url:
            cand.setdefault(rec["parent_asin"], []).append((f"{rec.get('user_id', 'u')}_{k}", url))
    prods = [p for p, v in cand.items() if len(v) >= per_product[0]]
    random.shuffle(prods)
    pool = []
    for p in prods:
        if len({x[0].split("_")[0] for x in cand[p]}) < per_product[0]:
            continue  # need photos from different customers
        got = []
        for key, url in cand[p][: per_product[1] * 2]:
            path = os.path.join(out_dir, f"{p}_{len(got)}.jpg")
            if _save_image(url, path):
                got.append(dict(source_id=f"{p}_{len(got)}", product_id=p, path=path))
            if len(got) >= per_product[1]:
                break
        if len(got) >= per_product[0]:
            pool += got
        if len({x["product_id"] for x in pool}) >= max_products:
            break
    return pool


def folder_pool(root, max_products=400, per_product=(2, 4), seed=0):
    """Pool from a local folder where files are named <productid>_<n>.<ext> (Stanford Online Products layout).

    Searches subfolders; use this after unzipping SOP.
    """
    random.seed(seed)
    by = {}
    for f in glob.glob(os.path.join(root, "**", "*.*"), recursive=True):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            by.setdefault(os.path.basename(f).rsplit("_", 1)[0], []).append(f)
    prods = [p for p, v in by.items() if len(v) >= per_product[0]]
    random.shuffle(prods)
    pool = []
    for p in prods[:max_products]:
        for f in sorted(by[p])[: per_product[1]]:
            pool.append(dict(source_id=os.path.basename(f), product_id=p, path=f))
    return pool


def split_by_group(df, fracs=(0.7, 0.15, 0.15), seed=0):
    """Split train/val/test so every product (group) lives in exactly one split."""
    rng = np.random.default_rng(seed)
    groups = rng.permutation(np.array(list(df["group"].unique()), dtype=object))
    n = len(groups)
    a, b = int(fracs[0] * n), int((fracs[0] + fracs[1]) * n)
    g = {"train": set(groups[:a]), "val": set(groups[a:b]), "test": set(groups[b:])}
    return tuple(df[df["group"].isin(g[k])].reset_index(drop=True) for k in ("train", "val", "test"))
