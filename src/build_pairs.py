"""Track A - build labelled image pairs: positives = edited copies, negatives = easy + HARD."""

import random, cv2, numpy as np, pandas as pd
from evidence import transform, pair_features


def build_pairs(pool, n_variants=3, seed=0, emb_fn=None):
    """pool: list of dict(source_id, product_id, path). 'group' = product_id so no product leaks across splits.

    emb_fn: optional callable(BGR image) -> unit vector. If given, every row also gets emb_cos.
    Every row also records 'path_a' (the base image) and 'path_b' (partner image, same as path_a for edited copies)."""
    random.seed(seed)
    cos = (lambda u, v: float(np.dot(u, v))) if emb_fn else None
    cache = {}

    def emb(path, img):
        if path not in cache:
            cache[path] = emb_fn(img)
        return cache[path]

    rows, by_prod = [], {}
    for p in pool:
        by_prod.setdefault(p["product_id"], []).append(p)
    for p in pool:
        a = cv2.imread(p["path"])
        base = dict(group=p["product_id"], source=p["source_id"], path_a=p["path"])
        ea = emb(p["path"], a) if emb_fn else None
        for k in range(n_variants):  # positives: same source, edited
            b, ops = transform(a)
            extra = dict(emb_cos=cos(ea, emb_fn(b))) if emb_fn else {}
            rows.append(
                dict(**base, path_b=p["path"], y=1, kind="same_source", ops="+".join(ops), **extra, **pair_features(a, b))
            )
        sib = [q for q in by_prod[p["product_id"]] if q["source_id"] != p["source_id"]]
        if sib:  # HARD negative: same product, other photo
            q = random.choice(sib)
            b = cv2.imread(q["path"])
            extra = dict(emb_cos=cos(ea, emb(q["path"], b))) if emb_fn else {}
            rows.append(
                dict(**base, path_b=q["path"], y=0, kind="same_product_other_photo", ops="", **extra, **pair_features(a, b))
            )
        q = random.choice(pool)  # easy negative: random other image
        b = cv2.imread(q["path"])
        extra = dict(emb_cos=cos(ea, emb(q["path"], b))) if emb_fn else {}
        rows.append(dict(**base, path_b=q["path"], y=0, kind="random_other", ops="", **extra, **pair_features(a, b)))
    return pd.DataFrame(rows)
