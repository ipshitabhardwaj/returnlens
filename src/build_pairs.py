"""Track A - build labelled image pairs: positives = edited copies, negatives = easy + HARD."""

import random, cv2, pandas as pd
from evidence import transform, pair_features


def build_pairs(pool, n_variants=3, seed=0):
    """pool: list of dict(source_id, product_id, path). 'group' = product_id so no product leaks across splits."""
    random.seed(seed)
    rows, by_prod = [], {}
    for p in pool:
        by_prod.setdefault(p["product_id"], []).append(p)
    for p in pool:
        a = cv2.imread(p["path"])
        base = dict(group=p["product_id"], source=p["source_id"])
        for k in range(n_variants):  # positives: same source, edited
            b, ops = transform(a)
            rows.append(dict(**base, y=1, kind="same_source", ops="+".join(ops), **pair_features(a, b)))
        sib = [q for q in by_prod[p["product_id"]] if q["source_id"] != p["source_id"]]
        if sib:  # HARD negative: same product, other photo
            b = cv2.imread(random.choice(sib)["path"])
            rows.append(
                dict(**base, y=0, kind="same_product_other_photo", ops="", **pair_features(a, b))
            )
        b = cv2.imread(random.choice(pool)["path"])  # easy negative: random other image
        rows.append(dict(**base, y=0, kind="random_other", ops="", **pair_features(a, b)))
    return pd.DataFrame(rows)
