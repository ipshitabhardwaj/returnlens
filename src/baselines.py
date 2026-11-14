"""Track A - baselines table. Thresholds are chosen on VALIDATION only, then locked for test."""

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

HARD = "same_product_other_photo"


def pick_threshold(val, col, max_fpr_hard=0.01):
    """Lowest threshold that keeps the false-positive rate on HARD negatives <= 1%."""
    hard = val[val.kind == HARD][col]
    for t in sorted(val[col].unique()):
        if (hard >= t).mean() <= max_fpr_hard:
            return float(t)
    return float(val[col].max()) + 1


def evaluate(df, col, thr):
    pred = (df[col] >= thr).astype(int)
    p, r, f, _ = precision_recall_fscore_support(df.y, pred, average="binary", zero_division=0)
    return dict(
        feature=col,
        thr=round(thr, 3),
        P=round(p, 3),
        R=round(r, 3),
        F1=round(f, 3),
        FPR_hard=round(float((df[df.kind == HARD][col] >= thr).mean()), 3),
    )


def table(val, test, cols=("phash_sim", "orb_good", "orb_inliers", "inlier_ratio")):
    return pd.DataFrame([evaluate(test, c, pick_threshold(val, c)) for c in cols])
