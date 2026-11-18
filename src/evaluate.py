"""Track A - evaluation: calibration, ablation, error gallery, graph-vs-pairwise chain experiment (E1-E4)."""

import cv2, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from baselines import pick_threshold, evaluate
from evidence import transform, pair_features
from graph_verifier import fit_verifier, build_graph, cluster_claims, FEATS

ABLATIONS = {
    "full": [],
    "- pHash": ["phash_sim"],
    "- ORB/RANSAC": ["orb_good", "orb_inliers", "inlier_ratio"],
    "- embedding": ["emb_cos"],
    "- image quality": ["min_blur", "min_side"],
}


def ece(y, p, bins=10):
    """Expected calibration error with equal-width bins."""
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean()) for b in range(bins) if (idx == b).any()))


def reliability_table(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)) - 1, 0, bins - 1)
    return pd.DataFrame(
        [dict(bin=b, n=int((idx == b).sum()), mean_p=float(p[idx == b].mean()), frac_pos=float(y[idx == b].mean()))
         for b in range(bins) if (idx == b).any()]
    )


def score_split(clf, df):
    return clf.predict_proba(df[clf.feats_])[:, 1]


def ablation(train, val, test, feats=None):
    """E3: drop one feature group at a time. Threshold chosen on val (hard-neg FPR <= 1%), reported on test."""
    feats = feats or [f for f in FEATS if f in train.columns]
    rows = []
    for name, drop in ABLATIONS.items():
        use = [f for f in feats if f not in drop]
        if not use or (drop and not set(drop) & set(feats)):
            continue
        clf = fit_verifier(train, use)
        v, t = val.copy(), test.copy()
        v["p"], t["p"] = score_split(clf, v), score_split(clf, t)
        thr = pick_threshold(v, "p")
        r = evaluate(t, "p", thr)
        rows.append(dict(variant=name, n_feats=len(use), AUC=round(roc_auc_score(t.y, t.p), 3),
                         Brier=round(brier_score_loss(t.y, t.p), 4), ECE=round(ece(t.y, t.p), 4),
                         P=r["P"], R=r["R"], F1=r["F1"], FPR_hard=r["FPR_hard"]))
    return pd.DataFrame(rows)


def error_gallery(test, p, k=10, thr=0.5):
    """The k highest-scoring negatives and the k lowest-scoring positives.

    A row is labelled false_positive / miss ONLY if it is a real error at `thr`; otherwise it is labelled
    hardest_negative / weakest_positive (the closest calls, still correct). Add a one-line reason by hand.
    """
    d = test.assign(p=np.asarray(p))
    fp = d[(d.y == 0)].sort_values("p", ascending=False).head(k)
    fp = fp.assign(error=np.where(fp.p >= thr, "false_positive", "hardest_negative"))
    fn = d[(d.y == 1)].sort_values("p").head(k)
    fn = fn.assign(error=np.where(fn.p < thr, "miss", "weakest_positive"))
    out = pd.concat([fp, fn])
    cols = [c for c in ["error", "kind", "ops", "p", "path_a", "path_b"] if c in out.columns]
    return out[cols].reset_index(drop=True)


def save_montage(rows, out_path, side=220):
    """Side-by-side image of each error pair (path_a | path_b) stacked vertically."""
    tiles = []
    for _, r in rows.iterrows():
        a, b = cv2.imread(r.path_a), cv2.imread(r.path_b)
        if a is None or b is None:
            continue
        a, b = cv2.resize(a, (side, side)), cv2.resize(b, (side, side))
        t = np.hstack([a, b])
        cv2.putText(t, f"{r.error} p={r.p:.2f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        tiles.append(t)
    if tiles:
        cv2.imwrite(out_path, np.vstack(tiles))
    return len(tiles)


def chain_experiment(pool, clf, emb_fn=None, n_chains=30, length=4, p_edge=0.8, seed=0):
    """E4 (SYNTHETIC customers): is a reuse CHAIN recovered better by the graph than by direct pairs?

    Each chain: customer 1 uploads photo X; customer 2 re-uploads an edited copy of X; customer 3 edits customer 2's
    copy again, and so on. The far ends of a long chain can look different directly, but are linked through the middle.
    Distractor: one other photo of the same product by another customer (should NOT join the chain).
    For the first claim we ask: which other claims do you say share its source?
      pairwise = direct neighbours with p >= p_edge;  graph = connected component of POSSIBLE_SOURCE_MATCH edges.
    """
    import random

    random.seed(seed)
    by_prod = {}
    for p in pool:
        by_prod.setdefault(p["product_id"], []).append(p)
    prods = [k for k, v in by_prod.items() if len(v) >= 2]
    random.shuffle(prods)
    res = []

    def prob(a, b):
        f = pair_features(a, b)
        if emb_fn is not None:
            ea, eb = emb_fn(a), emb_fn(b)
            f["emb_cos"] = float(np.dot(ea, eb))
        return float(clf.predict_proba(pd.DataFrame([f])[clf.feats_])[:, 1][0])

    for c, pid in enumerate(prods[:n_chains]):
        base, other = by_prod[pid][0], by_prod[pid][1]
        imgs = {f"c{c}_0": cv2.imread(base["path"])}
        for i in range(1, length):
            imgs[f"c{c}_{i}"] = transform(imgs[f"c{c}_{i - 1}"])[0]
        imgs[f"c{c}_d"] = cv2.imread(other["path"])
        ids = list(imgs)
        claims = [dict(claim_id=i, customer_id=f"cust_{i}", product_id=pid, image_id=f"img_{i}", is_catalogue=False) for i in ids]
        edges, direct = [], set()
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                p = prob(imgs[ids[x]], imgs[ids[y]])
                edges.append((f"img_{ids[x]}", f"img_{ids[y]}", p))
                if p >= p_edge and ids[x] == ids[0]:
                    direct.add(ids[y])
        truth = set(ids[1:length])
        G = build_graph(claims, edges)
        graph = cluster_claims(G, ids[0]) - {ids[0]}
        for name, pred in (("pairwise", direct), ("graph", graph)):
            tp = len(pred & truth)
            res.append(dict(method=name, chain=c, recall=tp / len(truth), precision=tp / len(pred) if pred else 1.0))
    r = pd.DataFrame(res)
    return r.groupby("method")[["recall", "precision"]].mean().round(3).assign(n_chains=len(r) // 2)
