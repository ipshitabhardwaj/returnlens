"""Track A - calibrated verifier + evidence graph."""

import networkx as nx, numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb

FEATS = ["phash_sim", "orb_good", "orb_inliers", "inlier_ratio", "emb_cos", "min_blur", "min_side"]


def _calibrated(model):
    return CalibratedClassifierCV(model, method="isotonic", cv=3)


def fit_verifier(train_df, feats=None, backend=None):
    """Fit + calibrate on a given training frame (use the train split by product). Returns the model with .feats_.

    backend: "lgbm" (default on Linux/Colab) or "hgb" (default on Windows) (scikit-learn HistGradientBoosting). If LightGBM's native library crashes on
    your machine (seen on some Windows installs: 'access violation'), this falls back to "hgb" automatically.
    Force it with the environment variable RETURNLENS_VERIFIER=hgb."""
    import os
    from sklearn.ensemble import HistGradientBoostingClassifier

    feats = feats or [f for f in FEATS if f in train_df.columns]
    import sys

    # LightGBM's native library crashes on some Windows setups, so default to scikit-learn there.
    backend = backend or os.environ.get("RETURNLENS_VERIFIER", "hgb" if sys.platform == "win32" else "lgbm")
    X, y = train_df[feats], train_df.y
    clf = None
    if backend == "lgbm":
        try:
            clf = _calibrated(lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, verbose=-1))
            clf.fit(X, y)
        except Exception as e:  # OSError on a broken native lib
            print("LightGBM failed (%s: %s); falling back to scikit-learn HistGradientBoosting" % (type(e).__name__, e))
            clf = None
    if clf is None:
        clf = _calibrated(HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=0))
        clf.fit(X, y)
    clf.feats_ = list(feats)
    return clf


def train_verifier(df):
    """df has FEATS + y + source_group. Split by SOURCE so transformed copies never leak."""
    grp = df["source_group"] if "source_group" in df else df["group"]
    tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=0).split(df, df.y, grp))
    return fit_verifier(df.iloc[tr]), df.iloc[te]


def evidence_uncertainty(row):
    """Quality-driven uncertainty (NOT fraud): blurry / tiny / texture-poor images."""
    u = 0.0
    u += 0.4 if row["min_blur"] < 50 else 0.0
    u += 0.3 if row["min_side"] < 200 else 0.0
    u += 0.3 if row["orb_good"] < 10 else 0.0
    return min(u, 1.0)


def build_graph(claims, edges):
    """claims: list of dict(claim_id, customer_id, product_id, image_id, is_catalogue)
    edges : list of (image_id_a, image_id_b, p_same_source)"""
    G = nx.Graph()
    for c in claims:
        G.add_node(c["claim_id"], kind="Claim")
        G.add_node(c["image_id"], kind="Image", catalogue=c["is_catalogue"])
        G.add_node(c["customer_id"], kind="Customer")
        G.add_node(c["product_id"], kind="Product")
        G.add_edge(c["claim_id"], c["image_id"], rel="HAS_IMAGE")
        G.add_edge(c["customer_id"], c["claim_id"], rel="HAS_CLAIM")
        G.add_edge(c["claim_id"], c["product_id"], rel="REFERENCES_PRODUCT")
    for a, b, p in edges:
        if p >= 0.8:
            G.add_edge(a, b, rel="POSSIBLE_SOURCE_MATCH", p=p)
    return G


def _cluster(G, claim_id):
    img = next(n for n in G[claim_id] if G.nodes[n]["kind"] == "Image")
    M = nx.Graph([(u, v) for u, v, d in G.edges(data=True) if d["rel"] == "POSSIBLE_SOURCE_MATCH"])
    cluster = nx.node_connected_component(M, img) if img in M else {img}
    claims, custs = set(), set()
    for im in cluster:
        for n in G[im]:
            if G.nodes[n]["kind"] == "Claim":
                claims.add(n)
                custs |= {m for m in G[n] if G.nodes[m]["kind"] == "Customer"}
    return img, cluster, claims, custs


def cluster_claims(G, claim_id):
    """All claims whose image is (transitively) linked to this claim's image, including itself."""
    return _cluster(G, claim_id)[2]


def query_uncertainty(img_bgr):
    """Evidence uncertainty from the QUERY photo alone (blur, tiny, texture-poor). Not a fraud signal."""
    import cv2

    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    u = 0.4 if cv2.Laplacian(g, cv2.CV_64F).var() < 50 else 0.0
    u += 0.3 if min(g.shape[:2]) < 200 else 0.0
    u += 0.3 if len(cv2.ORB_create(500).detect(g, None)) < 100 else 0.0
    return min(u, 1.0)


def graph_features(G, claim_id):
    """Cross-claim context for one claim: how many OTHER customers share its image cluster?"""
    img, cluster, claims, custs = _cluster(G, claim_id)
    return {
        "cluster_size": len(cluster),
        "n_claims": len(claims),
        "n_customers": len(custs),
        "cross_customer_reuse": int(len(custs) > 1),
        "catalogue_image": int(G.nodes[img].get("catalogue", False)),
    }
