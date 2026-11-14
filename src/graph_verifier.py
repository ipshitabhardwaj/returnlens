"""Track A - calibrated verifier + evidence graph."""

import networkx as nx, numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb

FEATS = ["phash_sim", "orb_good", "orb_inliers", "inlier_ratio", "emb_cos", "min_blur", "min_side"]


def train_verifier(df):
    """df has FEATS + y + source_group. Split by SOURCE so transformed copies never leak."""
    tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=0).split(df, df.y, df.source_group))
    clf = CalibratedClassifierCV(
        lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, verbose=-1),
        method="isotonic",
        cv=3,
    )
    clf.fit(df.iloc[tr][FEATS], df.iloc[tr].y)
    return clf, df.iloc[te]


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


def graph_features(G, claim_id):
    """Cross-claim context for one claim: how many OTHER customers share its image cluster?"""
    img = next(n for n in G[claim_id] if G.nodes[n]["kind"] == "Image")
    M = nx.Graph([(u, v) for u, v, d in G.edges(data=True) if d["rel"] == "POSSIBLE_SOURCE_MATCH"])
    cluster = nx.node_connected_component(M, img) if img in M else {img}
    claims, custs = set(), set()
    for im in cluster:
        for n in G[im]:
            if G.nodes[n]["kind"] == "Claim":
                claims.add(n)
                custs |= {m for m in G[n] if G.nodes[m]["kind"] == "Customer"}
    return {
        "cluster_size": len(cluster),
        "n_claims": len(claims),
        "n_customers": len(custs),
        "cross_customer_reuse": int(len(custs) > 1),
        "catalogue_image": int(G.nodes[img].get("catalogue", False)),
    }
