"""Track A - step A7: score one claim photo against a gallery and return the /evidence contract."""

import json, os, cv2, numpy as np, pandas as pd, imagehash
from PIL import Image
from evidence import pair_features
from graph_verifier import query_uncertainty


def _phash(img_bgr):
    return imagehash.phash(Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))


class EvidenceScorer:
    """gallery: list of dict(image_id, path, claim_id, customer_id, product_id, is_catalogue).

    clf: fitted verifier (has .feats_). emb_fn: optional callable(BGR)->unit vector (fine-tuned DINOv2).
    Candidates are the top-k gallery photos by embedding cosine (or by pHash if no embedder); each is verified
    with pair features and the calibrated verifier. reuse_risk = highest verified probability."""

    def __init__(self, gallery, clf, emb_fn=None, topk=5, p_edge=0.8):
        self.gallery, self.clf, self.emb_fn, self.topk, self.p_edge = gallery, clf, emb_fn, topk, p_edge
        self.imgs = [cv2.imread(g["path"]) for g in gallery]
        self.hashes = [_phash(i) for i in self.imgs]
        self.embs = np.stack([emb_fn(i) for i in self.imgs]) if emb_fn and gallery else None

    def _candidates(self, q, exclude_claim):
        if self.embs is not None:
            sims = self.embs @ self.emb_fn(q)
        else:
            hq = _phash(q)
            sims = np.array([1 - (hq - h) / 64.0 for h in self.hashes])
        order = [i for i in np.argsort(-sims) if self.gallery[i]["claim_id"] != exclude_claim]
        return order[: self.topk]

    def score(self, image, claim_id, customer_id=None):
        q = cv2.imread(image) if isinstance(image, str) else image
        if q is None:
            raise ValueError("could not read image")
        matches, p_max = [], 0.0
        cross, cat = False, False
        for i in self._candidates(q, claim_id):
            f = pair_features(q, self.imgs[i])
            if self.emb_fn:
                f["emb_cos"] = float(np.dot(self.emb_fn(q), self.embs[i]))
            p = float(self.clf.predict_proba(pd.DataFrame([f])[self.clf.feats_])[:, 1][0])
            p_max = max(p_max, p)
            if p >= self.p_edge:
                g = self.gallery[i]
                matches.append(dict(other_claim=g["claim_id"], p_same_source=round(p, 4), orb_inliers=int(f["orb_inliers"])))
                cross = cross or (customer_id is not None and g.get("customer_id") not in (None, customer_id))
                cat = cat or bool(g.get("is_catalogue", False))
        return dict(
            claim_id=claim_id,
            reuse_risk=round(p_max, 4),
            evidence_uncertainty=round(query_uncertainty(q), 2),
            cross_customer_reuse=bool(cross),
            catalogue_image=bool(cat),
            matches=sorted(matches, key=lambda m: -m["p_same_source"]),
        )


def save_artifacts(scorer, out_dir):
    import joblib

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(scorer.clf, os.path.join(out_dir, "verifier.joblib"))
    with open(os.path.join(out_dir, "gallery.json"), "w") as f:
        json.dump(scorer.gallery, f)


def load_scorer(out_dir, emb_fn=None, image_root=None, embedder=None, device=None, **kw):
    """Rebuild a scorer from save_artifacts output.

    image_root: folder holding the gallery images. Gallery paths saved in Colab ("/content/img_amazon/x.jpg") are
        remapped to image_root/x.jpg by file name.
    embedder: path to embedder.pt (fine-tuned DINOv2). Needed if the verifier was trained with the emb_cos feature.
    """
    import joblib

    clf = joblib.load(os.path.join(out_dir, "verifier.joblib"))
    gallery = json.load(open(os.path.join(out_dir, "gallery.json")))
    if image_root:
        gallery = [dict(g, path=os.path.join(image_root, os.path.basename(g["path"]))) for g in gallery]
    missing = [g["path"] for g in gallery if not os.path.exists(g["path"])]
    if missing:
        raise FileNotFoundError(f"{len(missing)} gallery images not found, e.g. {missing[0]} - set image_root")
    if emb_fn is None and embedder:
        import torch
        from retrieval import finetuned

        emb_fn = finetuned(embedder, device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if emb_fn is None and "emb_cos" in getattr(clf, "feats_", []):
        raise ValueError("this verifier uses the emb_cos feature: pass embedder=<path to embedder.pt> (or emb_fn)")
    return EvidenceScorer(gallery, clf, emb_fn=emb_fn, **kw)
