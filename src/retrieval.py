"""Track A - embedding helpers and retrieval evaluation (Recall@K of finding the SOURCE photo of an edited copy)."""

import cv2, numpy as np
from evidence import transform


def make_emb_fn(forward, device="cuda"):
    """Wrap a torch module (or callable) into emb_fn(BGR image) -> unit numpy vector. forward(batch)->(N,d)."""
    import torch

    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)

    @torch.no_grad()
    def fn(img_bgr):
        x = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (224, 224)).astype(np.float32) / 255.0
        x = torch.from_numpy(((x - mean) / std).transpose(2, 0, 1)).unsqueeze(0).to(device)
        z = forward(x)[0].float().cpu().numpy()
        return z / (np.linalg.norm(z) + 1e-9)

    return fn


def pretrained_dino(device="cuda", name="facebook/dinov2-small"):
    """Untouched DINOv2: CLS token, L2-normalised. This is the 'before fine-tuning' baseline."""
    import torch
    from transformers import AutoModel

    m = AutoModel.from_pretrained(name).to(device).eval()
    return make_emb_fn(lambda x: torch.nn.functional.normalize(m(pixel_values=x).last_hidden_state[:, 0], dim=-1), device)


def finetuned(ckpt, device="cuda"):
    """Load embedder.pt written by embed_ft.train."""
    import torch
    from embed_ft import Embedder

    m = Embedder().to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()
    return make_emb_fn(m, device)


def recall_at_k(gallery, queries, truth, ks=(1, 5, 10)):
    """gallery (G,d), queries (Q,d), truth (Q,) index of the true gallery item. Uses FAISS if installed."""
    gallery, queries = np.asarray(gallery, np.float32), np.asarray(queries, np.float32)
    kmax = min(max(ks), len(gallery))
    try:
        import faiss

        index = faiss.IndexFlatIP(gallery.shape[1])
        index.add(gallery)
        _, top = index.search(queries, kmax)
    except ImportError:
        top = np.argsort(-(queries @ gallery.T), axis=1)[:, :kmax]
    return {f"R@{k}": float((top[:, :k] == np.asarray(truth)[:, None]).any(1).mean()) for k in ks}


def retrieval_eval(pool, emb_fn, n_queries_per_image=1, seed=0):
    """Gallery = the original photos in pool. Queries = edited copies. Returns Recall@1/5/10.

    Only use TEST-split products here. Photos of the same product act as natural distractors."""
    import random

    random.seed(seed)
    imgs = [cv2.imread(p["path"]) for p in pool]
    gallery = np.stack([emb_fn(i) for i in imgs])
    qs, truth = [], []
    for j, img in enumerate(imgs):
        for _ in range(n_queries_per_image):
            qs.append(emb_fn(transform(img)[0]))
            truth.append(j)
    return recall_at_k(gallery, np.stack(qs), truth)
