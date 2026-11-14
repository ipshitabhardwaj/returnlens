"""Track A - classical evidence engine: transforms + pHash + ORB + RANSAC (CPU only)."""

import cv2, numpy as np, imagehash, random
from PIL import Image


def transform(img_bgr, rng=random):
    """Apply 1-3 random 'reuse' edits: resize, rotate, crop, jpeg, brightness, blur, border."""
    ops = rng.sample(
        ["resize", "rotate", "crop", "jpeg", "bright", "blur", "border"], rng.randint(1, 3)
    )
    out, names = img_bgr.copy(), []
    for op in ops:
        h, w = out.shape[:2]
        if op == "resize":
            s = rng.uniform(0.5, 1.2)
            out = cv2.resize(out, (int(w * s), int(h * s)))
        elif op == "rotate":
            M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-15, 15), 1.0)
            out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        elif op == "crop":
            f = rng.uniform(0.65, 0.95)
            ch, cw = int(h * f), int(w * f)
            y, x = rng.randint(0, h - ch), rng.randint(0, w - cw)
            out = out[y : y + ch, x : x + cw]
        elif op == "jpeg":
            _, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(15, 60)])
            out = cv2.imdecode(enc, 1)
        elif op == "bright":
            out = cv2.convertScaleAbs(out, alpha=rng.uniform(0.7, 1.3), beta=rng.uniform(-30, 30))
        elif op == "blur":
            out = cv2.GaussianBlur(out, (5, 5), rng.uniform(0.5, 1.5))
        elif op == "border":
            out = cv2.copyMakeBorder(out, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        names.append(op)
    return out, names


def phash_sim(a_bgr, b_bgr):
    ha = imagehash.phash(Image.fromarray(cv2.cvtColor(a_bgr, cv2.COLOR_BGR2RGB)))
    hb = imagehash.phash(Image.fromarray(cv2.cvtColor(b_bgr, cv2.COLOR_BGR2RGB)))
    return 1 - (ha - hb) / 64.0  # S_pHash = 1 - dH / B


def orb_geom(a_bgr, b_bgr, ratio=0.75, nfeat=1500):
    """ORB + ratio test + RANSAC homography. Returns (n_good, n_inliers, inlier_ratio)."""
    orb = cv2.ORB_create(nfeatures=nfeat)
    ga, gb = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (a_bgr, b_bgr))
    ka, da = orb.detectAndCompute(ga, None)
    kb, db = orb.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0, 0, 0.0
    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < ratio * m[1].distance]
    if len(good) < 8:
        return len(good), 0, 0.0
    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return len(good), 0, 0.0
    inl = int(mask.sum())
    return len(good), inl, inl / len(good)


def quality(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return {"blur_var": float(cv2.Laplacian(g, cv2.CV_64F).var()), "h": g.shape[0], "w": g.shape[1]}


def pair_features(a, b):
    n_good, n_inl, r_inl = orb_geom(a, b)
    qa, qb = quality(a), quality(b)
    return {
        "phash_sim": phash_sim(a, b),
        "orb_good": n_good,
        "orb_inliers": n_inl,
        "inlier_ratio": r_inl,
        "min_blur": min(qa["blur_var"], qb["blur_var"]),
        "min_side": min(qa["h"], qa["w"], qb["h"], qb["w"]),
    }
