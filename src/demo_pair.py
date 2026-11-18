"""Track A - demo helper: compare two photos and draw the ORB/RANSAC matches (used by app_track_a.py)."""

import cv2, numpy as np
from evidence import pair_features


def compare(a, b, max_draw=60):
    """Returns (features dict, match visualisation BGR, inlier mask count). RANSAC inliers are drawn green."""
    f = pair_features(a, b)
    orb = cv2.ORB_create(1500)
    ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    ka, da = orb.detectAndCompute(ga, None)
    kb, db = orb.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return f, np.hstack([cv2.resize(a, (320, 240)), cv2.resize(b, (320, 240))])
    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    mask = None
    if len(good) >= 8:
        src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    kw = dict(matchColor=(0, 200, 0), flags=2)
    if mask is not None:
        kw["matchesMask"] = mask.ravel().tolist()
    return f, cv2.drawMatches(a, ka, b, kb, good[:max_draw] if mask is None else good, None, **kw)
