import sys, pathlib, random
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import cv2, numpy as np
from evidence import transform, pair_features


def fake_photo(seed):
    r = np.random.default_rng(seed)
    img = cv2.resize((r.random((24, 32, 3)) * 255).astype(np.uint8), (480, 360), interpolation=cv2.INTER_CUBIC)
    for _ in range(25):
        cv2.circle(img, (int(r.integers(0, 480)), int(r.integers(0, 360))), int(r.integers(5, 30)),
                   [int(c) for c in r.integers(0, 255, 3)], -1)
    return img


def test_edited_copy_scores_higher_than_independent_photo():
    random.seed(0)
    a = fake_photo(1)
    b, _ = transform(a)
    same = pair_features(a, b)
    other = pair_features(a, fake_photo(999))
    assert same["orb_inliers"] > other["orb_inliers"]
    assert same["phash_sim"] > other["phash_sim"]
