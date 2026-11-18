"""Streamlit demo for Track A.  Run:  streamlit run app_track_a.py   (from the repo root)"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import cv2, numpy as np, streamlit as st
from demo_pair import compare

st.title("ReturnLens - Evidence check (Track A)")
st.caption("Upload a claim photo and an earlier photo. Matches shown are ORB keypoints kept by RANSAC (green).")
c1, c2 = st.columns(2)
fa, fb = c1.file_uploader("Claim photo", type=["jpg", "jpeg", "png"]), c2.file_uploader("Earlier photo", type=["jpg", "jpeg", "png"])
read = lambda f: cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR)
if fa and fb:
    a, b = read(fa), read(fb)
    f, vis = compare(a, b)
    st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    st.metric("pHash similarity", round(f["phash_sim"], 3))
    st.metric("RANSAC inliers", f["orb_inliers"])
    st.json({k: (round(v, 3) if isinstance(v, float) else v) for k, v in f.items()})
    st.info("A high score means 'possible same source', not proof of fraud. Poor photo quality raises uncertainty, not risk.")
