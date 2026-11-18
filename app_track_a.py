"""Streamlit demo for Track A (evidence check).  Run:  streamlit run app_track_a.py   (from the repo root)"""

import sys, os, glob

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
import cv2, numpy as np, streamlit as st
from demo_pair import compare

SAME_AT, DIFF_AT = 13, 6  # RANSAC inliers: >=13 = same-source cut-off picked on validation (baselines_table.csv)

st.set_page_config(page_title="ReturnLens - Evidence check", page_icon="🔍", layout="wide")
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, button {font-family: 'Inter', system-ui, sans-serif !important;}
.stApp {background: radial-gradient(1200px 600px at 10% -10%, #1e1b4b 0%, transparent 60%),
                    radial-gradient(900px 500px at 100% 0%, #0c4a6e 0%, transparent 55%), #0b1020;}
.block-container {padding-top: 1.6rem; max-width: 1180px;}
header[data-testid="stHeader"] {background: transparent;}
section[data-testid="stSidebar"] {background: #0f1530; border-right: 1px solid #ffffff14;}
.hero {position: relative; overflow: hidden; padding: 1.8rem 2rem; border-radius: 20px; margin-bottom: 1.2rem;
       background: linear-gradient(120deg, #2563eb 0%, #7c3aed 55%, #db2777 100%); color: #fff;
       box-shadow: 0 20px 50px -20px #7c3aedaa;}
.hero:after {content: ""; position: absolute; right: -60px; top: -60px; width: 260px; height: 260px; border-radius: 50%;
             background: radial-gradient(circle, #ffffff33, transparent 70%);}
.hero h1 {margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -.02em;}
.hero p {margin: .45rem 0 .9rem 0; opacity: .92; font-size: 1.02rem; max-width: 720px;}
.pill {display: inline-block; padding: .22rem .7rem; margin-right: .4rem; border-radius: 999px; font-size: .78rem;
       font-weight: 600; background: #ffffff26; border: 1px solid #ffffff40; backdrop-filter: blur(4px);}
.steps {display: flex; gap: .5rem; margin: .2rem 0 1.2rem 0; flex-wrap: wrap;}
.step {flex: 1; min-width: 120px; padding: .6rem .8rem; border-radius: 12px; background: #ffffff0d; border: 1px solid #ffffff14;
       font-size: .82rem; color: #cbd5e1;} .step b {display: block; color: #fff; font-size: .9rem;}
.verdict {display: flex; align-items: center; gap: 1.4rem; padding: 1.3rem 1.6rem; border-radius: 18px; margin: .4rem 0 1.2rem 0;
          border: 1px solid; backdrop-filter: blur(6px);}
.verdict .big {font-size: 3.2rem; font-weight: 800; line-height: 1; letter-spacing: -.03em;}
.verdict .big small {display: block; font-size: .72rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; opacity: .8; margin-top: .3rem;}
.verdict .txt {font-size: 1.35rem; font-weight: 700;} .verdict .sub {font-size: .9rem; font-weight: 400; opacity: .85; margin-top: .2rem;}
.v-same {background: linear-gradient(120deg, #7f1d1d66, #7f1d1d22); border-color: #ef4444; color: #fecaca;}
.v-mid {background: linear-gradient(120deg, #78350f66, #78350f22); border-color: #f59e0b; color: #fde68a;}
.v-diff {background: linear-gradient(120deg, #14532d66, #14532d22); border-color: #22c55e; color: #bbf7d0;}
.gauge {position: relative; height: 12px; border-radius: 999px; margin: 1.6rem 0 2.2rem 0;
        background: linear-gradient(90deg, #22c55e 0%, #22c55e 20%, #f59e0b 20%, #f59e0b 43%, #ef4444 43%, #ef4444 100%);}
.gauge .mk {position: absolute; top: -7px; width: 4px; height: 26px; border-radius: 2px; background: #fff; box-shadow: 0 0 0 3px #0b1020;}
.gauge .lb {position: absolute; top: 20px; font-size: .72rem; color: #94a3b8; transform: translateX(-50%); white-space: nowrap;}
.cards {display: grid; grid-template-columns: repeat(4, 1fr); gap: .8rem; margin: 1rem 0 1.4rem 0;}
.card {padding: 1rem 1.1rem; border-radius: 16px; background: #ffffff0d; border: 1px solid #ffffff1a;}
.card .k {font-size: .75rem; letter-spacing: .06em; text-transform: uppercase; color: #94a3b8; font-weight: 600;}
.card .v {font-size: 1.9rem; font-weight: 800; margin-top: .15rem; color: #fff;} .card .h {font-size: .75rem; color: #94a3b8; margin-top: .1rem;}
.sec {font-size: 1.05rem; font-weight: 700; margin: 1.2rem 0 .5rem 0; color: #e2e8f0;}
[data-testid="stImage"] img {border-radius: 14px; border: 1px solid #ffffff1f; box-shadow: 0 10px 30px -15px #000;}
[data-testid="stFileUploader"] section {border-radius: 14px; border: 1px dashed #ffffff40; background: #ffffff08;}
.stButton button {border-radius: 12px; border: 1px solid #ffffff26; background: #ffffff0f; font-weight: 600;}
.stButton button:hover {border-color: #7c3aed; background: #7c3aed33;}
.small {opacity: .7; font-size: .82rem;}
[data-testid="stToolbar"], [data-testid="stDeployButton"], #MainMenu {display: none;}
.stApp, .stApp label, .stApp p, section[data-testid="stSidebar"] * {color: #e2e8f0;}
.hero, .hero * {color: #fff !important;}
.verdict .txt, .verdict .sub, .verdict .big {color: inherit !important;}
</style>
<div class="hero"><h1>🔍 ReturnLens &middot; Evidence check</h1>
<p>Is this return-claim photo a reused or edited copy of an earlier photo? Track A looks for geometric proof, not just a similar look.</p>
<span class="pill">pHash</span><span class="pill">ORB keypoints</span><span class="pill">RANSAC geometry</span><span class="pill">Evidence only &middot; never auto-rejects</span></div>
<div class="steps"><div class="step"><b>1 &middot; Upload</b>claim photo + earlier photo</div>
<div class="step"><b>2 &middot; Look-alike</b>pHash similarity</div>
<div class="step"><b>3 &middot; Keypoints</b>ORB features + ratio test</div>
<div class="step"><b>4 &middot; Geometry</b>RANSAC inliers</div>
<div class="step"><b>5 &middot; Verdict</b>evidence for the router</div></div>
""",
    unsafe_allow_html=True,
)

read = lambda f: cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR)
rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# ---- sidebar: one-click demo pairs (data/demo, made for the group demo) ----
demo = sorted(glob.glob(os.path.join(ROOT, "data", "demo", "*.jpg")))
pair = None
with st.sidebar:
    st.header("Try a sample")
    edited = [p for p in demo if os.path.basename(p).split("_")[1] == "edited"]
    for p in edited:
        k = os.path.basename(p).split("_")[0]
        orig = next((q for q in demo if os.path.basename(q).startswith(f"{k}_original")), None)
        if orig and st.button(f"Edited copy #{k} vs its original", use_container_width=True):
            pair = (p, orig)
    if len(edited) >= 2:
        o = [q for q in demo if "_original_" in os.path.basename(q)]
        if len(o) >= 2 and st.button("Two unrelated products", use_container_width=True):
            pair = (o[0], o[-1])
    st.divider()
    st.markdown("**How to read it**")
    st.markdown(
        f"- **RANSAC inliers**: keypoints that agree on one geometric transform. **>= {SAME_AT}** = same source.\n"
        "- **pHash**: overall look. Drops with rotation/cropping, so it is only a weak signal.\n"
        "- **Photo quality**: blur and size. Poor quality raises *uncertainty*, not risk."
    )
    st.caption("Evidence only. This tool never rejects a claim; a human or the router decides.")

c1, c2 = st.columns(2)
fa = c1.file_uploader("Claim photo (new)", type=["jpg", "jpeg", "png"])
fb = c2.file_uploader("Earlier photo (gallery)", type=["jpg", "jpeg", "png"])

a = b = None
if pair:
    a, b = cv2.imread(pair[0]), cv2.imread(pair[1])
elif fa and fb:
    a, b = read(fa), read(fb)

if a is None or b is None:
    st.info("Upload a claim photo and an earlier photo, or pick a sample in the sidebar.")
    st.stop()

f, vis = compare(a, b)
inl = int(f["orb_inliers"])
if inl >= SAME_AT:
    cls, head, sub, icon = "v-same", "Likely the same source", "Strong geometric agreement between the two photos.", "🚩"
elif inl >= DIFF_AT:
    cls, head, sub, icon = "v-mid", "Inconclusive", "Some matching keypoints, not enough to call it.", "🤔"
else:
    cls, head, sub, icon = "v-diff", "No sign of reuse", "The two photos share no consistent geometry.", "✅"
st.markdown(
    f'<div class="verdict {cls}"><div class="big">{inl}<small>RANSAC inliers</small></div>'
    f'<div><div class="txt">{icon} {head}</div><div class="sub">{sub}</div></div></div>',
    unsafe_allow_html=True,
)
pos = min(inl, 30) / 30 * 100
st.markdown(
    f'<div class="gauge"><div class="mk" style="left:calc({pos}% - 2px)"></div>'
    f'<div class="lb" style="left:0%">0</div><div class="lb" style="left:{DIFF_AT/30*100}%">{DIFF_AT}</div>'
    f'<div class="lb" style="left:{SAME_AT/30*100}%">{SAME_AT} (cut-off)</div><div class="lb" style="left:100%">30+</div></div>',
    unsafe_allow_html=True,
)

i1, i2 = st.columns(2)
i1.image(rgb(a), caption="Claim photo", use_container_width=True)
i2.image(rgb(b), caption="Earlier photo", use_container_width=True)

st.markdown(
    f'<div class="cards">'
    f'<div class="card"><div class="k">RANSAC inliers</div><div class="v">{inl}</div><div class="h">hard negatives had at most 12</div></div>'
    f'<div class="card"><div class="k">Inlier ratio</div><div class="v">{f["inlier_ratio"]:.2f}</div><div class="h">share of matches that agree</div></div>'
    f'<div class="card"><div class="k">pHash similarity</div><div class="v">{f["phash_sim"]:.2f}</div><div class="h">drops with rotation / crop</div></div>'
    f'<div class="card"><div class="k">Photo sharpness</div><div class="v">{f["min_blur"]:.0f}</div><div class="h">low = harder to judge</div></div></div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="sec">Matching keypoints</div>', unsafe_allow_html=True)
st.image(rgb(vis), caption="Green lines = keypoints that agree on the same geometry (ORB + RANSAC)", use_container_width=True)
with st.expander("Raw features"):
    st.json({k: (round(v, 3) if isinstance(v, float) else v) for k, v in f.items()})
st.markdown(
    '<p class="small">A high score means "possible same source", not proof of fraud. '
    "Cut-off picked on a validation split of a small pilot (100 products, synthetic edits). "
    "Evidence only: a human or the router makes the decision.</p>",
    unsafe_allow_html=True,
)
