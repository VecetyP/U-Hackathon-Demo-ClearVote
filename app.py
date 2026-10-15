"""Local test bench: streamlit run app.py"""
import pathlib

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import ballot

ASSETS = pathlib.Path(__file__).parent / "tests" / "assets"
BENCHMARK = ASSETS / "benchmark_infographic.jpg"
SAMPLES = sorted(ASSETS.glob("sample_*.png"))
VERDICT_STYLE = {ballot.VERDICT_VALID: ("#0a7d32", "Valid ballot"),
                 ballot.VERDICT_INVALID: ("#c62828", "Invalid ballot"),
                 ballot.VERDICT_NOVOTE: ("#555", "No-vote ballot"),
                 ballot.VERDICT_NOBOX: ("#555", "No voting boxes found")}

st.set_page_config(page_title="Thai Ballot Verifier", layout="wide")
st.title("Thai Ballot Verification Engine")
st.caption("ตรวจบัตรเลือกตั้ง: กาถูกต้อง / เครื่องหมายผิดประเภท / ว่าง + ตรวจรอยขีดเขียนนอกช่อง")

c1, c2, c3 = st.columns([3, 1, 1])
upload = c1.file_uploader("Upload a ballot or test grid", type=["png", "jpg", "jpeg"])
first = c2.number_input("First candidate number", min_value=0, value=1, step=1,
                        help="Printed number of the first mark box (EC sample ballots start at 501).")
if c3.button("Run benchmark (32-box EC infographic)", use_container_width=True):
    st.session_state.src = "benchmark"
sample = c3.selectbox("...or a sample ballot", ["-"] + [p.name for p in SAMPLES])
if upload is not None and upload.file_id != st.session_state.get("last_upload"):
    st.session_state.last_upload = upload.file_id
    st.session_state.src = "upload"
if sample != st.session_state.get("last_sample"):
    st.session_state.last_sample = sample
    if sample != "-":
        st.session_state.src = "sample"

src = st.session_state.get("src")
if src == "upload" and upload is not None:
    img = cv2.imdecode(np.frombuffer(upload.getvalue(), np.uint8), cv2.IMREAD_COLOR)
elif src == "sample" and sample != "-":
    img = cv2.imread(str(ASSETS / sample))
elif src == "benchmark":
    img = cv2.imread(str(BENCHMARK))
else:
    st.info("Upload an image or run the benchmark.")
    st.stop()

if img is None:
    st.error("Could not read that file as an image.")
    st.stop()

res = ballot.analyze(img, first_number=int(first))
color, english = VERDICT_STYLE[res["verdict"]]

left, right = st.columns([3, 2])
left.image(cv2.cvtColor(res["overlay"], cv2.COLOR_BGR2RGB), use_container_width=True)

with right:
    st.markdown(f"<div style='background:{color};color:#fff;padding:18px;border-radius:10px;text-align:center'>"
                f"<div style='font-size:2.4rem;font-weight:700'>{res['verdict']}</div>"
                f"<div>{english}</div></div>", unsafe_allow_html=True)
    if res["voted_number"] is not None:
        st.subheader(f"เลือกหมายเลข {res['voted_number']}  (voted for #{res['voted_number']})")
        if res["number_crop"] is not None and res["number_crop"].size:
            st.image(cv2.cvtColor(res["number_crop"], cv2.COLOR_BGR2RGB),
                     caption="Printed row next to the marked box - confirm the number", width=320)
    n = res["counts"]
    m = st.columns(4)
    m[0].metric("Boxes", n["total"])
    m[1].metric("Valid", n["VALID"])
    m[2].metric("Invalid", n["INVALID"])
    m[3].metric("Empty", n["EMPTY"])
    if res["strays"]:
        st.error(f"Stray marks outside boxes: YES ({len(res['strays'])}) at "
                 + ", ".join(f"(x={x}, y={y})" for x, y, _, _ in res["strays"]))
    else:
        st.success("Stray marks outside boxes: NO")
    for w in res["warnings"]:
        st.warning(w)
    with st.expander("Per-box detail", expanded=False):
        st.dataframe(pd.DataFrame([{"#": "no-vote" if b["is_no_vote"] else str(b["number"]), "label": b["label"],
                                    "confidence": b["confidence"], "reason": b["reason"],
                                    "bbox (x,y,w,h)": str(b["bbox"])} for b in res["boxes"]]),
                     hide_index=True, use_container_width=True)
