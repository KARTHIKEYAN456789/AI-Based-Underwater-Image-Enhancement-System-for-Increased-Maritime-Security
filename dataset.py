"""Streamlit demo: AI-Based Underwater Image Enhancement for Maritime Security.

Upload an underwater image -> enhance it with the trained U-Net -> optionally run
YOLO object detection on both raw and enhanced versions to show that enhancement
improves detection of security-relevant objects.

Run:  ..\\.venv\\Scripts\\streamlit run app.py
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from infer import enhance_image

ROOT = Path(__file__).resolve().parent.parent

st.set_page_config(page_title="Underwater Image Enhancement", page_icon="🌊", layout="wide")


@lru_cache(maxsize=1)
def load_yolo():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")  # small pretrained detector, auto-downloads once


def run_detection(pil_img: Image.Image):
    """Return (annotated PIL image, number of detections)."""
    model = load_yolo()
    res = model.predict(np.array(pil_img.convert("RGB")), verbose=False)[0]
    annotated = Image.fromarray(res.plot()[:, :, ::-1])  # BGR -> RGB
    return annotated, len(res.boxes)


# ---------------------------------------------------------------- UI
st.title("🌊 AI-Based Underwater Image Enhancement")
st.caption("Enhancing degraded underwater imagery for improved maritime security & surveillance")

with st.sidebar:
    st.header("About")
    st.write(
        "A compact **U-Net** trained on the **UIEB** dataset removes the color cast "
        "and haze from underwater images. Enhancement is a preprocessing step that "
        "makes objects clearer for human operators and AI detectors."
    )
    run_detect = st.checkbox("Run object detection (YOLOv8)", value=False,
                             help="Compares detections on raw vs. enhanced image. "
                                  "Downloads a small model on first use.")
    st.divider()
    st.write("**Pipeline**")
    st.code("raw  ->  U-Net enhance  ->  YOLO detect", language="text")

uploaded = st.file_uploader("Upload an underwater image", type=["jpg", "jpeg", "png", "bmp"])

# fall back to a sample from the dataset if nothing is uploaded
sample = None
if uploaded is None:
    samples = sorted((ROOT / "data" / "raw-890").glob("*.png"))
    if samples:
        sample = samples[0]
        st.info("No upload yet — showing a sample image from the UIEB dataset. "
                "Upload your own image above to try it.")

if uploaded is not None or sample is not None:
    raw = Image.open(uploaded if uploaded is not None else sample).convert("RGB")

    with st.spinner("Enhancing image..."):
        enhanced = enhance_image(raw)

    st.subheader("Enhancement result")
    c1, c2 = st.columns(2)
    c1.image(raw, caption="Raw (degraded)", use_container_width=True)
    c2.image(enhanced, caption="Enhanced (U-Net)", use_container_width=True)

    # download button for the enhanced image
    buf = ROOT / "results" / "app_enhanced.png"
    buf.parent.mkdir(exist_ok=True)
    enhanced.save(buf)
    with open(buf, "rb") as f:
        st.download_button("Download enhanced image", f, "enhanced.png", "image/png")

    if run_detect:
        st.subheader("Object detection: raw vs. enhanced")
        with st.spinner("Running YOLOv8 detection..."):
            raw_det, n_raw = run_detection(raw)
            enh_det, n_enh = run_detection(enhanced)
        d1, d2 = st.columns(2)
        d1.image(raw_det, caption=f"Raw — {n_raw} object(s) detected", use_container_width=True)
        d2.image(enh_det, caption=f"Enhanced — {n_enh} object(s) detected", use_container_width=True)
        delta = n_enh - n_raw
        if delta > 0:
            st.success(f"Enhancement surfaced {delta} additional detection(s) — "
                       "clearer imagery helps the detector.")
        elif delta < 0:
            st.warning(f"Detected {-delta} fewer object(s) after enhancement on this image.")
        else:
            st.info("Same number of detections on this image. Try another for a clearer contrast.")
else:
    st.stop()
