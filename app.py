"""Streamlit web app: upload a photo and get an AI-generated caption.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st
from PIL import Image

from captioner import Captioner, model_files_exist
from viz import attention_overlay

st.set_page_config(page_title="AI Image Captioner", page_icon="🖼️", layout="centered")


@st.cache_resource(show_spinner="Loading model (first time takes ~20 s)...")
def get_captioner() -> Captioner:
    return Captioner()


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    method_label = st.radio("Decoding method", ["Beam search", "Greedy"],
                            help="Greedy picks the single best word each step. "
                                 "Beam search keeps several candidate sentences "
                                 "and usually writes better captions.")
    method = "beam" if method_label == "Beam search" else "greedy"
    beam_width = st.slider("Beam width", 2, 5, 3, disabled=method == "greedy")
    show_attention = st.toggle("Show where the model looked", value=True)

    st.header("About")
    st.write(
        "A pretrained **EfficientNetB0** CNN turns the photo into 49 region "
        "features; a **Transformer decoder** writes the caption word by word, "
        "attending to different regions for each word. Trained on the Kaggle "
        "[Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) dataset."
    )
    st.caption("Works best on everyday photos of people, animals and outdoor scenes.")

# ----------------------------------------------------------------- main page
st.title("🖼️ AI Image Caption Generator")
st.write("Upload a photo and the model describes it in a sentence.")

if not model_files_exist():
    st.error(
        "Model files not found. Train the model first (`python train.py` or the "
        "Colab notebook), then put `encoder.weights.h5`, `decoder.weights.h5`, "
        "`config.json` and `vocab.json` in the `model/` folder."
    )
    st.stop()

source = st.radio("Image source", ["Upload a photo", "Use camera"], horizontal=True)
if source == "Upload a photo":
    file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
else:
    file = st.camera_input("Take a photo")

if file is None:
    st.info("Waiting for an image...")
    st.stop()

try:
    image = Image.open(file)
    image.load()
except Exception:  # noqa: BLE001 - any unreadable file
    st.error("Could not read that file. Please upload a valid JPG or PNG image.")
    st.stop()

st.image(image, width="stretch")

captioner = get_captioner()
with st.spinner("Writing a caption..."):
    result, feats = captioner.caption(image, method=method, beam_width=beam_width)

if result.words:
    st.success(f"### {result.caption}.")
else:
    st.warning("The model could not produce a caption for this image.")
    st.stop()

if method == "beam" and len(result.alternatives) > 1:
    with st.expander("Other candidate captions (beam search)"):
        for text, score in result.alternatives:
            st.write(f"• {text}  \n<small>score {score:.3f}</small>", unsafe_allow_html=True)

if show_attention:
    st.subheader("Where the model looked for each word")
    maps = captioner.attention_maps(feats, result.token_ids)
    cols = st.columns(4)
    for i, (word, attn) in enumerate(zip(result.words, maps)):
        with cols[i % 4]:
            st.image(attention_overlay(image, attn), caption=word, width="stretch")
    st.caption("Red = high attention. Maps come from the last decoder layer, averaged over heads.")
