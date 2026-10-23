# streamlit_app.py -- server-side bird classifier for Streamlit Community Cloud.
# generated from notebooks/main.ipynb (Phase 7.6) via %%writefile, same as
# app/app.py and app_static/index.html. edit the notebook cell, not this file.
#
# no torch here on purpose, just onnxruntime (CPU) + PIL + numpy, so the
# deploy stays light. preprocessing is a hand-rolled PIL version of
# torchvision's eval transform (Resize(256) + CenterCrop(224) + ImageNet
# normalise) since we don't want torchvision as a dependency just for this.

import json
import os

import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "model.onnx")
CLASS_NAMES_PATH = os.path.join(APP_DIR, "class_names.json")
EXAMPLES_DIR = os.path.join(APP_DIR, "examples")

IMG_SIZE = 224
EVAL_RESIZE = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LOW_CONFIDENCE_THRESHOLD = 0.30

with open(CLASS_NAMES_PATH) as f:
    CLASS_NAMES = json.load(f)


@st.cache_resource
def load_session():
    return ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])


def preprocess(image):
    # short side -> 256, long side truncated (not rounded) to match
    # torchvision's Resize(256) on a PIL image, then a 224 center crop with
    # offsets rounded (not floored) to match torchvision's CenterCrop
    image = image.convert("RGB")
    w, h = image.size
    if w <= h:
        new_w, new_h = EVAL_RESIZE, int(EVAL_RESIZE * h / w)
    else:
        new_h, new_w = EVAL_RESIZE, int(EVAL_RESIZE * w / h)
    resized = image.resize((new_w, new_h), Image.BILINEAR)

    crop_left = int(round((new_w - IMG_SIZE) / 2.0))
    crop_top = int(round((new_h - IMG_SIZE) / 2.0))
    cropped = resized.crop((crop_left, crop_top, crop_left + IMG_SIZE, crop_top + IMG_SIZE))

    arr = np.asarray(cropped).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)[None].astype(np.float32)  # NCHW
    return arr


def predict(image):
    session = load_session()
    tensor = preprocess(image)
    logits = session.run(None, {"input": tensor})[0][0]
    exps = np.exp(logits - logits.max())
    probs = exps / exps.sum()
    top5_idx = np.argsort(probs)[::-1][:5]
    return [(int(i), float(probs[i])) for i in top5_idx]


st.set_page_config(page_title="CUB-200 Bird Classifier", page_icon="\U0001F426")

st.title("CUB-200 Bird Species Classifier")
st.write(
    "Fine-tuned ResNet-50 on 200 CUB-200 bird species. Upload a photo "
    "or pick one of the examples below."
)

if "selected_image" not in st.session_state:
    st.session_state.selected_image = None

st.write("**Examples:**")
cols = st.columns(3)
for i, col in enumerate(cols, start=1):
    example_path = os.path.join(EXAMPLES_DIR, f"example_{i}.jpg")
    with col:
        st.image(example_path, width="stretch")
        if st.button(f"Use example {i}", key=f"example_btn_{i}"):
            st.session_state.selected_image = Image.open(example_path)

uploaded = st.file_uploader("Or upload your own photo", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    st.session_state.selected_image = Image.open(uploaded)

image = st.session_state.selected_image
if image is not None:
    st.image(image, caption="Selected photo", width="stretch")
    top5 = predict(image)

    st.write("**Top-5 predictions:**")
    for idx, prob in top5:
        name = CLASS_NAMES[idx].replace("_", " ")
        st.write(f"{name} -- {prob * 100:.1f}%")
        st.progress(min(1.0, prob))

    if top5[0][1] < LOW_CONFIDENCE_THRESHOLD:
        st.warning(
            "Low confidence (top prediction under 30%). This photo might not "
            "be one of the 200 CUB-200 species, or the bird is hard to make out."
        )

st.markdown("---")
st.caption(
    "Model: resnet50_finetune_testsel (ResNet-50, fine-tuned). "
    "83.5% Top-1 on the CUB-200 test set (test-set selection protocol)."
)
