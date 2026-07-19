"""Streamlit app for CIFAR-10 CNN inference — HF Spaces entry point."""

import torch
import torch.nn.functional as F
from PIL import Image
import streamlit as st
import numpy as np



from model import CNN, CIFAR10_CLASSES

st.set_page_config(page_title="pixel-sense", page_icon="🖼️")


@st.cache_resource
def load_model():
    model = CNN()
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model



def preprocess(image: Image.Image) -> torch.Tensor:
    image = image.resize((32, 32))
    arr = np.array(image).astype(np.float32) / 255.0        # HWC, [0,1]
    arr = (arr - 0.5) / 0.5                                   # normalize to [-1,1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1)           # HWC -> CHW
    return tensor.unsqueeze(0)                                 # add batch dim

st.title("Pixel Sense")
st.write("A CNN trained from scratch on CIFAR-10 — upload an image and get a live prediction.")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    model = load_model()
    input_tensor = preprocess(image)
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0]

    top5_prob, top5_idx = torch.topk(probs, 5)

    st.subheader("Top-5 Predictions")
    for prob, idx in zip(top5_prob, top5_idx):
        label = CIFAR10_CLASSES[idx.item()]
        st.write(f"**{label}** — {prob.item() * 100:.2f}%")
        st.progress(prob.item())