"""Streamlit app for CIFAR-10 CNN inference — HF Spaces entry point."""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st

from model import CNN, CIFAR10_CLASSES

st.set_page_config(page_title="pixel-sense", page_icon="🖼️")


@st.cache_resource
def load_model():
    model = CNN()
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

st.title("🖼️ pixel-sense")
st.write("A CNN trained from scratch on CIFAR-10 — upload an image and get a live prediction.")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    model = load_model()
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0]

    top5_prob, top5_idx = torch.topk(probs, 5)

    st.subheader("Top-5 Predictions")
    for prob, idx in zip(top5_prob, top5_idx):
        label = CIFAR10_CLASSES[idx.item()]
        st.write(f"**{label}** — {prob.item() * 100:.2f}%")
        st.progress(prob.item())