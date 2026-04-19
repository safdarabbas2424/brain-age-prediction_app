import streamlit as st
import torch
import timm
import nibabel as nib
import numpy as np
import cv2
import torch.nn as nn

from PIL import Image
from torchvision import transforms


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class SwinAgePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=True,
            num_classes=0
        )
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, axial, coronal, sagittal):
        x1 = self.backbone(axial)
        x2 = self.backbone(coronal)
        x3 = self.backbone(sagittal)
        x = torch.cat([x1, x2, x3], dim=1)
        return self.head(x).squeeze(1)


@st.cache_resource
def load_model():
    model = SwinAgePredictor()
    model.load_state_dict(torch.load("best_swin_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model


model = load_model()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_and_extract_slices(file_path):
    img = nib.load(file_path)
    data = np.squeeze(img.get_fdata())

    axial = data[:, :, data.shape[2] // 2]
    coronal = data[:, data.shape[1] // 2, :]
    sagittal = data[data.shape[0] // 2, :, :]

    return axial, coronal, sagittal


def preprocess(slice_img):
    slice_img = np.nan_to_num(slice_img)
    norm = cv2.normalize(slice_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    resized = cv2.resize(norm, (224, 224))
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)


def predict(file_path):
    axial, coronal, sagittal = load_and_extract_slices(file_path)

    axial = transform(preprocess(axial)).unsqueeze(0).to(device)
    coronal = transform(preprocess(coronal)).unsqueeze(0).to(device)
    sagittal = transform(preprocess(sagittal)).unsqueeze(0).to(device)

    with torch.no_grad():
        age = model(axial, coronal, sagittal).item()

    return age


st.title("🧠 Brain Age Prediction")
st.write("Upload both Analyze-format MRI files: .hdr and .img")

hdr_file = st.file_uploader("Upload .hdr file", type=["hdr"])
img_file = st.file_uploader("Upload .img file", type=["img"])

if hdr_file is not None and img_file is not None:
    with open("temp.hdr", "wb") as f:
        f.write(hdr_file.read())

    with open("temp.img", "wb") as f:
        f.write(img_file.read())

    try:
        age = predict("temp.hdr")
        st.success(f"Predicted Brain Age: {age:.2f} years")
    except Exception as e:
        st.error(f"Error: {e}")