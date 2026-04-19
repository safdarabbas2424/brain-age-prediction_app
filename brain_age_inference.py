import os
import cv2
import torch
import timm
import nibabel as nib
import numpy as np
import torch.nn as nn
import urllib.request

from PIL import Image
from torchvision import transforms


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Model download settings
# -----------------------------
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "best_swin_model.pth")
MODEL_URL = "https://drive.google.com/uc?export=download&id=1OAW3tNWuMnNBSF4z6viKhjw8xr8c9UtG"


# -----------------------------
# Download model if not exists
# -----------------------------
def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully.")

    return MODEL_PATH


# -----------------------------
# Swin Transformer Model
# -----------------------------
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


# -----------------------------
# Load model
# -----------------------------
def load_model():
    model_path = download_model()

    model = SwinAgePredictor()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model


# -----------------------------
# Transform
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# Load MRI and extract slices
# -----------------------------
def load_and_extract_slices(file_path):
    img = nib.load(file_path)
    data = np.squeeze(img.get_fdata())

    axial = data[:, :, data.shape[2] // 2]
    coronal = data[:, data.shape[1] // 2, :]
    sagittal = data[data.shape[0] // 2, :, :]

    return axial, coronal, sagittal


# -----------------------------
# Preprocess slice
# -----------------------------
def preprocess_slice(slice_img, size=224):
    slice_img = np.nan_to_num(slice_img)
    norm = cv2.normalize(slice_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    resized = cv2.resize(norm, (size, size), interpolation=cv2.INTER_AREA)
    rgb_img = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb_img)


# -----------------------------
# Predict brain age
# -----------------------------
def predict_brain_age(file_path):
    model = load_model()

    axial, coronal, sagittal = load_and_extract_slices(file_path)

    axial_img = transform(preprocess_slice(axial)).unsqueeze(0).to(device)
    coronal_img = transform(preprocess_slice(coronal)).unsqueeze(0).to(device)
    sagittal_img = transform(preprocess_slice(sagittal)).unsqueeze(0).to(device)

    with torch.no_grad():
        predicted_age = model(axial_img, coronal_img, sagittal_img).item()

    return predicted_age
