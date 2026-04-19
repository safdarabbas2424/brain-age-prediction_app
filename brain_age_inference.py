import os
import cv2
import torch
import timm
import nibabel as nib
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from glob import glob
from torchvision import transforms


# -----------------------------
# 1. Device
# -----------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# -----------------------------
# 2. Model class
# -----------------------------
class SwinAgePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=False,
            num_classes=0
        )
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, axial, coronal, sagittal):
        x1 = self.backbone(axial)
        x2 = self.backbone(coronal)
        x3 = self.backbone(sagittal)
        x = torch.cat([x1, x2, x3], dim=1)
        return self.head(x).squeeze(1)


# -----------------------------
# 3. Load saved model
# -----------------------------
MODEL_PATH = "best_swin_model.pth"

model = SwinAgePredictor()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print("Model loaded successfully.")


# -----------------------------
# 4. Transform
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# 5. Load MRI and extract slices
# -----------------------------
def load_and_extract_slices(hdr_path):
    img = nib.load(hdr_path)
    data = np.squeeze(img.get_fdata())

    axial = data[:, :, data.shape[2] // 2]
    coronal = data[:, data.shape[1] // 2, :]
    sagittal = data[data.shape[0] // 2, :, :]

    return axial, coronal, sagittal


# -----------------------------
# 6. Preprocess one slice
# -----------------------------
def preprocess_slice(slice_img, size=224):
    slice_img = np.nan_to_num(slice_img)
    norm = cv2.normalize(slice_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    resized = cv2.resize(norm, (size, size), interpolation=cv2.INTER_AREA)
    rgb_img = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb_img)


# -----------------------------
# 7. Prediction function
# -----------------------------
def predict_brain_age(hdr_path):
    axial, coronal, sagittal = load_and_extract_slices(hdr_path)

    axial_img = transform(preprocess_slice(axial)).unsqueeze(0).to(device)
    coronal_img = transform(preprocess_slice(coronal)).unsqueeze(0).to(device)
    sagittal_img = transform(preprocess_slice(sagittal)).unsqueeze(0).to(device)

    with torch.no_grad():
        predicted_age = model(axial_img, coronal_img, sagittal_img).item()

    return predicted_age


# -----------------------------
# 8. Visualize slices (optional)
# -----------------------------
def show_slices(hdr_path):
    axial, coronal, sagittal = load_and_extract_slices(hdr_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(axial.T, cmap="gray", origin="lower")
    axes[0].set_title("Axial")
    axes[1].imshow(coronal.T, cmap="gray", origin="lower")
    axes[1].set_title("Coronal")
    axes[2].imshow(sagittal.T, cmap="gray", origin="lower")
    axes[2].set_title("Sagittal")
    plt.tight_layout()
    plt.show()


# -----------------------------
# 9. Test on one MRI file
# -----------------------------
sample_hdr = "/Users/safdarabbas/Downloads/Data MRI/disc1/OAS1_0001_MR1/PROCESSED/MPRAGE/T88_111/OAS1_0001_MR1_mpr_n4_anon_111_t88_gfc.hdr"

show_slices(sample_hdr)

predicted_age = predict_brain_age(sample_hdr)
print(f"Predicted Brain Age: {predicted_age:.2f} years")