# 🧠 Brain Age Prediction using Deep Learning

This project predicts brain age from MRI scans using deep learning models.

---

## 📌 Project Overview

- Input: 3D Brain MRI (Analyze format: .hdr + .img)
- Output: Predicted brain age
- Task: Regression
- Metric: Mean Absolute Error (MAE)

---

## 🧠 Models Used

- Custom CNN (baseline)
- ResNet-50 (pretrained)
- Swin Transformer (pretrained - best model)

---

## ⚙️ Pipeline

1. Load 3D MRI scan
2. Extract center slices:
   - Axial
   - Coronal
   - Sagittal
3. Normalize pixel values
4. Resize to 224x224
5. Convert grayscale → 3-channel
6. Pass through model
7. Predict brain age

---

## 📊 Evaluation

- Metric: Mean Absolute Error (MAE)
- Swin Transformer achieved best performance

---

## 📁 Project Structure
