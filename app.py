import streamlit as st
import os
from brain_age_inference import predict_brain_age


st.set_page_config(page_title="Brain Age Prediction", page_icon="🧠")

st.title("🧠 Brain Age Prediction")
st.write("Upload both MRI files: .hdr and .img")

hdr_file = st.file_uploader("Upload .hdr file", type=["hdr"])
img_file = st.file_uploader("Upload .img file", type=["img"])

if hdr_file is not None and img_file is not None:
    with open("temp.hdr", "wb") as f:
        f.write(hdr_file.read())

    with open("temp.img", "wb") as f:
        f.write(img_file.read())

    try:
        with st.spinner("Downloading model and predicting age..."):
            age = predict_brain_age("temp.hdr")

        st.success(f"Predicted Brain Age: {age:.2f} years")

    except Exception as e:
        st.error(f"Error: {e}")
