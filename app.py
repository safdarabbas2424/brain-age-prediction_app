import os
import shutil
import tempfile
import streamlit as st

from brain_age_inference import predict_brain_age


st.set_page_config(page_title="Brain Age Prediction", page_icon="🧠")

st.title("🧠 Brain Age Prediction")
st.write("Upload both MRI files: .hdr and .img")

hdr_file = st.file_uploader("Upload .hdr file", type=["hdr"])
img_file = st.file_uploader("Upload .img file", type=["img"])

if hdr_file is not None and img_file is not None:
    try:
        # Create temporary folder
        temp_dir = tempfile.mkdtemp()

        # Save both files with the SAME base name
        hdr_path = os.path.join(temp_dir, "scan.hdr")
        img_path = os.path.join(temp_dir, "scan.img")

        with open(hdr_path, "wb") as f:
            f.write(hdr_file.read())

        with open(img_path, "wb") as f:
            f.write(img_file.read())

        with st.spinner("Downloading model and predicting age..."):
            age = predict_brain_age(hdr_path)

        st.success(f"Predicted Brain Age: {age:.2f} years")

    except Exception as e:
        st.error(f"Error: {e}")

    finally:
        # Clean up temp folder
        if "temp_dir" in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
