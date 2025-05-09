import streamlit as st
import numpy as np
import nibabel as nib
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt
import gdown
import zipfile
import tempfile

# Title
st.title("Brain Tumor Segmentation using 3D U-Net")

# --- Download Model from Google Drive ---
def download_default_model():
    file_id = "1lV1SgafomQKwgv1NW2cjlpyb4LwZXFwX"  # Replace if needed
    output_path = "default_model.keras"
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return output_path

@st.cache_resource
def load_default_model():
    model_path = download_default_model()
    return load_model(model_path, compile=False)

default_model = load_default_model()

# --- Preprocess Function ---
def preprocess_nifti(file_path):
    image = nib.load(file_path).get_fdata()
    image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-5)
    return image

# --- Combine Channels ---
def combine_channels(t1n, t1c, t2f, t2w):
    for i, img in enumerate([t1n, t1c, t2f, t2w]):
        if img.shape != t1n.shape:
            raise ValueError(f"Image {i} has different shape: {img.shape}")
    combined = np.stack([t1n, t1c, t2f, t2w], axis=-1)
    combined = combined[56:184, 56:184, 13:141]  # Optional crop
    return combined

# --- Segmentation ---
def run_segmentation(model, input_image):
    if input_image.ndim != 4:
        raise ValueError("Expected shape (H, W, D, C), got " + str(input_image.shape))
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dim
    st.write("Input to model:", input_image.shape)
    prediction = model.predict(input_image)
    return np.argmax(prediction, axis=-1)[0]

# --- Sidebar: Upload Custom Model ---
st.sidebar.header("Upload Your Own Model")
uploaded_model = st.sidebar.file_uploader("Upload a Keras model (.keras)", type=["keras"])

if uploaded_model:
    with open("temp_model.keras", "wb") as f:
        f.write(uploaded_model.getbuffer())
    try:
        model = load_model("temp_model.keras", compile=False)
        st.sidebar.success("Custom model loaded!")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")
        model = default_model
else:
    model = default_model

# --- Upload ZIP ---
st.header("Upload a ZIP with NIfTI files: T1n, T1c, T2f, T2w")
uploaded_folder = st.file_uploader("Upload zip", type=["zip"])

if uploaded_folder:
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "input.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_folder.getbuffer())
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        # Locate files
        paths = {"t1n": None, "t1c": None, "t2f": None, "t2w": None, "mask": None}
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if "t1n" in file: paths["t1n"] = os.path.join(root, file)
                elif "t1c" in file: paths["t1c"] = os.path.join(root, file)
                elif "t2f" in file: paths["t2f"] = os.path.join(root, file)
                elif "t2w" in file: paths["t2w"] = os.path.join(root, file)
                elif "seg" in file: paths["mask"] = os.path.join(root, file)

        if all(paths[k] for k in ["t1n", "t1c", "t2f", "t2w"]):
            # Load and preprocess
            t1n = preprocess_nifti(paths["t1n"])
            t1c = preprocess_nifti(paths["t1c"])
            t2f = preprocess_nifti(paths["t2f"])
            t2w = preprocess_nifti(paths["t2w"])

            combined_image = combine_channels(t1n, t1c, t2f, t2w)
            st.write("Combined image shape:", combined_image.shape)

            st.write("Running segmentation...")
            seg_result = run_segmentation(model, combined_image)

            # Visualization
            slices = [75, 90, 100]
            fig, ax = plt.subplots(len(slices), 2, figsize=(10, 10))
            for i, z in enumerate(slices):
                ax[i, 0].imshow(np.rot90(combined_image[:, :, z, 0]), cmap="gray")
                ax[i, 0].set_title(f"Input - Slice {z}")
                ax[i, 1].imshow(np.rot90(seg_result[:, :, z]))
                ax[i, 1].set_title(f"Prediction - Slice {z}")
            st.pyplot(fig)

            # Save segmentation result
            output_file = "segmentation_result.nii.gz"
            nib.save(nib.Nifti1Image(seg_result.astype(np.uint8), np.eye(4)), output_file)

            # Download link
            with open(output_file, "rb") as f:
                st.download_button("Download Segmentation Result", f, file_name="segmentation_result.nii.gz")

            os.remove(output_file)
        else:
            st.error("Missing required NIfTI files (t1n, t1c, t2f, t2w)")
