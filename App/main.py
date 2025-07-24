import os
import streamlit as st
import numpy as np
from PIL import Image
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import base64

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/asl_final_model.h5"
#image_file = f"{working_dir}/background_image.png"

# Set background image
import base64

def set_background(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
set_background(f"{working_dir}/background_image.png")

st.markdown(
    """
    <style>
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: black !important;
    }
    
    /* Sidebar customization */
    section[data-testid="stSidebar"] {
        background-color: 	#009688;  /* Dark blue */
    
     /* Sidebar text in white */
    .stSidebar, .stSidebar h1, .stSidebar h2, .stSidebar h3,
    .stSidebar p, .stSidebar div, .stSidebar label {
        color: white !important;
    }
    
    .stButton > button {
        color:  rgba(255,255,255,0.7)  !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)

#loading the class names
#class_indices = json.load(open(f"{working_dir}/classes.json"))

# Load class names
@st.cache_resource
def load_class_names():
    with open(f"{working_dir}/class_names.json", "r") as f:
        return json.load(f)

# Preprocess image
def preprocess_image(img, target_size=(200, 200)):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Predict image
def predict_image(img, model, class_names):
    processed_img = preprocess_image(img)
    preds = model.predict(processed_img)
    class_idx = np.argmax(preds)
    class_name = class_names[str(class_idx)]
    confidence = preds[0][class_idx]
    return class_name, confidence

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="ASL Alphabet Classifier", page_icon="🤟", layout="centered")

st.markdown("<h1 style='text-align: center;'>🤟 ASL Alphabet Classifier 🤟</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image of a hand showing a sign, and let AI predict the alphabet!</p>", unsafe_allow_html=True)
st.markdown("---")

st.sidebar.title("📘 Instructions")
st.sidebar.info("1. Upload a JPG or PNG image showing a hand sign.\n\n2. Click **Classify**.\n\n3. View predicted alphabet and confidence score.")

upload_image = st.file_uploader("📤 Upload an ASL alphabet image", type=["jpg", "jpeg", "png"])

if upload_image is not None:
    image = Image.open(upload_image).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        resized_image = image.resize((200, 200), resample=Image.Resampling.LANCZOS)
        st.image(resized_image, caption="Uploaded Image", use_container_width=True)

    with col2:
        if st.button("Classify"):
            model = load_model()
            class_names = load_class_names()
            predicted_label, confidence = predict_image(image, model, class_names)
            st.success(f"Prediction: **{predicted_label}**")
            st.info(f"Confidence: **{confidence * 100:.2f}%**")

    st.markdown("<div style='text-align: center;'>👈 Upload an image to classify an ASL sign.</div>", unsafe_allow_html=True)
