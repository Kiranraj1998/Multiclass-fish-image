import os
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import time

# 1. Configuration
BASE_DIR = r"C:\Guvi\my-project\Multiclass fish image\fish-classification"
MODEL_DIR = os.path.join(BASE_DIR, "models")

# 2. Load Class Names
try:
    with open(os.path.join(MODEL_DIR, "class_names.json")) as f:
        class_names = json.load(f)
except FileNotFoundError:
    st.error("Class names file not found! Please train models first.")
    st.stop()

# 3. Model Loading
@st.cache_resource
def load_model(model_path):
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.info("Please ensure you've run train.py first")
        return None

# 4. App Interface
def main():
    st.set_page_config(page_title="Fish Classifier", layout="wide")
    
    # Model Selection
    st.sidebar.header("Model Selection")
    model_type = st.sidebar.radio(
        "Choose Model",
        ["CNN", "MobileNetV2"],
        index=0
    )
    model_path = os.path.join(
        MODEL_DIR,
        "cnn_model.h5" if model_type == "CNN" else "mobilenet_model.h5"
    )
    model = load_model(model_path)
    
    if not model:
        st.stop()  # Stop if model failed to load

    # Main UI
    st.title("🎣 Fish Species Classifier")
    upload = st.file_uploader(
        "Upload a fish image",
        type=["jpg", "jpeg", "png"]
    )

    if upload:
        # Preprocessing
        img = Image.open(upload).resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        with st.spinner('Analyzing...'):
            start_time = time.time()
            pred = model.predict(img_array)
            inference_time = time.time() - start_time

        # Results
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.subheader("Results")
            predicted_class = class_names[np.argmax(pred)]
            confidence = np.max(pred)
            
            st.metric("Predicted Species", predicted_class)
            st.metric("Confidence", f"{confidence:.2%}")
            st.metric("Inference Time", f"{inference_time:.3f}s")

            st.subheader("Class Probabilities")
            prob_data = {class_names[i]: float(pred[0][i]) 
                        for i in range(len(class_names))}
            st.bar_chart(prob_data)

    # Model Info
    with st.expander("ℹ️ Model Information"):
        st.markdown(f"""
        **Selected Model:** `{model_type}`
        
        - **Input Shape:** 224×224 RGB
        - **Classes Supported:** {len(class_names)} species
        - **Available Models:**
            - `CNN`: Custom convolutional neural network
            - `MobileNetV2`: Pretrained on ImageNet with fine-tuning
        """)

if __name__ == "__main__":
    main()