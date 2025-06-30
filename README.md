# 🐟 Multiclass Fish Image Classification

A deep learning project for classifying fish species using CNN and transfer learning models, deployed via Streamlit.

![Demo](https://via.placeholder.com/800x400?text=Fish+Classification+Demo) *(Replace with actual demo GIF)*

## 📌 Features

- **Two Model Architectures**
  - Custom CNN from scratch
  - Fine-tuned MobileNetV2 (transfer learning)
- **Data Augmentation** with rotation, flipping, and zoom
- **Model Comparison** with accuracy and inference time metrics
- **Streamlit Web App** for real-time predictions
- **Automatic Validation Split** during training

**Usage:**
**1:Preprocessing**
python preprocess.py

**2:Training**
python train.py
**Outputs:**

models/cnn_model.h5

models/mobilenet_model.h5

reports/model_comparison.md

**3.Evaluation**
python evaluate.py

**4.Streamlit app**
streamlit run app.py

**Model performance:**
                 accuracy  inference_time_ms
cnn_model        0.973957         158.631811
mobilenet_model  0.994666         395.220840
