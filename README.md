# 🛣️ Road Segmentation Analysis Platform

A web-based application built to compare and visualize the performance of three deep learning models for **road extraction** from satellite images. This tool provides a hands-on experience for analyzing outputs from:

- 🔹 U-Net  
- 🔹 Custom CNN  
- 🔹 Self-Attention U-Net

The application is hosted on **[Live Demo](https://huggingface.co/spaces/AdityaK-27/road-segmentation-app)** using **Gradio**.

---

## 📌 Overview

This project addresses the problem of **road segmentation from satellite imagery** — a critical task in urban planning, autonomous driving, and infrastructure monitoring. We trained and deployed three semantic segmentation models that detect roads from RGB satellite images and output binary or probabilistic masks.

The app allows users to:
- Upload a satellite image
- Choose between binary or soft masks
- Get results from 3 different models side-by-side
- Test with preloaded example images

---

## 🚀 Demo

👉 **[Try the Live App Here](https://huggingface.co/spaces/AdityaK-27/road-segmentation-app)**  
No sign-in required. Preloaded examples available.

---

## 🧠 Models Used

### 1️⃣ U-Net
A well-known encoder-decoder architecture with skip connections. Trained for 8 epochs using the DeepGlobe Road Extraction dataset.

- ✅ Fast and interpretable  
- ⚠️ Struggles with road discontinuities in complex images

### 2️⃣ Custom CNN
A lightweight architecture designed for fast inference and reduced computational load. Trained for 25 epochs.

- ✅ Efficient for quick prototyping  
- ⚠️ Occasionally misclassifies background as road (false positives)

### 3️⃣ Self-Attention U-Net
An advanced U-Net variant that incorporates self-attention layers to capture long-range dependencies. Trained for 25 epochs.

- ✅ Best overall performance in terms of segmentation accuracy and road continuity  
- ⚠️ Slower inference due to attention complexity

---

## 📂 Dataset

**Dataset Used:** [DeepGlobe Road Extraction Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset)  
- 📸 RGB satellite images in JPEG format  
- 🏷️ Corresponding binary road masks in PNG format  
- 📊 Total: 6,208 samples  
  - 5,600 for training  
  - 608 for validation

---

## 🛠️ Deployment Details

### 🔧 Frameworks and Tools
- TensorFlow / Keras
- Gradio for the front-end UI
- Hugging Face Spaces for hosting
- PIL & NumPy for image processing

### 📁 Folder Structure (for GitHub repo)

```
├── app.py
├── model/
│ ├── unet_model.h5
│ ├── custom_cnn_model-25.h5
│ └── u_netself_attention-25.h5
├── sample_inputs/
│ └── [example satellite images]
├── requirements.txt
├── README.md

yaml
Copy
Edit
```

---

## 🧪 How to Run Locally

Clone the repository and run the app locally:

```bash
git clone https://github.com/AdityaK-27/Road-Extraction-DL.git
cd Road-Extraction-DL
pip install -r requirements.txt
python app.py
```

The app will launch at `http://localhost:7860`.

---

## 📷 Sample Images

The app includes a collection of real satellite images for quick testing. These are located in the `sample_inputs/` folder and automatically loaded into the interface for one-click evaluation.

---

## 📈 Performance Summary

| Model                 | Epochs | Strengths                        | Limitations                        |
|----------------------|--------|----------------------------------|------------------------------------|
| U-Net                | 8      | Simple, effective on clean inputs | Misses disconnected roads          |
| Custom CNN           | 25     | Fast and efficient               | False positives on non-road objects|
| Self-Attention U-Net | 25     | Best segmentation and continuity | Slower inference time              |

---

## 📚 Acknowledgements

This project was developed as part of the Deep Learning course at **VIT Chennai**.

- **Student**: Aditya Kankarwal (22BLC1269)  
- **Instructor**: Ms. Suchita M  
- **Dataset**: DeepGlobe Road Extraction Challenge

