# 🛣️ Road Segmentation Analysis Platform

A web-based application for comparing three deep learning models for **road extraction** from satellite images. Built using TensorFlow and Gradio, the platform provides an interactive environment to visualize and analyze semantic segmentation outputs side-by-side.

**🔗 Live App**: [Launch on Hugging Face Spaces](https://huggingface.co/spaces/AdityaK-27/road-segmentation-app)

---

## 📌 Project Overview

Road segmentation from aerial and satellite imagery is a crucial task in geospatial analysis, urban planning, autonomous navigation, and disaster response. This application addresses this challenge using three distinct models trained on the DeepGlobe Road Extraction dataset.

Users can:
- Upload a custom satellite image or select from built-in examples
- Choose between binary and soft segmentation mask outputs
- Visually compare results from three different neural network architectures

---

## 🧠 Model Architectures

### 1. U-Net

The U-Net model is a widely recognized encoder-decoder architecture featuring skip connections to preserve spatial information. It is well-suited for segmentation tasks where data is limited and model interpretability is crucial.

- **Training**: 8 epochs on the DeepGlobe dataset
- **Strengths**:
  - Provides a reliable performance baseline
  - Effective on images with clear and continuous road features
- **Limitations**:
  - May struggle with detecting disconnected or occluded road segments
- **Rationale**:
  - Serves as a foundational benchmark for comparison with enhanced architectures

---

### 2. Custom CNN

This model is a lightweight convolutional neural network designed for fast inference and minimal computational overhead. It is ideal for environments where efficiency is a priority.

- **Training**: 25 epochs with data augmentation
- **Strengths**:
  - Faster training and prediction time
  - Low resource consumption
- **Limitations**:
  - May produce false positives in cluttered or urban settings
- **Rationale**:
  - Demonstrates how smaller architectures can be leveraged for practical road segmentation tasks

---

### 3. Self-Attention U-Net

An enhanced U-Net variant that integrates self-attention layers to better capture global context and spatial dependencies — particularly useful for long, winding, or fragmented roads.

- **Training**: 25 epochs using custom attention layers
- **Strengths**:
  - Delivers the most accurate segmentation results in this project
  - Handles complex road structures with greater continuity
- **Limitations**:
  - Slightly slower inference time due to attention computations
- **Rationale**:
  - Validates the benefit of modern attention mechanisms on top of classical segmentation models

---

## 📂 Dataset Details

- **Source**: [DeepGlobe Road Extraction (Kaggle)](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset)
- **Content**:
  - RGB satellite images (`.jpg`)
  - Binary road masks (`.png`)
- **Volume**: 6,208 images
  - 5,600 for training
  - 608 for validation
- **Preprocessing**:
  - All images resized to 256×256 resolution
  - Normalized pixel values (0–1 range)

---

## ⚙️ Technical Stack

| Component      | Technology         |
|----------------|--------------------|
| Model Training | TensorFlow / Keras |
| App Interface  | Gradio             |
| Hosting        | Hugging Face Spaces|
| Image I/O      | PIL, NumPy         |

---

## 🖼️ Sample Images

The application includes a collection of real satellite test images located in the `sample_inputs/` folder. These are automatically displayed in the app as example inputs for immediate testing and comparison.

---

## 📈 Performance Comparison

| Model                 | Epochs | Strengths                          | Limitations                         |
|----------------------|--------|------------------------------------|-------------------------------------|
| **U-Net**            | 8      | Effective on structured inputs     | Misses disconnected roads           |
| **Custom CNN**       | 25     | Fast and efficient                 | Occasional false positives          |
| **Self-Attention U-Net** | 25  | Best segmentation and continuity   | Slightly slower inference time      |

---

## 🚀 Deployment & Usage

### Option 1: Run Locally

Clone the repository and launch the app locally:

```bash
git clone https://github.com/AdityaK-27/Road-Extraction-DL.git
cd Road-Extraction-DL
pip install -r requirements.txt
python app.py
```

The app will launch at http://localhost:7860.

---

### Option 2: Use Online (Recommended)

Simply visit the hosted version on Hugging Face Spaces:

**🔗 Launch App**: [Launch on Hugging Face Spaces](https://huggingface.co/spaces/AdityaK-27/road-segmentation-app)

No setup or installation is required.

---

## 📁 Repository Structure
```
├── app.py                               # Main Gradio application
├── model/                               # Saved Keras model files (.h5)
│   ├── unet_model.h5
│   ├── custom_cnn_model-25.h5
│   └── u_netself_attention-25.h5
├── sample_inputs/                       # Example satellite images
│   └── test1.jpg, ...
├── requirements.txt                     # Python dependencies
├── runtime.txt                     
├── DL_Simple_CNN_model.ipynb            # Python Notebook
├── DL_U_NET2_(full).ipynb               # Python Notebook
├── DL_U_Net_Self_Attention(Full).ipynb  # Python Notebook
└── README.md
```

---

## 📚 Acknowledgements

This project was developed as part of the **Deep Learning** course at **VIT Chennai**.

- **Student**: Aditya Kankarwal 
- **Instructor**: Ms. Suchita M  
- **Dataset**: [DeepGlobe Road Extraction (Kaggle)](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset)

---

## 📬 Contact

For any queries, collaborations, or feedback, feel free to:

- 🔗 [Connect on LinkedIn](https://www.linkedin.com/in/aditya-kankarwal-68b626300/)  
- 🐛 [Raise an issue in this GitHub repository](https://github.com/AdityaK-27/Road-Extraction-DL/issues)

---
