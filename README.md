# Road-Extraction-DL
Comparative analysis of deep learning models (U-Net, ResUNet, VGG16 U-Net, Self-Attention U-Net, Custom CNN) for road extraction from satellite imagery using the DeepGlobe dataset.
# 🚧 Road Extraction from Satellite Imagery using Deep Learning

### 🔍 Semantic Segmentation with U-Net, ResUNet, VGG16 U-Net, Self-Attention U-Net, and Custom CNN

---

## 📌 Overview

This project implements and compares five deep learning models for road extraction from high-resolution satellite imagery using semantic segmentation. It uses the **DeepGlobe Road Extraction** dataset and evaluates each model on both qualitative and quantitative metrics.

---

## 🛰 Dataset

- **Source:** DeepGlobe Road Extraction Dataset
- **Images:** 6,208 RGB satellite images (JPEG)
- **Masks:** Binary grayscale masks (PNG, road = 1, background = 0)
- **Split:** 90% Training (5,600), 10% Validation (608)
- **Preprocessing:**
  - Resized to 256×256 using LANCZOS resampling
  - Normalized to [0, 1]
  - Custom data generator used for batching and augmentation

---

## 🧠 Models Implemented

| Model                | Description                                                 |
|---------------------|-------------------------------------------------------------|
| **U-Net**            | Baseline encoder-decoder with skip connections              |
| **Custom CNN**       | Lightweight architecture for faster training and inference  |
| **ResUNet**          | Adds residual blocks for better gradient flow and edge detail |
| **VGG16 U-Net**      | U-Net with pretrained VGG16 encoder for transfer learning   |
| **Self-Attention U-Net** | Integrates attention to capture long-range spatial dependencies |

---

## ⚙️ Training Configuration

- **Environment:** Google Colab (TPU/GPU)
- **Loss Function:** Binary Crossentropy (BCE), with Dice Loss for VGG16 in some runs
- **Optimizers:** Adam, SGD
- **Batch Sizes:** 4–16
- **Epochs:** 8–50
- **Callbacks Used:** `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`, `CSVLogger`

---

## 📊 Evaluation Metrics

- **Primary:** Intersection over Union (IoU)
- **Secondary:** Pixel Accuracy (used for trends only)

| Model               | Epochs | Performance Summary                                  |
|--------------------|--------|------------------------------------------------------|
| U-Net              | 8      | Moderate segmentation with road discontinuities      |
| Custom CNN         | 25     | Better continuity, some false positives              |
| ResUNet            | 10     | Fragmented output due to undertraining               |
| VGG16 U-Net        | 10     | Sharp edges but mild over-segmentation               |
| Self-Attention U-Net | 25   | Most accurate and connected segmentation             |

---

## 🖼 Qualitative Analysis

Visual inspection of predictions was conducted using an unlabeled test set to assess:
- Road continuity and connectivity
- Edge sharpness
- False positives and noise resilience

*(See `/results/` or report for outputs)*

---

## 🔬 Comparative Summary

✅ **Best Overall Performance:** Self-Attention U-Net  
📌 **Observations:**
- **Custom CNN**: Fast but less precise
- **VGG16 U-Net**: High feature quality, prone to over-segmentation
- **U-Net / ResUNet**: Need more epochs to improve

---

## 🔭 Future Work

- Incorporate transformer-based segmentation (e.g., SAM, Mask2Former)
- Add CRF or graph-based post-processing
- Deploy as a web app using Streamlit or Gradio
- Explore ensembling methods

## 📁 Project Structure

```
road-extraction-deep-learning/
├── data/
│   ├── train/                   # Training images and masks
│   ├── val/                     # Validation images and masks
│   └── test/                    # Test images (no ground truth)
│
├── models/                      # Model architecture scripts
│   ├── unet.py
│   ├── resunet.py
│   ├── custom_cnn.py
│   ├── vgg16_unet.py
│   └── attention_unet.py
│
├── notebooks/                   # Jupyter notebooks for each model
│   ├── train_unet.ipynb
│   ├── train_resunet.ipynb
│   ├── train_custom_cnn.ipynb
│   ├── train_vgg16_unet.ipynb
│   └── train_attention_unet.ipynb
│
├── utils/                       # Utility scripts (data loaders, metrics, etc.)
│   ├── data_generator.py
│   └── metrics.py
│
├── results/                     # Output predictions and training logs
│   ├── qualitative_outputs/     # Predicted masks from test images
│   └── training_logs/           # CSV logs, model checkpoints
│
├── report/
│   └── Road Extraction - Final Report.pdf
│
├── README.md
└── requirements.txt             # (Optional) Python package list
```


## 📄 Report & Resources

- 📘 **[Download Final Report](./report/Road Extraction - Final Report.pdf)**  
- 📂 Trained model weights: _[To be added if available]_  
- 💻 Full code and notebooks for each model in `/notebooks/`

