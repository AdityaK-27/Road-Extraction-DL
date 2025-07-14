# 🛰️ Road Extraction from Satellite Imagery using Deep Learning

_A Comparative Study of Multiple CNN Architectures on the DeepGlobe Dataset_

## 📌 Project Overview

This project presents a comprehensive implementation and comparative analysis of deep learning models for **road extraction from high-resolution satellite imagery**. The work focuses on **semantic segmentation** using five architectures:

- U-Net
- Custom CNN
- ResUNet
- VGG16 U-Net (Transfer Learning)
- Self-Attention U-Net

The study is part of the Deep Learning course (BCSE332L) at **Vellore Institute of Technology, Chennai**, submitted in **April 2025**.

---

## 📂 Dataset

**Dataset Used:** [DeepGlobe Road Extraction Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset)  
- 📸 RGB satellite images in JPEG format  
- 🏷️ Corresponding binary road masks in PNG format  
- 📊 Total: 6,208 samples  
  - 5,600 for training  
  - 608 for validation

---

## 🧠 Model Architectures

| Model               | Description |
|--------------------|-------------|
| **U-Net**           | Baseline encoder-decoder with skip connections |
| **Custom CNN**      | Lightweight model built from scratch |
| **ResUNet**         | U-Net enhanced with residual connections |
| **VGG16 U-Net**     | U-Net decoder with a pretrained VGG16 encoder |
| **Self-Attention U-Net** | U-Net with self-attention mechanism to capture global dependencies |

---

## ⚙️ Implementation Details

- **Input Size:** 256x256 (or 224x224 for VGG16)  
- **Normalization:** RGB images scaled to `[0, 1]`  
- **Loss Function:** Binary Crossentropy (BCE), Dice Loss (optional)  
- **Optimization:** Adam / SGD  
- **Augmentation:** Applied during runtime via a custom data generator  
- **Training Platform:** Google Colab (TPU/GPU enabled)  

### 🧪 Training Setup

| Model               | Batch Size | Epochs | Optimizer | Notes |
|--------------------|------------|--------|-----------|-------|
| U-Net              | 8          | 50     | Adam      | Baseline |
| Custom CNN         | 16         | 40     | SGD       | Fast training |
| ResUNet            | 8          | 30     | Adam      | Deeper network |
| VGG16 U-Net        | 4          | 35     | Adam      | Transfer learning |
| Self-Attention U-Net | 4        | 30     | Adam      | Highest accuracy |

---

## 📈 Evaluation Metrics

- **Intersection over Union (IoU)** – Primary metric for segmentation accuracy
- **Pixel-wise Accuracy** – Secondary metric (less reliable in imbalanced datasets)

---

## 🖼️ Results

### 🔢 Quantitative Evaluation

| Model               | IoU (Trend) | Comments |
|--------------------|-------------|----------|
| U-Net              | Moderate    | Struggled with thin roads |
| Custom CNN         | Fair        | Lightweight but prone to false positives |
| ResUNet            | Moderate    | Better edges, limited by fewer epochs |
| VGG16 U-Net        | High        | Sharp boundaries, slightly over-segmented |
| Self-Attention U-Net | Excellent | Best road continuity and precision |

### 📸 Qualitative Insights
- **U-Net:** Missed thinner roads  
- **Custom CNN:** Occasionally confused non-road areas  
- **ResUNet:** Incomplete road detection  
- **VGG16 U-Net:** Very sharp edges, sometimes too eager  
- **Self-Attention U-Net:** Precise and connected roads with long-range spatial awareness

---

## 🧾 Key Learnings

- Self-attention improves global context understanding
- Transfer learning boosts early-stage performance
- Lightweight models are suitable for constrained environments but less accurate
- Residual connections improve edge preservation

---

## 🚧 Limitations

- Limited epochs for ResUNet & VGG16 U-Net due to resource constraints
- No deployment or GIS integration (yet)
- Evaluation limited to IoU and pixel accuracy

---

## 🚀 Future Work

- Incorporate **Transformer-based segmentation models**
- Post-processing with **Conditional Random Fields (CRF)**
- **Model ensembling** for robust predictions
- Real-time deployment with **GIS integration**

---

## 📁 Repository Structure

```bash
.
├── models/
│   ├── unet.ipynb
│   ├── custom_cnn.ipynb
│   ├── resunet.ipynb
│   ├── vgg16_unet.ipynb
│   └── self_attention_unet.ipynb
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── utils/
│   ├── data_generator.py
│   └── preprocessing.py
├── results/
│   ├── predictions/
│   └── comparisons/
├── README.md
└── requirements.txt
```
## 👨‍💻 Author
- Aditya Kankarwal
- 📍 Electronics and Computer Engineering
- 🧑‍🎓 VIT Chennai | 22BLC1269
- 📅 April 2025

- Special thanks to Ms. Suchita M (Deep Learning Faculty), and teammates Manan & Harsh for collaboration and support.

