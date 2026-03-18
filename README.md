# 🧠 Deep Learning Framework for Alzheimer’s and Parkinson’s Classification

A deep learning-based research framework for automated classification of Alzheimer’s and Parkinson’s diseases using structural brain MRI scans.

This project focuses on accurate prediction and model interpretability for clinical decision support.

---

## 📌 Overview

This system uses transfer learning with advanced convolutional neural networks (CNNs) to classify brain MRI images.

It supports:
- Alzheimer’s disease detection (Demented vs Non-Demented)
- Parkinson’s disease detection (Parkinson vs Normal)

The framework is designed for research use and easy deployment using Google Colab.

---

## 🏗️ Core Architecture

Two pre-trained deep learning models are used:

### 🔹 ResNet50
- Deep residual network
- Handles complex brain image features
- Strong performance on medical imaging tasks

### 🔹 EfficientNet-B0
- Lightweight and efficient
- Optimized for limited hardware environments
- Good balance between speed and accuracy

---

## ⚙️ Key Features

### ✅ Binary Classification Logic

- **Alzheimer’s Disease**
  - Combines stages (Very Mild → Moderate) into one **“Demented”** class
  - Improves early-stage detection

- **Parkinson’s Disease**
  - Classifies into:
    - Normal
    - Parkinson

---

### 🧪 Data Processing Pipeline

- Image resizing: **224 × 224**
- Normalization for stable training
- Data augmentation:
  - Rotation
  - Zoom
  - Flipping
- Improves generalization and reduces overfitting

---

### ⚖️ Class Imbalance Handling

- Uses **class weighting**
- Helps model learn better from underrepresented classes
- Important for Alzheimer’s multi-stage data

---

### 🔍 Explainable AI (Grad-CAM)

- Implements **Gradient-weighted Class Activation Mapping**
- Generates heatmaps on MRI scans
- Shows regions used for prediction (e.g., hippocampus, cortex)

This improves:
- Model transparency  
- Clinical trust  
- Interpretability  

---

## 📊 Research Significance

This framework combines:
- High classification performance  
- Model interpretability  

It is suitable for:
- Medical imaging research  
- AI in healthcare studies  
- Clinical decision support exploration  

---

## 🧰 Tech Stack

- Python  
- TensorFlow / Keras  
- NumPy  
- OpenCV  
- Matplotlib  
- Google Colab  

---


