# Lightweight Multispectral Image Fusion

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A lightweight deep learning architecture designed for **multispectral image fusion** (e.g., Pansharpening, RGB + IR fusion, or hyperspectral enhancement). Built with efficiency in mind, this model minimizes parameters and FLOPs to make high-quality spatial and spectral fusion feasible on edge devices and resource-constrained hardware.

---

## 📌 Features

- **Efficient Architecture:** Optimized feature extraction and fusion blocks with reduced computational overhead.
- **Spectral & Spatial Preservation:** Preserves high-resolution spatial details while retaining accurate spectral fidelity.
- **Edge-Ready:** Low parameter count for fast inference on embedded systems, drones, and satellite edge hardware.
- **Flexible Data Support:** Compatible with satellite datasets (e.g., Sentinel-2, Landsat, WorldView) and thermal/infrared pairs.

---

## 📂 Project Structure

```text
Lightweight-Multispectral-Image-Fusion/
├── data/              # Sample multispectral & high-res images
├── models/            # Lightweight fusion network architectures
├── utils/             # Data loaders, loss functions, and metrics (ERGAS, SAM, PSNR)
├── train.py           # Training script
├── test.py            # Evaluation & inference script
└── requirements.txt   # Dependencies
