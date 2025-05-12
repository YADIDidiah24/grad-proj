# 3D Medical Image Segmentation Application

## Overview

The 3D Medical Image Segmentation Application is a sophisticated tool designed to facilitate the segmentation of 3D medical images using advanced neural network architectures. This application leverages deep learning techniques to accurately segment medical images, which is crucial for various medical diagnoses and treatment planning.

## Features

- **Multiple Neural Network Architectures**:
  - 3D U-Net
  - 3D V-Net
  - SA-Net (Self-Attention Network)
  - E1D3 U-Net (Enhanced 1D-3D U-Net)
  - HDC Net (Hybrid Dilated Convolution Network)

- **Dataset Handling**:
  - Loading and preprocessing of NIfTI format medical images and labels.
  - Resizing, normalizing, and binarizing images and labels.
  - Splitting datasets into training and testing sets.

- **Training and Evaluation**:
  - Data generators for efficient loading and preprocessing of 3D medical images.
  - Training multiple models with configurable parameters.
  - Evaluation metrics including Dice coefficient, accuracy, precision, recall, and F1 score.

- **Visualization**:
  - Tools for exploring dataset statistics, model architectures, and segmentation results.
  - 2D and 3D visualization of segmentation results.

- **User Interface**:
  - Intuitive and interactive web interface built with Streamlit.
  - Sidebar for navigation, dataset input, model selection, and training parameters.
  - Display of training progress, model metrics, and visualization of segmentation results.

- **Model Comparison and Export**:
  - Comparison of multiple trained models based on various performance metrics.
  - Exporting trained models in different formats (e.g., Pickle, Joblib, HDF5, Keras).

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- TensorFlow 2.x
- Keras
- NumPy
- SciPy
- NiBabel
- Streamlit
- Matplotlib
- Plotly
- Scikit-learn

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/3d-medical-image-segmentation.git
   cd 3d-medical-image-segmentation
