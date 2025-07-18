# # Geometric Shape Image Classification

## Overview

This project implements a full machine learning pipeline for synthetic image classification. It covers data generation, custom feature extraction, kernel engineering, classical and neural network modeling, and model evaluation—all built from scratch using only NumPy (plus Matplotlib/OpenCV for support tasks). The system is designed to focus on algorithms and engineering rather than reliance on high-level libraries.

## Features

- **Synthetic Dataset Generation:**  
  Creates large, labeled datasets of geometric shape images (circles, squares, triangles) with random placements, sizes, and colors, supporting user-defined complexity and balance.
- **Data Processing and Visualization:**  
  Includes tools for image scaling, overlap-checking, and advanced methods to display and inspect the generated data visually.
- **Patch-Based Feature Extraction:**  
  Extracts grid-based, color-agnostic spatial descriptors for each patch of an image, providing robust representations for classification.
- **Custom Kernel Engineering:**  
  Implements a set kernel comparing image patches for use in kernel SVMs, capturing structural similarities beyond pixel values.
- **Classic and Deep Learning Models:**  
  Contains a NumPy-based Support Vector Machine system and a fully custom CNN implementation with manual forward pass, pooling, and backprop only in the final dense layers.
- **Model Evaluation:**  
  Tools for ROC/AUC evaluation, confidence bands via quantile analysis, computation of baseline/majority accuracy, and standard accuracy utilities.

## Key logics Implemented From Scratch
  - Image dataset synthesis, scaling, shape placement, and label assignment
  - Grid-based spatial color feature computation (centroids, variance, skewness per color per patch)
  - Aggregated patch descriptors for color-agnostic classification
  - Grid-based set kernel for pairwise similarity
  - SVM kernel matrix computation without scikit-learn
  - Simple CNN with manual convolution, ReLU, pooling, flattening, and two-layer perceptron (fully NumPy-based)
  - Custom ROC/AUC/quantile evaluation utilities
