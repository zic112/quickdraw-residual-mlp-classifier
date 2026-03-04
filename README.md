# QuickDraw Residual MLP Classifier

A deep learning project for classifying sketches from the **Google QuickDraw dataset** using fully connected neural networks.

This repository explores how far **MLP-based architectures** can be pushed for image classification when combined with modern training techniques such as **residual connections, data augmentation, and advanced optimizers**.

The final model is a **Residual Multi-Layer Perceptron (ResMLP)** trained with **OneCycle learning rate scheduling, MixUp augmentation, and label smoothing**.

---

# Project Objective

The goal of this project is to classify hand-drawn sketches into **15 categories** using neural networks.

Unlike typical image classification pipelines that rely on convolutional neural networks (CNNs), this project focuses entirely on **MLP-based architectures** while respecting architectural constraints.

The project investigates:

- Shallow vs deep neural networks
- Width vs depth trade-offs
- Residual connections in MLPs
- Data augmentation for sketch data
- Optimization strategies for faster convergence

---

# Dataset

The dataset used is derived from the **Google QuickDraw Dataset**.

Each sample contains a simple grayscale sketch.

### Data Properties

| Property | Value |
|--------|------|
| Image size | 28 × 28 |
| Input features | 784 (flattened) |
| Number of classes | 15 |
| Format | `.npz` |

### Example Workflow
