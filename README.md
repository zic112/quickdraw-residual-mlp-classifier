# QuickDraw Residual MLP Classifier

A deep learning project for classifying sketches from the **Google
QuickDraw dataset** using fully connected neural networks.

This repository explores how far **MLP-based architectures** can be
pushed for image classification when combined with modern training
techniques such as **residual connections, data augmentation, and
advanced optimizers**.

The final model is a **Residual Multi-Layer Perceptron (ResMLP)**
trained with **OneCycle learning rate scheduling, MixUp augmentation,
and label smoothing**.

------------------------------------------------------------------------

# Project Objective

The goal of this project is to classify hand-drawn sketches into **15
categories** using neural networks.

Unlike typical image classification pipelines that rely on convolutional
neural networks (CNNs), this project focuses entirely on **MLP-based
architectures** while respecting architectural constraints.

The project investigates:

-   Shallow vs deep neural networks
-   Width vs depth trade-offs
-   Residual connections in MLPs
-   Data augmentation for sketch data
-   Optimization strategies for faster convergence

------------------------------------------------------------------------

# Dataset

The dataset used is derived from the **Google QuickDraw Dataset**.

Each sample contains a simple grayscale sketch.

## Data Properties

  Property            Value
  ------------------- -----------------
  Image size          28 × 28
  Input features      784 (flattened)
  Number of classes   15
  Format              `.npz`

### Workflow

28×28 image → Flatten → 784‑dimensional vector → MLP classifier

------------------------------------------------------------------------

# Model Architecture

The final model is a **Residual Multi‑Layer Perceptron (ResMLP)** with
stochastic skip connections.

### Champion Model Structure

Input (784)\
↓\
Linear Layer\
↓\
Residual Blocks (×10)\
• BatchNorm\
• GELU Activation\
• Dropout\
• Linear Layer\
↓\
Stochastic Skip Connection\
↓\
Output Layer (15 classes)

### Key Features

-   Fully connected architecture
-   Residual connections
-   Batch normalization
-   GELU activation
-   Dropout regularization
-   Stochastic depth (skip connections)

These components allow deeper networks while maintaining stable
gradients.

------------------------------------------------------------------------

# Training Pipeline

Several training improvements were applied to improve model performance.

## Data Augmentation

Sketch‑specific augmentation techniques:

-   Random affine transformations
-   Random perspective distortion
-   Random erasing
-   Gaussian noise injection

These augmentations help the model generalize better to variations in
sketches.

------------------------------------------------------------------------

## Regularization

To prevent overfitting:

-   Dropout
-   Label smoothing
-   MixUp augmentation

These techniques encourage smoother decision boundaries and improve
robustness.

------------------------------------------------------------------------

## Optimization Strategy

### Optimizer

AdamW

### Learning Rate Scheduler

OneCycleLR

Benefits:

-   Faster convergence
-   Stable training
-   Better generalization

Additional stabilization techniques:

-   Gradient clipping
-   Stochastic Weight Averaging (SWA)

------------------------------------------------------------------------

# Training Results

  Model      Description            Parameters   Validation Accuracy
  ---------- ---------------------- ------------ ---------------------
  Pancake    Wide shallow network   \~0.6M       \~72%
  Tower      Deeper architecture    \~1.2M       \~78%
  Champion   Residual MLP           \~1.28M      **\~83--84%**

Residual learning significantly improved convergence and final
performance.

------------------------------------------------------------------------

# Confusion Matrix Analysis

A confusion matrix was generated on the validation set.

The most confused classes were visually similar sketches with ambiguous
shapes.\
This confusion is likely caused by **aleatoric uncertainty**, meaning
the sketches themselves are ambiguous rather than the model lacking
capacity.

------------------------------------------------------------------------

# Inference Pipeline

The repository includes a script to generate predictions on the test
dataset.

Process:

1.  Load trained model weights
2.  Load test dataset
3.  Preprocess sketches
4.  Run model predictions
5.  Export predictions to CSV

Example output:

0,3,5,2,1,4,7,6,3,0,...

------------------------------------------------------------------------

# Repository Structure

quickdraw-residual-mlp-classifier/

│\
├── notebooks\
│ ├── training.ipynb\
│ └── inference.ipynb\
│\
├── saved_models\
│ └── mc5_trained.pt\
│\
├── predictions\
│ └── test_predictions.csv\
│\
├── data\
│ ├── quickdraw_train.npz\
│ └── quickdraw_test.npz\
│\
└── README.md

------------------------------------------------------------------------

# Running the Project

## Clone Repository

git clone
https://github.com/yourusername/quickdraw-residual-mlp-classifier

## Install Dependencies

pip install torch torchvision numpy matplotlib

## Train the Model

Open:

notebooks/training.ipynb

This notebook:

-   trains the models
-   evaluates validation accuracy
-   saves the best model

## Run Inference

Open:

notebooks/inference.ipynb

This notebook:

-   loads saved model
-   generates test predictions
-   outputs CSV predictions

------------------------------------------------------------------------

# Key Takeaways

This project demonstrates:

-   Deep MLPs can perform well on image data.
-   Residual connections improve gradient flow.
-   Data augmentation significantly improves generalization.
-   Modern optimizers and schedulers accelerate training.

------------------------------------------------------------------------

