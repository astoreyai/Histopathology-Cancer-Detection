# Histopathology Cancer Detection Notebooks

## Overview

This directory contains the main Jupyter Notebooks for training and testing a model to detect cancer in histopathology images. These notebooks are designed to be run in Kaggle, utilizing the data from Kaggle's Histopathologic Cancer Detection dataset.

### Notebooks

1. **`as_hcd_training.ipynb`**: 
   - Purpose: Train a CNN model to classify images for cancer detection.
   - Steps:
     - Clone repository and install requirements.
     - Load and preprocess data using the custom dataset and transformations.
     - Initialize model architecture and run the training loop, tracking with MLFlow.
     - Save the trained model for later testing.
   - Requirements: Kaggle dataset, Python libraries in `requirements.txt`.

2. **`as_hcd_testing.ipynb`**:
   - Purpose: Test the trained model on Kaggle's test data and create a submission file.
   - Steps:
     - Clone repository and install requirements.
     - Load pre-trained model and process test data.
     - Generate predictions and prepare submission file.
   - Requirements: Trained model file from training, Kaggle dataset, Python libraries in `requirements.txt`.

## Usage Instructions

1. Open the notebook in Kaggle.
2. Run the setup cells to clone the repository and install requirements.
3. Follow the instructions in each notebook to execute the training or testing process.

## Experiment Tracking

Experiments and metrics are tracked using MLFlow. All relevant parameters, metrics, and models are logged automatically during the execution of the training notebook. To view these logs, follow the MLFlow instructions in each notebook.

---
