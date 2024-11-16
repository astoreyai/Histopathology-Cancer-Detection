# Histopathology Cancer Detection

This repository provides a modular and reproducible framework for detecting cancerous cells in histopathology images using Convolutional Neural Networks (CNNs). It includes custom dataset loading, preprocessing, model training, validation, and prediction utilities, tailored for binary classification tasks in histopathology image analysis.

---

## Project Structure

```plaintext
├── data/                             # Placeholder for data (loaded from Kaggle or other sources)
├── mlruns/                           # Directory for MLflow experiment tracking
├── experiments/                      # Directory for experiment outputs and visualizations
├── scripts/                          # Core scripts for project implementation
│   ├── __init__.py                   # Initializes project modules
│   ├── config.py                     # Configuration file with dataset paths and hyperparameters
│   ├── data_utils.py                 # Dataset classes and utility functions
│   ├── eda.py                        # EDA script for dataset analysis and visualization
│   ├── model_utils.py                # Model utilities for saving/loading/checkpoints
│   ├── models.py                     # Modular implementation of CNN architectures
│   ├── preprocessing.py              # Image preprocessing pipeline
│   ├── testing_submission.py         # Generates predictions and submission files
│   ├── train_histopathology.py       # Main script for training the model
├── requirements.txt                  # Project dependencies
├── README.md                         # Main project documentation
├── LICENSE                           # License for the project
└── train_labels.csv                  # CSV file containing training labels
```

---

## Files Overview

### `data_utils.py`
- **Functions**:
  - `display_sample_images`: Visualize sample images for specific labels.
  - `calculate_class_distribution`: Analyze and visualize dataset label distribution.

- **Classes**:
  - `HistologyDataset`: Custom PyTorch dataset for loading and preprocessing histopathology images.
  - `HistopathologyDataModule`: Data module for managing training, validation, and test data with PyTorch Lightning.

---

### `eda.py`
- **Purpose**:
  - Conducts exploratory data analysis (EDA) using visualizations and statistics.
  - Generates a Sweetviz report for detailed insights into the dataset.

- **Features**:
  - Class distribution analysis.
  - Summary statistics and data insights.
  - Visual sampling of image labels.

---

### `models.py`
- **Classes**:
  - `BaselineCNN`: A CNN architecture for binary classification, featuring convolutional layers for feature extraction and fully connected layers for classification.

---

### `model_utils.py`
- **Functions**:
  - `save_model`: Saves the trained model checkpoint.
  - `load_model`: Loads a model checkpoint for inference or fine-tuning.

---

### `preprocessing.py`
- **Purpose**:
  - Preprocesses images through resizing, normalization, and augmentations.
  - Provides modular transformations for train, validation, and test datasets.

---

### `train_histopathology.py`
- **Purpose**:
  - Main script for training the CNN model.
  - Includes MLflow logging, early stopping, and checkpointing.

- **Features**:
  - GPU/CPU support.
  - Logging metrics and parameters to MLflow.
  - Automated saving of the best model.

---

### `testing_submission.py`
- **Purpose**:
  - Generates predictions using the trained model.
  - Saves predictions to `submission.csv` for Kaggle submissions.

---

## Getting Started

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/astoreyai/Histopathology_Cancer_Detection.git
   cd Histopathology_Cancer_Detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

### Usage

#### **Exploratory Data Analysis**
Run `eda.py` to analyze the dataset:
```bash
python scripts/eda.py
```

#### **Training**
Train the CNN model:
```bash
python scripts/train_histopathology.py
```

#### **Prediction**
Generate predictions for the test dataset:
```bash
python scripts/testing_submission.py
```

---

## Configuration

Modify `config.py` to update:
- Paths to training and test datasets.
- Image preprocessing parameters (target size, normalization values).
- Hyperparameters (learning rate, batch size, number of epochs).

---

## Experiment Tracking with MLflow

MLflow is integrated to log metrics, parameters, and artifacts during training. Launch the MLflow UI with:
```bash
mlflow ui
```
Access the UI at `http://localhost:5000`.

---

## Key Features
1. **Preprocessing Pipeline**:
   - Resizing, normalization, and augmentation.
   - Modular implementation for train, validation, and test splits.

2. **Reproducibility**:
   - Configurable scripts and parameter tracking.
   - MLflow integration for experiment comparison.

3. **Scalability**:
   - Modularized model definitions and utilities.
   - PyTorch Lightning integration for efficient training.

---

## Requirements

- Python 3.8+
- PyTorch
- PyTorch Lightning
- Pandas
- Matplotlib
- Seaborn
- Sweetviz

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.