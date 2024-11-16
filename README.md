# Histopathology Cancer Detection Project

This repository contains code, scripts, and resources for detecting cancer in histopathology images using deep learning. The project is structured to support reproducible experiments with MLflow tracking and modular scripts for data preprocessing, model training, validation, and testing.

---

## Project Overview

Histopathology image analysis plays a vital role in cancer detection. This project aims to automate the detection process using convolutional neural networks (CNNs) trained on histopathological image datasets. Key features include:

- **Preprocessing Pipelines**: Scalable preprocessing, including resizing, augmentation, and normalization.
- **Experiment Tracking**: MLflow integration for logging metrics, parameters, and artifacts.
- **Model Training**: PyTorch Lightning for modular, scalable training workflows.
- **Submission-Ready Predictions**: Easy-to-generate Kaggle submission files.

---

## Project Structure

```plaintext
├── data/                             # Placeholder for raw and processed datasets
├── mlruns/                           # MLflow tracking directory for experiment logs and artifacts
├── experiments/                      # Directory to store outputs and visualizations from experiments
├── scripts/                          # Modular scripts for the project
│   ├── __init__.py                   # Initializes project modules
│   ├── config.py                     # Configuration settings (paths, hyperparameters, etc.)
│   ├── data_utils.py                 # Dataset classes and preprocessing utilities
│   ├── eda.py                        # EDA script for dataset exploration
│   ├── model_utils.py                # Model utilities (metrics, saving/loading models)
│   ├── models.py                     # Modular implementation of model architectures
│   ├── preprocessing.py              # Preprocessing pipeline for image transformations
│   ├── testing_submission.py         # Generate predictions and create submission files
│   ├── train_histopathology.py       # Train the model with PyTorch Lightning
├── requirements.txt                  # Project dependencies
├── README.md                         # Main project documentation
├── LICENSE                           # License information
├── train_labels.csv                  # CSV containing training labels
```

---

## Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed, along with `pip`. Install the required dependencies using:

```bash
pip install -r requirements.txt
```

### Installation

Clone this repository and set up the environment:

```bash
git clone https://github.com/astoreyai/Histopathology_Cancer_Detection.git
cd Histopathology_Cancer_Detection
pip install -r requirements.txt
```

### Dataset

Place your dataset in the `data/` directory. Update `config.py` with the correct dataset paths.

---

## Workflow

### Exploratory Data Analysis (EDA)

Run `eda.py` or the notebook `as_hcd_eda.ipynb` to explore the dataset. The script generates:

- Class distribution plots
- Summary statistics
- Sweetviz reports for detailed insights

```bash
python scripts/eda.py
```

### Training the Model

Train the CNN using `train_histopathology.py`. The script handles data loading, training, validation, and logging to MLflow.

```bash
python scripts/train_histopathology.py
```

Key features:
- Automatically saves the best model checkpoint.
- Logs metrics, parameters, and artifacts for each experiment.
- Supports GPU acceleration and distributed training.

### Testing and Submissions

Generate predictions for the test dataset using `testing_submission.py`:

```bash
python scripts/testing_submission.py
```

This script saves predictions to `submission.csv` for easy Kaggle submissions.

---

## Key Components

### Scripts

#### `data_utils.py`
- Defines `HistologyDataset` and `HistopathologyDataModule`.
- Handles dataset splitting, augmentations, and batch preparation.

#### `preprocessing.py`
- Provides a robust preprocessing pipeline for image resizing, normalization, and augmentations.

#### `models.py`
- Implements modular CNN architectures for flexibility in experimentation.

#### `model_utils.py`
- Contains utility functions for training, evaluation, and logging.

#### `eda.py`
- Script for dataset analysis, visualization, and generating Sweetviz reports.

#### `train_histopathology.py`
- Manages the model training pipeline with PyTorch Lightning and MLflow integration.

#### `testing_submission.py`
- Script for loading the trained model, generating predictions, and creating submission files.

---

## Experiment Tracking with MLflow

### Setting Up MLflow

To launch the MLflow UI locally, run:

```bash
mlflow ui
```

Access the UI at `http://localhost:5000`.

### Logging

During training, the following are logged:
- **Metrics**: AUC, accuracy, loss
- **Parameters**: Learning rate, batch size, model configuration
- **Artifacts**: Checkpoints, visualizations, and predictions

---

## Dependencies

See `requirements.txt` for all dependencies. Major libraries include:

- PyTorch
- PyTorch Lightning
- Matplotlib
- Seaborn
- Pandas
- MLflow

---

## Contribution Guidelines

Contributions are welcome! Feel free to fork this repository and create pull requests.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

Special thanks to the Kaggle community for datasets and inspiration. This project was built with a focus on modularity, reproducibility, and scalability.