# Histopathology Cancer Detection Project

This repository contains code and resources for developing a deep learning model to detect cancer in histopathology images. The project includes scripts for data preprocessing, model training, validation, and testing, with an emphasis on reproducibility and experiment tracking through MLflow.

## Project Overview

The project is structured to support iterative improvements in preprocessing, modeling, and hyperparameter optimization. MLflow is used to log metrics, parameters, and artifacts for easy comparison and versioning across experiments, enhancing the reproducibility and reliability of results.

## Project Structure

```plaintext
├── data/                             # Placeholder for data (loaded from Kaggle or other sources)
├── notebooks/
│   ├── as_hcd_eda.ipynb              # EDA notebook utilizing Sweetviz for dataset analysis
│   ├── as_hcd_training.ipynb         # Training notebook with MLflow tracking
│   └── as_hcd_testing.ipynb          # Testing and submission notebook
├── scripts/
│   ├── data_utils.py                 # Dataset classes and transformations
│   ├── model_utils.py                # Model definitions and utilities
│   └── train_utils.py                # Training and evaluation utilities
├── README.md                         # Project documentation
├── LICENSE                           # License for the project
└── requirements.txt                  # Project dependencies
```

## Installation

Clone this repository and install the required packages:

```bash
git clone https://github.com/yourusername/histopathology-cancer-detection.git
cd histopathology-cancer-detection
pip install -r requirements.txt
```

## Workflow with MLflow

MLflow is integrated for experiment tracking, allowing systematic comparison of model configurations and performance across runs.

1. **MLflow Setup**:
   - Optionally, set up an MLflow Tracking Server for centralized logging and configure the tracking URI in notebooks or scripts.
   - Run MLflow locally for individual tracking:

     ```bash
     mlflow ui
     ```

   This command launches the MLflow UI at `localhost:5000`.

2. **Logging Experiments**:
   - **Metrics**: Logs training/validation losses, accuracy, and AUC scores at each epoch.
   - **Parameters**: Captures model parameters (learning rate, batch size, etc.) for reproducibility.
   - **Artifacts**: Saves trained model weights, configuration files, and experiment outputs.

3. **Experiment Management**:
   - The `as_hcd_training.ipynb` notebook logs each experiment with `mlflow.log_metric` and `mlflow.log_param`.
   - Use the MLflow UI to compare runs and evaluate the best configurations based on metrics and visualized graphs.

## Usage

### Exploratory Data Analysis (EDA)

Use `as_hcd_eda.ipynb` to conduct an initial analysis of the dataset:
   - **Data Overview**: Generates a detailed report with Sweetviz, including distribution analysis and image property statistics.
   - **Insights**: Visualizes key metrics, informing preprocessing steps and model selection.

### Model Training

The `as_hcd_training.ipynb` notebook handles data loading, preprocessing, model initialization, and training. Configuration options for paths, learning rate, batch size, and other parameters are modifiable within the notebook.

1. **Data Loading**: `data_utils.py` manages data preprocessing and transformations.
2. **Model Definition**: `model_utils.py` defines the CNN architecture, saving/loading methods, and device placement.
3. **Training**: `train_utils.py` includes functions for training, validation, and metric calculations, with MLflow logging integrated for each step.

### Model Testing and Submission

The `as_hcd_testing.ipynb` notebook tests the trained model on the test dataset, generating predictions. Results are saved in `submission.csv` for easy Kaggle submission.

### Project Modules

- **data_utils.py**: Defines dataset classes and transformations for image augmentation and processing.
- **model_utils.py**: Contains CNN model architecture, initialization, and saving/loading functions.
- **train_utils.py**: Implements functions for training, validation, and generating predictions, with integrated MLflow logging.