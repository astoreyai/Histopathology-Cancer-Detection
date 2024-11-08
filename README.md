# Histopathology Cancer Detection Project

This repository contains code and resources for developing a deep learning model to detect cancer in histopathology images. The project includes data preprocessing, model training, validation, and testing scripts with high reproducibility, leveraging utility scripts for dataset management, model architectures, and training workflows.

## Project Overview

This project is designed to allow iterative improvement through adjustments in preprocessing, modeling, and hyperparameters, with experiment tracking via MLflow.

## Project Structure

```plaintext
├── data/                             # Placeholder for data (loaded from Kaggle or other sources)
├── notebooks/
│   ├── train_notebook.ipynb          # Training notebook
│   └── test_notebook.ipynb           # Testing and submission notebook
├── scripts/
│   ├── data_utils.py                 # Dataset classes and transformations
│   ├── model_utils.py                # Model definitions and utilities
│   └── train_utils.py                # Training and evaluation utilities
├── README.md                         # Project documentation
├── LICENSE                           # License for the project
└── requirements.txt                  # Project dependencies
```

## Installation

Clone this repository and install the required packages with:

```bash
git clone https://github.com/yourusername/histopathology-cancer-detection.git
cd histopathology-cancer-detection
pip install -r requirements.txt
```

## Workflow with MLflow

This project uses MLflow to log experiments, track metrics, and save models, allowing systematic comparison of results. 

1. **Set up MLflow Tracking Server (Optional)**:
   If you want a centralized MLflow server, set it up and configure the tracking URI in the notebooks or scripts.

   ```bash
   mlflow ui
   ```

   This will start the MLflow UI locally on port 5000.

2. **Logging Experiments**:
   - **Training Metrics**: Training and validation losses, accuracy, and AUC scores are logged to MLflow in each epoch.
   - **Model Parameters**: Model hyperparameters (learning rate, batch size, etc.) are recorded for reproducibility.
   - **Artifacts**: The final model weights and configuration files are saved to MLflow for each experiment.

3. **Running Experiments**:
   - Track experiments directly from `train_notebook.ipynb` using `mlflow.log_metric` and `mlflow.log_param`.
   - Compare different runs within the MLflow UI to assess which configurations perform best.

## Usage

### Training the Model

The `train_notebook.ipynb` notebook handles data loading, preprocessing, model initialization, and training. Configurations like data paths, learning rates, and batch sizes are adjustable within the notebook.

1. **Data Loading**: `data_utils.py` manages data preprocessing and transformations using PyTorch.
2. **Model Definition**: `model_utils.py` defines the CNN architecture and model utility functions.
3. **Training Functions**: `train_utils.py` implements training and evaluation functions, tracking performance metrics at each step.

### Testing and Submission

The `test_notebook.ipynb` notebook performs testing and generates predictions. The results are output to `submission.csv` for easy submission.

### Project Modules

- **data_utils.py**: Defines the dataset classes, transformations, and data loading logic.
- **model_utils.py**: Contains model architecture definitions, saving/loading functions, and initialization.
- **train_utils.py**: Implements training, validation, and evaluation functions, including metric calculations and MLflow logging.
