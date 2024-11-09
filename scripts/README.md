# Histopathology Cancer Detection

This repository provides the code for detecting cancerous cells in histopathology images using a Convolutional Neural Network (CNN). It includes custom dataset loading, model definition, training, validation, and prediction utilities tailored for binary image classification of histopathology slides.

## Project Structure

- **data_utils.py**: Contains utility functions and dataset classes for loading, processing, and visualizing histopathology images.
- **model_utils.py**: Defines the CNN model architecture for binary classification tasks, along with functions for initializing, saving, and loading model states.
- **train_utils.py**: Provides training, validation, and prediction utilities for model training and evaluation.
- **config.py**: Configuration file specifying dataset paths, target image size, batch size, learning rate, and the number of training epochs.

## Files Overview

### data_utils.py
- **Functions**:
  - `load_labels`: Loads the labels CSV file for training images.
  - `get_transformations`: Defines transformations for image augmentation (e.g., resizing, horizontal flipping, normalization).
  - `display_sample_images`: Displays sample images for a given label (cancerous/non-cancerous) for visual inspection.
  - `calculate_mean_intensity`: Calculates pixel intensity distributions for sample images, aiding in dataset analysis.
  
- **Classes**:
  - `HistologyDataset`: Custom dataset for training/validation images, with transformations.
  - `HistologyTestDataset`: Custom dataset for test images, preparing them for prediction without labels.

### model_utils.py
- **Classes**:
  - `BaselineCNN`: A CNN model for binary classification. Includes convolutional layers for feature extraction and fully connected layers for classification.
  
- **Functions**:
  - `initialize_model`: Initializes and moves the model to the specified device (CPU/GPU).
  - `save_model`: Saves the model's state dictionary to a specified path.
  - `load_model`: Loads a saved model's state dictionary onto the specified device.

### train_utils.py
- **Functions**:
  - `train_one_epoch`: Trains the model for one epoch, updating weights and calculating training loss.
  - `validate`: Evaluates the model on validation data, calculating loss and AUC score.
  - `generate_predictions`: Generates predictions for test data, applying a threshold for binary classification.

### config.py
- Specifies configuration details:
  - Paths to training/test image directories and labels file.
  - Image target size, batch size, learning rate, and training epochs.

## Getting Started

### Requirements
- Python 3.x
- PyTorch
- pandas
- scikit-learn
- tqdm
- torchvision
- PIL (Pillow)

Install dependencies using:
```bash
pip install torch pandas scikit-learn tqdm torchvision pillow
```

### Usage

1. **Dataset Preparation**:
   - Organize histopathology images and labels according to paths specified in `config.py`.
  
2. **Training the Model**:
   - Run the training script, which loads images, trains the CNN model, and validates it on a separate dataset.

3. **Generating Predictions**:
   - Use the `generate_predictions` function in `train_utils.py` to make predictions on new images.

### Example Workflow
1. Load dataset labels and apply transformations.
2. Initialize the CNN model and set hyperparameters.
3. Train and validate the model, saving checkpoints periodically.
4. Evaluate model performance using AUC and loss metrics.

## Configuration
Modify the paths and parameters in `config.py` as necessary to adapt the code to different datasets or settings.

## License
MIT License.

