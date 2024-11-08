# scripts/model_utils.py

import torch
import torch.nn as nn
import os

class BaselineCNN(nn.Module):
    """
    Baseline CNN model for binary classification tasks.

    Args:
        input_shape (tuple): The shape of the input images as (channels, height, width).
        num_classes (int): The number of output classes. Defaults to 1 for binary classification.
    """
    def __init__(self, input_shape=(3, 224, 224), num_classes=1, **kwargs):
        super(BaselineCNN, self).__init__()
        # Convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fully connected layers for classification
        conv_output_size = 128 * (input_shape[1] // 8) * (input_shape[2] // 8)
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass for the model.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output predictions.
        """
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def initialize_model(input_shape=(3, 224, 224), num_classes=1, device="cpu", **kwargs):
    """
    Initialize the BaselineCNN model and move it to the specified device.

    Args:
        input_shape (tuple): Shape of the input image (channels, height, width).
        num_classes (int): Number of output classes.
        device (str): Device to load the model on ("cpu" or "cuda").

    Returns:
        BaselineCNN: Initialized model on the specified device.
    """
    model = BaselineCNN(input_shape=input_shape, num_classes=num_classes, **kwargs).to(device)
    return model

def save_model(model, path="baseline_cnn.pth", **kwargs):
    """
    Save the model's state dictionary to the specified path.

    Args:
        model (nn.Module): The model instance to save.
        path (str): Path where to save the model's state dictionary.
    """
    torch.save(model.state_dict(), path)

def load_model(model, path="baseline_cnn.pth", device="cpu", **kwargs):
    """
    Load the model's state dictionary from the specified path.

    Args:
        model (nn.Module): The model instance to load the weights into.
        path (str): Path to the model's state dictionary.
        device (str): Device to load the model onto ("cpu" or "cuda").

    Returns:
        nn.Module: Model loaded with weights on the specified device.
    """
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model
