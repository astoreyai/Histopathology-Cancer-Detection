# scripts/model_utils.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineCNN(nn.Module):
    """
    Baseline CNN model for binary classification with input images of size 96x96.
    Allows customization of key layer parameters using kwargs.
    """
    def __init__(self, input_shape=(3, 96, 96), num_classes=1, **kwargs):
        """
        Initializes the CNN model.

        Args:
            input_shape (tuple): Shape of the input image, default is (3, 96, 96).
            num_classes (int): Number of output classes, default is 1 for binary classification.
            **kwargs: Additional keyword arguments for layers.
                - kernel_size_conv (int or tuple): Kernel size for conv layers, default is 3.
                - stride_conv (int): Stride for conv layers, default is 1.
                - padding_conv (int): Padding for conv layers, default is 1.
                - kernel_size_pool (int): Kernel size for pooling layers, default is 2.
                - stride_pool (int): Stride for pooling layers, default is 2.
                - dropout_rate (float): Dropout rate for the dropout layer, default is 0.5.
        """
        super(BaselineCNN, self).__init__()
        
        # Retrieve kwargs for convolutional and pooling layers
        kernel_size_conv = kwargs.get('kernel_size_conv', 3)
        stride_conv = kwargs.get('stride_conv', 1)
        padding_conv = kwargs.get('padding_conv', 1)
        kernel_size_pool = kwargs.get('kernel_size_pool', 2)
        stride_pool = kwargs.get('stride_pool', 2)
        dropout_rate = kwargs.get('dropout_rate', 0.5)

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=kernel_size_conv, stride=stride_conv, padding=padding_conv),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=kernel_size_pool, stride=stride_pool),  # Output: (32, 48, 48)
            nn.Conv2d(32, 64, kernel_size=kernel_size_conv, stride=stride_conv, padding=padding_conv),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=kernel_size_pool, stride=stride_pool),  # Output: (64, 24, 24)
            nn.Conv2d(64, 128, kernel_size=kernel_size_conv, stride=stride_conv, padding=padding_conv),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=kernel_size_pool, stride=stride_pool)  # Output: (128, 12, 12)
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 128),  # Adjusted based on input shape
            nn.ReLU(),
            nn.Dropout(dropout_rate),
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
