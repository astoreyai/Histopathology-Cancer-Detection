# scripts/model_utils.py

import torch
import torch.nn as nn
import os

class BaselineCNN(nn.Module):
    """
    Baseline CNN model for binary classification.
    """
    def __init__(self, input_shape=(3, 224, 224), num_classes=1):
        super(BaselineCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (input_shape[1] // 8) * (input_shape[2] // 8), 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def initialize_model(input_shape=(3, 224, 224), num_classes=1, device="cpu"):
    """
    Initialize the BaselineCNN model and move it to the specified device.
    """
    model = BaselineCNN(input_shape=input_shape, num_classes=num_classes).to(device)
    return model

def save_model(model, path="baseline_cnn.pth"):
    """
    Save the model's state dictionary to the specified path.
    """
    torch.save(model.state_dict(), path)

def load_model(model, path="baseline_cnn.pth", device="cpu"):
    """
    Load the model's state dictionary from the specified path.
    """
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model
