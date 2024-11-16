"""
Module Initialization for Histopathology Cancer Detection Project.

This file initializes all key modules for the project, including data utilities,
model utilities, preprocessing pipelines, and configuration settings.
"""

# Import necessary components for external access
from .config import (
    TRAIN_DIR,
    TEST_DIR,
    LABELS_FILE,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    TARGET_SIZE,
)

from .data_utils import (
    HistologyDataset,
    HistopathologyDataModule,
    display_sample_images,
)

from .model_utils import BaselineCNN

from .preprocessing import Preprocessing

__all__ = [
    # Config exports
    "TRAIN_DIR",
    "TEST_DIR",
    "LABELS_FILE",
    "BATCH_SIZE",
    "LEARNING_RATE",
    "EPOCHS",
    "TARGET_SIZE",
    # Data utilities
    "HistologyDataset",
    "HistopathologyDataModule",
    "display_sample_images",
    # Model utilities
    "BaselineCNN",
    # Preprocessing pipeline
    "Preprocessing",
]
