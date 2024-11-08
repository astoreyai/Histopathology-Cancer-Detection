# scripts/train_utils.py

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, criterion, device="cpu", **kwargs):
    """
    Train the model for a single epoch.

    Args:
        model (nn.Module): The model to be trained.
        loader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates.
        criterion (nn.Module): Loss function.
        device (str): Device to perform training on ("cpu" or "cuda").

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.float().to(device)
        
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        
        # Calculate loss and perform backpropagation
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    return avg_loss

def validate(model, loader, criterion, device="cpu", **kwargs):
    """
    Validate the model on the validation dataset.

    Args:
        model (nn.Module): The model to be validated.
        loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        device (str): Device to perform validation on ("cpu" or "cuda").

    Returns:
        tuple: Average validation loss and AUC score.
    """
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.float().to(device)
            outputs = model(images).squeeze()
            
            # Calculate validation loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # Store outputs and labels for AUC calculation
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

    avg_loss = running_loss / len(loader)
    auc_score = roc_auc_score(all_labels, all_outputs)
    
    return avg_loss, auc_score

def generate_predictions(model, loader, device="cpu", threshold=0.5, **kwargs):
    """
    Generate predictions on the test dataset.

    Args:
        model (nn.Module): Trained model to generate predictions.
        loader (DataLoader): DataLoader for test data.
        device (str): Device to perform predictions on ("cpu" or "cuda").
        threshold (float): Threshold to classify predictions as 0 or 1. Default is 0.5.

    Returns:
        tuple: List of image IDs and corresponding predictions.
    """
    model.eval()
    predictions = []
    img_ids = []

    with torch.no_grad():
        for images, ids in tqdm(loader, desc="Generating Predictions"):
            images = images.to(device)
            outputs = model(images).squeeze()
            
            # Apply threshold to obtain binary predictions
            preds = (outputs > threshold).float().cpu().numpy()
            predictions.extend(preds)
            img_ids.extend(ids)

    return img_ids, predictions
