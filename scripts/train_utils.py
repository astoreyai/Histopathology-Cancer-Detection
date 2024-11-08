# scripts/train_utils.py

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.float().to(device)
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.float().to(device)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
    auc_score = roc_auc_score(all_labels, all_outputs)
    return running_loss / len(loader), auc_score

def generate_predictions(model, loader, device, threshold=0.5):
    """Generate predictions on the test dataset."""
    model.eval()
    predictions = []
    img_ids = []
    with torch.no_grad():
        for images, ids in tqdm(loader):
            images = images.to(device)
            outputs = model(images).squeeze()
            preds = (outputs > threshold).float().cpu().numpy()
            predictions.extend(preds)
            img_ids.extend(ids)
    return img_ids, predictions
