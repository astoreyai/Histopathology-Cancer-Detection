# scripts/train_utils.py

import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

def get_optimizer(model, learning_rate=0.001):
    """
    Initialize the Adam optimizer.
    """
    return optim.Adam(model.parameters(), lr=learning_rate)

def get_loss_function():
    """
    Define the Binary Cross-Entropy loss function for binary classification.
    """
    return nn.BCELoss()

def train_one_epoch(model, loader, optimizer, criterion, device="cpu"):
    """
    Train the model for one epoch.
    """
    model.train()
    running_loss = 0
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.float().to(device)
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def validate(model, loader, criterion, device="cpu"):
    """
    Validate the model and calculate loss and AUC score.
    """
    model.eval()
    running_loss = 0
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images, labels = images.to(device), labels.float().to(device)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
    auc_score = roc_auc_score(all_labels, all_outputs)
    return running_loss / len(loader), auc_score

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=5, device="cpu"):
    """
    Train and validate the model over a specified number of epochs.
    """
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
