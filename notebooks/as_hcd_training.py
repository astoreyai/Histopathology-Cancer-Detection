# %% [markdown]
# ### Histopathologic Cancer Detection with Baseline CNN Model
# 
# This code implements a Convolutional Neural Network (CNN) to classify histopathology images
# 
# as part of a binary classification task for detecting cancerous tissues.
# 
# The dataset consists of image files stored in directories, along with labels for training.

# %% [markdown]
# #### Import Libraries

# %%
import os  # For file and directory operations
import numpy as np  # Numerical operations
import pandas as pd  # Data handling in DataFrames
import torch  # PyTorch library for tensor operations
import torch.nn as nn  # Neural network components
import torch.optim as optim  # Optimization algorithms
import torchvision.transforms as transforms  # Image transformations for preprocessing
import matplotlib.pyplot as plt  # Plotting library
import seaborn as sns  # For advanced plotting
from torch.utils.data import Dataset, DataLoader  # Custom dataset and loader handling
from PIL import Image  # For image loading and processing
from sklearn.model_selection import train_test_split  # Splits data into training and validation sets
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report  # For evaluating model

# %%
# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# #### Configuration Constants

# %%
# Set directories for training and testing data, and the file path for labels

TRAIN_DIR = "/kaggle/input/histopathologic-cancer-detection/train"
TEST_DIR = "/kaggle/input/histopathologic-cancer-detection/test"
LABELS_FILE = "/kaggle/input/histopathologic-cancer-detection/train_labels.csv"

# Define target image size, batch size for data loading, learning rate, and the number of epochs

TARGET_SIZE = (96, 96)
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 5

# %% [markdown]
# #### Exploratory Data Analysis (EDA)

# %%
# Load labels DataFrame

labels_df = pd.read_csv(LABELS_FILE)

# Display basic information about the dataset

print("Dataset Info:")
print(labels_df.info())
print("\nDataset Head:")
print(labels_df.head())

# Check for class balance

print("\nClass Distribution:")
print(labels_df['label'].value_counts())

# Plot class distribution

sns.countplot(data=labels_df, x='label')
plt.title("Distribution of Labels")
plt.xlabel("Label (0 = Benign, 1 = Malignant)")
plt.ylabel("Count")
plt.show()

# %% [markdown]
# #### Sample Image Visualization

# %%
# Visualize some sample images from each class (benign and malignant)

def visualize_samples(dataframe, img_dir, num_samples=5):
    """
    Visualizes random samples of images for each class.

    Args:
    - dataframe: DataFrame containing image file names and labels.
    - img_dir: Directory where images are stored.
    - num_samples: Number of samples to display per class.
    """
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    for label in [0, 1]:  # 0: benign, 1: malignant
        sample_images = dataframe[dataframe['label'] == label].sample(num_samples)
        for i, img_id in enumerate(sample_images['id']):
            img_path = os.path.join(img_dir, f"{img_id}.tif")
            image = Image.open(img_path)
            ax = axes[label, i]
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(f"Label: {label}")
    plt.suptitle("Sample Images by Class")
    plt.show()

# Visualize sample images from each class

visualize_samples(labels_df, TRAIN_DIR)

# %% [markdown]
# #### Data Preprocessing Transformations

# %%
# Define image transformations for training and validation datasets

train_transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE),  # Resize images to target size of 96x96
    transforms.RandomHorizontalFlip(),  # Apply random horizontal flip for augmentation
    transforms.RandomRotation(20),  # Random rotation of the image
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust brightness and contrast
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize to ImageNet standards
])

val_transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE),  # Resize images to target size of 96x96
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# %% [markdown]
# #### Define Custom Dataset

# %%
class HistologyDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx, 0] + '.tif')
        image = Image.open(img_name).convert("RGB")
        label = self.dataframe.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

# %% [markdown]
# #### Load and Split Dataset

# %%
# Split data into training and validation sets

train_df, val_df = train_test_split(labels_df, test_size=0.2, stratify=labels_df['label'], random_state=42)

# Create training and validation datasets with respective transformations

train_dataset = HistologyDataset(dataframe=train_df, img_dir=TRAIN_DIR, transform=train_transform)
val_dataset = HistologyDataset(dataframe=val_df, img_dir=TRAIN_DIR, transform=val_transform)

# Initialize data loaders for batching and shuffling data during training and validation

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# %% [markdown]
# #### Baseline CNN Model

# %%
# Define the CNN model architecture for binary classification

class BaselineCNN(nn.Module):
    def __init__(self, input_shape=(3, 96, 96), num_classes=1):
        """
        Initializes the Baseline CNN architecture with convolutional and fully connected layers.
        Args:
        - input_shape: Shape of input images.
        - num_classes: Number of output classes (1 for binary classification).
        """
        super(BaselineCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # Convolutional layer with 32 filters
            nn.ReLU(),  # ReLU activation
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Convolutional layer with 64 filters
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Convolutional layer with 128 filters
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # Flatten the tensor
            nn.Linear(128 * (input_shape[1] // 8) * (input_shape[2] // 8), 128),  # Fully connected layer
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(128, num_classes),  # Output layer for binary classification
            nn.Sigmoid()  # Sigmoid activation for binary output
        )

    def forward(self, x):
        """
        Forward pass through the network.
        Args:
        - x: Input tensor (batch of images).
        Returns:
        - Output tensor with class scores.
        """
        x = self.conv_layers(x)  # Apply convolutional layers
        x = self.fc_layers(x)  # Apply fully connected layers
        return x

# %% [markdown]
# #### Training and Validation Functions

# %%
# Define functions to train and validate the model

def train_one_epoch(model, loader, optimizer, criterion):
    """
    Trains the model for one epoch.
    Args:
    - model: The neural network model.
    - loader: DataLoader for training data.
    - optimizer: Optimization algorithm.
    - criterion: Loss function.
    Returns:
    - Average loss over the training epoch.
    """
    model.train()  # Set model to training mode
    running_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.float().to(device)
        optimizer.zero_grad()  # Clear gradients
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagate
        optimizer.step()  # Update weights
        running_loss += loss.item()  # Accumulate loss
    return running_loss / len(loader)  # Average loss

def validate(model, loader, criterion):
    """
    Validates the model on the validation dataset.
    Args:
    - model: The neural network model.
    - loader: DataLoader for validation data.
    - criterion: Loss function.
    Returns:
    - Average validation loss and AUC score.
    """
    model.eval()  # Set model to evaluation mode
    running_loss = 0
    all_labels = []
    all_outputs = []
    with torch.no_grad():  # Disable gradient computation for validation
        for images, labels in loader:
            images, labels = images.to(device), labels.float().to(device)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)  # Compute loss
            running_loss += loss.item()  # Accumulate loss
            all_labels.extend(labels.cpu().numpy())  # Collect true labels
            all_outputs.extend(outputs.cpu().numpy())  # Collect predictions
    auc_score = roc_auc_score(all_labels, all_outputs)  # Calculate AUC score
    return running_loss / len(loader), auc_score  # Return average loss and AUC score

# %% [markdown]
# #### Model Initialization

# %%
# Initialize the model, optimizer, and loss function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
model = BaselineCNN().to(device)  # Instantiate and move model to device
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Define optimizer
criterion = nn.BCELoss()  # Use binary cross-entropy loss for binary classification

# %% [markdown]
# #### Training Loop

# %%
# Train the model for specified epochs and evaluate on validation data

for epoch in range(EPOCHS):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_auc = validate(model, val_loader, criterion)
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

# %% [markdown]
# #### Save Model

# %%
# Save the trained model's parameters for future use

torch.save(model.state_dict(), "baseline_cnn.pth")


