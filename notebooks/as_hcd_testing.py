# %% [markdown]
# ### Histopathologic Cancer Detection - Inference and Submission Notebook
# This notebook utilizes a pre-trained CNN model to generate predictions on histopathology test images.
# The predictions are saved in a CSV file for submission to Kaggle.
# 

# %% [markdown]
# #### Import Necessary Libraries

# %%
import os  # For file and directory handling
import numpy as np  # Numerical operations
import pandas as pd  # Data handling in DataFrames
import torch  # PyTorch library for tensor operations
from torch.utils.data import DataLoader, Dataset  # Custom dataset and loader handling
from torchvision import transforms  # Image transformations for consistency with training
import torch.nn as nn  # Neural network modules
from PIL import Image  # For image loading and manipulation
from tqdm import tqdm  # For displaying progress bars

# %% [markdown]
# #### Set up Device

# %%
# Use GPU if available for faster inference, otherwise default to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# #### Configuration Constants

# %%
# Define constants for paths, batch size, and image size to match the training configuration
TEST_DIR = "/kaggle/input/histopathologic-cancer-detection/test"  # Path to test images
MODEL_PATH = "/kaggle/input/as-week-1-baseline-cnn-training/baseline_cnn.pth"  # Path to saved model weights
SUBMISSION_FILE = "submission.csv"  # Name of the submission file
BATCH_SIZE = 32  # Batch size for loading test data
IMG_SIZE = (96, 96)  # Target image size, set to match the training image size

# %% [markdown]
# #### Define Test Data Transformations

# %%
# Define transformations to be applied on test images for consistency with the training setup
test_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),  # Resize images to 96x96
    transforms.ToTensor(),  # Convert image to tensor format
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

# %% [markdown]
# #### Define Custom Dataset for Test Images

# %%
class HistopathologyTestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        """
        Initializes the test dataset with the image directory and optional transformations.
        
        Args:
        - img_dir: Directory containing test images.
        - transform: Transformations to apply to images.
        """
        self.img_dir = img_dir
        self.transform = transform
        # List of image IDs in the test directory, excluding file extensions
        self.img_ids = [img_id.split(".")[0] for img_id in os.listdir(img_dir) if img_id.endswith(".tif")]

    def __len__(self):
        """Returns the total number of test samples."""
        return len(self.img_ids)

    def __getitem__(self, idx):
        """
        Retrieves an image by index and applies transformations.
        
        Args:
        - idx: Index of the image to retrieve.
        
        Returns:
        - Transformed image tensor and image ID.
        """
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_id + ".tif")
        image = Image.open(img_path).convert("RGB")  # Load image and convert to RGB
        
        if self.transform:
            image = self.transform(image)  # Apply transformations
            
        return image, img_id

# %% [markdown]
# #### Define Baseline CNN Model Architecture

# %%
class BaselineCNN(nn.Module):
    def __init__(self, input_shape=(3, 96, 96), num_classes=1):
        """
        Initializes the Baseline CNN model with convolutional and fully connected layers.
        
        Args:
        - input_shape: Shape of input images.
        - num_classes: Number of output classes (1 for binary classification).
        """
        super(BaselineCNN, self).__init__()
        # Define convolutional layers
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
        
        # Calculate the size of the flattened output for the first fully connected layer
        test_input = torch.rand(1, *input_shape)
        conv_out_size = self.conv_layers(test_input).view(-1).shape[0]
        
        # Define fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.Sigmoid()  # Sigmoid activation for binary classification
        )

    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
        - x: Input tensor (batch of images).
        
        Returns:
        - Output tensor with prediction scores.
        """
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# %% [markdown]
# #### Load the Pre-trained Model

# %%
# Initialize and load the pre-trained model's weights
model = BaselineCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()  # Set model to evaluation mode

# %% [markdown]
# #### Prepare Test Dataset and DataLoader

# %%
# Create a dataset and data loader for test images
test_dataset = HistopathologyTestDataset(img_dir=TEST_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# %% [markdown]
# #### Prediction Function

# %%
def generate_predictions(model, loader, threshold=0.5):
    """
    Generates predictions on the test dataset.
    
    Args:
    - model: Trained PyTorch model for inference.
    - loader: DataLoader for test images.
    - threshold: Classification threshold.
    
    Returns:
    - img_ids: List of image IDs.
    - predictions: List of binary predictions.
    """
    model.eval()  # Ensure model is in evaluation mode
    predictions = []
    img_ids = []

    with torch.no_grad():  # Disable gradient calculation for inference
        for images, ids in tqdm(loader):
            images = images.to(device)
            outputs = model(images).squeeze()
            preds = outputs.cpu().numpy()
            
            # Store predictions and corresponding image IDs
            predictions.extend(preds)
            img_ids.extend(ids)

    # Apply threshold to convert scores to binary labels
    predictions = [1 if x > threshold else 0 for x in predictions]
    return img_ids, predictions

# %% [markdown]
# #### Generate Predictions and Prepare Submission File

# %%
# Generate predictions on test dataset
img_ids, preds = generate_predictions(model, test_loader)

# Prepare submission DataFrame
submission_df = pd.DataFrame({
    "id": img_ids,
    "label": preds
})

# %% [markdown]
# #### Save to CSV for Submission

# %%
# Save predictions to CSV file for submission
submission_df.to_csv(SUBMISSION_FILE, index=False)
print(f"Submission file saved as {SUBMISSION_FILE}")


