# data_utils.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# Function to load labels CSV file
def load_labels(labels_file_path):
    """
    Load the labels from a CSV file.
    
    Args:
        labels_file_path (str): Path to the labels CSV file.
        
    Returns:
        pd.DataFrame: DataFrame containing the labels data.
        
    Raises:
        FileNotFoundError: If the CSV file is not found.
    """
    if not os.path.exists(labels_file_path):
        raise FileNotFoundError(f"Labels file {labels_file_path} not found.")
    return pd.read_csv(labels_file_path)


# Class for the Histology Dataset (Training/Validation)
class HistologyDataset(Dataset):
    """
    Custom Dataset class for loading histopathology images for training and validation.
    """
    def __init__(self, dataframe, img_dir, transform=None):
        """
        Initialize the dataset.
        
        Args:
            dataframe (pd.DataFrame): DataFrame with 'id' and 'label' columns.
            img_dir (str): Path to the image directory.
            transform (callable, optional): Optional transformations to be applied on an image.
        """
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Retrieve an image and its label by index.
        
        Args:
            idx (int): Index of the data item.
        
        Returns:
            Tuple[Image, int]: Transformed image and corresponding label.
        """
        img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx, 0] + '.tif')
        image = Image.open(img_name).convert("RGB")
        label = self.dataframe.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label


# Class for the Histology Test Dataset (Testing)
class HistologyTestDataset(Dataset):
    """
    Custom Dataset class for loading histopathology images for testing.
    """
    def __init__(self, img_dir, transform=None):
        """
        Initialize the dataset.
        
        Args:
            img_dir (str): Path to the image directory.
            transform (callable, optional): Optional transformations to be applied on an image.
        """
        self.img_dir = img_dir
        self.transform = transform
        # Image IDs are derived from filenames
        self.img_ids = [img_id.split(".")[0] for img_id in os.listdir(img_dir) if img_id.endswith(".tif")]

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.img_ids)

    def __getitem__(self, idx):
        """
        Retrieve an image by index.
        
        Args:
            idx (int): Index of the data item.
        
        Returns:
            Tuple[Image, str]: Transformed image and image ID.
        """
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_id + ".tif")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_id


# Define common transformations for training and validation
def get_transformations(target_size=(224, 224), is_train=True):
    """
    Get image transformations for training or validation data.
    
    Args:
        target_size (tuple): The desired target image size.
        is_train (bool): If True, returns transformations with augmentations; else, returns simple transformations.

    Returns:
        torchvision.transforms.Compose: Transformations for the dataset.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize(target_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def display_sample_images(df, img_dir, label, sample_size=5):
    """
    Display a sample of images for a specified label.

    Parameters:
        df (DataFrame): DataFrame containing 'id' and 'label' columns.
        img_dir (str): Directory path where the images are stored.
        label (int): Class label (1 for cancerous, 0 for non-cancerous).
        sample_size (int, optional): Number of images to display. Default is 5.

    Raises:
        ValueError: If label is not found in the DataFrame.
        FileNotFoundError: If image files are missing in the specified directory.
    """
    if label not in df['label'].unique():
        raise ValueError(f"Label {label} not found in DataFrame.")

    sample_df = df[df['label'] == label].sample(sample_size, random_state=42)
    fig, axes = plt.subplots(1, sample_size, figsize=(15, 5))

    for i, (_, row) in enumerate(sample_df.iterrows()):
        img_id = row['id']
        img_path = os.path.join(img_dir, img_id + '.tif')
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file {img_path} not found.")
        
        img = Image.open(img_path)
        axes[i].imshow(img)
        axes[i].axis("off")

    title = "Cancerous" if label == 1 else "Non-Cancerous"
    plt.suptitle(f"{sample_size} Sample {title} Images")
    plt.show()


def calculate_mean_intensity(df, img_dir, label, sample_size=10, **kwargs):
    """
    Calculate pixel intensity distribution for a subset of images with a specified label.

    Parameters:
        df (pd.DataFrame): DataFrame containing image IDs and labels.
        img_dir (str): Directory where images are stored.
        label (int): Label to filter images (1 for cancerous, 0 for non-cancerous).
        sample_size (int): Number of images to sample for calculating intensities.
        **kwargs: Additional keyword arguments for np.array operations.

    Returns:
        list: Flattened list of intensities for sampled images.
    """
    # Filter and sample the DataFrame for the specified label
    if label not in df['label'].unique():
        raise ValueError(f"Label {label} not found in DataFrame.")

    sample_df = df[df['label'] == label].sample(sample_size, random_state=42)
    
    intensities = []
    for img_id in sample_df['id']:
        img_path = os.path.join(img_dir, f"{img_id}.tif")
        
        # Convert image to grayscale and flatten to a list of intensities
        img = Image.open(img_path).convert('L')
        intensities.extend(np.array(img).flatten(**kwargs))
    
    return intensities