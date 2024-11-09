# data_utils.py

import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms

def load_labels(labels_file_path):
    """
    Load the labels from a CSV file.
    
    Args:
        labels_file_path (str): Path to the labels CSV file.
        
    Returns:
        pd.DataFrame: DataFrame containing the labels data.
    """
    return pd.read_csv(labels_file_path)

class HistologyDataset(Dataset):
    """
    Custom Dataset class for loading histopathology images for training and validation.
    """
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

class HistologyTestDataset(Dataset):
    """
    Custom Dataset class for loading histopathology images for testing.
    """
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_ids = [img_id.split(".")[0] for img_id in os.listdir(img_dir) if img_id.endswith(".tif")]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_id + ".tif")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_id

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
    - df (DataFrame): DataFrame containing 'id' and 'label' columns.
    - img_dir (str): Directory path where the images are stored.
    - label (int): Class label (1 for cancerous, 0 for non-cancerous).
    - sample_size (int, optional): Number of images to display. Default is 5.

    Raises:
    - ValueError: If label is not found in the DataFrame.
    - FileNotFoundError: If image files are missing in the specified directory.

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

def calculate_mean_intensity(img_path):
    """
    Calculate the mean pixel intensity for a single image.

    Parameters:
        img_path (str): Path to the image file.

    Returns:
        float: Mean intensity of the image.
    """
    img = Image.open(img_path).convert('L')  # Convert to grayscale
    return np.array(img).mean()