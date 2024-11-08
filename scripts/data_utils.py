# scripts/data_utils.py

import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

class HistologyDataset(Dataset):
    """
    Custom dataset class for histopathology images.
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

def get_transforms(phase='train'):
    """
    Define data augmentation and preprocessing transforms.
    """
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def load_data(labels_file, train_dir, test_size=0.2, random_state=42):
    """
    Load data and split into training and validation sets.
    """
    labels_df = pd.read_csv(labels_file)
    train_df, val_df = train_test_split(labels_df, test_size=test_size, stratify=labels_df['label'], random_state=random_state)
    return train_df, val_df
