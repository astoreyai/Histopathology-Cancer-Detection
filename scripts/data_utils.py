import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from scripts.config import TRAIN_DIR, TEST_DIR, LABELS_FILE, TARGET_SIZE, BATCH_SIZE

class HistologyDataset(Dataset):
    """
    Custom Dataset for loading histopathology images.
    """
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx, 0] + '.tif')
        try:
            image = Image.open(img_name).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image {img_name} not found. Skipping.")
            return None  # Return None to allow for skipping in the DataLoader collate function
        label = self.dataframe.iloc[idx, 1] if len(self.dataframe.columns) > 1 else -1  # -1 for test data
        if self.transform:
            image = self.transform(image)
        return image, label

class HistopathologyDataModule(LightningDataModule):
    """
    Lightning DataModule to manage data loading for training, validation, and testing.
    """
    def __init__(self, img_dir=TRAIN_DIR, labels_file=LABELS_FILE, test_dir=TEST_DIR, batch_size=BATCH_SIZE, target_size=TARGET_SIZE):
        super().__init__()
        self.img_dir = img_dir
        self.labels_file = labels_file
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.target_size = target_size

        # Define transformations
        self.train_transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        labels_df = pd.read_csv(self.labels_file)
        train_df, val_df = train_test_split(labels_df, test_size=0.2, stratify=labels_df['label'], random_state=42)
        self.train_dataset = HistologyDataset(train_df, self.img_dir, transform=self.train_transform)
        self.val_dataset = HistologyDataset(val_df, self.img_dir, transform=self.val_transform)
        
        # Initialize the test dataset if stage is "test"
        if stage == "test":
            test_df = pd.DataFrame({"id": os.listdir(self.test_dir)})
            self.test_dataset = HistologyDataset(test_df, self.test_dir, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        if hasattr(self, 'test_dataset'):
            return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        else:
            print("Test dataset not initialized. Call setup('test') before requesting test_dataloader.")
