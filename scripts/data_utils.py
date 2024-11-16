import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_lightning import LightningDataModule
from scripts.preprocessing import Preprocessing
from scripts.config import TRAIN_DIR, TEST_DIR, LABELS_FILE, BATCH_SIZE, TARGET_SIZE


class HistologyDataset(Dataset):
    """
    Custom dataset class for histopathology images, supporting preprocessing
    and dynamic transforms.
    """

    def __init__(self, dataframe, img_dir, preprocess_pipeline, mode: str):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing image IDs and labels.
            img_dir (str): Directory containing images.
            preprocess_pipeline (Preprocessing): Preprocessing pipeline object.
            mode (str): Mode for preprocessing ('train', 'val', 'test').
        """
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.preprocessor = preprocess_pipeline
        self.mode = mode

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['id']}.tif")
        label = row["label"] if "label" in row else -1  # Default -1 for unlabeled data

        # Load image and preprocess
        img = Image.open(img_path).convert("RGB")  # Convert to RGB
        preprocessed_img = self.preprocessor.preprocess_image(img, mode=self.mode)

        return preprocessed_img, torch.tensor(label, dtype=torch.float32)


class HistopathologyDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for managing train, validation, and test datasets
    with dynamic preprocessing and transforms.
    """

    def __init__(
        self,
        img_dir=TRAIN_DIR,
        labels_file=LABELS_FILE,
        test_dir=TEST_DIR,
        batch_size=BATCH_SIZE,
        target_size=TARGET_SIZE,
    ):
        """
        Args:
            img_dir (str): Directory containing training/validation images.
            labels_file (str): Path to the CSV file with image IDs and labels.
            test_dir (str): Directory containing test images.
            batch_size (int): Batch size for dataloaders.
            target_size (Tuple[int, int]): Target size for resizing images.
        """
        super().__init__()
        self.img_dir = img_dir
        self.labels_file = labels_file
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.target_size = target_size

        # Initialize preprocessing pipeline
        self.preprocessor = Preprocessing(input_size=target_size)

    def setup(self, stage=None):
        """
        Setup datasets for train, validation, and test phases.
        Args:
            stage (str): 'fit', 'test', or 'predict' to determine which datasets to set up.
        """
        # Load labels and split into train/validation sets
        labels_df = pd.read_csv(self.labels_file)
        train_df, val_df = train_test_split(
            labels_df, test_size=0.2, stratify=labels_df["label"], random_state=42
        )

        # Initialize datasets
        self.train_dataset = HistologyDataset(
            train_df, self.img_dir, self.preprocessor, mode="train"
        )
        self.val_dataset = HistologyDataset(
            val_df, self.img_dir, self.preprocessor, mode="val"
        )

        # Test dataset setup
        if stage == "test":
            test_files = [os.path.splitext(f)[0] for f in os.listdir(self.test_dir) if f.endswith(".tif")]
            test_df = pd.DataFrame({"id": test_files})
            self.test_dataset = HistologyDataset(
                test_df, self.test_dir, self.preprocessor, mode="test"
            )

    def train_dataloader(self):
        """Return the training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Return the validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Return the test DataLoader."""
        if hasattr(self, "test_dataset"):
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
        else:
            raise ValueError(
                "Test dataset not initialized. Call setup(stage='test') before requesting test_dataloader."
            )


def display_sample_images(dataset, label, sample_size=5):
    """
    Display original and preprocessed sample images for a specific label.

    Args:
        dataset (HistologyDataset): Dataset object to sample images from.
        label (int): Label to filter for (e.g., 0 or 1).
        sample_size (int): Number of images to display.
    """
    # Filter dataset for the given label
    label_data = dataset.dataframe[dataset.dataframe["label"] == label]

    # Ensure sample size doesn't exceed available data
    actual_sample_size = min(sample_size, len(label_data))
    if actual_sample_size == 0:
        print(f"No samples found for label {label}.")
        return

    label_data = label_data.sample(actual_sample_size)

    # Create a plot to show original and preprocessed images
    plt.figure(figsize=(15, 5))
    for i, (_, row) in enumerate(label_data.iterrows()):
        idx = dataset.dataframe.index.get_loc(row.name)
        preprocessed_img, lbl = dataset[idx]

        # Display preprocessed image
        plt.subplot(1, actual_sample_size, i + 1)
        plt.imshow(preprocessed_img.permute(1, 2, 0).numpy())
        plt.title(f"Label: {int(lbl.item())}")
        plt.axis("off")

    plt.suptitle(f"Samples for Label {label}", fontsize=16)
    plt.show()
