import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from typing import Tuple
import pandas as pd


class BaselineCNN(pl.LightningModule):
    """
    Baseline CNN model for binary classification in histopathology cancer detection.
    """

    def __init__(self, input_shape: Tuple[int, int, int] = (3, 96, 96), num_classes=1, learning_rate=1e-3):
        """
        Args:
            input_shape (Tuple[int, int, int]): Shape of input images (channels, height, width).
            num_classes (int): Number of output classes (default 1 for binary classification).
            learning_rate (float): Learning rate for the optimizer.
        """
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        # Define CNN architecture
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Calculate input size for the first FC layer dynamically
        fc_input_size = 128 * (input_shape[1] // 8) * (input_shape[2] // 8)
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # Metrics
        self.train_auc = torchmetrics.AUROC(task="binary")
        self.val_auc = torchmetrics.AUROC(task="binary")
        self.test_auc = torchmetrics.AUROC(task="binary")

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    def step(self, batch, stage: str):
        """
        Shared step logic for training, validation, and testing.
        """
        images, labels = batch
        logits = self(images).squeeze()
        loss = self.criterion(logits, labels.float())
        preds = torch.sigmoid(logits)
        
        if stage == "train":
            self.train_auc.update(preds, labels.int())
        elif stage == "val":
            self.val_auc.update(preds, labels.int())
        elif stage == "test":
            self.test_auc.update(preds, labels.int())

        return loss, preds

    def training_step(self, batch, batch_idx):
        loss, preds = self.step(batch, stage="train")
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_auc", self.train_auc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds = self.step(batch, stage="val")
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_auc", self.val_auc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds = self.step(batch, stage="test")
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_auc", self.test_auc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    @staticmethod
    def generate_predictions(model, dataloader, output_path="submission.csv"):
        """
        Generates predictions on a test DataLoader and saves them in CSV format.
        """
        model.eval()
        predictions = []

        device = next(model.parameters()).device
        with torch.no_grad():
            for batch in dataloader:
                images, ids = batch if len(batch) == 2 else (batch[0], None)
                images = images.to(device)
                outputs = model(images).squeeze()
                preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy().astype(int)
                if ids is not None:
                    predictions.extend(zip(ids, preds))
                else:
                    predictions.extend(enumerate(preds))

        submission_df = pd.DataFrame(predictions, columns=["id", "label"])
        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
