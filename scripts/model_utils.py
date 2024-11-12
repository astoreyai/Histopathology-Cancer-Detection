import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score
from pytorch_lightning import LightningModule
import pandas as pd

class BaselineCNN(LightningModule):
    """
    Baseline CNN model for binary classification in histopathology cancer detection.

    Args:
        input_shape (tuple): Shape of input images, typically (3, height, width).
        num_classes (int): Number of output classes.
        learning_rate (float): Learning rate for the optimizer.
    """
    def __init__(self, input_shape=(3, 96, 96), num_classes=1, learning_rate=1e-3):
        super(BaselineCNN, self).__init__()
        self.save_hyperparameters()  # Save hyperparameters for model reloading
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
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Define fully connected layers
        fc_input_size = 128 * (input_shape[1] // 8) * (input_shape[2] // 8)
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

        # Loss function
        self.criterion = nn.BCELoss()

    def forward(self, x):
        """
        Forward pass for the model.
        """
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    def training_step(self, batch, batch_idx):
        """
        Training step that calculates loss and logs it.
        """
        images, labels = batch
        outputs = self(images).squeeze()
        loss = self.criterion(outputs, labels.float())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step to calculate loss and AUC score, logging both metrics.
        """
        images, labels = batch
        outputs = self(images).squeeze()
        loss = self.criterion(outputs, labels.float())
        
        # Calculate AUC score
        try:
            auc_score = roc_auc_score(labels.cpu().numpy(), outputs.cpu().numpy())
        except ValueError:
            auc_score = 0  # Handle cases with a single class in batch
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_auc", auc_score, prog_bar=True)
        return {"val_loss": loss, "val_auc": auc_score}

    def test_step(self, batch, batch_idx):
        """
        Test step to calculate AUC score for evaluation on test data.
        """
        images, labels = batch
        outputs = self(images).squeeze()
        auc_score = roc_auc_score(labels.cpu().numpy(), outputs.cpu().numpy())
        self.log("test_auc", auc_score)
        return {"test_auc": auc_score}

    def configure_optimizers(self):
        """
        Sets up the optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def predict_step(self, batch, batch_idx):
        """
        Inference step to generate predictions for Kaggle submission.
        """
        images, ids = batch
        outputs = self(images).squeeze()
        preds = (outputs > 0.5).float()  # Threshold for binary classification
        return ids, preds

    def save_for_submission(self, dataloader, output_path="submission.csv"):
        """
        Generates predictions on a dataloader and saves them in CSV format for Kaggle submission.

        Args:
            dataloader (DataLoader): DataLoader for test data.
            output_path (str): Path to save the CSV output for submission.
        """
        self.eval()
        predictions = []

        for batch in dataloader:
            ids, preds = self.predict_step(batch, None)
            predictions.extend(zip(ids, preds.cpu().numpy().astype(int)))

        # Convert to DataFrame and save
        submission_df = pd.DataFrame(predictions, columns=["id", "label"])
        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
