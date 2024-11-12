import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score
import pandas as pd

class BaselineCNN(pl.LightningModule):
    """
    Baseline CNN model for binary classification in histopathology cancer detection.
    """
    def __init__(self, input_shape=(3, 96, 96), num_classes=1, learning_rate=1e-3):
        super(BaselineCNN, self).__init__()
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
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

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
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images).squeeze()
        loss = self.criterion(outputs, labels.float())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images).squeeze()
        loss = self.criterion(outputs, labels.float())
        try:
            auc_score = roc_auc_score(labels.cpu().numpy(), outputs.cpu().numpy())
        except ValueError:
            auc_score = 0
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_auc", auc_score, prog_bar=True)
        return {"val_loss": loss, "val_auc": auc_score}

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images).squeeze()
        try:
            auc_score = roc_auc_score(labels.cpu().numpy(), outputs.cpu().numpy())
        except ValueError:
            auc_score = 0
        self.log("test_auc", auc_score)
        return {"test_auc": auc_score}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def predict_step(self, batch, batch_idx):
        images, ids = batch
        outputs = self(images).squeeze()
        preds = (outputs > 0.5).float()
        return ids, preds

    def generate_predictions(self, dataloader, output_path="submission.csv"):
        """
        Generates predictions on a test DataLoader and saves them in CSV format for Kaggle submission.
        """
        self.eval()
        predictions = []
        
        # Ensure model and data are on the right device
        device = next(self.parameters()).device
        with torch.no_grad():
            for images, ids in dataloader:
                images = images.to(device)
                outputs = self(images).squeeze()
                preds = (outputs > 0.5).float()  # Binary classification threshold
                predictions.extend(zip(ids, preds.cpu().numpy().astype(int)))

        submission_df = pd.DataFrame(predictions, columns=["id", "label"])
        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
