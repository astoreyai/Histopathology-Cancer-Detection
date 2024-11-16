import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from scripts.data_utils import HistopathologyDataModule
from scripts.model_utils import BaselineCNN
from scripts.config import BATCH_SIZE, LEARNING_RATE, EPOCHS

def train_model():
    """
    Trains the histopathology CNN model with PyTorch Lightning.
    Handles logging, checkpoints, and validation during training.
    """
    print("Starting training...")

    # Setup MLflow Logger
    mlflow_logger = MLFlowLogger(
        experiment_name="Histopathology Cancer Detection - Training",
        tracking_uri="file:./experiments/mlruns"
    )

    # Initialize DataModule
    data_module = HistopathologyDataModule()

    # Initialize model
    model = BaselineCNN(input_size=data_module.target_size, learning_rate=LEARNING_RATE)

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="best_model",
        save_top_k=1,
        mode="min"
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min"
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        logger=mlflow_logger,
        accelerator="gpu",
        devices=2 if torch.cuda.is_available() else 1,
        strategy="ddp_notebook",
        callbacks=[checkpoint_callback, early_stopping]
    )

    # Train the model
    trainer.fit(model, data_module)

    # Save the best model
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")

    # Log the best model path to MLflow
    mlflow_logger.experiment.log_param(
        run_id=mlflow_logger.run_id,
        key="best_model_path",
        value=best_model_path
    )

    print("Training completed.")

# Run the training if called as a standalone script
if __name__ == "__main__":
    train_model()
