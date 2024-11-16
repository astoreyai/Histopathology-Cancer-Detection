import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from scripts.data_utils import HistopathologyDataModule
from scripts.model_utils import BaselineCNN
from scripts.config import BATCH_SIZE, LEARNING_RATE, EPOCHS, TARGET_SIZE

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
    data_module = HistopathologyDataModule(
        batch_size=BATCH_SIZE,
        target_size=TARGET_SIZE
    )

    # Initialize model
    model = BaselineCNN(input_shape=(3, *TARGET_SIZE), learning_rate=LEARNING_RATE)

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="best_model-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min"
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        logger=mlflow_logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        strategy="ddp_notebook" if torch.cuda.device_count() > 1 else None,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        log_every_n_steps=50,
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
