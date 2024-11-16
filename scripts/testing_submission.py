"""
Script for generating predictions and creating a submission file for
the Histopathology Cancer Detection project.
"""

import os
import torch
from scripts.data_utils import HistopathologyDataModule
from scripts.model_utils import BaselineCNN
from scripts.config import TEST_DIR, BATCH_SIZE, TARGET_SIZE

def main():
    """
    Main function to load the model, test data, generate predictions,
    and save the results to a CSV file.
    """
    print("Initializing data module...")
    # Instantiate data module and prepare the test dataset
    data_module = HistopathologyDataModule(
        batch_size=BATCH_SIZE,
        target_size=TARGET_SIZE,
        test_dir=TEST_DIR
    )
    data_module.setup(stage="test")

    print("Loading trained model checkpoint...")
    # Ensure the checkpoint exists before loading
    model_path = "checkpoints/best_model.ckpt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    model = BaselineCNN.load_from_checkpoint(model_path)
    model.eval()  # Set the model to evaluation mode

    print("Generating predictions...")
    # Generate predictions and save to a CSV file
    output_path = "submission.csv"
    try:
        model.generate_predictions(data_module.test_dataloader(), output_path=output_path)
        print(f"Predictions saved to {output_path}")
    except Exception as e:
        print(f"Error generating predictions: {e}")

    # Optional: Sync submission file to GitHub if running on Kaggle
    sync_to_github = False  # Set to True if needed
    if sync_to_github:
        print("Syncing submission file to GitHub...")
        os.system("git add submission.csv")
        os.system('git commit -m "Add submission file for Kaggle"')
        os.system("git push origin main")
        print("Submission file synced to GitHub.")

if __name__ == "__main__":
    main()
