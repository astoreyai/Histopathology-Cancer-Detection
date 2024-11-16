"""
Script for generating predictions and creating a submission file for
the Histopathology Cancer Detection project.
"""

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
    # Instantiate data module and load test dataset
    data_module = HistopathologyDataModule(batch_size=BATCH_SIZE, target_size=TARGET_SIZE)
    data_module.setup(stage="test")

    print("Loading trained model checkpoint...")
    # Load trained model checkpoint
    model_path = "checkpoints/best_model.ckpt"
    model = BaselineCNN.load_from_checkpoint(model_path)

    print("Generating predictions...")
    # Generate predictions and save to a CSV file
    output_path = "submission.csv"
    model.generate_predictions(data_module.test_dataloader(), output_path=output_path)

    print(f"Predictions saved to {output_path}")

    # Uncomment these lines if running on Kaggle and syncing to GitHub
    # print("Syncing submission file to GitHub...")
    # os.system("git add submission.csv")
    # os.system('git commit -m "Add submission file for Kaggle"')
    # os.system("git push origin main")

if __name__ == "__main__":
    main()
