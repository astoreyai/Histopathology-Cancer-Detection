# Histopathology-Cancer-Detection/scripts/config.py

# Path configurations
TRAIN_DIR = "/kaggle/input/histopathologic-cancer-detection/train"
TEST_DIR = "/kaggle/input/histopathologic-cancer-detection/test"
LABELS_FILE = "/kaggle/input/histopathologic-cancer-detection/train_labels.csv"

# Model and training configurations
TARGET_SIZE = (96, 96)
BATCH_SIZE = 32
LEARNING_RATE = 0.00007
EPOCHS = 5
NUM_CLASSES = 2

# Checkpoint and logging configurations
CHECKPOINT_PATH = "./checkpoints/"
MLFLOW_TRACKING_URI = "file:./experiments/mlruns"
EXPERIMENT_NAME = "Histopathology Cancer Detection"
