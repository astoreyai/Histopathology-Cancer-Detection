# data_utils.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from configs import TRAIN_DIR, EDA_SAMPLE_SIZE

def load_labels():
    """Load labels dataframe from the path specified in configs."""
    return pd.read_csv(LABELS_FILE)

def display_sample_images(df, img_dir, label, sample_size=5):
    """Display a sample of images for a given label."""
    sample_df = df[df['label'] == label].sample(sample_size, random_state=42)
    fig, axes = plt.subplots(1, sample_size, figsize=(15, 5))
    for i, row in enumerate(sample_df.iterrows()):
        img_id = row[1]['id']
        img_path = os.path.join(img_dir, img_id + '.tif')
        img = Image.open(img_path)
        axes[i].imshow(img)
        axes[i].axis("off")
    title = "Cancerous" if label == 1 else "Non-Cancerous"
    plt.suptitle(f"{sample_size} Sample {title} Images")
    plt.show()

def calculate_mean_intensity(df, img_dir, label, sample_size=EDA_SAMPLE_SIZE):
    """Calculate pixel intensity distribution for a subset of images."""
    sample_df = df[df['label'] == label].sample(sample_size, random_state=42)
    intensities = []
    for img_id in sample_df['id']:
        img_path = os.path.join(img_dir, img_id + '.tif')
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        intensities.extend(np.array(img).flatten())
    return intensities
