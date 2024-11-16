import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz as sv
from scripts.config import LABELS_FILE, TRAIN_DIR, TARGET_SIZE
from scripts.data_utils import HistologyDataset, display_sample_images
from scripts.preprocessing import Preprocessing

def run_eda():
    """
    Executes exploratory data analysis (EDA) on the dataset.
    Outputs class distributions, summary statistics, and Sweetviz reports.
    """
    print("Starting EDA...")

    # Load labels
    labels_df = pd.read_csv(LABELS_FILE)
    print(f"Loaded dataset with {len(labels_df)} entries.")

    # Initialize preprocessing pipeline
    preprocessor = Preprocessing(
        input_size=TARGET_SIZE,
        stain_correction=True,
        stain_method="macenko",
        mask_size=(32, 32),
        stain_reference_image_path="path/to/reference_image.tif"
    )

    # Display basic information
    print("\nDataset Info:")
    print(labels_df.info())
    print("\nSummary Statistics:")
    print(labels_df.describe())

    # Visualize class distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='label', data=labels_df, palette='pastel')
    plt.title('Class Distribution of Cancer Labels')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.show()

    # Create a dataset instance for sampling
    dataset = HistologyDataset(dataframe=labels_df, img_dir=TRAIN_DIR, preprocess_pipeline=preprocessor, mode="eda")

    # Display sample images for each label
    for label in labels_df["label"].unique():
        print(f"Samples for label {label}:")
        display_sample_images(dataset, label=label, sample_size=5)

    # Generate a Sweetviz report
    eda_report = sv.analyze(labels_df)
    eda_report_path = "eda_report.html"
    eda_report.show_html(filepath=eda_report_path, open_browser=False)
    print(f"Sweetviz report generated: {eda_report_path}")

    print("EDA completed.")

# Run the EDA if called as a standalone script
if __name__ == "__main__":
    run_eda()
