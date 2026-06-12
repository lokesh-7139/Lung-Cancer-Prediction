#!/usr/bin/env python3
"""
Lung Cancer Prediction from CT Scan Images
============================================
Converted from the original Jupyter Notebook:
    ``lung-cancer-prediction-on-image-data.ipynb``

This script reproduces the full notebook pipeline:

1. EDA — image-size distribution & sample previews
2. Data loading & preprocessing
3. Train / validation split
4. Three modelling approaches:
   a) SMOTE oversampling
   b) Class-weighted training
   c) Data augmentation + class weights
5. Evaluation & plots
"""

from __future__ import annotations

from src.data_loader import (
    get_image_size_distribution,
    load_data,
    show_preprocessing_preview,
    show_sample_images,
)
from src.logger import start_run_logging
from src.trainer import (
    split_data,
    train_with_augmentation,
    train_with_class_weights,
    train_with_smote,
)


def main() -> None:
    # Start run logging (captures console output to timestamped log file)
    start_run_logging()

    # ── 1. Exploratory Data Analysis ─────────────────────────────────────────
    print("═══ Image Size Distribution ═══")
    size_data = get_image_size_distribution()
    for category, sizes in size_data.items():
        print(f"\n{category}:")
        for size_key, count in sizes.items():
            print(f"  {size_key}: {count} images")

    print("\n═══ Sample Images (one per category) ═══")
    show_sample_images()

    print("\n═══ Preprocessing Preview ═══")
    show_preprocessing_preview(samples_per_category=3)

    # ── 2. Load & Prepare Data ───────────────────────────────────────────────
    print("\n═══ Loading Data ═══")
    X, y = load_data()

    # ── 3. Train / Validation Split ──────────────────────────────────────────
    print("\n═══ Splitting Data ═══")
    X_train, X_valid, y_train, y_valid = split_data(X, y)

    # ── 4. Approach 1: SMOTE ─────────────────────────────────────────────────
    train_with_smote(X_train, X_valid, y_train, y_valid)

    # ── 5. Approach 2: Class Weights ─────────────────────────────────────────
    train_with_class_weights(X_train, X_valid, y_train, y_valid)

    # ── 6. Approach 3: Data Augmentation + Class Weights ─────────────────────
    train_with_augmentation(X_train, X_valid, y_train, y_valid)

    print("\n═══ All experiments completed. ═══")


if __name__ == "__main__":
    main()