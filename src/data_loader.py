"""
Data-loading utilities.

Loads images from disk, analyses size distributions, and displays samples.
"""

from __future__ import annotations

import random
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.config import CATEGORIES, DATASET_DIR, IMG_SIZE


# ── EDA helpers ───────────────────────────────────────────────────────────────


def get_image_size_distribution() -> dict[str, dict[str, int]]:
    """Analyse the distribution of image sizes across all categories.

    Returns
    -------
    dict
        ``{category: {"<height> x <width>": count, ...}, ...}``
    """
    size_data: dict[str, dict[str, int]] = {}
    for category in CATEGORIES:
        path = DATASET_DIR / category
        temp_dict: dict[str, int] = {}
        for filepath in path.iterdir():
            img = cv2.imread(str(filepath))
            if img is None:
                continue
            height, width = img.shape[:2]
            key = f"{height} x {width}"
            temp_dict[key] = temp_dict.get(key, 0) + 1
        size_data[category] = temp_dict
    return size_data


def show_sample_images() -> None:
    """Display one sample image from each category (grayscale)."""
    for category in CATEGORIES:
        path = DATASET_DIR / category
        for filepath in path.iterdir():
            print(category)
            img = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
            plt.imshow(img, cmap="gray")
            plt.show()
            break


def show_preprocessing_preview(samples_per_category: int = 3) -> None:
    """Show side-by-side original → resized → blurred for each category."""
    for category in CATEGORIES:
        fig, axes = plt.subplots(samples_per_category, 3, figsize=(15, 15))
        fig.suptitle(category)

        path = DATASET_DIR / category
        count = 0
        for filepath in path.iterdir():
            img = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)

            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img_blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)

            axes[count, 0].imshow(img, cmap="gray")
            axes[count, 1].imshow(img_resized, cmap="gray")
            axes[count, 2].imshow(img_blurred, cmap="gray")
            count += 1
            if count == samples_per_category:
                break

        plt.show()


# ── Main data loader ──────────────────────────────────────────────────────────


def load_data() -> tuple[np.ndarray, np.ndarray]:
    """Load all images, resize to ``IMG_SIZE × IMG_SIZE``, and return arrays.

    Returns
    -------
    X : np.ndarray
        Shape ``(N, IMG_SIZE, IMG_SIZE, 1)``, ``float32`` in ``[0, 1]``.
    y : np.ndarray
        Integer class labels.
    """
    data: list[tuple[np.ndarray, int]] = []
    for class_num, category in enumerate(CATEGORIES):
        path = DATASET_DIR / category
        for filepath in path.iterdir():
            img = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append((img, class_num))

    random.shuffle(data)

    X = np.array([item[0] for item in data], dtype=np.float32)
    y = np.array([item[1] for item in data], dtype=np.int32)

    # Add channel dim and normalise to [0, 1]
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0

    print(f"X length: {len(X)}")
    print(f"y counts: {dict(Counter(y))}")

    return X, y