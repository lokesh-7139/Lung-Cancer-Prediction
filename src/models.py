"""
CNN model definitions for lung-cancer classification.

Provides a factory function that builds the same architecture used in the
original notebook.
"""

from __future__ import annotations

from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential

from src.config import (
    CONV1_FILTERS,
    CONV1_KERNEL,
    CONV2_FILTERS,
    CONV2_KERNEL,
    DENSE_UNITS,
    NUM_CLASSES,
    POOL_SIZE,
)


def build_cnn(input_shape: tuple[int, int, int]) -> Sequential:
    """Build the 3-class CNN classifier used throughout the notebook.

    Architecture::

        Conv2D(64, 3×3, relu) → MaxPool(2×2)
        Conv2D(64, 3×3, relu) → MaxPool(2×2)
        Flatten → Dense(16) → Dense(3, softmax)

    Parameters
    ----------
    input_shape:
        Shape of a single input sample, e.g. ``(256, 256, 1)``.

    Returns
    -------
    tensorflow.keras.Sequential
        Uncompiled Keras ``Sequential`` model.
    """
    model = Sequential(name="lung_cancer_cnn")

    model.add(Conv2D(CONV1_FILTERS, CONV1_KERNEL, activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=POOL_SIZE))

    model.add(Conv2D(CONV2_FILTERS, CONV2_KERNEL, activation="relu"))
    model.add(MaxPooling2D(pool_size=POOL_SIZE))

    model.add(Flatten())
    model.add(Dense(DENSE_UNITS))
    model.add(Dense(NUM_CLASSES, activation="softmax"))

    return model