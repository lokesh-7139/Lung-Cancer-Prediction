"""
Utility functions: evaluation metrics, plotting, and class-weight computation.
"""

from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from src.config import PLOTS_DIR


def compute_class_weights(y_train: np.ndarray) -> dict[int, float]:
    """Compute per-class weights inversely proportional to frequency.

    Formula (matches the notebook)::

        weight[c] = n_samples / (num_classes * count[c])

    Parameters
    ----------
    y_train:
        1-D array of integer training labels.

    Returns
    -------
    dict
        ``{class_label: weight, ...}``
    """
    counter = Counter(y_train)
    n_samples = len(y_train)
    num_classes = len(counter)
    return {
        cls: n_samples / (num_classes * count)
        for cls, count in counter.items()
    }


def evaluate_model(
    model,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    title: str = "Model Evaluation",
) -> np.ndarray:
    """Run predictions and print a classification report + confusion matrix.

    Parameters
    ----------
    model:
        A compiled Keras model.
    X_valid:
        Validation features.
    y_valid:
        Validation labels.
    title:
        Heading printed before the report.

    Returns
    -------
    np.ndarray
        Predicted class indices.
    """
    y_pred = model.predict(X_valid, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print(f"─── {title} ───")
    print(classification_report(y_valid, y_pred_classes))
    print(confusion_matrix(y_true=y_valid, y_pred=y_pred_classes))

    return y_pred_classes


def plot_training_history(history, name: str = "model") -> None:
    """Plot training & validation accuracy/loss from a Keras ``History``.

    The plots are saved to the ``plots/`` directory and displayed.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # -- Accuracy -----------------------------------------------------------
    plt.figure()
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(PLOTS_DIR / f"{name}_accuracy.png", dpi=150, bbox_inches="tight")
    plt.show()

    # -- Loss ---------------------------------------------------------------
    plt.figure()
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(PLOTS_DIR / f"{name}_loss.png", dpi=150, bbox_inches="tight")
    plt.show()


def print_label_counts(y_train: np.ndarray, y_valid: np.ndarray) -> None:
    """Print class distribution for train and validation sets."""
    print(f"Train   : {Counter(y_train)}")
    print(f"Valid   : {Counter(y_valid)}")