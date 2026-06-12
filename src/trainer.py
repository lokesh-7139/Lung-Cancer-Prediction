"""
Training routines for the three approaches used in the notebook:

1. SMOTE oversampling
2. Class-weighted training
3. Data augmentation + class-weighted training
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from src.config import (
    AUG_HORIZONTAL_FLIP,
    AUG_VERTICAL_FLIP,
    BATCH_SIZE,
    EPOCHS_AUGMENTATION,
    EPOCHS_CLASS_WEIGHT,
    EPOCHS_SMOTE,
    IMG_SIZE,
    MODELS_DIR,
    RANDOM_STATE,
    SMOTE_K_NEIGHBORS,
    TEST_SIZE,
)
from src.models import build_cnn
from src.utils import (
    compute_class_weights,
    evaluate_model,
    plot_training_history,
    print_label_counts,
)


def split_data(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train / validation split."""
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, random_state=RANDOM_STATE, stratify=y, test_size=TEST_SIZE,
    )
    print(f"Train: {len(X_train)}, shape: {X_train.shape}")
    print(f"Valid: {len(X_valid)}, shape: {X_valid.shape}")
    return X_train, X_valid, y_train, y_valid


# ── Approach 1: SMOTE ─────────────────────────────────────────────────────────


def train_with_smote(
    X_train: np.ndarray,
    X_valid: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Oversample the training set with SMOTE, then train a CNN."""
    print("\n═══ Approach 1: SMOTE Oversampling ═══")
    print_label_counts(y_train, y_valid)

    # Flatten for SMOTE
    n_train = X_train.shape[0]
    X_train_flat = X_train.reshape(n_train, IMG_SIZE * IMG_SIZE * 1)

    print(f"Before SMOTE: {Counter(y_train)}")
    smote = SMOTE(k_neighbors=SMOTE_K_NEIGHBORS)
    X_train_sampled, y_train_sampled = smote.fit_resample(X_train_flat, y_train)
    print(f"After SMOTE : {Counter(y_train_sampled)}")

    # Reshape back to image format
    X_train_sampled = X_train_sampled.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    print(f"X_train shape          : {X_train.shape}")
    print(f"X_train_sampled shape  : {X_train_sampled.shape}")

    # Build, compile, train
    model = build_cnn(input_shape=X_train.shape[1:])
    model.summary()
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    history = model.fit(
        X_train_sampled, y_train_sampled,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS_SMOTE,
        validation_data=(X_valid, y_valid),
        verbose=2,
    )

    # Save model (modern .keras format)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "model_smote.keras")

    # Evaluate
    evaluate_model(model, X_valid, y_valid, title="SMOTE Model Results")
    plot_training_history(history, name="smote")

    return model, history


# ── Approach 2: Class Weights ─────────────────────────────────────────────────


def train_with_class_weights(
    X_train: np.ndarray,
    X_valid: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Train a CNN on the original (imbalanced) data using per-class weights."""
    print("\n═══ Approach 2: Class-Weighted Training ═══")
    print_label_counts(y_train, y_valid)

    class_weights = compute_class_weights(y_train)
    print("Class weights:", class_weights)

    model = build_cnn(input_shape=X_train.shape[1:])
    model.summary()
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS_CLASS_WEIGHT,
        validation_data=(X_valid, y_valid),
        class_weight=class_weights,
        verbose=2,
    )

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "model_class_weighted.keras")

    evaluate_model(model, X_valid, y_valid, title="Class-Weighted Model Results")
    plot_training_history(history, name="class_weighted")

    return model, history


# ── Approach 3: Data Augmentation + Class Weights ─────────────────────────────


def _build_augmentation_layer() -> tf.keras.Sequential:
    """Build a Keras preprocessing model for real-time data augmentation.

    Replaces the deprecated ``ImageDataGenerator`` with modern Keras
    preprocessing layers.
    """
    # Map config booleans to the appropriate flip mode.
    if AUG_HORIZONTAL_FLIP and AUG_VERTICAL_FLIP:
        mode = "horizontal_and_vertical"
    elif AUG_HORIZONTAL_FLIP:
        mode = "horizontal"
    elif AUG_VERTICAL_FLIP:
        mode = "vertical"
    else:
        mode = "horizontal"  # fallback; RandomFlip is required

    augmentation = tf.keras.Sequential(
        [tf.keras.layers.RandomFlip(mode)],
        name="data_augmentation",
    )
    return augmentation


def train_with_augmentation(
    X_train: np.ndarray,
    X_valid: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Train a CNN using real-time data augmentation and class weights.

    Uses ``tf.data.Dataset`` pipelines with Keras preprocessing layers,
    replacing the deprecated ``ImageDataGenerator``.
    """
    print("\n═══ Approach 3: Data Augmentation + Class Weights ═══")
    print_label_counts(y_train, y_valid)

    class_weights = compute_class_weights(y_train)
    print("Class weights:", class_weights)

    # Build augmentation layer
    augmentation = _build_augmentation_layer()

    # Build tf.data pipelines (modern replacement for ImageDataGenerator)
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(buffer_size=len(X_train), seed=RANDOM_STATE)
        .batch(BATCH_SIZE)
        .map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = build_cnn(input_shape=X_train.shape[1:])
    model.summary()
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    history = model.fit(
        train_ds,
        epochs=EPOCHS_AUGMENTATION,
        validation_data=val_ds,
        class_weight=class_weights,
        verbose=2,
    )

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "model_augmented.keras")

    evaluate_model(model, X_valid, y_valid, title="Augmentation Model Results")
    plot_training_history(history, name="augmented")

    return model, history
