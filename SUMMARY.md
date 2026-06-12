# Lung Cancer Prediction — Project Summary

> A complete technical walkthrough of the end-to-end deep-learning pipeline for
> classifying lung CT scans into **Benign**, **Malignant**, and **Normal**
> categories.

---

## 1. Problem Statement

Lung cancer remains one of the leading causes of cancer-related deaths
worldwide. Early and accurate detection from CT imaging can significantly
improve patient outcomes. This project builds an automated image classifier
that takes a lung CT scan and predicts whether it shows **benign**,
**malignant**, or **normal** tissue.

The core challenge is **class imbalance** — certain categories have far fewer
samples than others. Three distinct strategies are implemented and compared to
address this.

---

## 2. High-Level Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Image Load  │───▶│ Preprocessing│───▶│   Training   │───▶│  Evaluation  │
│  & EDA       │    │  & Splitting │    │  (3 Models)  │    │  & Plotting  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
   data_loader.py      data_loader.py      trainer.py          utils.py
                        + config.py        + models.py
```

**Entry point:** `main.py` orchestrates the full pipeline in sequence.

---

## 3. Module Responsibilities

```
src/
├── config.py        # Central configuration: paths, hyperparameters, constants
├── logger.py        # Timestamped run-logging (mirrors console to log files)
├── data_loader.py   # Loads images from disk, EDA helpers, preprocessing
├── models.py        # CNN architecture definition (Keras Sequential)
├── trainer.py       # Training routines for all three approaches
└── utils.py         # Class-weight computation, evaluation metrics, plotting
```

### 3.1 `config.py`

All tuneable parameters live here. Changing a value here propagates through
the entire pipeline — no magic numbers scattered across files.

| Category     | Key Constants                                                                        |
| ------------ | ------------------------------------------------------------------------------------ |
| Paths        | `PROJECT_ROOT`, `MODELS_DIR`, `PLOTS_DIR`, `LOGS_DIR`, `DATASET_DIR`                 |
| Image        | `IMG_SIZE=256`, `IMG_CHANNELS=1` (grayscale)                                         |
| Split        | `TEST_SIZE=0.25`, `RANDOM_STATE=10`                                                  |
| Training     | `BATCH_SIZE=8`, `EPOCHS_SMOTE=10`, `EPOCHS_CLASS_WEIGHT=10`, `EPOCHS_AUGMENTATION=5` |
| Architecture | `CONV1_FILTERS=64`, `CONV2_FILTERS=64`, `DENSE_UNITS=16`, `NUM_CLASSES=3`            |
| SMOTE        | `SMOTE_K_NEIGHBORS=5`                                                                |
| Augmentation | `AUG_HORIZONTAL_FLIP=True`, `AUG_VERTICAL_FLIP=True`                                 |

### 3.2 `logger.py`

Provides `start_run_logging()` which:

1. Creates a timestamped log file in `logs/` (e.g. `run_20260612_211130.log`).
2. Redirects `stdout` and `stderr` through a `_TeeFilter` that mirrors
   everything to both the terminal and the log file.
3. Filters out TensorFlow/Keras internal noise messages from the log file.

Called once at the very start of `main()`.

### 3.3 `data_loader.py`

Handles all image I/O and exploratory data analysis:

| Function                        | Purpose                                                                      |
| ------------------------------- | ---------------------------------------------------------------------------- |
| `get_image_size_distribution()` | Scans all categories and returns a dict of image dimension counts            |
| `show_sample_images()`          | Displays one grayscale sample per category using Matplotlib                  |
| `show_preprocessing_preview()`  | Side-by-side comparison: original → resized → Gaussian-blurred               |
| `load_data()`                   | Loads all images, resizes to 256×256, normalises to [0, 1], returns `(X, y)` |

**Preprocessing applied:**

1. Read as grayscale (single channel).
2. Resize to `IMG_SIZE × IMG_SIZE` (256 × 256).
3. Cast to `float32` and divide by 255 → values in `[0.0, 1.0]`.
4. Reshape to `(N, 256, 256, 1)` — channel-last format for Keras.

### 3.4 `models.py`

Defines the CNN architecture used across all three approaches via
`build_cnn(input_shape)`:

```
Input (256 × 256 × 1)
  │
  ├─ Conv2D(64 filters, 3×3 kernel, ReLU activation)
  ├─ MaxPooling2D(2×2)
  │
  ├─ Conv2D(64 filters, 3×3 kernel, ReLU activation)
  ├─ MaxPooling2D(2×2)
  │
  ├─ Flatten
  ├─ Dense(16 units)
  └─ Dense(3 units, Softmax)  →  probability over [Benign, Malignant, Normal]
```

The model is returned **uncompiled** — the loss function and optimizer are
configured in `trainer.py` so each approach can be tuned independently.

### 3.5 `trainer.py`

Contains the three training routines. Each one:

1. Prints the class distribution.
2. Applies its specific strategy (SMOTE / class weights / augmentation).
3. Builds and compiles the CNN.
4. Trains the model.
5. Saves the trained model to `models/`.
6. Evaluates on the validation set.
7. Plots training history to `plots/`.

### 3.6 `utils.py`

Shared helper functions:

| Function                               | Purpose                                                                 |
| -------------------------------------- | ----------------------------------------------------------------------- |
| `compute_class_weights(y_train)`       | Calculates inverse-frequency weights: `n_samples / (n_classes × count)` |
| `evaluate_model(model, X, y)`          | Runs predictions, prints classification report + confusion matrix       |
| `plot_training_history(history)`       | Plots accuracy and loss curves, saves to `plots/`                       |
| `print_label_counts(y_train, y_valid)` | Prints `Counter` of class labels for train/valid splits                 |

---

## 4. The Three Training Approaches

### 4.1 Approach 1 — SMOTE Oversampling

**Problem:** The training data is imbalanced — some classes have many more
samples than others.

**Solution:** SMOTE (Synthetic Minority Over-sampling Technique) generates
synthetic samples for under-represented classes by interpolating between
existing minority-class neighbours.

**Pipeline:**

```
X_train (flatten) → SMOTE(k=5) → X_train_sampled (reshape) → CNN training
```

- Training data is flattened to 1D vectors for SMOTE.
- After oversampling, reshaped back to `(N, 256, 256, 1)`.
- After this step, all classes have equal representation.
- Model trained for 10 epochs with Adam optimizer.
- Loss: `sparse_categorical_crossentropy`.

**Best for:** When you want perfectly balanced training data and can tolerate
the synthetic samples.

### 4.2 Approach 2 — Class-Weighted Training

**Problem:** Same imbalance, but we don't want to modify the data.

**Solution:** Assign higher loss weights to under-represented classes so the
model pays more attention to them during training.

**Pipeline:**

```
Counter(y_train) → compute_class_weights() → model.fit(class_weight=weights)
```

- Weight formula: `weight[c] = total_samples / (num_classes × count[c])`
- Example: if class 0 has 100 samples and class 2 has 500, class 0 gets a
  higher weight.
- Original (imbalanced) data is used directly.
- No data modification — the imbalance is handled in the loss function.

**Best for:** Keeping the original data distribution intact while still
accounting for imbalance.

### 4.3 Approach 3 — Data Augmentation + Class Weights

**Problem:** Small dataset + imbalance → overfitting risk.

**Solution:** Combine real-time data augmentation (to increase effective
dataset size) with class-weighted training (to handle imbalance).

**Pipeline:**

```
X_train, y_train
  → tf.data.Dataset.from_tensor_slices()
  → .shuffle()
  → .batch(BATCH_SIZE)
  → .map(augmentation)        # RandomFlip applied on-the-fly
  → .prefetch(AUTOTUNE)
  → model.fit(class_weight=weights)
```

- Uses modern Keras `RandomFlip` preprocessing layer (replaces deprecated
  `ImageDataGenerator`).
- Augmentations applied: horizontal flip, vertical flip (configurable).
- Each epoch sees different augmented versions of the same images.
- Validation data is **not** augmented — only batched and prefetched.

**Best for:** Maximising generalisation with limited training data.

---

## 5. Data Flow Diagram

```
                         ┌─────────────────┐
                         │  CT Scan Images  │
                         │  (3 categories)  │
                         └────────┬────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │     load_data()            │
                    │  • Read grayscale          │
                    │  • Resize 256×256          │
                    │  • Normalise [0, 1]        │
                    │  • Return X, y             │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │     split_data()           │
                    │  • Stratified 75/25 split  │
                    └─────────────┬─────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
    ┌─────────▼─────────┐ ┌──────▼──────┐ ┌──────────▼──────────┐
    │   train_with_     │ │  train_with_│ │   train_with_       │
    │   smote()         │ │  class_     │ │   augmentation()    │
    │                   │ │  weights()  │ │                     │
    │  SMOTE → CNN      │ │  CNN+weights│ │  Aug+weights → CNN  │
    └─────────┬─────────┘ └──────┬──────┘ └──────────┬──────────┘
              │                   │                   │
              ▼                   ▼                   ▼
         model_smote.keras  model_class_      model_augmented.keras
                            weighted.keras
              │                   │                   │
              └───────────────────┼───────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │      evaluate_model()      │
                    │  • Classification report   │
                    │  • Confusion matrix        │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │   plot_training_history()  │
                    │  • Accuracy curves         │
                    │  • Loss curves             │
                    └───────────────────────────┘
```

---

## 6. CNN Architecture Details

| Layer                  | Output Shape   | Parameters        |
| ---------------------- | -------------- | ----------------- |
| Conv2D (64, 3×3, ReLU) | (254, 254, 64) | 640               |
| MaxPool2D (2×2)        | (127, 127, 64) | 0                 |
| Conv2D (64, 3×3, ReLU) | (125, 125, 64) | 36,928            |
| MaxPool2D (2×2)        | (62, 62, 64)   | 0                 |
| Flatten                | (245,760)      | 0                 |
| Dense (16)             | (16)           | 3,932,176         |
| Dense (3, Softmax)     | (3)            | 51                |
| **Total**              |                | **~3.97M params** |

**Key observations:**

- The vast majority of parameters (~99%) are in the first Dense layer due to
  flattening a 62×62×64 feature map.
- The convolutional layers are lightweight (only ~37K params).
- This is an intentionally simple architecture matching the original notebook.
  For production use, consider adding dropout, batch normalisation, or using a
  pre-trained backbone.

---

## 7. Training Configuration

All three approaches use the same core settings:

| Setting          | Value                             |
| ---------------- | --------------------------------- |
| Optimiser        | Adam (default lr=0.001)           |
| Loss             | `sparse_categorical_crossentropy` |
| Metric           | `accuracy`                        |
| Batch size       | 8                                 |
| Validation split | 25% (stratified)                  |

Differences per approach:

|               | SMOTE                 | Class Weights         | Augmentation + Weights |
| ------------- | --------------------- | --------------------- | ---------------------- |
| Epochs        | 10                    | 10                    | 5                      |
| Data          | Synthetic oversampled | Original (imbalanced) | Augmented on-the-fly   |
| Class weights | No                    | Yes                   | Yes                    |
| Augmentation  | No                    | No                    | RandomFlip (H+V)       |
| Data pipeline | NumPy arrays          | NumPy arrays          | `tf.data.Dataset`      |

---

## 8. Outputs

After running `python main.py`, the following files are generated:

```
logs/
└── run_YYYYMMDD_HHMMSS.log     # Full console output

models/
├── model_smote.keras            # SMOTE-trained model
├── model_class_weighted.keras   # Class-weighted model
└── model_augmented.keras        # Augmented model

plots/
├── smote_accuracy.png           # SMOTE accuracy curves
├── smote_loss.png               # SMOTE loss curves
├── class_weighted_accuracy.png  # Class-weighted accuracy curves
├── class_weighted_loss.png      # Class-weighted loss curves
├── augmented_accuracy.png       # Augmented accuracy curves
└── augmented_loss.png           # Augmented loss curves
```

**How to read the evaluation output:**

- **Classification Report** — Precision, recall, F1-score per class. Higher is
  better. Look for balance across classes.
- **Confusion Matrix** — Rows are true labels, columns are predicted. The
  diagonal should be dominant (correct predictions).
- **Accuracy/Loss Curves** — Training vs. validation. If training accuracy
  keeps rising while validation plateaus or drops, the model is overfitting.

---

## 9. Technology Stack

| Component        | Library            | Version |
| ---------------- | ------------------ | ------- |
| Deep Learning    | TensorFlow / Keras | ≥ 2.12  |
| Image Processing | OpenCV             | ≥ 4.5   |
| Metrics          | scikit-learn       | ≥ 1.0   |
| Oversampling     | imbalanced-learn   | ≥ 0.9   |
| Plotting         | Matplotlib         | ≥ 3.5   |
| Language         | Python             | ≥ 3.10  |

---

## 10. File Summary

| File                 | Lines | Purpose                                      |
| -------------------- | ----- | -------------------------------------------- |
| `main.py`            | ~70   | Entry point — orchestrates the full pipeline |
| `src/config.py`      | ~52   | All configuration constants                  |
| `src/logger.py`      | ~69   | Timestamped run-logging with noise filtering |
| `src/data_loader.py` | ~115  | Image loading, EDA, preprocessing            |
| `src/models.py`      | ~48   | CNN architecture definition                  |
| `src/trainer.py`     | ~190  | Three training routines + augmentation layer |
| `src/utils.py`       | ~80   | Evaluation, plotting, class-weight utilities |
| `requirements.txt`   | ~12   | Python dependencies                          |
| `README.md`          | —     | Project overview with setup instructions     |
| `.gitignore`         | —     | Excludes data, models, logs, plots from git  |
| `LICENSE`            | —     | CC BY 4.0 (attribution to Kaggle community)  |
| `SUMMARY.md`         | —     | This document                                |

---
