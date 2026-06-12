<div align="center">

# 🫁 Lung Cancer Prediction from CT Scan Images

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey?style=flat)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat)

**Deep-learning pipeline for classifying lung CT scans into Benign, Malignant, and Normal categories.**

</div>

---

## 📋 Overview

This project implements an end-to-end image classification pipeline originally developed as a Jupyter Notebook, refactored into production-quality Python modules. It trains a CNN on lung CT scan images using three different strategies to handle class imbalance:

| #   | Approach                         | Strategy                                                   |
| --- | -------------------------------- | ---------------------------------------------------------- |
| 1   | **SMOTE**                        | Synthetic oversampling of minority classes before training |
| 2   | **Class Weights**                | Cost-sensitive learning with per-class weight penalties    |
| 3   | **Augmentation + Class Weights** | Real-time data augmentation combined with class weights    |

## 📁 Project Structure

```
Lung Cancer/
├── main.py                  # Entry point — runs the full pipeline
├── requirements.txt         # Python dependencies
├── README.md
├── LICENSE
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py            # All paths, hyperparameters, and constants
│   ├── data_loader.py       # Image loading, EDA, and preprocessing
│   ├── logger.py            # Timestamped run-logging with TF noise filtering
│   ├── models.py            # CNN architecture (Keras Sequential)
│   ├── trainer.py           # Training routines for all three approaches
│   └── utils.py             # Evaluation metrics, plotting, class-weight helpers
├── data/                    # (not tracked) CT scan images
│   ├── Bengin cases/
│   ├── Malignant cases/
│   └── Normal cases/
├── models/                  # (not tracked) Saved .keras model files
├── plots/                   # (not tracked) Accuracy/loss plots
└── logs/                    # (not tracked) Timestamped run logs
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or later
- CUDA + cuDNN (for GPU acceleration, optional but recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/lokesh-7139/Lung-Cancer-Prediction.git
cd Lung-Cancer-Prediction

# Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset

Place your CT scan images inside the `data/` directory following this structure:

```
data/
├── Bengin cases/       # Benign CT scan images
├── Malignant cases/    # Malignant CT scan images
└── Normal cases/       # Normal CT scan images
```

> **Note:** The `data/` directory is git-ignored. You must supply your own dataset.

### Running

```bash
python main.py
```

This will:

1. Run exploratory data analysis (image size distribution, sample previews)
2. Load and preprocess all images (resize to 256×256, grayscale, normalise)
3. Split data into train / validation sets (stratified, 75/25)
4. Train three models (SMOTE → Class-Weighted → Augmentation + Class-Weighted)
5. Generate evaluation reports (classification report, confusion matrix)
6. Save plots and trained models to disk

## 🏗️ Architecture

```
Input (256 × 256 × 1)
  │
  ├─ Conv2D(64, 3×3, ReLU)
  ├─ MaxPool(2×2)
  │
  ├─ Conv2D(64, 3×3, ReLU)
  ├─ MaxPool(2×2)
  │
  ├─ Flatten
  ├─ Dense(16)
  └─ Dense(3, Softmax)  →  [Benign, Malignant, Normal]
```

## ⚙️ Configuration

All hyperparameters are centralised in `src/config.py`:

| Parameter             | Default | Description                        |
| --------------------- | ------- | ---------------------------------- |
| `IMG_SIZE`            | 256     | Input image dimensions             |
| `BATCH_SIZE`          | 8       | Training batch size                |
| `EPOCHS_SMOTE`        | 10      | Epochs for SMOTE approach          |
| `EPOCHS_CLASS_WEIGHT` | 10      | Epochs for class-weighted approach |
| `EPOCHS_AUGMENTATION` | 5       | Epochs for augmentation approach   |
| `TEST_SIZE`           | 0.25    | Validation split ratio             |
| `SMOTE_K_NEIGHBORS`   | 5       | SMOTE nearest-neighbour count      |

## 📊 Outputs

| Output         | Location                   | Description                   |
| -------------- | -------------------------- | ----------------------------- |
| Trained models | `models/*.keras`           | Keras native format           |
| Accuracy plots | `plots/*_accuracy.png`     | Train vs. validation accuracy |
| Loss plots     | `plots/*_loss.png`         | Train vs. validation loss     |
| Run logs       | `logs/run_<timestamp>.log` | Full console output mirror    |

## 🛠️ Tech Stack

- **TensorFlow / Keras** — CNN model definition and training
- **OpenCV** — Image loading and preprocessing
- **scikit-learn** — Train/test splitting, classification metrics
- **imbalanced-learn** — SMOTE oversampling
- **Matplotlib** — Plotting training history

## 📝 License

This project is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Model architecture and dataset: based on work originally shared by the Kaggle community
- Lung CT Scan images (Benign / Malignant / Normal)
