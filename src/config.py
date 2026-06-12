"""
Configuration constants for the Lung Cancer Prediction project.

Centralises all paths, hyper-parameters, and model settings in a single
location so that experiments are fully reproducible from one file.
"""

from __future__ import annotations

from pathlib import Path

# ── Project Root ──────────────────────────────────────────────────────────────
# Paths relative to the project root (where main.py lives).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = PROJECT_ROOT / "models"
PLOTS_DIR: Path = PROJECT_ROOT / "plots"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

# ── Dataset Paths ─────────────────────────────────────────────────────────────
DATASET_DIR: Path = PROJECT_ROOT / "data"

CATEGORIES: list[str] = ["Bengin cases", "Malignant cases", "Normal cases"]
NUM_CLASSES: int = len(CATEGORIES)  # 3

# ── Image Settings ────────────────────────────────────────────────────────────
IMG_SIZE: int = 256
IMG_CHANNELS: int = 1  # Grayscale

# ── Train / Validation Split ──────────────────────────────────────────────────
TEST_SIZE: float = 0.25
RANDOM_STATE: int = 10

# ── Model Training ────────────────────────────────────────────────────────────
BATCH_SIZE: int = 8
EPOCHS_SMOTE: int = 10
EPOCHS_CLASS_WEIGHT: int = 10
EPOCHS_AUGMENTATION: int = 5

# ── CNN Architecture ──────────────────────────────────────────────────────────
CONV1_FILTERS: int = 64
CONV1_KERNEL: tuple[int, int] = (3, 3)
CONV2_FILTERS: int = 64
CONV2_KERNEL: tuple[int, int] = (3, 3)
POOL_SIZE: tuple[int, int] = (2, 2)
DENSE_UNITS: int = 16

# ── SMOTE ─────────────────────────────────────────────────────────────────────
SMOTE_K_NEIGHBORS: int = 5

# ── Data Augmentation ─────────────────────────────────────────────────────────
AUG_HORIZONTAL_FLIP: bool = True
AUG_VERTICAL_FLIP: bool = True