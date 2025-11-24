"""
utils.py

Utility functions for the TFT BTC up/down project.

This module provides:
- Reproducibility helpers (set_seed)
- Device selection (get_device)
- Basic classification metrics computation for training & evaluation
- Plotting helpers for training curves
"""

from __future__ import annotations

import os
import random
from typing import Dict, Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import matplotlib.pyplot as plt


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy and PyTorch to improve reproducibility.

    Args:
        seed: Integer seed value to use across libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # If CUDA is available, seed all GPUs as well
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # These flags can make CUDA operations more deterministic,
    # at the cost of some performance. You can comment them out
    # if you don't care about strict determinism.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Get the default computation device (GPU if available, else CPU).

    Returns:
        torch.device("cuda") if a GPU is available, otherwise torch.device("cpu").
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_numpy_1d(x: Any) -> np.ndarray:
    """
    Helper to convert a variety of input types to a 1D NumPy array.

    Accepts:
        - NumPy arrays
        - Python lists/tuples
        - PyTorch tensors

    Returns:
        1D NumPy array of shape (N,).
    """
    if isinstance(x, np.ndarray):
        arr = x
    elif torch.is_tensor(x):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)

    # Flatten to 1D if necessary
    if arr.ndim > 1:
        arr = arr.reshape(-1)

    return arr


def compute_classification_metrics(
    y_true: Any,
    y_prob: Any,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute basic binary classification metrics from true labels and predicted probabilities.

    Args:
        y_true:
            Ground-truth labels. Can be:
                - 1D NumPy array
                - PyTorch tensor
                - list / tuple
            Values should be 0 or 1.
        y_prob:
            Predicted probabilities for the positive class (UP), in [0, 1].
            Can have the same types as y_true.
        threshold:
            Decision threshold for converting probabilities to class predictions.
            If y_prob >= threshold -> predict 1, else 0.

    Returns:
        Dictionary with:
            - "accuracy"
            - "precision"
            - "recall"
            - "f1"
            - "auc" (ROC AUC; NaN if it cannot be computed)
    """
    y_true_arr = _to_numpy_1d(y_true)
    y_prob_arr = _to_numpy_1d(y_prob)

    if y_true_arr.shape != y_prob_arr.shape:
        raise ValueError(
            f"Shapes of y_true {y_true_arr.shape} and y_prob {y_prob_arr.shape} do not match."
        )

    # Clip probabilities to a valid range just in case
    y_prob_arr = np.clip(y_prob_arr, 0.0, 1.0)

    # Binary predictions from threshold
    y_pred_arr = (y_prob_arr >= threshold).astype(int)

    # Basic metrics
    accuracy = accuracy_score(y_true_arr, y_pred_arr)
    precision = precision_score(y_true_arr, y_pred_arr, zero_division=0)
    recall = recall_score(y_true_arr, y_pred_arr, zero_division=0)
    f1 = f1_score(y_true_arr, y_pred_arr, zero_division=0)

    # ROC AUC (may fail if only one class is present)
    try:
        auc = roc_auc_score(y_true_arr, y_prob_arr)
    except ValueError:
        # e.g. if y_true is all 0s or all 1s
        auc = float("nan")

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
    }


def plot_training_curves(history: Dict[str, list], out_path: str) -> None:
    """
    Plot training & validation loss and F1-score curves over epochs.

    Expects a history dict with (at least):
        - "epoch"          : list of epoch indices (1-based or 0-based)
        - "train_loss"
        - "val_loss"
        - "train_f1"
        - "val_f1"

    Args:
        history:
            Dictionary with lists of per-epoch metrics.
        out_path:
            File path (PNG) where the plot will be saved.
    """
    # Use "epoch" if provided, otherwise infer from train_loss length
    if "epoch" in history and len(history["epoch"]) > 0:
        epochs = history["epoch"]
    else:
        n = len(history.get("train_loss", []))
        epochs = list(range(1, n + 1))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # --- Loss curves ---
    if "train_loss" in history and "val_loss" in history:
        ax1.plot(epochs, history["train_loss"], label="Train loss")
        ax1.plot(epochs, history["val_loss"], label="Val loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("BCE loss")
        ax1.set_title("Loss over epochs")
        ax1.legend()
    else:
        ax1.set_visible(False)

    # --- F1-score curves ---
    if "train_f1" in history and "val_f1" in history:
        ax2.plot(epochs, history["train_f1"], label="Train F1")
        ax2.plot(epochs, history["val_f1"], label="Val F1")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("F1-score")
        ax2.set_title("F1 over epochs")
        ax2.legend()
    else:
        ax2.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)