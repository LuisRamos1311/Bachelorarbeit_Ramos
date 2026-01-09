"""
utils.py

Utility functions for the TFT BTC forecasting project.

This module provides:
- Reproducibility helpers (set_seed)
- Device selection (get_device)
- Basic classification AND regression metrics
- Plotting helpers for training curves, confusion matrix, ROC, and score histograms
"""

from __future__ import annotations

import os
import random
import time
import json
import matplotlib.patheffects as pe
from typing import Any, Dict, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    r2_score,
)
import matplotlib.pyplot as plt


# ============================================================
# 1. SEED CONTROL & DEVICE / PATH HELPERS
# ============================================================

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make CuDNN deterministic (may slightly reduce speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Pick CPU or GPU device.

    Args:
        prefer_gpu: If True, return CUDA device if available.

    Returns:
        torch.device
    """
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_timestamp() -> str:
    """
    Return a compact timestamp string for filenames/logs.

    Format: YYYYMMDD_HHMMSS (e.g. 20251130_153045)
    """
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    """
    Create directory `path` if it doesn't already exist.

    Safe to call multiple times.
    """
    os.makedirs(path, exist_ok=True)


# ============================================================
# 2. NUMPY CONVERSION HELPER
# ============================================================

def _to_numpy_1d(x: Any) -> np.ndarray:
    """
    Convert a torch.Tensor / np.ndarray / list to a 1D NumPy array.

    Args:
        x: Input array-like or tensor.

    Returns:
        1D np.ndarray
    """
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        arr = x
    else:
        arr = np.asarray(x)

    return arr.reshape(-1)


# ============================================================
# 3. CLASSIFICATION METRICS (for 0/1 targets + probabilities)
# ============================================================

def compute_classification_metrics(
    y_true: Any,
    y_prob: Any,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute standard binary classification metrics.

    Args:
        y_true:
            Ground-truth labels (0/1). Can be torch tensors or arrays.
        y_prob:
            Model outputs interpreted as probabilities or scores for the
            positive class. Values will be clipped to [0, 1] for safety.
        threshold:
            Decision threshold on y_prob to binarize predictions.

    Returns:
        Dictionary with accuracy, precision, recall, F1, and ROC AUC.
    """
    y_true_arr = _to_numpy_1d(y_true)
    y_prob_arr = _to_numpy_1d(y_prob)

    if y_true_arr.shape != y_prob_arr.shape:
        raise ValueError(
            f"y_true and y_prob must have the same shape, "
            f"got {y_true_arr.shape} vs {y_prob_arr.shape}"
        )

    # Clip to [0, 1] for numerical stability
    y_prob_clipped = np.clip(y_prob_arr, 0.0, 1.0)

    # Hard predictions at the given threshold
    y_pred = (y_prob_clipped >= threshold).astype(int)

    acc = accuracy_score(y_true_arr, y_pred)
    prec = precision_score(y_true_arr, y_pred, zero_division=0)
    rec = recall_score(y_true_arr, y_pred, zero_division=0)
    f1 = f1_score(y_true_arr, y_pred, zero_division=0)

    # ROC AUC: requires both classes present; handle degenerate case
    try:
        auc = roc_auc_score(y_true_arr, y_prob_clipped)
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc),
    }


# ============================================================
# 4. REGRESSION METRICS (for continuous returns)
# ============================================================

def compute_regression_metrics(
    y_true: Any,
    y_pred: Any,
    direction_threshold: float = 0.0,
) -> Dict[str, float]:
    """
    Compute regression metrics for continuous targets (e.g., returns),
    plus directional accuracy/F1 based on the sign of predictions.

    Args:
        y_true:
            Ground-truth continuous values (e.g., 1-day forward returns).
        y_pred:
            Predicted continuous values (same shape as y_true).
        direction_threshold:
            Threshold used to classify a value as "UP" (1) vs "DOWN" (0).
            Typically 0.0: positive return => UP, else DOWN.

    Returns:
        Dictionary with mse, mae, rmse, r2, direction_accuracy, direction_f1.
    """
    y_true_arr = _to_numpy_1d(y_true).astype(float)
    y_pred_arr = _to_numpy_1d(y_pred).astype(float)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, "
            f"got {y_true_arr.shape} vs {y_pred_arr.shape}"
        )

    diff = y_true_arr - y_pred_arr
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))

    # R^2 can fail if y_true has zero variance; handle gracefully
    try:
        r2 = float(r2_score(y_true_arr, y_pred_arr))
    except ValueError:
        r2 = float("nan")

    # Directional metrics: compare sign relative to threshold
    y_true_dir = (y_true_arr > direction_threshold).astype(int)
    y_pred_dir = (y_pred_arr > direction_threshold).astype(int)

    direction_acc = float(accuracy_score(y_true_dir, y_pred_dir))
    direction_f1 = float(f1_score(y_true_dir, y_pred_dir, zero_division=0))

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "direction_accuracy": direction_acc,
        "direction_f1": direction_f1,
    }


# ============================================================
# 5. TRAINING CURVE PLOTTING
# ============================================================

def plot_training_curves(history: Dict[str, Sequence[float]], out_path: str) -> None:
    """
    Plot training/validation curves from a history dict.

    The function is flexible: it will always plot train/val loss if present,
    and then try to plot ONE additional metric pair, in this priority order:
        - F1-score     (train_f1 / val_f1)
        - RMSE         (train_rmse / val_rmse)
        - MAE          (train_mae / val_mae)
        - Direction Acc (train_direction_accuracy / val_direction_accuracy)
        - R^2          (train_r2 / val_r2)

    Args:
        history:
            Dict with lists keyed by metric names, e.g.
            "epoch", "train_loss", "val_loss", "train_f1", "val_f1", etc.
        out_path:
            File path where the PNG figure should be saved.
    """
    if not history:
        raise ValueError("History is empty – nothing to plot.")

    epochs = history.get("epoch")
    if not epochs:
        # Fallback: infer number of epochs from train_loss length
        n_epochs = len(history.get("train_loss", []))
        epochs = list(range(1, n_epochs + 1))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax1, ax2 = axes

    # ---- Loss subplot ----
    if "train_loss" in history and "val_loss" in history:
        ax1.plot(epochs, history["train_loss"], label="Train loss")
        ax1.plot(epochs, history["val_loss"], label="Val loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss over epochs")
        ax1.legend()
    else:
        ax1.set_visible(False)

    # ---- Second subplot: choose best metric to show ----
    metric_candidates = [
        ("train_f1", "val_f1", "F1-score"),
        ("train_rmse", "val_rmse", "RMSE"),
        ("train_mae", "val_mae", "MAE"),
        ("train_direction_accuracy", "val_direction_accuracy", "Direction accuracy"),
        ("train_r2", "val_r2", "R²"),
    ]

    plotted = False
    for train_key, val_key, label in metric_candidates:
        if train_key in history and val_key in history:
            if len(history[train_key]) == 0 or len(history[val_key]) == 0:
                continue
            ax2.plot(epochs, history[train_key], label=f"Train {label}")
            ax2.plot(epochs, history[val_key], label=f"Val {label}")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel(label)
            ax2.set_title(f"{label} over epochs")
            ax2.legend()
            plotted = True
            break

    if not plotted:
        ax2.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ============================================================
# 6. CONFUSION MATRIX & ROC CURVE PLOTTING
# ============================================================

def plot_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    out_path: str,
    class_names: Sequence[str] = ("DOWN", "UP"),
    title: str = "Confusion matrix",
) -> None:
    """
    Plot a 2x2 confusion matrix for binary classification.

    Args:
        y_true:
            Ground-truth labels (0/1).
        y_pred:
            Predicted labels (0/1).
        out_path:
            Output PNG path.
        class_names:
            Names for the classes [negative, positive].
        title:
            Title for the plot.
    """
    y_true_arr = _to_numpy_1d(y_true)
    y_pred_arr = _to_numpy_1d(y_pred)

    cm = confusion_matrix(y_true_arr, y_pred_arr)
    if cm.shape != (2, 2):
        raise ValueError(
            f"Expected a 2x2 confusion matrix for binary classification, "
            f"got shape {cm.shape}."
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    # Write counts into cells
    # Write counts into cells
    fmt = "d"

    # Use the colormap's normalization to decide text color:
    # norm(x) ~ 0 -> very light cell, norm(x) ~ 1 -> very dark cell
    norm = im.norm

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            brightness = norm(value)

            # If the cell is light, use black text; if dark, use white text
            text_color = "black" if brightness < 0.5 else "white"

            txt = ax.text(
                j,
                i,
                format(value, fmt),
                ha="center",
                va="center",
                color=text_color,
                fontsize=12,
                fontweight="bold",
            )

            # Optional: add a thin outline in the opposite color for extra contrast
            outline_color = "white" if text_color == "black" else "black"
            txt.set_path_effects(
                [pe.withStroke(linewidth=1.5, foreground=outline_color)]
            )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_roc_curve(
    y_true: Any,
    y_score: Any,
    out_path: str,
    title: str = "ROC curve",
) -> None:
    """
    Plot ROC curve for binary classification or directional evaluation.

    Args:
        y_true:
            Ground-truth binary labels (0/1).
        y_score:
            Scores for the positive class. These can be probabilities in [0, 1]
            or any real-valued scores where larger means "more likely UP".
        out_path:
            Output PNG path.
        title:
            Title for the plot.
    """
    y_true_arr = _to_numpy_1d(y_true)
    y_score_arr = _to_numpy_1d(y_score)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    try:
        fpr, tpr, _ = roc_curve(y_true_arr, y_score_arr)
        auc = roc_auc_score(y_true_arr, y_score_arr)
    except ValueError:
        # Degenerate case: e.g. only one class present
        fpr, tpr, auc = np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random baseline")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ============================================================
# 7. SCORE / PROBABILITY HISTOGRAM
# ============================================================

def plot_probability_histogram(
    y_prob: Any,
    out_path: str,
    threshold: float | None = None,
    title: str = "Predicted scores",
    bins: int = 50,
) -> None:
    """
    Plot a histogram of model scores/probabilities.

    Args:
        y_prob:
            Probabilities or scores (1D array-like).
        out_path:
            Output PNG path.
        threshold:
            Optional vertical line to draw (e.g., classification threshold).
        title:
            Plot title.
        bins:
            Number of histogram bins.
    """
    scores = _to_numpy_1d(y_prob)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scores, bins=bins, alpha=0.7, color="steelblue", edgecolor="black")
    ax.set_xlabel("Score / probability")
    ax.set_ylabel("Count")
    ax.set_title(title)

    if threshold is not None:
        ax.axvline(threshold, color="red", linestyle="--", label=f"threshold={threshold:.3f}")
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_json(data: Dict[str, Any], path: str) -> None:
    """
    Save a dictionary as a pretty-printed JSON file.

    Args:
        data:
            Dictionary of metrics or configuration to serialize.
        path:
            File path where the JSON should be written.
    """
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Convert any NumPy types to Python scalars for JSON compatibility
    def default(o):
        if isinstance(o, (np.generic,)):
            return o.item()
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=default)