"""
evaluate_tft.py

Evaluate a trained Temporal Fusion Transformer (TFT) model on the BTC dataset.

Supports both:
    - TASK_TYPE = "classification"  (binary up/down via target_up)
    - TASK_TYPE = "regression"      (continuous 1-day return via target_return)

For classification:
    * Runs threshold search on the validation set to maximize a target metric
      (e.g., F1) and then evaluates the test set using that threshold.
    * Produces probability histogram, confusion matrix, and ROC curve.

For regression:
    * Computes regression metrics (MSE, RMSE, MAE, R^2).
    * Computes directional metrics based on sign(return > UP_THRESHOLD).
    * Produces predicted-return histogram, true vs predicted scatter plot,
      and a confusion matrix for the directional decisions.
"""

import os
from typing import Tuple, Dict, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config import (
    MODEL_CONFIG,
    TRAINING_CONFIG,
    TASK_TYPE,
    EVAL_THRESHOLD,
    AUTO_TUNE_THRESHOLD,
    THRESHOLD_SEARCH_MIN,
    THRESHOLD_SEARCH_MAX,
    THRESHOLD_SEARCH_STEPS,
    THRESHOLD_TARGET_METRIC,
    BEST_MODEL_PATH,
    EXPERIMENTS_DIR,
    PLOTS_DIR,
    UP_THRESHOLD,
    SEQ_LENGTH,
)

from data_pipeline import prepare_datasets
from tft_model import TemporalFusionTransformer
import utils


# ---------------------------------------------------------------------------
# Helper: dataloaders
# ---------------------------------------------------------------------------

def create_eval_dataloaders(
    val_dataset,
    test_dataset,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for validation and test sets.
    We typically don't need the training set here.
    """
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    return val_loader, test_loader


# ---------------------------------------------------------------------------
# Helper: inference loops
# ---------------------------------------------------------------------------

def _unpack_batch(batch, device: torch.device):
    """
    Handle both dataset variants:
      - (x_past, y)
      - (x_past, x_future, y)
    """
    if len(batch) == 2:
        x_past, y = batch
        x_future = None
    else:
        x_past, x_future, y = batch

    x_past = x_past.to(device)
    y = y.to(device)

    if x_future is not None:
        x_future = x_future.to(device)

    return x_past, x_future, y


def run_inference_classification(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run inference for the classification task.

    Returns:
        y_true      : array of true labels (0/1)
        y_prob      : array of predicted probabilities (sigmoid outputs)
        avg_loss    : average BCEWithLogits loss over the dataset
    """
    model.eval()
    all_true = []
    all_prob = []
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in data_loader:
            x_past, x_future, y = _unpack_batch(batch, device)

            logits = model(x_past) if x_future is None else model(x_past, x_future)
            logits = logits.view(-1)
            y_flat = y.view(-1)

            loss = criterion(logits, y_flat)
            total_loss += loss.item() * y_flat.size(0)
            total_count += y_flat.size(0)

            probs = torch.sigmoid(logits)

            all_true.append(y_flat.cpu())
            all_prob.append(probs.cpu())

    y_true = torch.cat(all_true).numpy()
    y_prob = torch.cat(all_prob).numpy()
    avg_loss = total_loss / max(1, total_count)

    return y_true, y_prob, avg_loss


def run_inference_regression(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run inference for the regression task.

    Returns:
        y_true      : array of true continuous targets (returns)
        y_pred      : array of predicted continuous values
        avg_loss    : average MSE loss over the dataset
    """
    model.eval()
    all_true = []
    all_pred = []
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in data_loader:
            x_past, x_future, y = _unpack_batch(batch, device)

            outputs = model(x_past) if x_future is None else model(x_past, x_future)
            outputs = outputs.view(-1)
            y_flat = y.view(-1)

            loss = criterion(outputs, y_flat)
            total_loss += loss.item() * y_flat.size(0)
            total_count += y_flat.size(0)

            all_true.append(y_flat.cpu())
            all_pred.append(outputs.cpu())

    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()
    avg_loss = total_loss / max(1, total_count)

    return y_true, y_pred, avg_loss


# ---------------------------------------------------------------------------
# Helper plots for regression
# ---------------------------------------------------------------------------

def plot_return_histogram(
    values: np.ndarray,
    save_path: str,
    title: str = "Predicted Returns Histogram",
    bins: int = 50,
) -> None:
    """
    Simple histogram for continuous returns.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, alpha=0.7, edgecolor="black")
    plt.title(title)
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_true_vs_pred_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    title: str = "True vs Predicted Returns",
) -> None:
    """
    Scatter plot of true vs predicted returns.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.4, s=10)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="y = x")
    plt.title(title)
    plt.xlabel("True Return")
    plt.ylabel("Predicted Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def main() -> None:
    print("[evaluate_tft] Using device:", end=" ")
    device = utils.get_device()
    print(device)

    # ------------------------------------------------------------------
    # 1. Prepare datasets & loaders
    # ------------------------------------------------------------------
    print("[evaluate_tft] Preparing datasets.")

    # Use global SEQ_LENGTH from config (Option A)
    train_dataset, val_dataset, test_dataset, _ = prepare_datasets(
        seq_length=SEQ_LENGTH
    )

    print(f"[evaluate_tft] Train samples: {len(train_dataset)}")
    print(f"[evaluate_tft] Val samples:   {len(val_dataset)}")
    print(f"[evaluate_tft] Test samples:  {len(test_dataset)}")

    val_loader, test_loader = create_eval_dataloaders(
        val_dataset,
        test_dataset,
        batch_size=TRAINING_CONFIG.batch_size,
    )

    # ------------------------------------------------------------------
    # 2. Build model & load weights
    # ------------------------------------------------------------------
    print(f"[evaluate_tft] Loading model from {BEST_MODEL_PATH} .")

    # Must match train_tft: model = TemporalFusionTransformer(MODEL_CONFIG)
    model = TemporalFusionTransformer(MODEL_CONFIG).to(device)

    state_dict = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)

    # Timestamp for this evaluation
    eval_id = f"tft_eval_{utils.get_timestamp()}"
    utils.ensure_dir(EXPERIMENTS_DIR)
    utils.ensure_dir(PLOTS_DIR)

    # ------------------------------------------------------------------
    # 3. Branch by TASK_TYPE
    # ------------------------------------------------------------------
    if TASK_TYPE == "classification":
        # -------------------- Classification evaluation ----------------
        print("[evaluate_tft] TASK_TYPE='classification' -> binary up/down.")
        criterion = nn.BCEWithLogitsLoss()

        print("[evaluate_tft] Running inference on validation set for threshold tuning.")
        y_val_true, y_val_prob, val_loss = run_inference_classification(
            model, val_loader, device, criterion
        )

        if AUTO_TUNE_THRESHOLD:
            thresholds = np.linspace(
                THRESHOLD_SEARCH_MIN,
                THRESHOLD_SEARCH_MAX,
                THRESHOLD_SEARCH_STEPS,
            )

            best_threshold = None
            best_metric_value = -np.inf
            best_metrics: Dict[str, float] | None = None

            for th in thresholds:
                metrics = utils.compute_classification_metrics(
                    y_true=y_val_true,
                    y_prob=y_val_prob,
                    threshold=th,
                )
                metric_value = metrics.get(THRESHOLD_TARGET_METRIC, metrics.get("f1", 0.0))

                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_threshold = th
                    best_metrics = metrics

            threshold = best_threshold
            print("[evaluate_tft] Validation threshold search (optimize "
                  f"{THRESHOLD_TARGET_METRIC}):")
            print(f"  Best threshold on val: {threshold:.3f}")
            print(f"  Val loss:              {val_loss:.4f}")
            assert best_metrics is not None
            for k, v in best_metrics.items():
                print(f"  Val {k}: {v:.4f}")
        else:
            threshold = EVAL_THRESHOLD
            best_metrics = utils.compute_classification_metrics(
                y_true=y_val_true,
                y_prob=y_val_prob,
                threshold=threshold,
            )
            print("[evaluate_tft] Using fixed eval threshold on val:")
            print(f"  Threshold: {threshold:.3f}")
            print(f"  Val loss:  {val_loss:.4f}")
            for k, v in best_metrics.items():
                print(f"  Val {k}: {v:.4f}")

        # Validation probability histogram
        val_hist_path = os.path.join(PLOTS_DIR, f"{eval_id}_val_prob_hist.png")
        utils.plot_probability_histogram(
            y_prob=y_val_prob,
            out_path=val_hist_path,
            threshold=threshold,
            title="Validation Probability Histogram",
        )
        print(f"[evaluate_tft] Validation probability histogram saved to {val_hist_path}")

        # -------------------- Test set evaluation ----------------------
        print("[evaluate_tft] Running inference on test set with tuned threshold.")
        y_test_true, y_test_prob, test_loss = run_inference_classification(
            model, test_loader, device, criterion
        )

        test_metrics = utils.compute_classification_metrics(
            y_true=y_test_true,
            y_prob=y_test_prob,
            threshold=threshold,
        )

        print("[evaluate_tft] Test metrics (using tuned threshold):")
        print(f"  Test loss:      {test_loss:.4f}")
        for k, v in test_metrics.items():
            print(f"  {k.capitalize()}: {v:.4f}")
        print(f"  Threshold used: {threshold:.3f}")

        # Test probability histogram
        test_hist_path = os.path.join(PLOTS_DIR, f"{eval_id}_test_prob_hist.png")
        utils.plot_probability_histogram(
            y_prob=y_test_prob,
            out_path=test_hist_path,
            threshold=threshold,
            title="Test Probability Histogram",
        )
        print(f"[evaluate_tft] Test probability histogram saved to {test_hist_path}")

        # Confusion matrix
        y_test_pred = (y_test_prob >= threshold).astype(int)
        cm_path = os.path.join(PLOTS_DIR, f"{eval_id}_confusion_matrix.png")
        utils.plot_confusion_matrix(
            y_true=y_test_true,
            y_pred=y_test_pred,
            out_path=cm_path,
            title="Test Confusion Matrix",
        )
        print(f"[evaluate_tft] Confusion matrix saved to {cm_path}")

        # ROC curve
        roc_path = os.path.join(PLOTS_DIR, f"{eval_id}_roc_curve.png")
        utils.plot_roc_curve(
            y_true=y_test_true,
            y_score=y_test_prob,
            out_path=roc_path,
            title="Test ROC Curve",
        )
        print(f"[evaluate_tft] ROC curve saved to {roc_path}")

        # Save metrics to JSON
        metrics_out: Dict[str, Any] = {
            "task_type": TASK_TYPE,
            "val_loss": float(val_loss),
            "test_loss": float(test_loss),
            "threshold_used": float(threshold),
            "val_metrics": best_metrics,
            "test_metrics": test_metrics,
        }

    else:
        # ------------------------ Regression evaluation ----------------
        print("[evaluate_tft] TASK_TYPE='regression' -> continuous return.")
        criterion = nn.MSELoss()

        # --- Validation (no threshold tuning, just metrics) ---
        print("[evaluate_tft] Running inference on validation set (regression).")
        y_val_true, y_val_pred, val_loss = run_inference_regression(
            model, val_loader, device, criterion
        )

        val_reg_metrics = utils.compute_regression_metrics(
            y_true=y_val_true,
            y_pred=y_val_pred,
            direction_threshold=UP_THRESHOLD,
        )

        print("[evaluate_tft] Validation regression metrics:")
        print(f"  Val loss (MSE): {val_loss:.6f}")
        for k, v in val_reg_metrics.items():
            print(f"  Val {k}: {v:.6f}")

        # --- Test set ---
        print("[evaluate_tft] Running inference on test set (regression).")
        y_test_true, y_test_pred, test_loss = run_inference_regression(
            model, test_loader, device, criterion
        )

        test_reg_metrics = utils.compute_regression_metrics(
            y_true=y_test_true,
            y_pred=y_test_pred,
            direction_threshold=UP_THRESHOLD,
        )

        print("[evaluate_tft] Test regression metrics:")
        print(f"  Test loss (MSE): {test_loss:.6f}")
        for k, v in test_reg_metrics.items():
            print(f"  Test {k}: {v:.6f}")

        # --- Plots specific to regression ---

        # Histogram of predicted returns
        test_hist_path = os.path.join(PLOTS_DIR, f"{eval_id}_test_pred_return_hist.png")
        plot_return_histogram(
            values=y_test_pred,
            save_path=test_hist_path,
            title="Test Predicted 1-Day Returns Histogram",
        )
        print(f"[evaluate_tft] Test predicted-return histogram saved to {test_hist_path}")

        # True vs predicted scatter
        scatter_path = os.path.join(PLOTS_DIR, f"{eval_id}_test_true_vs_pred.png")
        plot_true_vs_pred_scatter(
            y_true=y_test_true,
            y_pred=y_test_pred,
            save_path=scatter_path,
            title="Test True vs Predicted Returns",
        )
        print(f"[evaluate_tft] True vs predicted scatter saved to {scatter_path}")

        # Directional confusion matrix (up/down based on UP_THRESHOLD)
        y_test_true_dir = (y_test_true > UP_THRESHOLD).astype(int)
        y_test_pred_dir = (y_test_pred > UP_THRESHOLD).astype(int)
        cm_path = os.path.join(PLOTS_DIR, f"{eval_id}_direction_confusion_matrix.png")
        utils.plot_confusion_matrix(
            y_true=y_test_true_dir,
            y_pred=y_test_pred_dir,
            out_path=cm_path,
            title=f"Test Direction Confusion (threshold={UP_THRESHOLD:.4f})",
        )
        print(f"[evaluate_tft] Directional confusion matrix saved to {cm_path}")

        # Save metrics to JSON
        metrics_out = {
            "task_type": TASK_TYPE,
            "up_threshold": float(UP_THRESHOLD),
            "val_loss_mse": float(val_loss),
            "test_loss_mse": float(test_loss),
            "val_metrics": val_reg_metrics,
            "test_metrics": test_reg_metrics,
        }

    # ------------------------------------------------------------------
    # 4. Persist metrics JSON
    # ------------------------------------------------------------------
    metrics_path = os.path.join(EXPERIMENTS_DIR, f"{eval_id}_test_metrics.json")
    utils.save_json(metrics_out, metrics_path)
    print(f"[evaluate_tft] Test metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()