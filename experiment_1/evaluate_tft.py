"""
evaluate_tft.py

Evaluate a trained Temporal Fusion Transformer (TFT) model on the BTC dataset.

Supports both:
    - TASK_TYPE = "classification"  (binary up/down via target_up)
    - TASK_TYPE = "regression"      (multi-horizon continuous returns)

For classification:
    * Runs threshold search on the validation set to maximize a target metric
      (e.g., F1) and then evaluates the test set using that threshold.
    * Produces probability histogram, confusion matrix, and ROC curve.

For regression (multi-horizon):
    * Model outputs a vector of returns for FORECAST_HORIZONS
      (e.g. [r_{t+1}, r_{t+3}, r_{t+7}]).
    * Computes aggregate regression metrics over all horizons.
    * Computes per-horizon metrics (RMSE, MAE, R^2, directional accuracy/F1).
    * Uses the 1-day horizon for UP/DOWN directional confusion matrix and ROC.
    * Produces a histogram of predicted 1-day returns and multi-panel
      true vs predicted scatter plots per horizon.
"""

import os
from typing import Tuple, Dict, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from experiment_1.config import (
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
    FORECAST_HORIZONS,
)

from experiment_1.data_pipeline import prepare_datasets
from experiment_1.tft_model import TemporalFusionTransformer
from experiment_1 import utils


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
            logits_flat = logits.view(-1)
            y_flat = y.view(-1)

            loss = criterion(logits_flat, y_flat)
            total_loss += loss.item() * y_flat.size(0)
            total_count += y_flat.size(0)

            probs = torch.sigmoid(logits_flat)

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
    Run inference for the regression task (multi-horizon).

    Returns:
        y_true : shape (N, H)   true continuous targets (returns)
        y_pred : shape (N, H)   predicted continuous values
        avg_loss : scalar       average MSE loss over samples
    """
    model.eval()
    all_true: list[torch.Tensor] = []
    all_pred: list[torch.Tensor] = []
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in data_loader:
            x_past, x_future, y = _unpack_batch(batch, device)

            outputs = model(x_past) if x_future is None else model(x_past, x_future)
            # outputs: (B, H), y: (B, H)
            loss = criterion(outputs, y)

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

            all_true.append(y.cpu())
            all_pred.append(outputs.cpu())

    if total_count == 0:
        raise RuntimeError("DataLoader has zero samples in run_inference_regression().")

    y_true_all = torch.cat(all_true, dim=0).numpy()  # (N, H)
    y_pred_all = torch.cat(all_pred, dim=0).numpy()  # (N, H)
    avg_loss = total_loss / total_count

    return y_true_all, y_pred_all, avg_loss


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
    values = np.asarray(values).reshape(-1)
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, alpha=0.7, edgecolor="black")
    plt.title(title)
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_multi_horizon_true_vs_pred_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: list[int],
    save_path: str,
    title_prefix: str = "True vs Predicted Returns",
    rmse_per_horizon: list[float] | None = None,
) -> None:
    """
    Multi-panel scatter: one subplot per horizon.

    Args:
        y_true:  shape (N, H)
        y_pred:  shape (N, H)
        horizons: list of horizon lengths (must match H)
        save_path: output PNG path
        title_prefix: base title string
        rmse_per_horizon: optional list of RMSE values to show in titles
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"y_true and y_pred must have same shape, "
            f"got {y_true_arr.shape} vs {y_pred_arr.shape}"
        )

    if y_true_arr.ndim != 2:
        raise ValueError(
            f"Expected 2D arrays (N, H) for multi-horizon scatter, "
            f"got ndim={y_true_arr.ndim}"
        )

    n_samples, n_horizons = y_true_arr.shape
    if n_horizons != len(horizons):
        raise ValueError(
            f"Number of horizons ({len(horizons)}) does not match "
            f"y_true/y_pred second dim ({n_horizons})."
        )

    fig, axes = plt.subplots(
        1, n_horizons, figsize=(5 * n_horizons, 5), squeeze=False
    )

    for i, h in enumerate(horizons):
        ax = axes[0, i]
        yt = y_true_arr[:, i]
        yp = y_pred_arr[:, i]

        ax.scatter(yt, yp, alpha=0.4, s=10)

        min_val = min(np.min(yt), np.min(yp))
        max_val = max(np.max(yt), np.max(yp))
        ax.plot([min_val, max_val], [min_val, max_val], "r--", label="y = x")

        title = f"{title_prefix} (h={h}d)"
        if rmse_per_horizon is not None and i < len(rmse_per_horizon):
            title += f"\nRMSE={rmse_per_horizon[i]:.4f}"

        ax.set_title(title)
        ax.set_xlabel("True Return")
        if i == 0:
            ax.set_ylabel("Predicted Return")
        ax.grid(True, alpha=0.3)

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

    # Use global SEQ_LENGTH from config
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
                metric_value = metrics.get(
                    THRESHOLD_TARGET_METRIC, metrics.get("f1", 0.0)
                )

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
        print("[evaluate_tft] TASK_TYPE='regression' -> multi-horizon continuous returns.")
        criterion = nn.MSELoss()

        # --- Validation (no threshold tuning, just metrics) ---
        print("[evaluate_tft] Running inference on validation set (regression).")
        y_val_true, y_val_pred, val_loss = run_inference_regression(
            model, val_loader, device, criterion
        )
        # y_val_true, y_val_pred: (N_val, H)

        # Aggregate metrics over all horizons
        val_agg_metrics = utils.compute_regression_metrics(
            y_true=y_val_true,
            y_pred=y_val_pred,
            direction_threshold=UP_THRESHOLD,
        )

        # Per-horizon metrics
        val_per_horizon: Dict[int, Dict[str, float]] = {}
        for i, h in enumerate(FORECAST_HORIZONS):
            m = utils.compute_regression_metrics(
                y_true=y_val_true[:, i],
                y_pred=y_val_pred[:, i],
                direction_threshold=UP_THRESHOLD,
            )
            val_per_horizon[h] = m

        print("[evaluate_tft] Validation regression metrics (aggregate over all horizons):")
        print(f"  Val loss (MSE): {val_loss:.6f}")
        for k, v in val_agg_metrics.items():
            print(f"  Val {k}: {v:.6f}")

        print("[evaluate_tft] Validation regression metrics per horizon:")
        for h in FORECAST_HORIZONS:
            m = val_per_horizon[h]
            print(f"  Horizon {h}d:")
            for k, v in m.items():
                print(f"    {k}: {v:.6f}")

        # --- Test set ---
        print("[evaluate_tft] Running inference on test set (regression).")
        y_test_true, y_test_pred, test_loss = run_inference_regression(
            model, test_loader, device, criterion
        )
        # y_test_true, y_test_pred: (N_test, H)

        test_agg_metrics = utils.compute_regression_metrics(
            y_true=y_test_true,
            y_pred=y_test_pred,
            direction_threshold=UP_THRESHOLD,
        )

        test_per_horizon: Dict[int, Dict[str, float]] = {}
        for i, h in enumerate(FORECAST_HORIZONS):
            m = utils.compute_regression_metrics(
                y_true=y_test_true[:, i],
                y_pred=y_test_pred[:, i],
                direction_threshold=UP_THRESHOLD,
            )
            test_per_horizon[h] = m

        print("[evaluate_tft] Test regression metrics (aggregate over all horizons):")
        print(f"  Test loss (MSE): {test_loss:.6f}")
        for k, v in test_agg_metrics.items():
            print(f"  Test {k}: {v:.6f}")

        print("[evaluate_tft] Test regression metrics per horizon:")
        for h in FORECAST_HORIZONS:
            m = test_per_horizon[h]
            print(f"  Horizon {h}d:")
            for k, v in m.items():
                print(f"    {k}: {v:.6f}")

        # --- Plots specific to regression ---

        # 1) Histogram of predicted 1-day returns (first horizon)
        test_hist_path = os.path.join(
            PLOTS_DIR, f"{eval_id}_test_pred_return_hist_1d.png"
        )
        plot_return_histogram(
            values=y_test_pred[:, 0],  # 1-day horizon
            save_path=test_hist_path,
            title="Test Predicted 1-Day Returns Histogram",
        )
        print(
            f"[evaluate_tft] Test predicted 1-day return histogram saved to {test_hist_path}"
        )

        # 2) True vs predicted scatter per horizon
        rmse_list = [test_per_horizon[h]["rmse"] for h in FORECAST_HORIZONS]
        scatter_path = os.path.join(
            PLOTS_DIR, f"{eval_id}_test_true_vs_pred_multi_horizon.png"
        )
        plot_multi_horizon_true_vs_pred_scatter(
            y_true=y_test_true,
            y_pred=y_test_pred,
            horizons=FORECAST_HORIZONS,
            save_path=scatter_path,
            title_prefix="Test True vs Predicted Returns",
            rmse_per_horizon=rmse_list,
        )
        print(
            f"[evaluate_tft] Multi-horizon true vs predicted scatter saved to {scatter_path}"
        )

        # 3) Directional confusion matrix & ROC using 1-day horizon
        y_test_true_1d = y_test_true[:, 0]
        y_test_pred_1d = y_test_pred[:, 0]

        y_test_true_dir = (y_test_true_1d > UP_THRESHOLD).astype(int)
        y_test_pred_dir = (y_test_pred_1d > UP_THRESHOLD).astype(int)

        cm_dir_path = os.path.join(
            PLOTS_DIR, f"{eval_id}_direction_confusion_matrix_1d.png"
        )
        utils.plot_confusion_matrix(
            y_true=y_test_true_dir,
            y_pred=y_test_pred_dir,
            out_path=cm_dir_path,
            title=f"Test Direction Confusion",
        )
        print(
            f"[evaluate_tft] Direction (1-day) confusion matrix saved to {cm_dir_path}"
        )

        # Optional: ROC curve for directional decision (using continuous 1d return as score)
        roc_dir_path = os.path.join(
            PLOTS_DIR, f"{eval_id}_direction_roc_curve_1d.png"
        )
        utils.plot_roc_curve(
            y_true=y_test_true_dir,
            y_score=y_test_pred_1d,
            out_path=roc_dir_path,
            title="Test Direction ROC (1-day horizon)",
        )
        print(
            f"[evaluate_tft] Direction (1-day) ROC curve saved to {roc_dir_path}"
        )

        # Pack metrics into JSON-friendly structure
        metrics_out: Dict[str, Any] = {
            "task_type": TASK_TYPE,
            "up_threshold": float(UP_THRESHOLD),
            "forecast_horizons": list(FORECAST_HORIZONS),
            "val": {
                "avg_loss_mse": float(val_loss),
                "aggregate": val_agg_metrics,
                "per_horizon": {
                    str(h): val_per_horizon[h] for h in FORECAST_HORIZONS
                },
            },
            "test": {
                "avg_loss_mse": float(test_loss),
                "aggregate": test_agg_metrics,
                "per_horizon": {
                    str(h): test_per_horizon[h] for h in FORECAST_HORIZONS
                },
            },
        }

    # ------------------------------------------------------------------
    # 4. Save evaluation metrics to JSON
    # ------------------------------------------------------------------
    metrics_path = os.path.join(EXPERIMENTS_DIR, f"{eval_id}_metrics.json")
    utils.save_json(metrics_out, metrics_path)
    print(f"[evaluate_tft] Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()