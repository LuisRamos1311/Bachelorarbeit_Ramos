"""
evaluate_tft.py

Evaluate a trained Temporal Fusion Transformer (TFT) model on the BTC dataset.

Experiment 3 setup:
    - TASK_TYPE = "classification"  (3-class DOWN / FLAT / UP via direction_3c)
    - 1-day-ahead prediction (single-horizon), but the code structure
      is compatible with later multi-horizon extensions.

Behaviour:
    * Runs threshold search on the validation set to maximize a target metric
      (e.g., F1) for UP-vs-REST and then evaluates the test set using that threshold.
    * Produces probability histograms, confusion matrices, and ROC curve.
    * Saves all metrics into a JSON file in EXPERIMENTS_DIR, including
      label configuration (log vs simple returns, threshold, horizons).
"""

import os
from typing import Tuple, Dict, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from experiment_3.config import (
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
    SEQ_LENGTH,
    NUM_CLASSES,
    # Labeling configuration (log-returns experiment)
    USE_LOG_RETURNS,
    DIRECTION_THRESHOLD,
    FORECAST_HORIZONS,
    USE_MULTI_HORIZON,
)

from experiment_3.data_pipeline import prepare_datasets
from experiment_3.tft_model import TemporalFusionTransformer
from experiment_3 import utils


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
    Run inference for the 3-class classification task.

    Returns:
        y_true : array of true integer labels, shape (N,)
        y_prob : array of predicted class probabilities, shape (N, C)
        avg_loss : average CrossEntropy loss over the dataset
    """
    model.eval()
    all_true: list[torch.Tensor] = []
    all_prob: list[torch.Tensor] = []
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in data_loader:
            x_past, x_future, y = _unpack_batch(batch, device)

            # Forward pass -> logits, shape (B, NUM_CLASSES)
            logits = model(x_past) if x_future is None else model(x_past, x_future)

            # Targets are already (B,) int64 in the dataset
            y_flat = y.view(-1).long()

            # CE expects (B, C) logits and (B,) labels
            loss = criterion(logits, y_flat)
            batch_size = y_flat.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

            # Convert logits -> probabilities
            probs = torch.softmax(logits, dim=1)

            all_true.append(y_flat.cpu())
            all_prob.append(probs.cpu())

    y_true = torch.cat(all_true, dim=0).numpy()          # (N,)
    y_prob = torch.cat(all_prob, dim=0).numpy()          # (N, C)
    avg_loss = total_loss / max(1, total_count)

    return y_true, y_prob, avg_loss


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def main() -> None:
    print("[evaluate_tft] Using device:", end=" ")
    device = utils.get_device()
    print(device)

    # Show key labeling configuration so you can see at a glance
    print("[evaluate_tft] Label config:")
    print(f"  USE_LOG_RETURNS    = {USE_LOG_RETURNS}")
    print(f"  DIRECTION_THRESHOLD = {DIRECTION_THRESHOLD}")
    print(f"  FORECAST_HORIZONS   = {FORECAST_HORIZONS}")
    print(f"  USE_MULTI_HORIZON   = {USE_MULTI_HORIZON}")

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
    model = TemporalFusionTransformer(MODEL_CONFIG).to(device)

    # Load best checkpoint (weights only)
    state = torch.load(BEST_MODEL_PATH, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)

    # Create a unique ID for this evaluation run
    eval_id = f"tft_eval_{utils.get_timestamp()}"
    utils.ensure_dir(EXPERIMENTS_DIR)
    utils.ensure_dir(PLOTS_DIR)

    metrics_out: Dict[str, Any] = {
        # High-level run metadata will be filled below
    }

    # ------------------------------------------------------------------
    # 3. Evaluation paths by TASK_TYPE
    # ------------------------------------------------------------------
    if TASK_TYPE == "classification":
        # 3-class direction classification: DOWN / FLAT / UP
        criterion = nn.CrossEntropyLoss()

        print("[evaluate_tft] TASK_TYPE='classification' -> 3-class direction "
              "(0=DOWN, 1=FLAT, 2=UP).")

        # === Validation set: multi-class metrics ===
        print("[evaluate_tft] Running inference on validation set (multi-class).")
        y_val_true, y_val_prob, val_loss = run_inference_classification(
            model, val_loader, device, criterion
        )

        # Multi-class metrics (macro averages, etc.)
        val_mc_metrics = utils.compute_multiclass_metrics(
            y_true=y_val_true,
            y_prob=y_val_prob,
        )

        print("[evaluate_tft] Validation multi-class metrics:")
        print(f"  Val loss (CE): {val_loss:.4f}")
        for k, v in val_mc_metrics.items():
            print(f"  Val {k}: {v:.4f}")

        # === Validation: UP vs REST threshold tuning ===
        # For UP-vs-REST we treat the UP class (index NUM_CLASSES-1) as the positive class.
        up_class_index = NUM_CLASSES - 1
        y_val_up_true = (y_val_true == up_class_index).astype(int)
        y_val_up_prob = y_val_prob[:, up_class_index]

        if AUTO_TUNE_THRESHOLD:
            thresholds = np.linspace(
                THRESHOLD_SEARCH_MIN,
                THRESHOLD_SEARCH_MAX,
                THRESHOLD_SEARCH_STEPS,
            )

            best_threshold = None
            best_metric_value = -np.inf
            best_up_metrics: Dict[str, float] | None = None

            for th in thresholds:
                metrics = utils.compute_classification_metrics(
                    y_true=y_val_up_true,
                    y_prob=y_val_up_prob,
                    threshold=th,
                )
                metric_value = metrics.get(
                    THRESHOLD_TARGET_METRIC, metrics.get("f1", 0.0)
                )

                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_threshold = th
                    best_up_metrics = metrics

            up_threshold = float(best_threshold)
            val_up_metrics = best_up_metrics or {}
            print("[evaluate_tft] Validation UP-vs-REST threshold search "
                  f"(optimize {THRESHOLD_TARGET_METRIC}):")
            print(f"  Best P(UP) threshold on val: {up_threshold:.3f}")
            for k, v in val_up_metrics.items():
                print(f"  Val UP-vs-REST {k}: {v:.4f}")
        else:
            # Use fixed threshold from config
            up_threshold = float(EVAL_THRESHOLD)
            val_up_metrics = utils.compute_classification_metrics(
                y_true=y_val_up_true,
                y_prob=y_val_up_prob,
                threshold=up_threshold,
            )
            print("[evaluate_tft] Validation UP-vs-REST (fixed threshold):")
            print(f"  Using P(UP) threshold: {up_threshold:.3f}")
            for k, v in val_up_metrics.items():
                print(f"  Val UP-vs-REST {k}: {v:.4f}")

        # Plot validation P(UP) histogram
        val_hist_path = os.path.join(PLOTS_DIR, f"{eval_id}_val_up_prob_hist.png")
        utils.plot_probability_histogram(
            y_prob=y_val_up_prob,
            out_path=val_hist_path,
            threshold=up_threshold,
            title="Val P(UP) Histogram (UP vs REST)",
        )
        print(f"[evaluate_tft] Validation P(UP) histogram saved to {val_hist_path}")

        # === Test set: multi-class + UP-vs-REST using chosen threshold ===
        print("[evaluate_tft] Running inference on test set (multi-class).")
        y_test_true, y_test_prob, test_loss = run_inference_classification(
            model, test_loader, device, criterion
        )

        test_mc_metrics = utils.compute_multiclass_metrics(
            y_true=y_test_true,
            y_prob=y_test_prob,
        )

        print("[evaluate_tft] Test multi-class metrics:")
        print(f"  Test loss (CE): {test_loss:.4f}")
        for k, v in test_mc_metrics.items():
            print(f"  Test {k}: {v:.4f}")

        # Binary UP vs REST on test
        y_test_up_true = (y_test_true == up_class_index).astype(int)
        y_test_up_prob = y_test_prob[:, up_class_index]

        test_up_metrics = utils.compute_classification_metrics(
            y_true=y_test_up_true,
            y_prob=y_test_up_prob,
            threshold=up_threshold,
        )

        print("[evaluate_tft] Test UP-vs-REST metrics "
              f"(using P(UP) threshold={up_threshold:.3f}):")
        for k, v in test_up_metrics.items():
            print(f"  Test UP-vs-REST {k}: {v:.4f}")

        # ---- Plots for classification ----

        # 1) Test P(UP) histogram
        test_hist_path = os.path.join(PLOTS_DIR, f"{eval_id}_test_up_prob_hist.png")
        utils.plot_probability_histogram(
            y_prob=y_test_up_prob,
            out_path=test_hist_path,
            threshold=up_threshold,
            title="Test P(UP) Histogram (UP vs REST)",
        )
        print(f"[evaluate_tft] Test P(UP) histogram saved to {test_hist_path}")

        # 2) 3-class confusion matrix (argmax over logits)
        y_test_pred_mc = np.argmax(y_test_prob, axis=1)
        cm_3c_path = os.path.join(
            PLOTS_DIR, f"{eval_id}_test_confusion_matrix_3class.png"
        )
        utils.plot_confusion_matrix(
            y_true=y_test_true,
            y_pred=y_test_pred_mc,
            out_path=cm_3c_path,
            title="Test Direction Confusion (3-class)",
            class_names=["DOWN", "FLAT", "UP"],
        )
        print(f"[evaluate_tft] 3-class confusion matrix saved to {cm_3c_path}")

        # 3) Binary UP vs REST confusion matrix
        y_test_pred_up_bin = (y_test_up_prob >= up_threshold).astype(int)
        cm_bin_path = os.path.join(
            PLOTS_DIR, f"{eval_id}_test_confusion_matrix_up_vs_rest.png"
        )
        utils.plot_confusion_matrix(
            y_true=y_test_up_true,
            y_pred=y_test_pred_up_bin,
            out_path=cm_bin_path,
            title="Test UP vs REST Confusion",
            class_names=["NOT_UP", "UP"],
        )
        print(f"[evaluate_tft] UP-vs-REST confusion matrix saved to {cm_bin_path}")

        # 4) ROC curve for UP vs REST
        roc_path = os.path.join(PLOTS_DIR, f"{eval_id}_test_roc_up_vs_rest.png")
        utils.plot_roc_curve(
            y_true=y_test_up_true,
            y_score=y_test_up_prob,
            out_path=roc_path,
            title="Test ROC (UP vs REST)",
        )
        print(f"[evaluate_tft] ROC curve (UP vs REST) saved to {roc_path}")

        # ---- Pack metrics into JSON-friendly structure ----
        metrics_out = {
            "task_type": TASK_TYPE,
            "num_classes": NUM_CLASSES,
            "class_encoding": {0: "DOWN", 1: "FLAT", 2: "UP"},

            # Labeling / experiment config (critical for comparing runs)
            "label_config": {
                "use_log_returns": bool(USE_LOG_RETURNS),
                "direction_threshold": float(DIRECTION_THRESHOLD),
                "forecast_horizons": list(FORECAST_HORIZONS),
                "use_multi_horizon": bool(USE_MULTI_HORIZON),
            },

            "val_loss_ce": float(val_loss),
            "test_loss_ce": float(test_loss),
            "val_multiclass": val_mc_metrics,
            "test_multiclass": test_mc_metrics,
            "up_class_index": int(up_class_index),
            "up_prob_threshold": float(up_threshold),
            "val_up_vs_rest": val_up_metrics,
            "test_up_vs_rest": test_up_metrics,
        }

    else:
        raise NotImplementedError(
            f"TASK_TYPE='{TASK_TYPE}' is not supported yet in evaluate_tft.py"
        )

    # ------------------------------------------------------------------
    # 4. Save evaluation metrics to JSON
    # ------------------------------------------------------------------
    metrics_path = os.path.join(EXPERIMENTS_DIR, f"{eval_id}_metrics.json")
    utils.save_json(metrics_out, metrics_path)
    print(f"[evaluate_tft] Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()