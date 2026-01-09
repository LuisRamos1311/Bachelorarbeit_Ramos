"""
evaluate_tft.py

Evaluate a trained Temporal Fusion Transformer (TFT) model on the BTC dataset.

Experiment 4 setup (binary big-move classification):
    - TASK_TYPE = "classification"
    - Target = direction_2c_bigmove (0 = DOWN, 1 = UP)
    - Only days with |r_1d| > DIRECTION_THRESHOLD are kept in the dataset
      (small-move days are dropped in data_pipeline.prepare_datasets).

Behaviour:
    * Runs the model on the validation set, performs a grid search over
      P(UP) thresholds and selects the one that maximizes a chosen metric
      (e.g. F1) via THRESHOLD_TARGET_METRIC.
    * Evaluates the test set with that threshold.
    * Produces probability histograms, a binary confusion matrix and a ROC curve.
    * Saves all metrics into a JSON file in EXPERIMENTS_DIR, including
      label configuration (log vs simple returns, threshold, horizons).
"""

import os
from typing import Tuple, Dict, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from experiment_4.config import (
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

from experiment_4.data_pipeline import prepare_datasets
from experiment_4.tft_model import TemporalFusionTransformer
from experiment_4 import utils


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
    Run inference for a C-class classification task (C >= 2).

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

    if total_count == 0:
        raise RuntimeError("No samples seen in run_inference_classification.")

    y_true = torch.cat(all_true, dim=0).numpy()          # (N,)
    y_prob = torch.cat(all_prob, dim=0).numpy()          # (N, C)
    avg_loss = total_loss / float(total_count)

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

    # Use global SEQ_LENGTH from config (kept in sync with train_tft)
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

    # ------------------------------------------------------------------
    # 3. Evaluation paths by TASK_TYPE
    # ------------------------------------------------------------------
    if TASK_TYPE == "classification":
        # Binary big-move classification (0 = DOWN, 1 = UP).
        # Use the same optional class weights as in train_tft.
        pos_weight = getattr(TRAINING_CONFIG, "pos_weight", 1.0)
        weight_tensor = None
        if pos_weight is not None and float(pos_weight) != 1.0:
            # Class 0 (DOWN) -> weight 1.0
            # Class 1 (UP)   -> weight pos_weight
            weight_tensor = torch.tensor(
                [1.0, float(pos_weight)],
                dtype=torch.float32,
                device=device,
            )

        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        print(
            "[evaluate_tft] TASK_TYPE='classification' -> binary big-move "
            "direction (0=DOWN, 1=UP)."
        )
        if weight_tensor is not None:
            print(f"[evaluate_tft] Class weights: {weight_tensor.tolist()}")

        # --------------------------------------------------------------
        # 3a. Validation: run model and tune P(UP) threshold
        # --------------------------------------------------------------
        print("[evaluate_tft] Running inference on validation set (binary).")
        y_val_true, y_val_prob_full, val_loss = run_inference_classification(
            model, val_loader, device, criterion
        )

        # Positive class index = last column (for consistency with training)
        up_class_index = NUM_CLASSES - 1  # 1 for binary 0/1
        y_val_up_prob = y_val_prob_full[:, up_class_index]
        y_val_true_bin = y_val_true.astype(int)

        if AUTO_TUNE_THRESHOLD:
            thresholds = np.linspace(
                THRESHOLD_SEARCH_MIN,
                THRESHOLD_SEARCH_MAX,
                THRESHOLD_SEARCH_STEPS,
            )

            best_threshold: float | None = None
            best_metric_value = -np.inf
            best_val_metrics: Dict[str, float] | None = None

            for th in thresholds:
                metrics = utils.compute_classification_metrics(
                    y_true=y_val_true_bin,
                    y_prob=y_val_up_prob,
                    threshold=float(th),
                )
                metric_value = metrics.get(
                    THRESHOLD_TARGET_METRIC, metrics.get("f1", 0.0)
                )

                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_threshold = float(th)
                    best_val_metrics = metrics

            if best_threshold is None:
                # Fallback – should not normally happen
                best_threshold = float(EVAL_THRESHOLD)
                best_val_metrics = utils.compute_classification_metrics(
                    y_true=y_val_true_bin,
                    y_prob=y_val_up_prob,
                    threshold=best_threshold,
                )

            up_threshold = best_threshold
            val_bin_metrics = best_val_metrics or {}

            print(
                "[evaluate_tft] Validation threshold search on P(UP) "
                f"(optimize {THRESHOLD_TARGET_METRIC}):"
            )
            print(f"  Best P(UP) threshold on val: {up_threshold:.3f}")
            for k, v in val_bin_metrics.items():
                print(f"  Val {k}: {v:.4f}")
        else:
            # Use fixed threshold from config (no tuning)
            up_threshold = float(EVAL_THRESHOLD)
            val_bin_metrics = utils.compute_classification_metrics(
                y_true=y_val_true_bin,
                y_prob=y_val_up_prob,
                threshold=up_threshold,
            )
            print("[evaluate_tft] Validation (fixed threshold):")
            print(f"  Using P(UP) threshold: {up_threshold:.3f}")
            for k, v in val_bin_metrics.items():
                print(f"  Val {k}: {v:.4f}")

        # Plot validation P(UP) histogram
        val_hist_path = os.path.join(PLOTS_DIR, f"{eval_id}_val_up_prob_hist.png")
        utils.plot_probability_histogram(
            y_prob=y_val_up_prob,
            out_path=val_hist_path,
            threshold=up_threshold,
            title="Val P(UP) Histogram (UP vs DOWN)",
        )
        print(f"[evaluate_tft] Validation P(UP) histogram saved to {val_hist_path}")

        # --------------------------------------------------------------
        # 3b. Test: evaluate with chosen threshold
        # --------------------------------------------------------------
        print("[evaluate_tft] Running inference on test set (binary).")
        y_test_true, y_test_prob_full, test_loss = run_inference_classification(
            model, test_loader, device, criterion
        )

        y_test_up_prob = y_test_prob_full[:, up_class_index]
        y_test_true_bin = y_test_true.astype(int)

        test_bin_metrics = utils.compute_classification_metrics(
            y_true=y_test_true_bin,
            y_prob=y_test_up_prob,
            threshold=up_threshold,
        )

        print(
            "[evaluate_tft] Test binary metrics "
            f"(using P(UP) threshold={up_threshold:.3f}):"
        )
        print(f"  Test loss (CE): {test_loss:.4f}")
        for k, v in test_bin_metrics.items():
            print(f"  Test {k}: {v:.4f}")

        # ---- Plots for binary classification ----

        # 1) Test P(UP) histogram
        test_hist_path = os.path.join(PLOTS_DIR, f"{eval_id}_test_up_prob_hist.png")
        utils.plot_probability_histogram(
            y_prob=y_test_up_prob,
            out_path=test_hist_path,
            threshold=up_threshold,
            title="Test P(UP) Histogram (UP vs DOWN)",
        )
        print(f"[evaluate_tft] Test P(UP) histogram saved to {test_hist_path}")

        # 2) Binary confusion matrix (using tuned threshold)
        y_test_pred_bin = (y_test_up_prob >= up_threshold).astype(int)
        cm_bin_path = os.path.join(
            PLOTS_DIR, f"{eval_id}_test_confusion_matrix_binary.png"
        )
        utils.plot_confusion_matrix(
            y_true=y_test_true_bin,
            y_pred=y_test_pred_bin,
            out_path=cm_bin_path,
            title="Test Direction Confusion (binary big-move)",
            class_names=["DOWN", "UP"],
        )
        print(f"[evaluate_tft] Binary confusion matrix saved to {cm_bin_path}")

        # 3) ROC curve for UP vs DOWN
        roc_path = os.path.join(PLOTS_DIR, f"{eval_id}_test_roc_up_vs_down.png")
        utils.plot_roc_curve(
            y_true=y_test_true_bin,
            y_score=y_test_up_prob,
            out_path=roc_path,
            title="Test ROC (UP vs DOWN)",
        )
        print(f"[evaluate_tft] ROC curve (UP vs DOWN) saved to {roc_path}")

        # ---- Pack metrics into JSON-friendly structure ----
        metrics_out: Dict[str, Any] = {
            "task_type": TASK_TYPE,
            "num_classes": NUM_CLASSES,
            "class_encoding": {0: "DOWN", 1: "UP"},

            # Labeling / experiment config (critical for comparing runs)
            "label_config": {
                "use_log_returns": bool(USE_LOG_RETURNS),
                "direction_threshold": float(DIRECTION_THRESHOLD),
                "forecast_horizons": list(FORECAST_HORIZONS),
                "use_multi_horizon": bool(USE_MULTI_HORIZON),
                "binary_big_move": True,
            },

            "up_class_index": int(up_class_index),
            "probability_threshold": float(up_threshold),

            "val_loss_ce": float(val_loss),
            "test_loss_ce": float(test_loss),
            "val_binary": val_bin_metrics,
            "test_binary": test_bin_metrics,
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
