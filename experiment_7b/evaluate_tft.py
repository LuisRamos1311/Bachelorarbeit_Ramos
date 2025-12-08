"""
evaluate_tft.py

Evaluate a trained Temporal Fusion Transformer (TFT) model on the BTC dataset.

Experiment / generic single-horizon setup:
    - TASK_TYPE = "classification"  (3-class DOWN / FLAT / UP via direction_3c)
    - Single-horizon H-step-ahead prediction where
          H = FORECAST_HORIZONS[0]
      e.g. for Experiment 6 with hourly data:
          FREQUENCY = "1h", FORECAST_HORIZONS = [24]
          -> "next 24 hours" direction.

    - UP-vs-REST analysis with:
        * Manual threshold sweep over a fixed grid for P(UP)
        * Selection of τ* on validation using THRESHOLD_SELECTION_METRIC
          (e.g. balanced_accuracy)
        * Long-only trading metrics based on the H-step forward return
          aligned with each sample.

Behaviour:
    * Computes full multi-class metrics on val/test.
    * Sweeps thresholds τ ∈ UP_THRESHOLD_GRID on P(UP) for UP-vs-REST:
        - classification metrics (accuracy, precision, recall, F1, etc.)
        - trading metrics (avg return while in position, cumulative return,
          Sharpe, hit ratio) from a long-only strategy.
    * Chooses τ* on the validation set according to THRESHOLD_SELECTION_METRIC.
    * Evaluates test UP-vs-REST and trading metrics at τ*.
    * Produces probability histograms, confusion matrices, and ROC curve.
    * Saves all metrics (including the per-threshold sweep) into JSON files.
"""

import os
from typing import Tuple, Dict, Any, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from experiment_7b.config import (
    MODEL_CONFIG,
    TRAINING_CONFIG,
    TASK_TYPE,
    EVAL_THRESHOLD,
    AUTO_TUNE_THRESHOLD,
    BEST_MODEL_PATH,
    EXPERIMENTS_DIR,
    PLOTS_DIR,
    SEQ_LENGTH,
    NUM_CLASSES,
    # Labeling configuration
    USE_LOG_RETURNS,
    DIRECTION_THRESHOLD,
    FORECAST_HORIZONS,
    USE_MULTI_HORIZON,
    FREQUENCY,
    # UP-vs-REST threshold grid + selection metric
    UP_THRESHOLD_GRID,
    THRESHOLD_SELECTION_METRIC,
    NON_OVERLAPPING_TRADES,
)

from experiment_7b.data_pipeline import prepare_datasets
from experiment_7b.tft_model import TemporalFusionTransformer
from experiment_7b import utils


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
    Run inference for multi-class classification and return:
        y_true (N,), y_prob (N, C), avg_loss (scalar)
    """
    model.eval()
    total_loss = 0.0
    total_count = 0

    all_true: List[torch.Tensor] = []
    all_prob: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in data_loader:
            x_past, x_future, y = _unpack_batch(batch, device)

            if x_future is not None:
                logits = model(x_past, x_future)
            else:
                logits = model(x_past)

            loss = criterion(logits, y)
            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

            probs = torch.softmax(logits, dim=1)

            all_true.append(y.detach().cpu())
            all_prob.append(probs.detach().cpu())

    if total_count == 0:
        raise RuntimeError("DataLoader is empty in run_inference_classification.")

    y_true = torch.cat(all_true, dim=0).numpy()
    y_prob = torch.cat(all_prob, dim=0).numpy()
    avg_loss = total_loss / float(total_count)

    return y_true, y_prob, avg_loss


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def main() -> None:
    print("[evaluate_tft] Using device:", end=" ")
    device = utils.get_device()
    print(device)

    # Horizon description (mirrors train_tft.py)
    if FORECAST_HORIZONS:
        horizon_steps = int(FORECAST_HORIZONS[0])
    else:
        horizon_steps = 1

    if FREQUENCY.upper() == "D":
        horizon_desc = f"{horizon_steps}-day-ahead"
    elif FREQUENCY.lower() in {"1h", "h", "hourly"}:
        horizon_desc = f"{horizon_steps}-hour-ahead"
    else:
        horizon_desc = f"{horizon_steps}-step-ahead"

    # Show key labeling configuration so you can see at a glance
    print("[evaluate_tft] Label config:")
    print(f"  USE_LOG_RETURNS     = {USE_LOG_RETURNS}")
    print(f"  DIRECTION_THRESHOLD = {DIRECTION_THRESHOLD}")
    print(f"  FORECAST_HORIZONS   = {FORECAST_HORIZONS} "
          f"(first horizon: {horizon_desc})")
    print(f"  USE_MULTI_HORIZON   = {USE_MULTI_HORIZON}")
    print(f"  FREQUENCY           = '{FREQUENCY}'")
    print(f"  NON_OVERLAPPING_TRADES = {NON_OVERLAPPING_TRADES}")

    # ------------------------------------------------------------------
    # 1. Prepare datasets & loaders
    # ------------------------------------------------------------------
    print("[evaluate_tft] Preparing datasets (with forward returns).")

    # Use global SEQ_LENGTH from config and request forward returns
    (
        train_dataset,
        val_dataset,
        test_dataset,
        _,
        forward_returns,
    ) = prepare_datasets(
        seq_length=SEQ_LENGTH,
        return_forward_returns=True,
    )

    val_forward_returns = forward_returns["val"]
    test_forward_returns = forward_returns["test"]

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

    metrics_out: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 3. Evaluation paths by TASK_TYPE
    # ------------------------------------------------------------------
    if TASK_TYPE == "classification":
        # 3-class direction classification: DOWN / FLAT / UP
        criterion = nn.CrossEntropyLoss()

        print(
            "[evaluate_tft] TASK_TYPE='classification' -> 3-class direction "
            "(0=DOWN, 1=FLAT, 2=UP)."
        )

        # === Validation set: multi-class metrics ===
        print("[evaluate_tft] Running inference on validation set (multi-class).")
        y_val_true, y_val_prob, val_loss = run_inference_classification(
            model, val_loader, device, criterion
        )

        val_mc_metrics = utils.compute_multiclass_metrics(
            y_true=y_val_true,
            y_prob=y_val_prob,
        )

        print("[evaluate_tft] Validation multi-class metrics:")
        print(f"  Val loss (CE): {val_loss:.4f}")
        for k, v in val_mc_metrics.items():
            print(f"  Val {k}: {v:.4f}")

        # === Test set: multi-class metrics ===
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

        # ------------------------------------------------------------------
        # 3a. UP-vs-REST: threshold sweep on P(UP) + trading metrics
        # ------------------------------------------------------------------
        up_class_index = NUM_CLASSES - 1  # 2 in the 3-class setup

        # Binary labels for UP vs REST
        y_val_up_true = (y_val_true == up_class_index).astype(int)
        y_test_up_true = (y_test_true == up_class_index).astype(int)

        # Scores = P(UP)
        y_val_up_prob = y_val_prob[:, up_class_index]
        y_test_up_prob = y_test_prob[:, up_class_index]

        # For trading metrics we *optionally* switch to non-overlapping trades:
        # we sub-sample returns & scores every H = horizon_steps steps.
        # Classification metrics still use the full series (no sub-sampling).
        if NON_OVERLAPPING_TRADES and horizon_steps > 1:
            print(
                "[evaluate_tft] Using non-overlapping trades for trading metrics "
                f"(horizon_steps={horizon_steps})."
            )
            (
                val_forward_returns_tr,
                y_val_up_prob_tr,
            ) = utils.select_non_overlapping_trades(
                returns=val_forward_returns,
                scores=y_val_up_prob,
                horizon=horizon_steps,
            )
            (
                test_forward_returns_tr,
                y_test_up_prob_tr,
            ) = utils.select_non_overlapping_trades(
                returns=test_forward_returns,
                scores=y_test_up_prob,
                horizon=horizon_steps,
            )
        else:
            # Fallback: original overlapping-trades behaviour
            val_forward_returns_tr = val_forward_returns
            test_forward_returns_tr = test_forward_returns
            y_val_up_prob_tr = y_val_up_prob
            y_test_up_prob_tr = y_test_up_prob

        # Threshold grid (fallback to EVAL_THRESHOLD if grid is empty)
        threshold_grid: List[float] = list(UP_THRESHOLD_GRID) or [float(EVAL_THRESHOLD)]

        print("[evaluate_tft] Threshold sweep on P(UP) for UP-vs-REST:")
        print(f"  Grid: {threshold_grid}")
        print(f"  Selection metric (val): {THRESHOLD_SELECTION_METRIC}")
        print(f"  AUTO_TUNE_THRESHOLD   : {AUTO_TUNE_THRESHOLD}")

        selection_key = THRESHOLD_SELECTION_METRIC.lower().strip()

        sweep_records: List[Dict[str, Any]] = []

        best_threshold: float | None = None
        best_val_score: float = -np.inf
        best_val_binary: Dict[str, float] | None = None
        best_test_binary: Dict[str, float] | None = None
        best_val_trading: Dict[str, float] | None = None
        best_test_trading: Dict[str, float] | None = None

        # Sweep over grid
        for thr in threshold_grid:
            thr = float(thr)

            # Binary classification metrics on *full* series (UP vs REST)
            val_bin_metrics = utils.compute_classification_metrics(
                y_true=y_val_up_true,
                y_prob=y_val_up_prob,
                threshold=thr,
            )
            test_bin_metrics = utils.compute_classification_metrics(
                y_true=y_test_up_true,
                y_prob=y_test_up_prob,
                threshold=thr,
            )

            # Long-only trading metrics for this threshold
            # (possibly on non-overlapping trades)
            val_trading = utils.compute_trading_metrics(
                returns=val_forward_returns_tr,
                positions=utils.positions_from_threshold(
                    y_prob_up=y_val_up_prob_tr,
                    threshold=thr,
                ),
            )
            test_trading = utils.compute_trading_metrics(
                returns=test_forward_returns_tr,
                positions=utils.positions_from_threshold(
                    y_prob_up=y_test_up_prob_tr,
                    threshold=thr,
                ),
            )

            # Score used to select τ* on validation.
            # We allow THRESHOLD_SELECTION_METRIC to refer either to a
            # classification metric (e.g. "balanced_accuracy", "f1")
            # or to a trading metric (e.g. "sharpe", "cumulative_return").
            if selection_key in val_bin_metrics:
                # Use a classification metric as selection score
                val_score = float(val_bin_metrics[selection_key])
            elif selection_key in val_trading:
                # Use a trading metric as selection score
                val_score = float(val_trading[selection_key])
            else:
                # If the name is not found in either dict, raise a clear error
                raise ValueError(
                    f"Unknown THRESHOLD_SELECTION_METRIC='{THRESHOLD_SELECTION_METRIC}'. "
                    f"Available classification keys: {list(val_bin_metrics.keys())}, "
                    f"trading keys: {list(val_trading.keys())}"
                )

            record = {
                "threshold": thr,
                "val_binary": val_bin_metrics,
                "test_binary": test_bin_metrics,
                "val_trading": val_trading,
                "test_trading": test_trading,
                "selection_score": float(val_score),
            }
            sweep_records.append(record)

            if AUTO_TUNE_THRESHOLD and np.isfinite(val_score):
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_threshold = thr
                    best_val_binary = val_bin_metrics
                    best_test_binary = test_bin_metrics
                    best_val_trading = val_trading
                    best_test_trading = test_trading

        # If auto-tuning is disabled or failed, fall back to EVAL_THRESHOLD
        if not AUTO_TUNE_THRESHOLD or best_threshold is None:
            best_threshold = float(EVAL_THRESHOLD)
            print(
                f"[evaluate_tft] Auto-tuning disabled or failed, "
                f"falling back to EVAL_THRESHOLD={best_threshold:.3f}"
            )

            val_up_metrics = utils.compute_classification_metrics(
                y_true=y_val_up_true,
                y_prob=y_val_up_prob,
                threshold=best_threshold,
            )
            test_up_metrics = utils.compute_classification_metrics(
                y_true=y_test_up_true,
                y_prob=y_test_up_prob,
                threshold=best_threshold,
            )
            val_trading_best = utils.compute_trading_metrics(
                returns=val_forward_returns_tr,
                positions=utils.positions_from_threshold(
                    y_prob_up=y_val_up_prob_tr,
                    threshold=best_threshold,
                ),
            )
            test_trading_best = utils.compute_trading_metrics(
                returns=test_forward_returns_tr,
                positions=utils.positions_from_threshold(
                    y_prob_up=y_test_up_prob_tr,
                    threshold=best_threshold,
                ),
            )
        else:
            print(
                f"[evaluate_tft] Best threshold on val ({selection_key}): "
                f"{best_threshold:.3f} with score={best_val_score:.4f}"
            )
            val_up_metrics = best_val_binary or {}
            test_up_metrics = best_test_binary or {}
            val_trading_best = best_val_trading or {}
            test_trading_best = best_test_trading or {}

        up_threshold = float(best_threshold)

        # Print summary metrics at τ*
        print(
            f"[evaluate_tft] Validation UP-vs-REST metrics "
            f"(τ* = {up_threshold:.3f}):"
        )
        for k, v in val_up_metrics.items():
            print(f"  Val {k}: {v:.4f}")

        print(
            f"[evaluate_tft] Test UP-vs-REST metrics "
            f"(τ* = {up_threshold:.3f}):"
        )
        for k, v in test_up_metrics.items():
            print(f"  Test {k}: {v:.4f}")

        # Trading summary at τ*
        print(
            f"[evaluate_tft] Long-only trading metrics at τ* "
            f"({horizon_desc} horizon):"
        )
        print("  Validation:")
        for k, v in val_trading_best.items():
            print(f"    {k}: {v:.6f}")
        print("  Test:")
        for k, v in test_trading_best.items():
            print(f"    {k}: {v:.6f}")

        # ---- Plots: histograms, confusion matrices, ROC ----

        # 1) Validation P(UP) histogram
        val_hist_path = os.path.join(PLOTS_DIR, f"{eval_id}_val_up_prob_hist.png")
        utils.plot_probability_histogram(
            y_prob=y_val_up_prob,
            out_path=val_hist_path,
            threshold=up_threshold,
            title="Val P(UP) Histogram (UP vs REST)",
        )
        print(f"[evaluate_tft] Validation P(UP) histogram saved to {val_hist_path}")

        # 2) Test P(UP) histogram
        test_hist_path = os.path.join(PLOTS_DIR, f"{eval_id}_test_up_prob_hist.png")
        utils.plot_probability_histogram(
            y_prob=y_test_up_prob,
            out_path=test_hist_path,
            threshold=up_threshold,
            title="Test P(UP) Histogram (UP vs REST)",
        )
        print(f"[evaluate_tft] Test P(UP) histogram saved to {test_hist_path}")

        # 3) 3-class confusion matrix (argmax over logits) on test
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

        # 4) Binary UP vs REST confusion matrix (test, at τ*)
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

        # 5) ROC curve for UP vs REST (test)
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
                "frequency": FREQUENCY,
                "horizon_steps": int(horizon_steps),
                "non_overlapping_trades": bool(NON_OVERLAPPING_TRADES),
            },

            "seq_length": int(SEQ_LENGTH),
            "eval_id": eval_id,
            "model_checkpoint": BEST_MODEL_PATH,

            # Multi-class CE losses
            "val_loss_ce": float(val_loss),
            "test_loss_ce": float(test_loss),

            # Multi-class metrics
            "val_multiclass": val_mc_metrics,
            "test_multiclass": test_mc_metrics,

            # UP-vs-REST configuration and metrics
            "up_vs_rest": {
                "up_class_index": int(up_class_index),
                "auto_tune_threshold": bool(AUTO_TUNE_THRESHOLD),
                "selection_metric": THRESHOLD_SELECTION_METRIC,
                "selected_threshold": float(up_threshold),
                "val_binary_metrics": val_up_metrics,
                "test_binary_metrics": test_up_metrics,
                "val_trading_metrics": val_trading_best,
                "test_trading_metrics": test_trading_best,
                # Full sweep for later plotting / analysis
                "threshold_sweep": sweep_records,
            },
        }

        # Also save just the sweep as a separate JSON for convenience
        sweep_path = os.path.join(
            EXPERIMENTS_DIR, f"{eval_id}_up_vs_rest_threshold_sweep.json"
        )
        utils.save_json(sweep_records, sweep_path)
        print(f"[evaluate_tft] Threshold sweep saved to {sweep_path}")

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