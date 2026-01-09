"""
evaluate_tft.py

Evaluate a trained Temporal Fusion Transformer (TFT) model on the BTC dataset.

Experiment / generic single-horizon setup:
evaluation supports:
classification (old path)
quantile multi-horizon forecasting (new 9c path): pinball loss, MAE@trade horizon, and score-based threshold sweep.

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

from final_model.config import (
    # Model + training
    MODEL_CONFIG,
    TRAINING_CONFIG,
    TASK_TYPE,
    BEST_MODEL_PATH,

    # Paths
    EXPERIMENTS_DIR,
    PLOTS_DIR,

    # Data
    SEQ_LENGTH,
    NUM_CLASSES,
    USE_LOG_RETURNS,
    DIRECTION_THRESHOLD,
    FORECAST_HORIZONS,
    USE_MULTI_HORIZON,
    FREQUENCY,

    # Quantile forecasting (9c)
    FORECAST_HORIZON,
    QUANTILES,
    N_QUANTILES,

    # Thresholding / trading knobs (9b/9c)
    EVAL_THRESHOLD,
    AUTO_TUNE_THRESHOLD,
    THRESHOLD_SELECTION_METRIC,
    UP_THRESHOLD_GRID,   # kept for backward compatibility
    SCORE_GRID,
    SIGNAL_HORIZON,
    SCORE_EPS,
    ACTIVE_THRESHOLD_GRID,
    ACTIVE_SELECTION_METRIC,
    ACTIVE_AUTO_TUNE,

    # Trading/costs
    NON_OVERLAPPING_TRADES,
    TRADING_DAYS_PER_YEAR,
    COST_BPS,
    SLIPPAGE_BPS,

    # Baselines
    RANDOM_BASELINE_RUNS,
    RANDOM_SEED,
)
from final_model import config as cfg
from final_model.data_pipeline import prepare_datasets
from final_model.tft_model import TemporalFusionTransformer
from final_model import utils


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

def _nearest_quantile_index(quantiles: list[float], q: float) -> int:
    """Return the index of the quantile value closest to q."""
    if len(quantiles) == 0:
        raise ValueError("quantiles list is empty.")
    return int(np.argmin(np.abs(np.asarray(quantiles, dtype=float) - float(q))))


def _to_simple_returns(r: np.ndarray) -> np.ndarray:
    """
    Trading utilities compound via prod(1+r), so they expect SIMPLE returns.
    If targets are log-returns, convert: simple = exp(log) - 1.
    """
    r = np.asarray(r, dtype=float)
    if USE_LOG_RETURNS:
        return np.expm1(r)
    return r


def run_inference_quantile(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    quantiles: list[float],
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run inference for quantile multi-horizon forecasting and return:
        y_true (N, H), y_pred (N, H, Q), avg_pinball_loss (scalar)
    """
    model.eval()
    total_loss = 0.0
    total_count = 0

    all_true: list[torch.Tensor] = []
    all_pred: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in data_loader:
            x_past, x_future, y = _unpack_batch(batch, device)  # y: (B, H) float

            if x_future is not None:
                y_hat = model(x_past, x_future)  # (B, H, Q)
            else:
                y_hat = model(x_past)

            if y_hat.ndim != 3:
                raise RuntimeError(f"Expected y_hat to have 3 dims (B,H,Q), got shape={tuple(y_hat.shape)}")
            if y_hat.shape[0] != y.shape[0] or y_hat.shape[1] != y.shape[1]:
                raise RuntimeError(
                    f"Shape mismatch: y_hat={tuple(y_hat.shape)} vs y={tuple(y.shape)}"
                )

            loss = utils.pinball_loss(y, y_hat, quantiles)
            batch_size = y.size(0)
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

            all_true.append(y.detach().cpu())
            all_pred.append(y_hat.detach().cpu())

    if total_count == 0:
        raise RuntimeError("DataLoader is empty in run_inference_quantile.")

    y_true = torch.cat(all_true, dim=0).numpy()
    y_pred = torch.cat(all_pred, dim=0).numpy()
    avg_loss = total_loss / float(total_count)

    return y_true, y_pred, avg_loss


def _build_score_mu_iqr(
    y_pred: np.ndarray,
    quantiles: list[float],
    signal_idx: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """
    Score for trading (9c):
        mu  = q50 at signal horizon
        iqr = q90 - q10 at signal horizon
        score = mu / (iqr + eps)

    Returns (score, mu, iqr, indices_dict)
    """
    if y_pred.ndim != 3:
        raise ValueError(f"Expected y_pred shape (N,H,Q), got {y_pred.shape}")

    q10_i = _nearest_quantile_index(quantiles, 0.1)
    q50_i = _nearest_quantile_index(quantiles, 0.5)
    q90_i = _nearest_quantile_index(quantiles, 0.9)

    q10 = y_pred[:, signal_idx, q10_i]
    mu = y_pred[:, signal_idx, q50_i]
    q90 = y_pred[:, signal_idx, q90_i]

    iqr = q90 - q10
    iqr_safe = np.where(iqr > eps, iqr, eps)
    score = mu / iqr_safe

    return score, mu, iqr, {"q10": q10_i, "q50": q50_i, "q90": q90_i}

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

    # Experiment 9: data integrity flags (self-documenting logs)
    print("[evaluate_tft] Integrity flags (Experiment 9a):")
    print(f"  DAILY_FEATURE_LAG_DAYS      = {getattr(cfg, 'DAILY_FEATURE_LAG_DAYS', None)}")
    print(f"  DEBUG_DATA_INTEGRITY        = {getattr(cfg, 'DEBUG_DATA_INTEGRITY', None)}")
    print(f"  DROP_LAST_H_IN_EACH_SPLIT   = {getattr(cfg, 'DROP_LAST_H_IN_EACH_SPLIT', None)}")

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
    # Preflight: refuse to evaluate if no trained checkpoint is present
    if not os.path.isfile(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"[evaluate_tft] Best model checkpoint not found: {BEST_MODEL_PATH}\n"
            f"Run train_tft.py first (it should create the checkpoint under the 'standard/models' folder)."
        )
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
                trading_days_per_year=TRADING_DAYS_PER_YEAR,
            )
            test_trading = utils.compute_trading_metrics(
                returns=test_forward_returns_tr,
                positions=utils.positions_from_threshold(
                    y_prob_up=y_test_up_prob_tr,
                    threshold=thr,
                ),
                trading_days_per_year=TRADING_DAYS_PER_YEAR,
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
                trading_days_per_year=TRADING_DAYS_PER_YEAR,
            )
            test_trading_best = utils.compute_trading_metrics(
                returns=test_forward_returns_tr,
                positions=utils.positions_from_threshold(
                    y_prob_up=y_test_up_prob_tr,
                    threshold=best_threshold,
                ),
                trading_days_per_year=TRADING_DAYS_PER_YEAR,
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

        # ===========================
        # Experiment 9b: net-of-cost backtest + baselines (trade-step series)
        # ===========================

        # 1) Strategy positions at τ* on the trade-step probability series
        pos_val = utils.positions_from_threshold(y_prob_up=y_val_up_prob_tr, threshold=up_threshold)
        pos_test = utils.positions_from_threshold(y_prob_up=y_test_up_prob_tr, threshold=up_threshold)

        val_long_rate = float(np.mean(pos_val))
        test_long_rate = float(np.mean(pos_test))

        # 2) Strategy net-of-cost equity + metrics
        val_costs = utils.apply_costs(pos_val, cost_bps=COST_BPS, slippage_bps=SLIPPAGE_BPS)
        test_costs = utils.apply_costs(pos_test, cost_bps=COST_BPS, slippage_bps=SLIPPAGE_BPS)

        val_net_returns = pos_val.astype(float) * val_forward_returns_tr - val_costs
        test_net_returns = pos_test.astype(float) * test_forward_returns_tr - test_costs

        val_equity = utils.equity_curve(val_forward_returns_tr, pos_val, val_costs)
        test_equity = utils.equity_curve(test_forward_returns_tr, pos_test, test_costs)

        strategy_net_val = utils.compute_backtest_metrics(
            equity=val_equity,
            net_returns=val_net_returns,
            trading_days_per_year=TRADING_DAYS_PER_YEAR,
            positions=pos_val,
        )
        strategy_net_test = utils.compute_backtest_metrics(
            equity=test_equity,
            net_returns=test_net_returns,
            trading_days_per_year=TRADING_DAYS_PER_YEAR,
            positions=pos_test,
        )

        # 3) Buy & hold baseline (net-of-cost) on same return series
        buy_hold_net_val = utils.buy_and_hold_baseline(
            returns=val_forward_returns_tr,
            cost_bps=COST_BPS,
            slippage_bps=SLIPPAGE_BPS,
            trading_days_per_year=TRADING_DAYS_PER_YEAR,
        )
        buy_hold_net_test = utils.buy_and_hold_baseline(
            returns=test_forward_returns_tr,
            cost_bps=COST_BPS,
            slippage_bps=SLIPPAGE_BPS,
            trading_days_per_year=TRADING_DAYS_PER_YEAR,
        )

        # 4) Random exposure baseline summary (same long-rate as strategy)
        random_baseline_val = utils.random_exposure_baseline(
            returns=val_forward_returns_tr,
            long_rate=val_long_rate,
            runs=RANDOM_BASELINE_RUNS,
            seed=RANDOM_SEED,
            cost_bps=COST_BPS,
            slippage_bps=SLIPPAGE_BPS,
            trading_days_per_year=TRADING_DAYS_PER_YEAR,
        )
        random_baseline_test = utils.random_exposure_baseline(
            returns=test_forward_returns_tr,
            long_rate=test_long_rate,
            runs=RANDOM_BASELINE_RUNS,
            seed=RANDOM_SEED,
            cost_bps=COST_BPS,
            slippage_bps=SLIPPAGE_BPS,
            trading_days_per_year=TRADING_DAYS_PER_YEAR,
        )

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

        print(
            f"[evaluate_tft] Net-of-cost backtest at τ* "
            f"(cost={COST_BPS}bps, slippage={SLIPPAGE_BPS}bps, annual={TRADING_DAYS_PER_YEAR}):"
        )

        print(
            "  Strategy (net): "
            f"Val Sharpe={strategy_net_val['sharpe']:.4f}, "
            f"MDD={strategy_net_val['max_drawdown']:.4f}, "
            f"CumRet={strategy_net_val['cumulative_return']:.4f} | "
            f"Test Sharpe={strategy_net_test['sharpe']:.4f}, "
            f"MDD={strategy_net_test['max_drawdown']:.4f}, "
            f"CumRet={strategy_net_test['cumulative_return']:.4f}"
        )

        print(
            "  Buy&Hold (net): "
            f"Val Sharpe={buy_hold_net_val['sharpe']:.4f}, "
            f"MDD={buy_hold_net_val['max_drawdown']:.4f}, "
            f"CumRet={buy_hold_net_val['cumulative_return']:.4f} | "
            f"Test Sharpe={buy_hold_net_test['sharpe']:.4f}, "
            f"MDD={buy_hold_net_test['max_drawdown']:.4f}, "
            f"CumRet={buy_hold_net_test['cumulative_return']:.4f}"
        )

        print(
            "  Random baseline (same long-rate): "
            f"Test p95 Sharpe={random_baseline_test['sharpe_p95']:.4f}, "
            f"p95 CumRet={random_baseline_test['cumulative_return_p95']:.4f}"
        )

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
                # Experiment 9b
                "costs_and_baselines": {
                    "trading_days_per_year": int(TRADING_DAYS_PER_YEAR),
                    "cost_bps": float(COST_BPS),
                    "slippage_bps": float(SLIPPAGE_BPS),

                    "strategy_long_rate": {
                        "val": float(val_long_rate),
                        "test": float(test_long_rate),
                    },

                    "strategy_net": {
                        "val": strategy_net_val,
                        "test": strategy_net_test,
                    },
                    "buy_hold_net": {
                        "val": buy_hold_net_val,
                        "test": buy_hold_net_test,
                    },
                    "random_baseline_summary": {
                        "val": random_baseline_val,
                        "test": random_baseline_test,
                    },
                },
            },
        }

        # Also save just the sweep as a separate JSON for convenience
        sweep_path = os.path.join(
            EXPERIMENTS_DIR, f"{eval_id}_up_vs_rest_threshold_sweep.json"
        )
        utils.save_json(sweep_records, sweep_path)
        print(f"[evaluate_tft] Threshold sweep saved to {sweep_path}")

    elif TASK_TYPE in ("quantile_forecast", "regression_quantile"):
        print(
            f"[evaluate_tft] TASK_TYPE='{TASK_TYPE}' -> multi-horizon quantile forecast "
            f"(H={FORECAST_HORIZON}, QUANTILES={QUANTILES}, output_size={FORECAST_HORIZON * N_QUANTILES})."
        )

        H = int(getattr(cfg, "FORECAST_HORIZON", FORECAST_HORIZON))
        quantiles = list(getattr(cfg, "QUANTILES", QUANTILES))
        signal_h = int(getattr(cfg, "SIGNAL_HORIZON", SIGNAL_HORIZON))
        if signal_h < 1 or signal_h > H:
            raise ValueError(f"SIGNAL_HORIZON={signal_h} must be in [1, {H}]")
        signal_idx = signal_h - 1

        eps = float(getattr(cfg, "SCORE_EPS", SCORE_EPS))

        # --- Inference ---
        print("[evaluate_tft] Running inference on validation set (quantile).")
        y_val_true, y_val_pred, val_pinball = run_inference_quantile(
            model, val_loader, device, quantiles
        )

        print("[evaluate_tft] Running inference on test set (quantile).")
        y_test_true, y_test_pred, test_pinball = run_inference_quantile(
            model, test_loader, device, quantiles
        )

        # --- Forecast metrics (median MAE at SIGNAL_HORIZON) ---
        q50_i = _nearest_quantile_index(quantiles, 0.5)
        val_mae = float(np.mean(np.abs(y_val_pred[:, signal_idx, q50_i] - y_val_true[:, signal_idx])))
        test_mae = float(np.mean(np.abs(y_test_pred[:, signal_idx, q50_i] - y_test_true[:, signal_idx])))

        print("[evaluate_tft] Validation forecast metrics:")
        print(f"  Val pinball: {val_pinball:.6f}")
        print(f"  Val MAE@{signal_h}: {val_mae:.6f}")

        print("[evaluate_tft] Test forecast metrics:")
        print(f"  Test pinball: {test_pinball:.6f}")
        print(f"  Test MAE@{signal_h}: {test_mae:.6f}")

        # --- Build score series (full resolution; later downsample for trading if needed) ---
        score_val, mu_val, iqr_val, q_idx = _build_score_mu_iqr(
            y_pred=y_val_pred, quantiles=quantiles, signal_idx=signal_idx, eps=eps
        )
        score_test, mu_test, iqr_test, _ = _build_score_mu_iqr(
            y_pred=y_test_pred, quantiles=quantiles, signal_idx=signal_idx, eps=eps
        )

        # True returns for trading should come from the true target at SIGNAL_HORIZON
        # (this makes evaluation correct even if SIGNAL_HORIZON != FORECAST_HORIZON).
        val_returns = _to_simple_returns(y_val_true[:, signal_idx])
        test_returns = _to_simple_returns(y_test_true[:, signal_idx])

        # --- Non-overlapping trades ---
        trade_horizon_steps = int(signal_h)
        if NON_OVERLAPPING_TRADES and trade_horizon_steps > 1:
            print(
                "[evaluate_tft] Using non-overlapping trades for trading metrics "
                f"(horizon_steps={trade_horizon_steps})."
            )
            idx = np.arange(0, val_returns.shape[0], trade_horizon_steps)

            val_returns_tr = val_returns[idx]
            score_val_tr = score_val[idx]
            mu_val_tr = mu_val[idx]

            idx_t = np.arange(0, test_returns.shape[0], trade_horizon_steps)
            test_returns_tr = test_returns[idx_t]
            score_test_tr = score_test[idx_t]
            mu_test_tr = mu_test[idx_t]
        else:
            val_returns_tr = val_returns
            test_returns_tr = test_returns
            score_val_tr = score_val
            score_test_tr = score_test
            mu_val_tr = mu_val
            mu_test_tr = mu_test

        # --- Threshold sweep over SCORE (select τ* on val by ACTIVE_SELECTION_METRIC) ---
        # Build threshold grid (absolute or percentile-based)
        grid_mode = str(getattr(cfg, "ACTIVE_THRESHOLD_GRID_MODE", "absolute")).lower().strip()

        raw_grid = list(getattr(cfg, "ACTIVE_THRESHOLD_GRID", ACTIVE_THRESHOLD_GRID)) \
                   or list(getattr(cfg, "SCORE_GRID", SCORE_GRID)) \
                   or [float(EVAL_THRESHOLD)]

        if grid_mode in ("percentile", "percentiles", "pct", "pctl", "quantile", "quantiles"):
            # raw_grid contains percentiles in [0.0, 1.0]
            percentiles = np.array(raw_grid, dtype=float)
            percentiles = np.clip(percentiles, 0.0, 1.0)

            # IMPORTANT: use the same validation scores you actually trade on (non-overlapping)
            finite_scores = np.asarray(score_val_tr, dtype=float)
            finite_scores = finite_scores[np.isfinite(finite_scores)]

            threshold_vals = np.quantile(finite_scores, percentiles)

            # de-duplicate + sort (quantiles can coincide if distribution is tight)
            threshold_grid = sorted(set(float(x) for x in threshold_vals))
        else:
            # raw_grid already contains literal threshold values
            threshold_grid = [float(x) for x in raw_grid]

        selection_metric = str(getattr(cfg, "ACTIVE_SELECTION_METRIC", ACTIVE_SELECTION_METRIC))
        auto_tune = bool(getattr(cfg, "ACTIVE_AUTO_TUNE", ACTIVE_AUTO_TUNE))
        selection_key = selection_metric.lower().strip()

        print("[evaluate_tft] Threshold sweep on score = mu/(iqr+eps) for LONG-only:")
        print(f"  Grid mode: {grid_mode}")
        if grid_mode in ("percentile", "percentiles", "pct", "pctl", "quantile", "quantiles"):
            print(f"  Percentiles: {raw_grid}")
            print(f"  Thresholds:  {threshold_grid}")
        else:
            print(f"  Grid: {threshold_grid}")
        print(f"  Selection metric (val): {selection_metric}")
        print(f"  AUTO_TUNE_THRESHOLD   : {auto_tune}")
        print(f"  Score quantile idx: {q_idx}  (mu=q50, iqr=q90-q10)")

        sweep_records: list[dict[str, Any]] = []

        best_threshold: float | None = None
        best_val_score: float = -np.inf
        best_val_trading: dict[str, float] | None = None
        best_test_trading: dict[str, float] | None = None

        for thr in threshold_grid:
            thr = float(thr)

            # LONG rule (9c): score >= thr AND mu > 0
            pos_val = ((score_val_tr >= thr) & (mu_val_tr > 0)).astype(int)
            pos_test = ((score_test_tr >= thr) & (mu_test_tr > 0)).astype(int)

            val_trading = utils.compute_trading_metrics(
                returns=val_returns_tr,
                positions=pos_val,
                trading_days_per_year=TRADING_DAYS_PER_YEAR,
            )
            test_trading = utils.compute_trading_metrics(
                returns=test_returns_tr,
                positions=pos_test,
                trading_days_per_year=TRADING_DAYS_PER_YEAR,
            )

            if selection_key in val_trading:
                val_score = float(val_trading[selection_key])
            else:
                raise ValueError(
                    f"Unknown selection metric '{selection_metric}'. "
                    f"Not found in trading metrics keys={list(val_trading.keys())}"
                )

            sweep_records.append(
                {
                    "threshold": thr,
                    "val_trading": val_trading,
                    "test_trading": test_trading,
                    "selection_score": val_score,
                    "val_long_rate": float(np.mean(pos_val)),
                    "test_long_rate": float(np.mean(pos_test)),
                }
            )

            if auto_tune and np.isfinite(val_score) and val_score > best_val_score:
                best_val_score = val_score
                best_threshold = thr
                best_val_trading = val_trading
                best_test_trading = test_trading

        # Fallback if autotune disabled/failed
        if (not auto_tune) or (best_threshold is None):
            best_threshold = float(EVAL_THRESHOLD)
            print(
                f"[evaluate_tft] Auto-tuning disabled or failed, "
                f"falling back to EVAL_THRESHOLD={best_threshold:.3f}"
            )
            pos_val = ((score_val_tr >= best_threshold) & (mu_val_tr > 0)).astype(int)
            pos_test = ((score_test_tr >= best_threshold) & (mu_test_tr > 0)).astype(int)
            best_val_trading = utils.compute_trading_metrics(
                returns=val_returns_tr,
                positions=pos_val,
                trading_days_per_year=TRADING_DAYS_PER_YEAR,
            )
            best_test_trading = utils.compute_trading_metrics(
                returns=test_returns_tr,
                positions=pos_test,
                trading_days_per_year=TRADING_DAYS_PER_YEAR,
            )
        else:
            print(
                f"[evaluate_tft] Best threshold on val ({selection_key}): "
                f"{best_threshold:.3f} with score={best_val_score:.6f}"
            )

        thr_star = float(best_threshold)

        # --- Net-of-cost backtest + baselines (same structure as 9b) ---
        pos_val = ((score_val_tr >= thr_star) & (mu_val_tr > 0)).astype(int)
        pos_test = ((score_test_tr >= thr_star) & (mu_test_tr > 0)).astype(int)

        val_long_rate = float(np.mean(pos_val))
        test_long_rate = float(np.mean(pos_test))

        val_costs = utils.apply_costs(pos_val, cost_bps=COST_BPS, slippage_bps=SLIPPAGE_BPS)
        test_costs = utils.apply_costs(pos_test, cost_bps=COST_BPS, slippage_bps=SLIPPAGE_BPS)

        val_net_returns = pos_val.astype(float) * val_returns_tr - val_costs
        test_net_returns = pos_test.astype(float) * test_returns_tr - test_costs

        val_equity = utils.equity_curve(val_returns_tr, pos_val, val_costs)
        test_equity = utils.equity_curve(test_returns_tr, pos_test, test_costs)

        strategy_net_val = utils.compute_backtest_metrics(
            equity=val_equity,
            net_returns=val_net_returns,
            trading_days_per_year=TRADING_DAYS_PER_YEAR,
            positions=pos_val,
        )
        strategy_net_test = utils.compute_backtest_metrics(
            equity=test_equity,
            net_returns=test_net_returns,
            trading_days_per_year=TRADING_DAYS_PER_YEAR,
            positions=pos_test,
        )

        buy_hold_net_val = utils.buy_and_hold_baseline(
            returns=val_returns_tr,
            cost_bps=COST_BPS,
            slippage_bps=SLIPPAGE_BPS,
            trading_days_per_year=TRADING_DAYS_PER_YEAR,
        )
        buy_hold_net_test = utils.buy_and_hold_baseline(
            returns=test_returns_tr,
            cost_bps=COST_BPS,
            slippage_bps=SLIPPAGE_BPS,
            trading_days_per_year=TRADING_DAYS_PER_YEAR,
        )

        random_baseline_val = utils.random_exposure_baseline(
            returns=val_returns_tr,
            long_rate=val_long_rate,
            runs=RANDOM_BASELINE_RUNS,
            seed=RANDOM_SEED,
            cost_bps=COST_BPS,
            slippage_bps=SLIPPAGE_BPS,
            trading_days_per_year=TRADING_DAYS_PER_YEAR,
        )
        random_baseline_test = utils.random_exposure_baseline(
            returns=test_returns_tr,
            long_rate=test_long_rate,
            runs=RANDOM_BASELINE_RUNS,
            seed=RANDOM_SEED,
            cost_bps=COST_BPS,
            slippage_bps=SLIPPAGE_BPS,
            trading_days_per_year=TRADING_DAYS_PER_YEAR,
        )

        # --- Print summaries ---
        print(f"[evaluate_tft] Long-only trading metrics at τ* (signal={signal_h}):")
        print("  Validation:")
        for k, v in (best_val_trading or {}).items():
            print(f"    {k}: {v:.6f}")
        print("  Test:")
        for k, v in (best_test_trading or {}).items():
            print(f"    {k}: {v:.6f}")

        print(
            f"[evaluate_tft] Net-of-cost backtest at τ* "
            f"(cost={COST_BPS}bps, slippage={SLIPPAGE_BPS}bps, annual={TRADING_DAYS_PER_YEAR}):"
        )
        print(
            "  Strategy (net): "
            f"Val Sharpe={strategy_net_val['sharpe']:.4f}, "
            f"MDD={strategy_net_val['max_drawdown']:.4f}, "
            f"CumRet={strategy_net_val['cumulative_return']:.4f} | "
            f"Test Sharpe={strategy_net_test['sharpe']:.4f}, "
            f"MDD={strategy_net_test['max_drawdown']:.4f}, "
            f"CumRet={strategy_net_test['cumulative_return']:.4f}"
        )
        print(
            "  Buy&Hold (net): "
            f"Val Sharpe={buy_hold_net_val['sharpe']:.4f}, "
            f"MDD={buy_hold_net_val['max_drawdown']:.4f}, "
            f"CumRet={buy_hold_net_val['cumulative_return']:.4f} | "
            f"Test Sharpe={buy_hold_net_test['sharpe']:.4f}, "
            f"MDD={buy_hold_net_test['max_drawdown']:.4f}, "
            f"CumRet={buy_hold_net_test['cumulative_return']:.4f}"
        )
        print(
            "  Random baseline (same long-rate): "
            f"Test p95 Sharpe={random_baseline_test['sharpe_p95']:.4f}, "
            f"p95 CumRet={random_baseline_test['cumulative_return_p95']:.4f}"
        )

        # ------------------------------------------------------------------
        # 9c Reporting pack: replace score histograms
        # ------------------------------------------------------------------

        # Step 2.1 — Create 4 plot file paths (we generate the first two in Steps 2.2–2.3)
        forecast_band_path = os.path.join(PLOTS_DIR, f"{eval_id}_test_forecast_band.png")
        threshold_sweep_plot_path = os.path.join(PLOTS_DIR, f"{eval_id}_threshold_sweep.png")
        equity_curve_path = os.path.join(PLOTS_DIR, f"{eval_id}_test_equity_curve.png")
        signal_confusion_path = os.path.join(PLOTS_DIR, f"{eval_id}_test_signal_confusion.png")

        # Step 2.2 — Build the forecast series for plotting (test only, at signal horizon)
        y_true_sig = y_test_true[:, signal_idx]
        y_pred_sig_q = y_test_pred[:, signal_idx, :]  # (N, Q)

        utils.plot_quantile_forecast_band(
            y_true=y_true_sig,
            y_pred_q=y_pred_sig_q,
            quantiles=quantiles,
            out_path=forecast_band_path,
            title=f"Test forecast band (signal={signal_h})",
            q_low=0.1,
            q_mid=0.5,
            q_high=0.9,
            window=500,
        )

        # Step 2.3 — Threshold sweep plot (from existing sweep_records)
        utils.plot_threshold_sweep(
            sweep_records=sweep_records,
            out_path=threshold_sweep_plot_path,
            title=f"Threshold sweep (val {selection_metric})",
            selected_threshold=thr_star,
        )

        # Step 2.4 — Equity curve plot (test only)
        bh_pos_test = np.ones_like(test_returns_tr, dtype=int)
        bh_costs_test = utils.apply_costs(
            bh_pos_test,
            cost_bps=COST_BPS,
            slippage_bps=SLIPPAGE_BPS,
        )
        bh_equity_test = utils.equity_curve(test_returns_tr, bh_pos_test, bh_costs_test)

        utils.plot_equity_curves(
            equity_strategy=test_equity,
            equity_buy_hold=bh_equity_test,
            out_path=equity_curve_path,
            title=f"Test equity curves (net)  τ*={thr_star:.3f}",
        )

        # Step 2.5 — Signal confusion matrix (2×2)
        y_true_up = (test_returns_tr > 0).astype(int)
        y_pred_long = (pos_test == 1).astype(int)

        utils.plot_signal_confusion_matrix(
            actual_up=y_true_up,
            model_long=y_pred_long,
            out_path=signal_confusion_path,
            title=f"Test signal confusion (τ*={thr_star:.3f})",
        )

        # Step 2.6 — Write the 2 tables
        forecast_table_path = os.path.join(EXPERIMENTS_DIR, f"{eval_id}_forecast_table.csv")
        trading_table_path = os.path.join(EXPERIMENTS_DIR, f"{eval_id}_trading_table.csv")

        # Optional: coverage of [q10, q90] at signal horizon on test
        coverage_q10_q90_test: float | None = None
        if isinstance(q_idx, dict) and ("q10" in q_idx) and ("q90" in q_idx):
            q10_i = int(q_idx["q10"])
            q90_i = int(q_idx["q90"])
            if 0 <= q10_i < y_pred_sig_q.shape[1] and 0 <= q90_i < y_pred_sig_q.shape[1]:
                lo = y_pred_sig_q[:, q10_i]
                hi = y_pred_sig_q[:, q90_i]
                coverage_q10_q90_test = float(np.mean((y_true_sig >= lo) & (y_true_sig <= hi)))

        # Number of long trades (non-overlapping if NON_OVERLAPPING_TRADES is enabled)
        n_trades_val = int(np.sum(pos_val == 1))
        n_trades_test = int(np.sum(pos_test == 1))

        # Forecast table: val/test pinball + MAE@signal (+ optional test coverage)
        forecast_rows = [
            {
                "split": "val",
                "H": int(H),
                "signal_horizon": int(signal_h),
                "pinball": float(val_pinball),
                "mae_at_signal": float(val_mae),
            },
            {
                "split": "test",
                "H": int(H),
                "signal_horizon": int(signal_h),
                "pinball": float(test_pinball),
                "mae_at_signal": float(test_mae),
                "coverage_q10_q90": coverage_q10_q90_test if coverage_q10_q90_test is not None else "",
            },
        ]
        utils.save_table_csv(forecast_rows, forecast_table_path)

        # Trading table: τ*, long-rate, net Sharpe/MDD/CumRet, hit-rate, number of trades (val/test)
        trading_rows = [
            {
                "split": "val",
                "tau_star": float(thr_star),
                "signal_horizon": int(signal_h),
                "long_rate": float(val_long_rate),
                "n_trades": n_trades_val,
                "sharpe_net": float(strategy_net_val.get("sharpe", 0.0)),
                "max_drawdown_net": float(strategy_net_val.get("max_drawdown", 0.0)),
                "cumulative_return_net": float(strategy_net_val.get("cumulative_return", 0.0)),
                "hit_rate_net": float(strategy_net_val.get("hit_rate", 0.0)),
                "avg_trade_return_net": float(strategy_net_val.get("avg_trade_return", 0.0)),
            },
            {
                "split": "test",
                "tau_star": float(thr_star),
                "signal_horizon": int(signal_h),
                "long_rate": float(test_long_rate),
                "n_trades": n_trades_test,
                "sharpe_net": float(strategy_net_test.get("sharpe", 0.0)),
                "max_drawdown_net": float(strategy_net_test.get("max_drawdown", 0.0)),
                "cumulative_return_net": float(strategy_net_test.get("cumulative_return", 0.0)),
                "hit_rate_net": float(strategy_net_test.get("hit_rate", 0.0)),
                "avg_trade_return_net": float(strategy_net_test.get("avg_trade_return", 0.0)),
            },
        ]
        utils.save_table_csv(trading_rows, trading_table_path)

        # --- Save outputs ---
        metrics_out.update(
            {
                "task_type": TASK_TYPE,
                "forecast": {
                    "H": int(H),
                    "quantiles": [float(q) for q in quantiles],
                    "signal_horizon": int(signal_h),
                    "q_indices": q_idx,
                    "val_pinball": float(val_pinball),
                    "test_pinball": float(test_pinball),
                    "val_mae_at_signal": float(val_mae),
                    "test_mae_at_signal": float(test_mae),
                },
                "signal": {
                    "score_definition": "score = mu / (iqr + eps), long if score>=thr and mu>0",
                    "eps": float(eps),
                    "selected_threshold": float(thr_star),
                    "threshold_sweep": sweep_records,
                    "val_trading_metrics": best_val_trading or {},
                    "test_trading_metrics": best_test_trading or {},
                },
                "costs_and_baselines": {
                    "trading_days_per_year": int(TRADING_DAYS_PER_YEAR),
                    "cost_bps": float(COST_BPS),
                    "slippage_bps": float(SLIPPAGE_BPS),
                    "strategy_long_rate": {"val": float(val_long_rate), "test": float(test_long_rate)},
                    "strategy_net": {"val": strategy_net_val, "test": strategy_net_test},
                    "buy_hold_net": {"val": buy_hold_net_val, "test": buy_hold_net_test},
                    "random_baseline_summary": {"val": random_baseline_val, "test": random_baseline_test},
                },
            }
        )

        sweep_path = os.path.join(EXPERIMENTS_DIR, f"{eval_id}_score_threshold_sweep.json")
        utils.save_json(sweep_records, sweep_path)

    else:
        raise NotImplementedError(
            f"TASK_TYPE='{TASK_TYPE}' is not supported yet in evaluate_tft.py"
        )

    # ------------------------------------------------------------------
    # 4. Save evaluation metrics to JSON
    # ------------------------------------------------------------------
    metrics_path = os.path.join(EXPERIMENTS_DIR, f"{eval_id}_metrics.json")
    utils.save_json(metrics_out, metrics_path)
    if TASK_TYPE == "quantile_forecast":
        print("[evaluate_tft] 9c summary (quantile_forecast)")

        cov_txt = f", coverage_q10_q90={coverage_q10_q90_test:.3f}" if coverage_q10_q90_test is not None else ""
        print(
            f"  Forecast: val pinball={val_pinball:.6f}, MAE@{signal_h}={val_mae:.6f} | "
            f"test pinball={test_pinball:.6f}, MAE@{signal_h}={test_mae:.6f}{cov_txt}"
        )

        print(
            f"  Trading (net, test): τ*={thr_star:.3f}, long_rate={test_long_rate:.3f}, trades={n_trades_test} | "
            f"Sharpe={strategy_net_test['sharpe']:.4f}, MDD={strategy_net_test['max_drawdown']:.4f}, "
            f"CumRet={strategy_net_test['cumulative_return']:.4f}, Hit={strategy_net_test['hit_rate']:.3f}"
        )
        print(
            f"  Buy&Hold (net, test): Sharpe={buy_hold_net_test['sharpe']:.4f}, "
            f"MDD={buy_hold_net_test['max_drawdown']:.4f}, CumRet={buy_hold_net_test['cumulative_return']:.4f}"
        )

        print("  Saved artifacts:")
        for p in [
            forecast_band_path,
            threshold_sweep_plot_path,
            equity_curve_path,
            signal_confusion_path,
            forecast_table_path,
            trading_table_path,
            sweep_path,
            metrics_path,
        ]:
            print(f"    - {p}")
    else:
        print(f"[evaluate_tft] Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()