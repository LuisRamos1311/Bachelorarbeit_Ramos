"""
evaluate_tft.py

Evaluate a trained Temporal Fusion Transformer (TFT) model on the BTC dataset.

This codebase now supports ONLY:
    - TASK_TYPE="quantile_forecast"

Evaluation:
    - pinball loss on (B, H, Q) quantile outputs
    - MAE on the median (q=0.5) at SIGNAL_HORIZON
    - score-based threshold sweep for trading metrics
    - saves metrics + plots to standard/experiments and standard/plots
"""

import os
from typing import Tuple, Dict, Any
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from final_model.config import (
    # Model + training
    MODEL_CONFIG,
    TRAINING_CONFIG,
    BEST_MODEL_PATH,

    # Paths
    EXPERIMENTS_DIR,
    PLOTS_DIR,

    # Data
    SEQ_LENGTH,
    USE_LOG_RETURNS,
    FREQUENCY,
    DAILY_FEATURE_LAG_DAYS,
    DEBUG_DATA_INTEGRITY,
    DROP_LAST_H_IN_EACH_SPLIT,

    # Quantile forecasting
    FORECAST_HORIZON,
    QUANTILES,

    # Thresholding / trading knobs
    SIGNAL_HORIZON,
    SCORE_EPS,
    ACTIVE_THRESHOLD_GRID,
    ACTIVE_THRESHOLD_GRID_MODE,
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
    Build the uncertainty-aware trading score at `signal_idx` from quantile predictions.

    Definitions (at the signal horizon):
        mu    = q50 (median)
        iqr   = q90 - q10
        score = mu / max(iqr, eps)

    `eps` is used as a lower bound on the denominator to avoid division-by-zero when the
    predicted interval collapses.

    Returns (score, mu, iqr, indices_dict)
    """
    if y_pred.ndim != 3:
        raise ValueError(f"Expected y_pred shape (N,H,Q), got {y_pred.shape}")

    q10_i = utils.nearest_quantile_index(quantiles, 0.1)
    q50_i = utils.nearest_quantile_index(quantiles, 0.5)
    q90_i = utils.nearest_quantile_index(quantiles, 0.9)

    q10 = y_pred[:, signal_idx, q10_i]
    mu = y_pred[:, signal_idx, q50_i]
    q90 = y_pred[:, signal_idx, q90_i]

    iqr = q90 - q10
    iqr_safe = np.where(iqr > eps, iqr, eps)
    score = mu / iqr_safe

    return score, mu, iqr, {"q10": q10_i, "q50": q50_i, "q90": q90_i}

def _to_simple_returns(r: np.ndarray) -> np.ndarray:
    """
    Convert returns to *simple returns* for trading metrics.

    If USE_LOG_RETURNS=True, targets are log-returns, so convert via expm1.
    If USE_LOG_RETURNS=False, they are already simple returns.
    """
    r = np.asarray(r, dtype=float)
    return np.expm1(r) if USE_LOG_RETURNS else r

# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def main() -> None:
    print("[evaluate_tft] Using device:", end=" ")
    device = utils.get_device()
    print(device)

    # Horizon description (quantile-only)
    horizon_steps = int(SIGNAL_HORIZON)

    if FREQUENCY.upper() == "D":
        horizon_desc = f"{horizon_steps}-day-ahead"
    elif FREQUENCY.lower() in {"1h", "h", "hourly"}:
        horizon_desc = f"{horizon_steps}-hour-ahead"
    else:
        horizon_desc = f"{horizon_steps}-step-ahead"

    print(
        "[evaluate_tft] Quantile config:\n"
        f"  USE_LOG_RETURNS        = {USE_LOG_RETURNS}\n"
        f"  FORECAST_HORIZON       = {FORECAST_HORIZON}\n"
        f"  SIGNAL_HORIZON         = {SIGNAL_HORIZON} (=> {horizon_desc})\n"
        f"  QUANTILES              = {list(QUANTILES)}\n"
        f"  FREQUENCY              = '{FREQUENCY}'\n"
        f"  NON_OVERLAPPING_TRADES = {NON_OVERLAPPING_TRADES}\n"
        "  Integrity flags:\n"
        f"    DAILY_FEATURE_LAG_DAYS    = {DAILY_FEATURE_LAG_DAYS}\n"
        f"    DEBUG_DATA_INTEGRITY      = {DEBUG_DATA_INTEGRITY}\n"
        f"    DROP_LAST_H_IN_EACH_SPLIT = {DROP_LAST_H_IN_EACH_SPLIT}"
    )

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

    # Backward-compat note:
    # This codebase now saves weights-only checkpoints (state_dict directly).
    # If you have a legacy dict checkpoint, re-save it as weights-only.
    if isinstance(state, dict) and "state_dict" in state:
        raise RuntimeError(
            "[evaluate_tft] Detected legacy checkpoint format (dict with 'state_dict'). "
            "This codebase now expects weights-only checkpoints. "
            "Re-save the legacy checkpoint as weights-only and point BEST_MODEL_PATH to it."
        )
    model.load_state_dict(state)

    # Create a unique ID for this evaluation run
    eval_id = f"tft_eval_{utils.get_timestamp()}"
    utils.ensure_dir(EXPERIMENTS_DIR)
    utils.ensure_dir(PLOTS_DIR)

    metrics_out: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 3. Quantile-forecast evaluation
    # ------------------------------------------------------------------
    if str(getattr(MODEL_CONFIG, "task_type", "quantile_forecast")) == "quantile_forecast":
        print(
            f"[evaluate_tft] multi-horizon quantile forecast "
            f"(H={FORECAST_HORIZON}, QUANTILES={list(QUANTILES)}, output_size={FORECAST_HORIZON * len(QUANTILES)})."
        )

        H = int(FORECAST_HORIZON)
        quantiles = list(QUANTILES)
        signal_h = int(SIGNAL_HORIZON)
        if signal_h < 1 or signal_h > H:
            raise ValueError(f"SIGNAL_HORIZON={signal_h} must be in [1, {H}]")
        signal_idx = signal_h - 1

        eps = float(SCORE_EPS)

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
        q50_i = utils.nearest_quantile_index(quantiles, 0.5)
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
        grid_mode = str(ACTIVE_THRESHOLD_GRID_MODE).lower().strip()

        raw_grid = list(ACTIVE_THRESHOLD_GRID)
        if not raw_grid:
            raise RuntimeError(
                "No threshold grid provided (ACTIVE_THRESHOLD_GRID empty). "
                "Threshold sweep is required."
            )

        if grid_mode in ("percentile", "percentiles", "pct", "pctl", "quantile", "quantiles"):
            # raw_grid contains percentiles in [0.0, 1.0]
            percentiles = np.array(raw_grid, dtype=float)
            percentiles = np.clip(percentiles, 0.0, 1.0)

            # IMPORTANT: use the same validation scores you actually trade on (non-overlapping)
            finite_scores = np.asarray(score_val_tr, dtype=float)
            finite_scores = finite_scores[np.isfinite(finite_scores)]

            if finite_scores.size == 0:
                raise RuntimeError(
                    "All validation scores are non-finite; cannot build percentile threshold grid (no fallback)."
                )
            threshold_vals = np.quantile(finite_scores, percentiles)

            # de-duplicate + sort (quantiles can coincide if distribution is tight)
            threshold_grid = sorted(set(float(x) for x in threshold_vals))
        else:
            # raw_grid already contains literal threshold values
            threshold_grid = [float(x) for x in raw_grid]

        selection_metric = str(ACTIVE_SELECTION_METRIC)
        auto_tune = bool(ACTIVE_AUTO_TUNE)
        selection_key = selection_metric.lower().strip()

        print("[evaluate_tft] Threshold sweep on score = mu/max(iqr,eps) for LONG-only:")
        print(f"  Grid mode: {grid_mode}")
        if grid_mode in ("percentile", "percentiles", "pct", "pctl", "quantile", "quantiles"):
            print(f"  Percentiles: {raw_grid}")
            print(f"  Thresholds:  {threshold_grid}")
        else:
            print(f"  Grid: {threshold_grid}")
        print(f"  Selection metric (val): {selection_metric}")
        print(f"  ACTIVE_AUTO_TUNE      : {auto_tune}")
        print(f"  Score quantile idx: {q_idx}  (mu=q50, iqr=q90-q10)")

        sweep_records: list[dict[str, Any]] = []

        best_threshold: float | None = None
        best_val_score: float = -np.inf
        best_val_trading: dict[str, float] | None = None
        best_test_trading: dict[str, float] | None = None

        for thr in threshold_grid:
            thr = float(thr)

            # Long-only signal rule: score >= thr AND mu > 0
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

        # Guardrail: threshold selection is required
        if (not auto_tune) or (best_threshold is None):
            raise RuntimeError(
                "Threshold selection is required (no fallback). "
                "Enable ACTIVE_AUTO_TUNE and ensure the threshold sweep produces a valid best_threshold."
            )

        thr_star = float(best_threshold)

        # --- Net-of-cost backtest + baselines ---
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
        # Reporting plots (forecast band, threshold sweep, equity curve, signal confusion)
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
                "task_type": "quantile_forecast",
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
                    "score_definition": "score = mu / max(iqr, eps), long if score>=thr and mu>0",
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

    # ------------------------------------------------------------------
    # 4. Save evaluation metrics to JSON
    # ------------------------------------------------------------------
    metrics_path = os.path.join(EXPERIMENTS_DIR, f"{eval_id}_metrics.json")
    utils.save_json(metrics_out, metrics_path)

if __name__ == "__main__":
    main()