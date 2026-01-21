"""
utils.py

Utility functions for the TFT BTC forecasting project.

This module provides:
- Reproducibility helpers (set_seed)
- Device selection (get_device)
- Quantile-forecast helpers (pinball loss, MAE on median at horizon)
- Trading/backtest utilities (positions, costs, equity curve, metrics, baselines)
- Plotting/reporting helpers (training curves, forecast bands, threshold sweeps, equity curves, signal confusion matrix)
"""

from __future__ import annotations
import os
import random
import time
import json
from typing import Any, Dict, Sequence, Optional
import numpy as np
import torch
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

def guard_dir_missing_or_empty(path: str, *, display_name: str = "standard") -> None:
    """
    Safety guard:
    - If `path` does not exist: create it.
    - If `path` exists and is empty: OK.
    - If `path` exists and is non-empty: raise (refuse to run).
    - If `path` exists but is not a directory: raise.
    """
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise RuntimeError(
                f"Refusing to run: '{display_name}' path exists but is not a directory: {path}"
            )

        # Non-empty means *any* entry exists (files, folders, hidden files, etc.)
        with os.scandir(path) as it:
            if any(True for _ in it):
                raise RuntimeError(
                    f"Refusing to run: '{display_name}' folder already exists and is non-empty: {path}\n"
                    f"Rename/move it (catalog it) and re-run, so a fresh '{display_name}' can be created."
                )
    else:
        os.makedirs(path, exist_ok=False)

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
# 4. TRADING HELPERS
# ============================================================

def pinball_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    quantiles: Sequence[float],
) -> torch.Tensor:
    """
    Pinball (quantile) loss for multi-horizon forecasting.

    Shapes:
      y_true: (B, H)
      y_pred: (B, H, Q)
      quantiles: length Q (e.g. [0.1, 0.5, 0.9])

    Returns:
      scalar tensor (mean over batch, horizons, quantiles)
    """
    if not torch.is_tensor(y_pred):
        y_pred = torch.as_tensor(y_pred)

    # Put y_true on same device/dtype as predictions (important for training)
    if not torch.is_tensor(y_true):
        y_true = torch.as_tensor(y_true, device=y_pred.device, dtype=y_pred.dtype)
    else:
        y_true = y_true.to(device=y_pred.device, dtype=y_pred.dtype)

    if y_true.ndim != 2:
        raise ValueError(f"pinball_loss expects y_true to have shape (B,H), got {tuple(y_true.shape)}")
    if y_pred.ndim != 3:
        raise ValueError(f"pinball_loss expects y_pred to have shape (B,H,Q), got {tuple(y_pred.shape)}")

    B, H = y_true.shape
    if y_pred.shape[0] != B or y_pred.shape[1] != H:
        raise ValueError(
            f"pinball_loss shape mismatch: y_true is (B,H)=({B},{H}) "
            f"but y_pred is {tuple(y_pred.shape)}"
        )

    Q = y_pred.shape[2]
    if len(quantiles) != Q:
        raise ValueError(
            f"pinball_loss quantiles mismatch: got {len(quantiles)} quantiles "
            f"but y_pred has Q={Q}"
        )

    q = torch.tensor(quantiles, device=y_pred.device, dtype=y_pred.dtype).view(1, 1, Q)

    # err = y - y_hat
    err = y_true.unsqueeze(-1) - y_pred

    # max((q-1)*err, q*err)
    loss = torch.maximum((q - 1.0) * err, q * err)
    return loss.mean()

def _closest_quantile_index(quantiles: Sequence[float], target: float = 0.5) -> int:
    """
    Return index of quantile closest to `target` (default median=0.5).
    """
    if len(quantiles) == 0:
        raise ValueError("quantiles must be non-empty.")
    diffs = [abs(float(q) - target) for q in quantiles]
    return int(np.argmin(diffs))

def nearest_quantile_index(quantiles: Sequence[float], q: float = 0.5) -> int:
    """
    Return index of the quantile value closest to q.

    This is the project-wide public helper; internally it reuses _closest_quantile_index
    to avoid duplicating logic.
    """
    return _closest_quantile_index(quantiles, target=q)

def mae_on_median_at_horizon(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    quantiles: Sequence[float],
    horizon_step: int | None = None,
) -> float:
    """
    Compute MAE using the median (q≈0.5) forecast.

    Shapes:
      y_true: (B,H)
      y_pred: (B,H,Q)

    Args:
      horizon_step:
        If None -> use last step (H).
        If provided -> 1-based horizon step (e.g. 24 means the 24h-ahead step).

    Returns:
      float MAE
    """
    if not torch.is_tensor(y_pred):
        y_pred = torch.as_tensor(y_pred)
    if not torch.is_tensor(y_true):
        y_true = torch.as_tensor(y_true, device=y_pred.device, dtype=y_pred.dtype)
    else:
        y_true = y_true.to(device=y_pred.device, dtype=y_pred.dtype)

    if y_true.ndim != 2 or y_pred.ndim != 3:
        raise ValueError(
            f"mae_on_median_at_horizon expects y_true (B,H) and y_pred (B,H,Q), "
            f"got y_true={tuple(y_true.shape)} y_pred={tuple(y_pred.shape)}"
        )

    B, H = y_true.shape
    q_idx = _closest_quantile_index(quantiles, target=0.5)

    if horizon_step is None:
        h_idx = H - 1
    else:
        if horizon_step < 1 or horizon_step > H:
            raise ValueError(f"horizon_step must be in [1, {H}], got {horizon_step}")
        h_idx = horizon_step - 1

    y_med = y_pred[:, h_idx, q_idx]
    err = torch.abs(y_med - y_true[:, h_idx])
    return float(err.mean().item())

def compute_trading_metrics(
    returns: Any,
    positions: Any,
    trading_days_per_year: int = 252,
) -> Dict[str, float]:
    """
    Compute simple long-only trading metrics for a given return and position series.

    Args:
        returns:
            1D array-like of forward returns aligned to the trading decision step (e.g. future_return_H).
        positions:
            1D array-like of {0, 1} positions. Must have same shape as returns.
        trading_days_per_year:
            Used for annualizing the Sharpe ratio.
            For crypto with 24h decision steps, this is typically 365 (see config.TRADING_DAYS_PER_YEAR).

    Returns:
        Dict with:
            - avg_daily_return: mean strategy return per decision step (historical name; 'daily' when step=24h)
            - cumulative_return: compounded return over the period
            - sharpe: annualized Sharpe using `trading_days_per_year`
            - hit_ratio: fraction of positive returns when in position
            - avg_return_in_position: mean return conditional on being in position
            - in_position_rate: fraction of steps with non-zero position
    """
    r = _to_numpy_1d(returns)
    p = _to_numpy_1d(positions)

    if r.shape != p.shape:
        raise ValueError(
            f"returns and positions must have the same shape, "
            f"got {r.shape} vs {p.shape}"
        )

    if r.size == 0:
        return {
            "avg_daily_return": 0.0,
            "cumulative_return": 0.0,
            "sharpe": 0.0,
            "hit_ratio": 0.0,
            "avg_return_in_position": 0.0,
        }

    # Strategy per-step returns: only earn the return when in position
    strat_r = r * p

    # Average per-step return of the strategy
    avg_daily = float(strat_r.mean())

    # Cumulative return over the period (product of (1 + r_t) - 1)
    cumulative = float(np.prod(1.0 + strat_r) - 1.0)

    # Sharpe ratio (simple, using sample std dev)
    std_daily = float(strat_r.std(ddof=1))
    if std_daily > 0.0:
        sharpe = (avg_daily / std_daily) * np.sqrt(trading_days_per_year)
    else:
        sharpe = 0.0

    # In-position mask (works for long-only and long/short: position != 0)
    in_pos = (p != 0)
    in_position_rate = float(in_pos.mean())

    # Hit ratio (ONLY when in a position) — avoids counting "no-trade" zeros
    if in_pos.any():
        hit_ratio = float((strat_r[in_pos] > 0).mean())
        avg_return_in_position = float(strat_r[in_pos].mean())
    else:
        hit_ratio = 0.0
        avg_return_in_position = 0.0

    return {
        "avg_daily_return": avg_daily,
        "cumulative_return": cumulative,
        "sharpe": sharpe,
        "hit_ratio": hit_ratio,
        "avg_return_in_position": avg_return_in_position,
        "in_position_rate": in_position_rate
    }

def _bps_to_return(bps: float) -> float:
    """Convert basis points to decimal return. 1 bp = 0.0001."""
    return float(bps) / 10000.0


def apply_costs(
    positions: Any,
    cost_bps: float,
    slippage_bps: float,
) -> np.ndarray:
    """
    Compute per-step transaction cost as a return drag based on position changes.

    Costs are charged on:
      - entry  (0 -> 1)
      - exit   (1 -> 0)

    Args:
        positions: 1D array-like of {0,1}.
        cost_bps: fee/commission in basis points.
        slippage_bps: slippage in basis points.

    Returns:
        1D np.ndarray of costs (same length as positions), in decimal returns.
    """
    p = _to_numpy_1d(positions).astype(int)
    if p.size == 0:
        return np.asarray(p, dtype=float)

    per_side = _bps_to_return(cost_bps + slippage_bps)

    prev = np.concatenate(([0], p[:-1]))
    changed = (p != prev)

    # Charge cost whenever position changes (entry or exit)
    costs = changed.astype(float) * per_side
    return costs


def equity_curve(
    returns: Any,
    positions: Any,
    costs: Any,
) -> np.ndarray:
    """
    Build an equity curve from returns, positions, and per-step costs.

    net_r[t] = positions[t] * returns[t] - costs[t]
    equity[t+1] = equity[t] * (1 + net_r[t])

    Returns:
        Equity array of length (N+1), starting at 1.0
    """
    r = _to_numpy_1d(returns)
    p = _to_numpy_1d(positions).astype(float)
    c = _to_numpy_1d(costs).astype(float)

    if not (r.shape == p.shape == c.shape):
        raise ValueError(f"returns, positions, costs must match shapes. Got {r.shape}, {p.shape}, {c.shape}")

    equity = np.empty(r.size + 1, dtype=float)
    equity[0] = 1.0

    net_r = p * r - c
    for i in range(r.size):
        equity[i + 1] = equity[i] * (1.0 + net_r[i])

    return equity


def compute_backtest_metrics(
    equity: Any,
    net_returns: Any,
    trading_days_per_year: int = 365,
    positions: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Compute backtest metrics from equity and net returns.

    Includes at least:
      - cumulative return
      - Sharpe (annualized)
      - CAGR
      - volatility (annualized)
      - max drawdown
      - hit rate
      - avg trade return

    If `positions` is provided, hit rate / avg trade return are computed on steps where position==1.
    Otherwise they are computed on non-zero net returns.
    """
    eq = np.asarray(equity, dtype=float)
    nr = _to_numpy_1d(net_returns).astype(float)

    if eq.ndim != 1:
        raise ValueError(f"equity must be 1D, got shape {eq.shape}")
    if eq.size != nr.size + 1:
        raise ValueError(f"equity must have length N+1 where N=len(net_returns). Got {eq.size} vs {nr.size}")

    if nr.size == 0:
        return {
            "cumulative_return": 0.0,
            "sharpe": 0.0,
            "cagr": 0.0,
            "volatility": 0.0,
            "max_drawdown": 0.0,
            "hit_rate": 0.0,
            "avg_trade_return": 0.0,
        }

    # Cumulative return
    cumulative_return = float(eq[-1] - 1.0)

    # Volatility + Sharpe
    mean_r = float(nr.mean())
    std_r = float(nr.std(ddof=1)) if nr.size > 1 else 0.0
    volatility = float(std_r * np.sqrt(trading_days_per_year)) if std_r > 0 else 0.0
    sharpe = float((mean_r / std_r) * np.sqrt(trading_days_per_year)) if std_r > 0 else 0.0

    # CAGR: assumes each step represents 1 / trading_days_per_year years (e.g. 24h non-overlapping steps on crypto -> 365).
    years = nr.size / float(trading_days_per_year)
    if years > 0 and eq[-1] > 0:
        cagr = float(eq[-1] ** (1.0 / years) - 1.0)
    else:
        cagr = 0.0

    # Max drawdown
    running_max = np.maximum.accumulate(eq)
    drawdowns = (eq / running_max) - 1.0
    max_drawdown = float(-drawdowns.min())  # positive magnitude

    # Hit rate + avg trade return
    if positions is not None:
        p = _to_numpy_1d(positions).astype(int)
        if p.shape != nr.shape:
            raise ValueError(f"positions must have same shape as net_returns. Got {p.shape} vs {nr.shape}")
        trade_mask = (p == 1)
    else:
        trade_mask = (nr != 0)

    if np.any(trade_mask):
        trade_returns = nr[trade_mask]
        hit_rate = float((trade_returns > 0).mean())
        avg_trade_return = float(trade_returns.mean())
    else:
        hit_rate = 0.0
        avg_trade_return = 0.0

    return {
        "cumulative_return": cumulative_return,
        "sharpe": sharpe,
        "cagr": cagr,
        "volatility": volatility,
        "max_drawdown": max_drawdown,
        "hit_rate": hit_rate,
        "avg_trade_return": avg_trade_return,
    }


def buy_and_hold_baseline(
    returns: Any,
    cost_bps: float,
    slippage_bps: float,
    trading_days_per_year: int = 365,
) -> Dict[str, float]:
    """
    Buy & hold baseline on the same trade-step return series.

    Equivalent to staying long for every decision step (positions = 1).
    This is the natural comparator for any long/flat timing strategy.

    Cost is charged once on entry (no forced exit by default).
    """
    r = _to_numpy_1d(returns)
    p = np.ones_like(r, dtype=int)

    costs = apply_costs(p, cost_bps=cost_bps, slippage_bps=slippage_bps)
    net_r = p.astype(float) * r - costs
    eq = equity_curve(r, p, costs)

    return compute_backtest_metrics(eq, net_r, trading_days_per_year=trading_days_per_year, positions=p)


def random_exposure_baseline(
    returns: Any,
    long_rate: float,
    runs: int,
    seed: int,
    cost_bps: float,
    slippage_bps: float,
    trading_days_per_year: int = 365,
) -> Dict[str, float]:
    """
    Random exposure baseline with the SAME long-rate as the strategy.

    Returns summary stats of Sharpe and cumulative return distribution:
      mean/median/p95
    """
    r = _to_numpy_1d(returns)
    n = r.size
    if n == 0:
        return {
            "runs": int(runs),
            "long_rate": float(long_rate),
            "sharpe_mean": 0.0,
            "sharpe_median": 0.0,
            "sharpe_p95": 0.0,
            "cumulative_return_mean": 0.0,
            "cumulative_return_median": 0.0,
            "cumulative_return_p95": 0.0,
        }

    lr = float(np.clip(long_rate, 0.0, 1.0))
    rng = np.random.default_rng(int(seed))

    sharpe_vals = np.empty(int(runs), dtype=float)
    cumret_vals = np.empty(int(runs), dtype=float)

    n_long = int(round(lr * n))
    base = np.zeros(n, dtype=int)
    base[:n_long] = 1

    for i in range(int(runs)):
        p = base.copy()
        rng.shuffle(p)

        costs = apply_costs(p, cost_bps=cost_bps, slippage_bps=slippage_bps)
        net_r = p.astype(float) * r - costs
        eq = equity_curve(r, p, costs)
        m = compute_backtest_metrics(eq, net_r, trading_days_per_year=trading_days_per_year, positions=p)

        sharpe_vals[i] = m["sharpe"]
        cumret_vals[i] = m["cumulative_return"]

    return {
        "runs": int(runs),
        "long_rate": float(lr),
        "sharpe_mean": float(np.mean(sharpe_vals)),
        "sharpe_median": float(np.median(sharpe_vals)),
        "sharpe_p95": float(np.percentile(sharpe_vals, 95)),
        "cumulative_return_mean": float(np.mean(cumret_vals)),
        "cumulative_return_median": float(np.median(cumret_vals)),
        "cumulative_return_p95": float(np.percentile(cumret_vals, 95)),
    }


# ============================================================
# 5. TRAINING CURVE PLOTTING
# ============================================================

def plot_training_curves(history: Dict[str, Sequence[float]], out_path: str) -> None:
    """
    Plot training/validation curves from a history dict (quantile-only).

    Expected keys (produced by train_tft.py):
      - epoch (optional)
      - train_loss, val_loss
      - train_mae,  val_mae

    If MAE keys are missing, only the loss subplot is shown.
    """
    if not history:
        raise ValueError("History is empty – nothing to plot.")

    epochs = history.get("epoch")
    if not epochs:
        n_epochs = len(history.get("train_loss", []))
        epochs = list(range(1, n_epochs + 1))

    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax1, ax2 = axes

    # ---- Loss subplot ----
    if "train_loss" in history and "val_loss" in history and len(history["train_loss"]) > 0:
        ax1.plot(epochs, history["train_loss"], label="Train loss")
        ax1.plot(epochs, history["val_loss"], label="Val loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Pinball loss")
        ax1.set_title("Loss over epochs")
        ax1.legend(loc="best")
    else:
        ax1.set_visible(False)

    # ---- MAE subplot ----
    if "train_mae" in history and "val_mae" in history and len(history["train_mae"]) > 0:
        ax2.plot(epochs, history["train_mae"], label="Train MAE")
        ax2.plot(epochs, history["val_mae"], label="Val MAE")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MAE")
        ax2.set_title("MAE over epochs")
        ax2.legend(loc="best")
    else:
        ax2.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ============================================================
# 8. REPORTING PACK HELPERS
# ============================================================

def save_table_csv(rows: Sequence[Dict[str, Any]], out_path: str) -> None:
    """
    Save a list of dictionaries as a CSV table.

    This is intentionally lightweight (no pandas dependency) because it is
    meant for thesis/report artifacts.

    Args:
        rows:
            Sequence of row dicts. All keys found across all rows become
            columns in the CSV.
        out_path:
            Output CSV path.
    """
    if rows is None:
        raise ValueError("rows must not be None")
    rows_list = list(rows)
    if len(rows_list) == 0:
        raise ValueError("rows is empty; refusing to write an empty table.")

    # Collect columns in a stable order: keys of first row, then any new keys in later rows
    fieldnames: list[str] = []
    seen = set()
    for k in rows_list[0].keys():
        fieldnames.append(str(k))
        seen.add(str(k))

    for r in rows_list[1:]:
        for k in r.keys():
            ks = str(k)
            if ks not in seen:
                fieldnames.append(ks)
                seen.add(ks)

    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        import csv  # local import keeps top-level import changes minimal
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows_list:
            # Convert numpy scalars to Python scalars for nicer CSV output
            clean_row = {}
            for k in fieldnames:
                v = r.get(k, "")
                if isinstance(v, (np.generic,)):
                    v = v.item()
                clean_row[k] = v
            writer.writerow(clean_row)

def plot_quantile_forecast_band(
    y_true: Any,
    y_pred_q: Any,
    quantiles: Sequence[float],
    out_path: str,
    title: str = "Forecast band",
    q_low: float = 0.1,
    q_mid: float = 0.5,
    q_high: float = 0.9,
    window: int | None = 500,
) -> None:
    """
    Plot actual series against predicted median with a (q_low, q_high) uncertainty band.

    Expected shapes:
      - y_true: (N,)
      - y_pred_q: (N, Q)  where Q = len(quantiles)

    If window is provided, plot only the LAST `window` points to keep figures readable.
    """
    y_true_arr = _to_numpy_1d(y_true)

    if isinstance(y_pred_q, torch.Tensor):
        y_pred_arr = y_pred_q.detach().cpu().numpy()
    else:
        y_pred_arr = np.asarray(y_pred_q)

    if y_pred_arr.ndim != 2:
        raise ValueError(f"y_pred_q must have shape (N,Q), got {y_pred_arr.shape}")

    if y_pred_arr.shape[0] != y_true_arr.shape[0]:
        raise ValueError(
            f"y_true and y_pred_q must have same length, got "
            f"{y_true_arr.shape[0]} vs {y_pred_arr.shape[0]}"
        )

    low_i = _closest_quantile_index(quantiles, q_low)
    mid_i = _closest_quantile_index(quantiles, q_mid)
    high_i = _closest_quantile_index(quantiles, q_high)

    y_low = y_pred_arr[:, low_i]
    y_mid = y_pred_arr[:, mid_i]
    y_high = y_pred_arr[:, high_i]

    if window is not None and window > 0 and y_true_arr.shape[0] > window:
        y_true_arr = y_true_arr[-window:]
        y_low = y_low[-window:]
        y_mid = y_mid[-window:]
        y_high = y_high[-window:]

    x = np.arange(y_true_arr.shape[0])

    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y_true_arr, label="Actual")
    ax.plot(x, y_mid, label=f"Pred q{q_mid:g}")
    ax.fill_between(x, y_low, y_high, alpha=0.2, label=f"Band q{q_low:g}–q{q_high:g}")

    ax.set_title(title)
    ax.set_xlabel("Time index (windowed)")
    ax.set_ylabel("Return")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_threshold_sweep(
    sweep_records: Sequence[Dict[str, Any]],
    out_path: str,
    title: str = "Threshold sweep",
    selected_threshold: float | None = None,
) -> None:
    """
    Plot threshold sweep results.

    Expected record format (quantile_forecast):
      - "threshold": float
      - "selection_score": float (computed in evaluate_tft.py)
      - optional: "val_long_rate": float

    The y-axis is the sweep 'selection_score' (computed in evaluate_tft.py).
    If 'val_long_rate' is present, it is plotted on a secondary axis.
    """
    if sweep_records is None:
        raise ValueError("sweep_records must not be None")
    recs = list(sweep_records)
    if len(recs) == 0:
        raise ValueError("sweep_records is empty; nothing to plot.")

    thr = np.asarray([float(r["threshold"]) for r in recs], dtype=float)
    score = np.asarray([float(r.get("selection_score", np.nan)) for r in recs], dtype=float)

    val_long_rate = np.asarray(
        [float(r.get("val_long_rate", np.nan)) for r in recs],
        dtype=float,
    )

    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(thr, score, marker="o", label="Selection score")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Selection score")
    ax1.set_title(title)

    # Optional overlay: validation long-rate if present (9c)
    if np.isfinite(val_long_rate).any():
        ax2 = ax1.twinx()
        ax2.plot(thr, val_long_rate, marker="x", linestyle="--", label="Val long-rate")
        ax2.set_ylabel("Val long-rate")
        # Build a combined legend
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="best")
    else:
        ax1.legend(loc="best")

    if selected_threshold is not None:
        ax1.axvline(float(selected_threshold), linestyle="--", label="Selected threshold")
        # Re-draw legend to include the vline label
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels, loc="best")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_equity_curves(
    equity_strategy: Any,
    equity_buy_hold: Any,
    out_path: str,
    title: str = "Equity curves (net)",
) -> None:
    """
    Plot net equity curve of the strategy vs buy&hold (both as growth of $1).

    Args:
        equity_strategy: (N,) equity values
        equity_buy_hold: (N,) equity values
    """
    eq_s = _to_numpy_1d(equity_strategy).astype(float)
    eq_b = _to_numpy_1d(equity_buy_hold).astype(float)

    if eq_s.shape != eq_b.shape:
        raise ValueError(f"Equity curves must have same shape, got {eq_s.shape} vs {eq_b.shape}")

    x = np.arange(eq_s.shape[0])

    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, eq_s, label="Strategy (net)")
    ax.plot(x, eq_b, label="Buy & hold (net)")
    ax.set_title(title)
    ax.set_xlabel("Time index")
    ax.set_ylabel("Equity (growth of $1)")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_signal_confusion_matrix(
    actual_up: Any,
    model_long: Any,
    out_path: str,
    title: str = "Trading signal confusion matrix",
) -> None:
    """
    Plot a 2x2 confusion matrix for the trading signal (decision vs outcome).

    Rows: realized outcome (actual_up = 1 if realized forward return > 0 else 0)
    Cols: strategy decision (model_long = 1 if long else 0)
    """
    y_true = _to_numpy_1d(actual_up).astype(int)
    y_pred = _to_numpy_1d(model_long).astype(int)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"actual_up and model_long must have the same shape, got {y_true.shape} vs {y_pred.shape}"
        )

    # Force binary {0,1}
    y_true = (y_true != 0).astype(int)
    y_pred = (y_pred != 0).astype(int)

    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    cbar = ax.figure.colorbar(im, ax=ax)

    # Thesis-like labels
    outcome_labels = ["Down", "Up"]
    decision_labels = ["Flat", "Long"]

    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=decision_labels,
        yticklabels=outcome_labels,
        ylabel="Realized outcome",
        xlabel = "Model decision",
        title=title,
    )

    # Adjust the fontsize of labels
    ax.set_xlabel("Model decision", fontsize=14)
    ax.set_ylabel("Realized outcome", fontsize=14)
    ax.set_title(title, fontsize=15)

    ax.tick_params(axis="both", which="major", labelsize=12)
    cbar.ax.tick_params(labelsize=12)

    # Make the numbers readable on dark cells
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            txt_color = "white" if val > thresh else "black"
            ax.text(
                j, i, f"{val}",
                ha="center", va="center",
                fontsize=13, fontweight="bold",
                color=txt_color,
            )

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
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # Convert any NumPy types to Python scalars for JSON compatibility
    def default(o):
        if isinstance(o, (np.generic,)):
            return o.item()
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=default)