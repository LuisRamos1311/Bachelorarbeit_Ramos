# Experiment 9d (BTC): Static Regime-Conditioned TFT  
**Upgrade over Experiment 9c:** add **static regime features** and inject them as **context** into TFT gating/variable selection to improve cross-regime robustness.

This experiment keeps the **9c core setup** (24h-ahead quantile forecasting + uncertainty-aware threshold trading) and introduces **static covariates** (a per-sample regime vector) to condition the TFT.

---

## 1) Goal

1. Keep the **probabilistic forecasting** framing from 9c:
   - predict **24h-ahead returns** via quantiles (q10/q50/q90)
2. Improve **decision usefulness** (trading signal quality) by conditioning the model on a compact, leak-free **regime snapshot**:
   - volatility / trend / drawdown type descriptors computed from *past* data only
3. Evaluate across two very different regimes:
   - **2022 (bear)**
   - **2024 (bull)**

---

## 2) Data, Features & Targets

### Data (hourly BTC)
- Frequency: **1h**
- Lookback window: **96 hours** (4 days)
- Forecast horizon: **24 hours ahead** (signal horizon = 24)

> The code supports multi-horizon lists, but these runs focus on the 24h horizon (H=24).

### Features

We use three feature “channels”:

#### A) Past covariates (time-varying, length=96)
- OHLCV-derived market features (price + volume)
- Technical indicators (e.g. RSI/MACD/ATR/ROC style)
- Optional external features (e.g. on-chain, sentiment) depending on your config

These are scaled with train-only statistics (MinMax for price/volume-style columns, StandardScaler for indicator-style columns).

#### B) Known-future covariates (time-varying, length=24)
- Calendar / seasonality signals (hour/day/week/month, weekend flags)
- Optional event flags (e.g. halving-window indicators)

These are “known at decision time” and are safe to feed at t+1…t+24.

#### C) NEW in 9d: Static regime covariates (one vector per sample)
A per-sample vector summarizing the regime at the **end of the encoder window** (time t), built from *past-only rolling statistics*. Typical examples:

- rolling volatility (short/medium window)
- rolling trend vs moving average
- rolling drawdown (peak-to-trough)
- rolling volume z-score / volume regime ratios

**Leakage guardrail:** these are computed using only information available up to time t (no negative shifts, no use of future close).

Static features are scaled with a StandardScaler fit on the training split only.

### Target (24h-ahead return)
- Predict 24h log return:

  \[
  y(t) = \log\left(\frac{close(t+24)}{close(t)}\right)
  \]

- Train as **quantile regression** with Q = [0.1, 0.5, 0.9] using **pinball loss**.

---

## 3) Model (TFT) & What Changed in 9d

### 3.1 Core TFT (same idea as 9c)
- Input projections for past/future features
- Variable Selection Network (VSN) to learn feature importances
- LSTM encoder for temporal dynamics
- Self-attention over encoder outputs
- Gated Residual Networks (GRNs) + gating for stable training
- Quantile output head → (q10, q50, q90)

### 3.2 NEW in 9d: Static encoder + context injection
9d adds:

1. **Static encoder**
   - Encodes x_static (B, S) → static_context (B, hidden)

2. **Context-conditioned GRNs + VSN weights**
   - Inject static_context *inside* GRNs via a learned context projection before the nonlinearity
   - Condition VSN weight networks on static_context (regime-dependent feature selection)

Intuition:
- In high-volatility / bear regimes, the model can learn to:
  - widen uncertainty, lower confidence scores, reduce exposure
- In stable / trending regimes, it can:
  - tighten uncertainty and increase confidence when signals are cleaner

---

## 4) Trading Rule & Evaluation (same framework as 9c)

### 4.1 Uncertainty-aware score
At signal horizon H=24:

- μ = q50  
- IQR = q90 − q10  
- score = μ / (IQR + ε)

### 4.2 Long-only decision rule
- Go long if: **(score ≥ τ) AND (μ > 0)**
- Otherwise: stay in cash

We evaluate with:
- **non-overlapping 24h trades** (step size = 24h)
- **net-of-cost** results (transaction cost + slippage applied on position changes)

### 4.3 Reporting pack
Each evaluation produces:
- Training curves (loss + MAE)
- Forecast band plot (actual vs q50 with [q10,q90])
- Threshold sweep plot (validation selection score vs τ)
- Net equity curves (strategy vs buy&hold)
- Confusion matrix (long vs no-long vs realized outcome)
- Two tables:
  - forecast metrics CSV
  - trading metrics CSV
- JSON artifacts for reproducibility (metrics + sweep)

---

## 5) Results: 9c vs 9d (2022 bear, 2024 bull)

### 5.1 Trading results (test, net-of-cost)
| Test year | Exp | τ* | Long rate | # trades | Sharpe (net) | CumRet (net) | MaxDD (net) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2022 (bear) | 9c | 0.0711 | 0.338 | 122 | 0.158 | -0.016 | 0.311 |
| 2022 (bear) | 9d | 0.0402 | 0.036 | 13 | -0.237 | -0.053 | 0.201 |
| 2024 (bull) | 9c | 0.0264 | 0.873 | 316 | 1.729 | 1.093 | 0.212 |
| 2024 (bull) | 9d | 0.0225 | 0.503 | 182 | 2.240 | 1.126 | 0.211 |

**Buy & hold reference (net, same test years):**
- 2022: CumRet **-0.640**, Sharpe **-1.280**, MaxDD **0.668**
- 2024: CumRet **1.110**, Sharpe **1.682**, MaxDD **0.262**

### 5.2 Forecast quality (test)
| Test year | Exp | Pinball (test) | MAE@24 (test) | Coverage q10–q90 |
|---|---:|---:|---:|---:|
| 2022 (bear) | 9c | 0.006122 | 0.025485 | 0.935 |
| 2022 (bear) | 9d | 0.005480 | 0.023224 | 0.859 |
| 2024 (bull) | 9c | 0.004462 | 0.019419 | 0.745 |
| 2024 (bull) | 9d | 0.004436 | 0.019440 | 0.769 |

### 5.3 Interpretation

#### 2024 (bull)
- 9d achieves a **large Sharpe improvement** over 9c while keeping drawdown similar.
- The main mechanism is **selectivity**:
  - long rate drops from ~0.87 (9c) to ~0.50 (9d),
  - trades drop from 316 → 182,
  - average net trade return increases materially.
- This matches the equity curve behavior: the strategy avoids some chop/drawdown and stays out during lower-confidence periods, while still capturing enough upside.

#### 2022 (bear)
- 9d improves *forecast* metrics (pinball + MAE) and moves interval coverage closer to nominal,
  but the trading layer becomes **overly conservative**:
  - long rate ~0.036 (only 13 trades),
  - strategy ends slightly negative net-of-cost.
- 9c traded more (~0.34 long rate) and stayed closer to flat net-of-cost in 2022.

**What this means:**  
The 9d model update is working (it changes regime selectivity), but the **threshold selection / score calibration** is now the main bottleneck for bear-regime performance.

---

## 6) Conclusion

### What improved from 9c → 9d
- **Bull-market selectivity and risk-adjusted performance improved strongly** (2024):
  - higher Sharpe, lower volatility, similar drawdown, comparable or slightly higher net return.
- Forecast uncertainty behavior becomes more regime-sensitive, and coverage moves closer to nominal.

### What remains unsolved
- In **bear regimes (2022)** the strategy becomes too “cash-heavy” and the few trades taken have negative expectancy.
- This is likely not a pure forecasting problem (forecast metrics improved), but a **signal calibration / thresholding** problem.

---

## 7) Reproducing the experiment

1. Configure splits and feature toggles in `config.py`
   - choose a test year (e.g. 2024) and matching validation window
2. Train:
   - `python train_tft.py`
3. Evaluate:
   - `python evaluate_tft.py`
4. Collect artifacts:
   - `*_metrics.json`, `*_score_threshold_sweep.json`
   - `*_forecast_table.csv`, `*_trading_table.csv`
   - plots: equity/forecast band/sweep/confusion/training curves

Tip: to run the regime stress test, re-point the test period to 2022 and re-run evaluation with the same trained weights.