# Experiment 9c (BTC): Multi-Horizon Quantile Forecasting + Percentile Threshold Trading  
**Baseline experiment for future upgrades** — with a direct comparison against **Experiment 9b**.

This folder implements the evolution from **Experiment 9b → Experiment 9c**:

- **Experiment 9b**: 3-class direction classification (`DOWN / FLAT / UP`) and a trading rule based on `P(UP)` thresholding.
- **Experiment 9c**: **multi-horizon quantile regression** (uncertainty-aware forecasting) and a trading rule based on a **risk-adjusted score** derived from quantiles.
- **Update (finalized)**: thesis-friendly reporting with **5 plots + 2 tables** per evaluation run (instead of “probability-style histograms” that did not fit 9c well).

Goal: a **reproducible, interpretable baseline** that can be evaluated fairly across regimes (bear vs bull) and compared to 9b.

---

## 1) What changed from 9b to 9c?

### Experiment 9b (classification baseline)
**Task**: classify the 24h-ahead move into:
- `DOWN / FLAT / UP`, using `DIRECTION_THRESHOLD = 0.005` (±0.5%) on 24h returns (log returns if enabled)

**Trading signal (9b)**:
1. Compute `P(UP)` on validation.
2. Sweep thresholds `τ` (probability thresholds).
3. Select `τ*` that maximizes **validation Sharpe**.
4. Apply `τ*` on test using a **long-only**, non-overlapping strategy (hold = 24h).

This is easy to interpret (confusion matrices, ROC), but the trading signal depends heavily on how well `P(UP)` separates regimes.

---

### Experiment 9c (quantile forecasting baseline)
**Task**: multi-horizon quantile forecasting with:
- horizons `H = 24` (1h steps → up to 24h)
- quantiles `Q = [0.1, 0.5, 0.9]`
- loss = **pinball loss** (quantile loss)

Instead of a single point estimate, the model predicts a distribution:
- `q10`, `q50` (median), `q90` for each horizon.

---

## 2) Trading rule in 9c (uncertainty-aware score)

For the selected trading horizon (24h-ahead), define:

- `μ = q50`  
- `IQR = q90 − q10`  
- `score = μ / (IQR + ε)`

This acts like a **risk-adjusted forecast**:
- big positive median + tight uncertainty band → high score
- wide uncertainty → score shrinks

**Long-only decision rule**
- enter long if `(score ≥ τ) AND (μ > 0)`
- otherwise stay out

We evaluate using **non-overlapping trades** (step size = 24 hours).

---

## 3) Why percentile thresholding matters (and what it does / does not do)

Raw score values shift across years (different volatility/regimes), so fixed numeric grids are unstable.

Example (from recent runs):
- 2022 scores were around ~0.07
- 2024 scores were around ~0.026–0.034

### Percentile grid (recommended baseline)
Instead of hard-coded score values, we define a list of validation-score percentiles:

`PERCENTILES = [0.10, 0.20, ..., 0.90, 0.95]`

Evaluation converts them into thresholds:
- `τ_p = percentile(score_val, p)`
- sweep over `{τ_p}` and choose `τ*` by **validation Sharpe**

**Important note:** percentile thresholds ensure comparability of the *grid* across regimes,  
but they **do not force selectivity**. If the best percentile is low (e.g., 0.10), the strategy can still become “mostly long”.

---

## 4) Updated reporting pack (final)

Experiment 9c now produces thesis-friendly diagnostics:

### Figures (max 5)
1. **Training curves** (loss + MAE over epochs)
2. **Test forecast band** (actual vs `q50` with `[q10, q90]` uncertainty band at signal horizon)
3. **Threshold sweep** (validation Sharpe vs threshold; shows threshold choice)
4. **Test equity curves (net)** (strategy vs buy&hold)
5. **Test signal confusion matrix** (2×2: predicted position vs realized outcome)

### Tables (2)
- `*_forecast_table.csv` (forecast metrics summary)
- `*_trading_table.csv` (trading metrics summary)

JSON artifacts remain for reproducibility:
- `*_metrics.json`
- `*_score_threshold_sweep.json`

---

## 5) Results: 9c across regimes (bear 2022 vs bull 2024)

Assumptions:
- long-only, non-overlapping trades (24h step)
- net-of-cost: **cost=5bps**, **slippage=2bps**
- annualization: 365

### 9c headline results (net-of-cost, test)
| Test year | τ* | Long rate (test) | Strategy net Sharpe | Strategy net CumRet | Strategy net MDD | Buy&Hold net Sharpe | Buy&Hold net CumRet | Random baseline p95 CumRet |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022 (bear) | 0.0711 | 0.338 | 0.1578 | -0.0162 | 0.3114 | -1.2803 | -0.6398 | 0.0607 |
| 2024 (bull) | 0.0264 | 0.873 | 1.7291 | 1.0932 | 0.2120 | 1.6817 | 1.1104 | 1.4002 |

### 9c forecast quality (signal=24h, test)
| Test year | Test pinball | Test MAE@24 |
|---|---:|---:|
| 2022 | 0.006122 | 0.025485 |
| 2024 | 0.004462 | 0.019419 |

**Interpretation**
- **2022 (bear)**: 9c behaves like a *risk filter* — it avoids much of the drawdown vs buy&hold, and ends near flat net-of-cost.
- **2024 (bull)**: 9c performs similarly to buy&hold because the optimal validation threshold is at a low percentile (high exposure). Profitability is high, but it does not prove strong “selective alpha” on its own.

---

## 6) Comparison: 9b vs 9c (what to take away)

Below are the *previously recorded* 9b summary numbers (kept for continuity).  
Note that 9b and 9c use different learning targets (classification vs quantiles), so you should compare:
- 9b: classification quality (MC F1) + trading metrics
- 9c: pinball/MAE (forecast quality) + trading metrics

### Key comparison table (net-of-cost, test)
| Experiment | Test year | Val year | τ* | Net Sharpe (test) | Net CumRet (test) | Buy&Hold Net Sharpe (test) | Buy&Hold Net CumRet (test) | Test MC F1 | Test pinball | Test MAE@24 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 9b (classification) | 2022 (bear) | 2021 | 0.3500 | -1.4911 | -0.6361 | -1.5899 | -0.7093 | 0.3032 | — | — |
| 9b (classification) | 2024 (bull) | 2023 | 0.4500 | 1.2140 | 0.3845 | 1.4293 | 0.8405 | 0.3058 | — | — |
| 9c (quantile + percentile τ) | 2022 (bear) | 2021 | 0.0711 | 0.1578 | -0.0162 | -1.2803 | -0.6398 | — | 0.0061 | 0.0255 |
| 9c (quantile + percentile τ) | 2024 (bull) | 2023 | 0.0264 | 1.7291 | 1.0932 | 1.6817 | 1.1104 | — | 0.0045 | 0.0194 |

### Practical recommendation
If you must pick one baseline for future development:
- **Use 9c** if you care about probabilistic forecasting + uncertainty-aware risk control (closest to TFT’s original multi-horizon quantile idea).
- Keep **9b** as a secondary baseline for direction interpretability (classic confusion matrix / ROC storytelling).

---

## 7) Conclusion: How good is Experiment 9c as a predicting tool?

**As a forecasting model (predicting 24h returns):**
Experiment 9c produces stable point-forecast metrics (pinball loss / MAE) and provides uncertainty estimates via quantiles (q10/q50/q90). This makes it a valid *probabilistic forecasting* tool in the spirit of TFT-style quantile prediction. However, across regimes the uncertainty calibration is not stable: in the 2022 bear regime the prediction intervals are overly conservative (very high q10–q90 coverage), while in the 2024 bull regime the intervals become too narrow (coverage below nominal). This indicates that the model’s uncertainty estimates are not yet reliable as “true probabilities” across different market conditions.

**As a trading-oriented predictor (decision usefulness):**
The model is most useful as a *risk filter* rather than a strong alpha generator. In 2022 it meaningfully reduces drawdowns compared to buy & hold (capital preservation behavior). In 2024 it performs similarly to buy & hold because the validation-selected threshold leads to high market exposure; in other words, strong results in a bull market can be explained largely by being long most of the time rather than by selective predictive skill. The random baseline with matched exposure remains an important stress test: if the strategy does not consistently beat that baseline, profitability is not strong evidence of genuine predictive edge.

**Overall assessment:**
Experiment 9c is a solid baseline for thesis work because it is (1) probabilistic and multi-horizon (quantile) rather than purely directional, (2) evaluated with realistic net-of-cost backtests, and (3) tested across different regimes. At the current stage, its strongest demonstrated benefit is regime-dependent risk control. The main limitation is that predictive selectivity and calibration stability across years are not yet strong enough to claim consistent trading alpha.

