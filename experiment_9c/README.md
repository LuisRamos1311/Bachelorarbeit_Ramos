# Experiments 9c: from Direction Classification to Multi‑Horizon Quantile Forecasting (with Percentile Thresholding)

This README documents the evolution from **Experiment 9b** to **Experiment 9c** in `experiment_9c/`:
- **9b**: 3‑class *direction* classification (`DOWN / FLAT / UP`) + an *UP-vs-REST* trading signal tuned by threshold sweep.
- **9c**: **multi‑horizon quantile regression** (uncertainty‑aware forecasting) + a **score-based trading signal** derived from predicted median and spread.
- **Update**: **percentile‑based thresholds** for 9c so threshold tuning stays comparable across different years/regimes.

The goal is to have a **fair comparison** between models (9b vs 9c) and across regimes (a “good” test year vs a “bad” test year).

---

## 1) What changed from 9b to 9c?

### Experiment 9b (baseline)
**Task:** 3‑class classification of the 24h‑ahead move:
- Labels are created using log returns (or returns) over **24 hours** and a direction threshold `DIRECTION_THRESHOLD=0.005` (±0.5%).
- Model outputs class probabilities.
- For trading, we focus on the probability of **UP**, i.e. `P(up)`.

**Trading signal (9b):**
1. Compute `P(up)` on validation.
2. Sweep a grid of probability thresholds `τ` (e.g., 0.35 … 0.65).
3. Select `τ*` that maximizes **validation Sharpe**.
4. Apply `τ*` on test for a long‑only strategy (non‑overlapping trades at horizon=24h).

### Experiment 9c (new goal)
**Task:** **multi‑horizon quantile regression** (H=24 horizons, quantiles = 0.1 / 0.5 / 0.9).
- Instead of “UP/DOWN/FLAT”, the model predicts a **distribution** of future returns:
  - `q10`, `q50` (median), `q90` for each horizon.
- Forecast loss: **pinball loss** (quantile loss).

**Trading score (9c):** use uncertainty to scale the signal  
For the chosen trading horizon (24h‑ahead), define:
- `μ = q50` (median predicted return)
- `IQR = q90 − q10`
- **score = μ / (IQR + ε)**

This behaves like a “risk‑adjusted predicted return”: large positive median with tight spread → higher score.

---

## 2) Why percentile thresholds were added in 9c

With absolute score thresholds, the “right” number is **not stable across years** because the score distribution shifts.
Example from your runs:
- 2022 scores are around **0.06–0.075**
- 2024 scores are around **0.025–0.045**

So a fixed grid like `[0.015, 0.02, …]` might be **too low** in one year (making you almost always long) and too high in another.

### Percentile thresholding (9c update)
Instead of defining a fixed list of score thresholds, we define a list of **percentiles** on the **validation score distribution**, for example:

- 2024 run: `[0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]`
- 2022 run: `[0.10, 0.20, …, 0.95]`

The evaluation script converts those percentiles into actual thresholds:
- `τ_p = percentile(score_val, p)`
- then runs the threshold sweep over `{τ_p}` and picks the best `τ*` by validation Sharpe.

**Benefit:** “top X% of scores” is comparable across years and avoids accidental “always long” thresholds.

---

## 3) Which metrics JSON for 9b is which?

You uploaded two 9b `*_metrics.json` files. They correspond to different evaluation runs (different years):

- **2024 split (train→2022, val=2023, test=2024)**  
  File: `4f9955e4-c2d3-469c-8eef-0789e2304cf4.json`  
  Contains `eval_id = tft_eval_20260104_184138` and has **positive Buy&Hold test performance** (bull-ish year).

- **2022 split (train→2020, val=2021, test=2022)**  
  File: `b5714d8a-0378-404c-9be5-5b6f06db6f67.json`  
  Contains `eval_id = tft_eval_20260105_181946` and has **negative Buy&Hold test performance** (bear-ish year).

---

## 4) Results: 9b vs 9c on a “bad year” (2022) and a “good year” (2024)

### Key comparison table (net of cost)
Numbers below use the **net-of-cost** strategy results reported by your evaluation scripts (cost=5bps, slippage=2bps, annual=365).

| Experiment                            | Test year   |   Val year |   Threshold (tau*) |   Net Sharpe (test) |   Net CumRet (test) |   Buy&Hold Net Sharpe (test) |   Buy&Hold Net CumRet (test) |   Test MC F1 |   Test pinball |   Test MAE@24 |
|:--------------------------------------|:------------|-----------:|-------------------:|--------------------:|--------------------:|-----------------------------:|-----------------------------:|-------------:|---------------:|--------------:|
| 9b (classification)                   | 2022 (bear) |       2021 |             0.3500 |             -1.4911 |             -0.6361 |                      -1.5899 |                      -0.7093 |       0.3032 |       nan      |      nan      |
| 9b (classification)                   | 2024 (bull) |       2023 |             0.4500 |              1.2140 |              0.3845 |                       1.4293 |                       0.8405 |       0.3058 |       nan      |      nan      |
| 9c (quantile forecast + percentile τ) | 2022 (bear) |       2021 |             0.0711 |              0.1578 |             -0.0162 |                      -1.2803 |                      -0.6398 |     nan      |         0.0061 |        0.0255 |
| 9c (quantile forecast + percentile τ) | 2024 (bull) |       2023 |             0.0311 |              0.7794 |              0.2233 |                       1.6817 |                       1.1104 |     nan      |         0.0045 |        0.0194 |

**How to read this table**
- **9b**: multiclass F1 provides a classification quality snapshot, but trading uses UP‑vs‑REST thresholding.
- **9c**: pinball/MAE measure forecast quality; trading is based on the score threshold.

---

## 5) What changed in 9c after percentile thresholds?

Before the percentile update, your threshold tuning could pick thresholds that effectively made the strategy **always in position** (often matching Buy&Hold exactly), which makes the backtest look “great” in bull years and “terrible” in bear years—without proving the model is generating a *selective* signal.

After switching to percentile thresholds, the strategy becomes more selective (higher thresholds), which:
- **hurts bull-year returns** (less market exposure),
- but **dramatically reduces bear-year drawdowns** (less exposure when signals are weak).

Here’s the same 9c model family, comparing the earlier “fixed grid picked τ≈0.0 (always long)” behavior vs the new percentile tuning:

|   Year | Mode                    |   Net Sharpe |   Net CumRet |   Bh Sharpe |   Bh CumRet |
|-------:|:------------------------|-------------:|-------------:|------------:|------------:|
|   2022 | fixed (tau=0.0)         |      -1.2803 |      -0.6398 |     -1.2803 |     -0.6398 |
|   2022 | percentile (tau*=0.071) |       0.1578 |      -0.0162 |     -1.2803 |     -0.6398 |
|   2024 | fixed (tau=0.0)         |       1.6817 |       1.1104 |      1.6817 |      1.1104 |
|   2024 | percentile (tau*=0.031) |       0.7794 |       0.2233 |      1.6817 |      1.1104 |

---

## 6) Interpretation (what this means for “best baseline”)

### 2024 (good year)
- **9b** achieves higher net Sharpe than percentile‑9c, but both underperform Buy&Hold in this year.
- **9c (percentile)** is *not* an “always-long” proxy anymore, so it no longer inherits the full bull-market performance. This is expected and is the price of making the signal meaningful.

### 2022 (bad year)
- **9b** struggles: net Sharpe ≈ −1.49, large negative cumulative return.
- **9c (percentile)** is far more robust: it nearly sidesteps the drawdown (net CumRet close to 0) and improves risk-adjusted performance relative to Buy&Hold.

### Recommendation (practical)
If you must pick **one** baseline for “future updates”:

- Pick **9c + percentile thresholds** if your priority is **robustness across regimes** and a forecasting setup that naturally supports uncertainty, risk controls, and multi‑horizon extensions.
- Keep **9b** as a *secondary baseline* if your priority is **direction classification**, because it’s easier to interpret and can still produce strong bull-year trading performance.

A good workflow is to track *both*:
- **Forecasting baseline:** 9c (pinball loss / MAE + calibration)
- **Directional trading baseline:** 9b (UP-vs-REST trading Sharpe)

---

## 7) Suggested default percentile grids going forward

Your percentile grids were different between the 2022 and 2024 runs. For a single consistent setting, start with:

```python
PERCENTILES = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99]
```

Why:
- Covers “moderate selectivity” (0.50–0.80) and “high conviction” (0.90–0.99).
- Helps you see if performance is stable as you trade less.

---

## 8) How to run

Train:
```bash
python train_tft.py
```

Evaluate:
```bash
python evaluate_tft.py
```

Outputs:
- `experiment_9c/plots/` – learning curves + score histograms
- `experiment_9c/experiments/` – `*_metrics.json` + `*_score_threshold_sweep.json`

---

## 9) Notes / caveats

- **Validation year differs** between the “2022 test” run (val=2021) and the “2024 test” run (val=2023). That’s intended in your rolling split setup, but it means `τ*` is tuned on different regimes.
- Percentile thresholds are computed on **validation scores** (no test leakage). In live trading you may want a *rolling* percentile of recent scores to keep long-rate stable over time.