# Experiment 6 – Hourly 24h-Ahead Directional TFT on BTC

## 1. Goal

Build on the daily TFT experiments (Exp. 1–5) and test whether a **Temporal Fusion Transformer** can better forecast BTC direction when it sees **hourly data** and predicts the **24-hour-ahead** move instead of the next daily close.  
In parallel:

1. Fix a flaw in the trading backtest for 24h horizons (overlapping trades),
2. Tidy up the training procedure with **slightly stronger regularisation and fewer epochs**.

Experiments 1–5 defined and stress-tested the daily baseline (regression → 3-class direction on log-returns, different thresholds, and various decision rules / time splits).   
Experiment 6 is the continuation of that line, but on **hourly BTC**.

---

## 2. Data & Targets

### 2.1 Data

- Source: `BTCUSD_hourly.csv` (CryptoDataDownload-style).
- Period used for this experiment: **2018-01-01 to ~2025-XX-XX**.
- Frequency: **1h** candlesticks.
- Time splits (by calendar year, same philosophy as daily experiments):

  - **Train**: 2018–2023  
  - **Val**: 2024  
  - **Test**: 2025  

- Sliding window:

  - Lookback: `SEQ_LENGTH = 96` hours (4 days).
  - Forecast horizon: `HORIZON = 24` hours (1 day ahead, but on the hourly grid).

- Sample counts (after building sequences and dropping rows that lack future data):

  - Train: **~49k** samples  
  - Val: **~8.7k** samples  
  - Test: **~8.0k** samples  

- Label balance (FULL 2018–2025) with the final configuration:

  - DOWN: **37.1%**  
  - FLAT: **22.2%**  
  - UP:   **40.7%**  

  Train/val/test splits have very similar proportions.

### 2.2 Features

Feature engineering is conceptually the same as in the daily experiments, but applied to **hourly** candles:

**Past covariates (per hour)**

- OHLCV:
  - `open`, `high`, `low`, `close`
  - `volume_btc`, `volume_usd`
- Technical indicators (computed on hourly closes):
  - `roc_10`, `atr_14`
  - `macd`, `macd_signal`, `macd_hist`
  - `rsi_14`
- Calendar / halving:
  - Hour of day, day of week, month, weekend flag
  - Halving window indicators (aligned to dates, then broadcast to hours)

Scaling reuses the same scheme as the daily TFT:

- Price / volume columns → `MinMaxScaler`
- Indicator columns → `StandardScaler`

**Known future covariates (t+24)**

Because the horizon is 24 hours, the “known future” vector contains features for the **end of the forecast window** (t+24h):

- `dow_next`, `is_weekend_next`, `month_next`, `is_halving_window_next`
- Plus one extra known-future flag (final hourly setup uses 5 future features in total).

These are obtained via a 24-step `shift(-24)` and form a 5-dim vector per sample.

### 2.3 Target: 24h Log-Return Direction (3-Class)

I keep the **log-return** framework from Experiment 3, but with a **24h horizon** instead of 1 day. :contentReference[oaicite:1]{index=1}  

For each hour *t*:

1. Compute **24h log return**:

\[
\text{log\_return\_24h}(t) =
\log\left(\frac{\text{close}_{t+24}}{\text{close}_t}\right)
\]

2. Apply a symmetric dead-zone with `DIRECTION_THRESHOLD = 0.005` (≈0.5%):

- `0 = DOWN` if `log_return_24h < −0.005`  
- `1 = FLAT` if `|log_return_24h| ≤ 0.005`  
- `2 = UP`   if `log_return_24h > +0.005`

Config flags for this experiment:

- `USE_LOG_RETURNS = True`
- `FORECAST_HORIZONS = [24]`
- `USE_MULTI_HORIZON = False` (single-horizon classification)
- `FREQUENCY = "1h"`
- `TARGET_COLUMN = "direction_3c"`
- `NUM_CLASSES = 3`

---

## 3. Model & Training

### 3.1 Model

I reuse the **simplified Temporal Fusion Transformer** encoder from earlier experiments:

- Variable Selection Network (VSN) over 12 past features
- LSTM encoder:
  - `hidden_size = 64`
- Multi-head self-attention:
  - `num_heads = 4`
- Gated Residual Networks (GRNs):
  - post-LSTM encoder
  - attention output
  - temporal feed-forward block
  - future-covariate encoder
  - decision GRN (fusing history + future)
- Future covariate encoder for 5 known-future features.
- Final linear layer:
  - `output_size = 3` → logits for `[DOWN, FLAT, UP]`.

Compared to the daily TFT, the **architecture is unchanged**; only the input sequence length and the semantics of the horizon change.

### 3.2 Training (final Experiment-6 configuration)

Script: `experiment_6/train_tft.py`

- Task: `TASK_TYPE = "classification"`
- Label: `TARGET_COLUMN = "direction_3c"` (24h log-return)
- Loss: `CrossEntropyLoss`
- Optimizer: Adam
- Hyperparameters (after the regularisation update):

  - Batch size: `64`
  - Learning rate: `1e-3`
  - **Weight decay**: `1e-4` (was `1e-5`)
  - **Dropout** (all GRNs & MLPs): `0.2` (was `0.1`)
  - **Epochs**: `12` (was `20`)

- Model selection:

  - **Best validation macro-F1** (3-class) across epochs.
  - The corresponding checkpoint is saved as `experiment_6/models/tft_btc_best.pth`.

The training script also logs per-epoch train/val loss and F1 and saves training curves as `plots/..._training_curves.png`.

---

## 4. Experiment Journey & Design Decisions

This experiment went through **three main phases**, all under the umbrella of “Experiment 6”. Each step fixed a specific issue or design choice rather than starting from scratch.

### 4.1 Phase 1 – From Daily (Exp. 5) to Hourly 24h-Ahead TFT

**Motivation**

Daily Experiment 5 used:

- daily candles (2014–2025)  
- 1-day-ahead 3-class log-return direction  
- TFT with 30-day input window  

It consistently produced:

- Test 3-class macro-F1 ≈ **0.37**, macro-AUC ≈ **0.55**,  
- UP-vs-REST AUC ≈ **0.50** (almost random),  
- A degenerate decision rule where the F1-optimal threshold essentially predicted **UP on almost every day**. :contentReference[oaicite:2]{index=2}  

The hypothesis for Experiment 6 was:

> Maybe **hourly structure** (intraday volatility, micro-trends) contains more predictability for the **next 24 hours** than a single daily close.

**Changes in Phase 1**

- Switch from daily to **hourly data (1h)**.
- Use a **96-hour lookback** and **24-hour ahead** horizon.
- Keep the 3-class log-return label (`DOWN/FLAT/UP` with ±0.5% dead-zone).
- Keep the same TFT architecture and almost identical training config:
  - dropout `0.1`, weight decay `1e-5`, `20` epochs.

**Initial results (overlapping 24h trades)**

- 3-class metrics (test):

  - Accuracy ≈ **0.39**
  - Macro-F1 ≈ **0.34**
  - Macro-AUC ≈ **0.55**

  → Very similar to the daily baseline.

- UP-vs-REST (tuned on balanced accuracy, threshold τ ≈ `0.50`):

  - Test accuracy ≈ **0.60**
  - Balanced accuracy ≈ **0.51**
  - F1 ≈ **0.24**
  - AUC ≈ **0.53**, positive_rate ≈ **0.15**

  → Slightly informative but still weak.

- **Trading metrics (buggy):**

  The initial trading backtest opened a new 24h-ahead position **every hour**, creating **24 overlapping trades** at any given time. With this overlapping-trade logic, the validation cumulative return inflated to **~×72**, which is clearly unrealistic and not comparable to any daily strategy.

**Conclusion of Phase 1**

- Purely from a predictive standpoint, hourly data **did not magically solve** the problem:
  - test macro-F1 and AUC were in the same ballpark as the daily baseline.
- However, the model behaved slightly less pathologically in UP-vs-REST (less “always UP”).
- The trading backtest needed to be fixed before any serious comparison.

---

### 4.2 Phase 2 – Fixing Overlapping 24h Trades (Non-Overlapping)

**Problem**

For a 24h horizon on **hourly** data, opening a fresh position at every hour applies the *same* 24h view 24 times in parallel. This:

- artificially amplifies exposure and cumulative return,  
- makes Sharpe and hit ratios hard to interpret,  
- is inconsistent with the **1-trade-per-day** interpretation used on daily data.

**Fix**

Introduce a config flag:

- `NON_OVERLAPPING_TRADES = True`

and change the trading evaluation in `evaluate_tft.py` so that:

- For each **24-hour horizon**, only evaluate a decision **once per 24 hours**.
- When `NON_OVERLAPPING_TRADES = True` and `HORIZON = 24`, the code:

  - samples every 24th prediction,
  - opens at most **one position at a time**,
  - holds it for the full 24-hour period.

**Impact on results**

- **Classification metrics** (3-class and UP-vs-REST) are *unchanged* by this change (the model and its outputs are the same).
- **Trading metrics** become much more realistic:

  - Validation cumulative return dropped from “×72” to around **+15–20%**,  
  - Test cumulative return around **+10%**,  
  - Sharpe ratios in the **0.3–0.6** range instead of enormous values.

This step did **not improve predictive skill**, but it was crucial to ensure that:

> Trading performance is measured under **non-overlapping, one-trade-per-day behaviour**, directly comparable to daily experiments.

---

### 4.3 Phase 3 – Mild Regularisation & Shorter Training

**Observation**

Even with hourly data and fixed trading:

- Train F1 kept climbing to ~0.79,
- Val F1 hovered around **0.33–0.36**,
- Val loss rose above **2.0**,

which is classic **overfitting**. The best model (by val macro-F1) was around epochs 10–16, but the curves looked ugly and I was wasting epochs.

**Changes**

In `config.py`:

- Increase model dropout:

  - `dropout: 0.2` (was `0.1`)

- Increase weight decay:

  - `weight_decay: 1e-4` (was `1e-5`)

- Reduce epochs:

  - `num_epochs: 12` (was `20`)

No architecture changes were made; this was purely a **regularisation / training-schedule cleanup**.

**Behaviour with new config**

- Train F1 still increases steadily, but only up to ~0.65 at epoch 12.
- Val F1 peaks earlier (~epoch 5) at **0.40** and then gradually decays.
- Because I already save the **best validation macro-F1** checkpoint, the effective model used for evaluation is the epoch-5 one, which sits **before** the curves diverge too much.

---

## 5. Final Results (Experiment 6 – Hourly, Non-Overlapping, Regularised)

All numbers below refer to the **final configuration**:

- 1h data, 96-hour input window, 24h-ahead 3-class log-return label,
- non-overlapping 24h trades,
- dropout 0.2, weight decay 1e-4, 12 epochs.

### 5.1 3-Class Direction Metrics

**Validation (2024)**

- Cross-entropy loss: **1.13**
- Accuracy: **0.43**
- Macro precision: **0.43**
- Macro recall: **0.40**
- **Macro-F1: 0.40**
- **Macro-AUC: 0.57**

**Test (2025)**

- Cross-entropy loss: **1.20**
- Accuracy: **0.39**
- Macro precision: **0.39**
- Macro recall: **0.37**
- **Macro-F1: 0.36**
- **Macro-AUC: 0.55**

Compared to the daily Experiment-5 baseline (1-day ahead on daily candles):

- Exp. 5 (daily): test macro-F1 ≈ **0.37**, macro-AUC ≈ **0.55**. :contentReference[oaicite:3]{index=3}  
- Exp. 6 (hourly): test macro-F1 ≈ **0.36**, macro-AUC ≈ **0.55**.

So **predictive power is essentially the same**; hourly data does not significantly improve the ability to distinguish DOWN / FLAT / UP.

### 5.2 UP-vs-REST (Binary) Metrics

I convert the 3 classes to a binary label:

- Positive: `UP`
- Negative: `{DOWN, FLAT}`

On the validation set, I sweep thresholds on `P(UP)`:

\[
\tau \in \{0.10, 0.20, 0.30, 0.40, 0.50\}
\]

and select the one that **maximises balanced accuracy**. For this run:

- Best threshold: **τ\* = 0.40**.
- This is more conservative than the daily experiments (where τ\* often collapsed towards 0.10).

**Validation (τ\* = 0.40)**

- Accuracy: **0.57**
- Balanced accuracy: **0.55**
- Precision (UP): **0.48**
- Recall (UP): **0.40**
- F1 (UP): **0.44**
- Macro-F1 (binary): **0.54**
- AUC (UP vs REST): **0.55**
- Positive rate: **0.35** (model trades on ~35% of samples)

**Test (τ\* = 0.40)**

- Accuracy: **0.56**
- Balanced accuracy: **0.50**
- Precision (UP): **0.38**
- Recall (UP): **0.24**
- F1 (UP): **0.29**
- Macro-F1 (binary): **0.49**
- AUC (UP vs REST): **0.51**
- Positive rate: **0.24**

Compared to the earlier hourly run (smaller regularisation, τ\* ≈ 0.50), this configuration:

- **Improves UP-class F1** on test (from ~0.24 → ~0.29),
- Uses a **more active** trading threshold (positive rate ~24% instead of ~15%),
- Still delivers only **near-random AUC (~0.51)**, so ranking quality remains weak.

### 5.3 Non-Overlapping 24h Trading Metrics

Using τ\* = 0.40 and **non-overlapping trades**:

- For each 24h block I either:
  - **enter long BTC for the next 24 hours** if P(UP) ≥ τ\*, or  
  - stay **flat**.

**Validation (2024)**

- Average daily return: **0.00063** (~0.06% / day)
- Cumulative return (year): **+18.9%**
- Sharpe ratio: **0.57**
- Hit ratio (positive days while in position): **0.18**
- Average return when in position: **0.00195**

**Test (2025)**

- Average daily return: **−0.00031** (~−0.03% / day)
- Cumulative return: **−11.6%**
- Sharpe ratio: **−0.45**
- Hit ratio: **0.096**
- Average return when in position: **−0.00151**

Interpretation:

- The strategy is in the market roughly **a quarter of the time** (positive_rate ≈ 0.24).
- Performance is **not consistently positive**: mildly profitable on the 2024 validation year, but negative in 2025.
- Given the weak AUC, these results are consistent with **no reliable edge** beyond noise and regime differences.

---

## 6. Summary & Role as Baseline

Experiment 6 achieved three things:

1. **Hourly data and 24h horizons**  
   - I verified that a TFT can be trained and evaluated on **1h BTC data** with a **24h-ahead log-return label** using the same architecture as the daily experiments.
   - The model’s **predictive skill (macro-F1, AUC)** is **very similar** to the daily baseline; hourly data did **not** unlock a large edge.

2. **Correct trading evaluation for 24h horizons**  
   - I identified and fixed a subtle but important issue: overlapping 24h trades on hourly data.
   - The new `NON_OVERLAPPING_TRADES` option enforces **one position per 24h horizon**, making trading metrics comparable across frequencies and horizons.

3. **Cleaner regularised training**  
   - Increasing dropout and weight decay and reducing the max epoch count gave a **more reasonable training curve** (less obvious runaway overfitting).
   - Validation metrics improved slightly (val macro-F1 from ~0.36 → ~0.40) without changing test metrics dramatically.
   - This configuration is now a **sensible, cost-effective default** for future TFT experiments.

### Final conclusion

- **Experiment 5 (daily)** remains an important reference, but  
- the **final version of Experiment 6**—hourly data, 24h-ahead 3-class log-return direction, non-overlapping trades, and mild regularisation—is the **preferred TFT baseline** for subsequent work in this project.

Future experiments (e.g. class-weighted loss, richer features, multi-horizon outputs, or alternative targets) will be built on top of this **hourly Experiment-6 setup** unless explicitly stated otherwise.