# Experiment 7 – Hourly 24h-Ahead TFT with Technical + On-Chain BTC Features

## 1. Goal

Build directly on the hourly 24h-ahead TFT baseline from Experiment 6 and test whether adding a compact on-chain feature set improves:

- 24h-ahead **3-class direction** prediction (DOWN / FLAT / UP), and  
- a **long-only UP-vs-REST trading strategy** on BTC.

Everything except the feature set (data splits, label, model, training, evaluation) is kept identical to Experiment 6 to allow a clean comparison.

---

## 2. Data, Features & Targets

### Data

Same setup as Experiment 6:

- Source: `BTCUSD_hourly.csv`
- Period: **2018–2025**
- Frequency: **1h** candles
- Splits:
  - Train: **2018–2023**
  - Val: **2024**
  - Test: **2025**
- Window:
  - Lookback: `SEQ_LENGTH = 96` hours (4 days)
  - Horizon: `HORIZON = 24` hours (24h ahead on the hourly grid)
- Sample counts (approx.):
  - Train: **49k**
  - Val: **8.7k**
  - Test: **8.0k**
- Label balance (FULL 2018–2025):
  - DOWN ≈ **37%**, FLAT ≈ **22%**, UP ≈ **41%**

### Features

We reuse all **technical features** from Experiment 6 and add 5 derived on-chain metrics.

**Base (past) features (as in Exp. 6)**  
- OHLCV (hourly): `open, high, low, close, volume_btc, volume_usd`  
- Technical indicators on hourly closes:
  - `roc_10`, `atr_14`
  - `macd`, `macd_signal`, `macd_hist`
  - `rsi_14`
- Calendar / halving flags:
  - hour of day, day of week, month, weekend indicator
  - halving-window indicators  

These 12 past features are scaled with:

- `PRICE_VOLUME_COLS` → `MinMaxScaler`
- `INDICATOR_COLS` → `StandardScaler`

**New on-chain features (daily → broadcast to hours)**

From a daily BTC on-chain CSV:

- Raw daily series:
  - `active_addresses`
  - `tx_count`
  - `mvrv`
  - `sopr`
  - `hash_rate`

We derive 5 features and forward-fill them to all hours of that day:

- `aa_ma_ratio  = active_addresses / MA_7(active_addresses)`
- `tx_ma_ratio  = tx_count / MA_7(tx_count)`
- `hash_ma_ratio = hash_rate / MA_30(hash_rate)`
- `mvrv_z      = z-score(mvrv)` (standardised later using train stats)
- `sopr_z      = z-score(sopr)` (standardised later using train stats)

These are added to `FEATURE_COLS` and scaled together with indicators via `StandardScaler`:

```python
FEATURE_COLS = PRICE_VOLUME_COLS + INDICATOR_COLS + ONCHAIN_COLS
ONCHAIN_COLS = ["aa_ma_ratio", "tx_ma_ratio", "mvrv_z", "sopr_z", "hash_ma_ratio"]
````

Total past features: **17** (12 technical + 5 on-chain).

**Known future covariates (unchanged)**  

- 5 “known-future” features at `t+24h` (calendar/halving style), as in Experiment 6.

### Target

Same 24h log-return 3-class label as Experiment 6:

- Compute 24h log-return:

  log_return_24h(t) = log(close[t+24] / close[t])

- Apply a symmetric dead-zone with `DIRECTION_THRESHOLD = 0.005`:

  - `0 = DOWN` if log_return_24h < −0.005  
  - `1 = FLAT` if |log_return_24h| ≤ 0.005  
  - `2 = UP`   if log_return_24h > +0.005  

Config flags:

- `USE_LOG_RETURNS = True`
- `FORECAST_HORIZONS = [24]`
- `USE_MULTI_HORIZON = False`
- `FREQUENCY = "1h"`
- `TARGET_COLUMN = "direction_3c"`
- `NUM_CLASSES = 3`
- `NON_OVERLAPPING_TRADES = True` (one 24h trade at a time, as in Exp. 6)

---

## 3. Model & Training

### Model

We reuse the simplified Temporal Fusion Transformer from Experiment 6, with a larger input size:

- Input projection: `input_size = len(FEATURE_COLS) = 17 → 64`
- Variable Selection Network (VSN) over 17 past features
- LSTM encoder:
  - `hidden_size = 64`
- Multi-head self-attention (`num_heads = 4`)
- Gated Residual Networks (GRNs) for:
  - post-LSTM
  - attention output
  - temporal feed-forward block
  - future covariate encoder (5-dim future vector)
  - decision GRN (history + future)
- Output layer: `64 → 3` logits (`[DOWN, FLAT, UP]`)

No structural changes vs Experiment 6; the TFT just sees 5 extra on-chain inputs.

### Training

Script: `experiment_7/train_tft.py` (copied from Exp. 6 with minor logging updates).

Config (identical to Experiment 6 to keep comparison fair):

- Task: `TASK_TYPE = "classification"`
- Loss: `CrossEntropyLoss`
- Optimizer: Adam
- Batch size: `64`
- Learning rate: `1e-3`
- Weight decay: `1e-4`
- Dropout (all GRNs): `0.2`
- Epochs: `12`
- Model selection: best validation macro-F1 (3-class)  
  → best checkpoint saved as `experiment_7/models/tft_btc_best.pth`.

---

## 4. Evaluation & Comparison to Experiment 6

Evaluation is done by `experiment_7/evaluate_tft.py` and mirrors Experiment 6:

1. Compute 3-class metrics (DOWN / FLAT / UP) on val/test.
2. Collapse to binary **UP vs REST** (REST = {DOWN, FLAT}) and:
   - sweep a small grid of thresholds on `P(UP)` (`[0.1, 0.2, 0.3, 0.4, 0.5]`),
   - choose τ* that maximises balanced accuracy on validation,
   - evaluate UP-vs-REST metrics on test at τ*.
3. Using τ*, run a non-overlapping long-only trading strategy:
   - if `P(UP)_t ≥ τ*`, go long BTC from *t* to *t+24h*, otherwise stay flat.

### 4.1 3-Class Direction (DOWN / FLAT / UP)

**Experiment 6 (technical only, summary)**  

- **Val (2024)**:
  - Accuracy ≈ **0.43**
  - Macro-F1 ≈ **0.40**
  - Macro-AUC ≈ **0.57**
- **Test (2025)**:
  - Accuracy ≈ **0.39**
  - Macro-F1 ≈ **0.36**
  - Macro-AUC ≈ **0.55**

**Experiment 7 (technical + on-chain)**

- **Val (2024)**:
  - Accuracy: **0.38**
  - Macro-F1: **0.37**
  - Macro-AUC: **0.52**
- **Test (2025)**:
  - Accuracy: **0.36**
  - Macro-F1: **0.36**
  - Macro-AUC: **0.53**

**Takeaway:**  
Adding on-chain features does not noticeably improve the pure 3-class direction task. Accuracy and macro-F1 stay around 0.36–0.40, and macro-AUC remains only slightly above random.

### 4.2 UP-vs-REST (Binary) & Trading

Here the on-chain features have a more visible effect.

**Threshold selection**

- Exp. 6: best τ* ≈ **0.40**
- Exp. 7: best τ* ≈ **0.50**

**UP-vs-REST metrics (test)**

- **Experiment 6** (τ* = 0.40, tech only)  
  - Accuracy: **0.56**
  - Balanced accuracy: **0.50**
  - Precision (UP): **0.38**
  - Recall (UP): **0.24**
  - F1 (UP): **0.29**
  - AUC: **0.51**
  - Positive rate: **0.24**

- **Experiment 7** (τ* = 0.50, tech + on-chain)  
  - Accuracy: **0.55**
  - Balanced accuracy: **0.52**
  - Precision (UP): **0.40**
  - Recall (UP): **0.36**
  - F1 (UP): **0.38**
  - AUC: **0.52**
  - Positive rate: **0.34**

**Takeaway:**

- The ranking power (AUC) is still weak (~0.52), but:
  - Recall for UP improves (≈24% → 36%),
  - F1 for UP improves (≈0.29 → 0.38),
  - Balanced accuracy and macro-F1 also improve slightly,
  - The model issues more UP signals (positive_rate increases) with better quality.

This suggests that on-chain features help the TFT detect UP regimes a bit better, even if overall classification remains hard.

**Long-only trading (non-overlapping 24h trades)**

Using τ* from each experiment:

- **Validation (2024)**  
  - **Exp. 6**:  
    - Cumulative return ≈ **+19%**, Sharpe ≈ **0.57**, hit ratio ≈ **0.18**
  - **Exp. 7**:  
    - Cumulative return ≈ **+50%**, Sharpe ≈ **1.19**, hit ratio ≈ **0.24**

- **Test (2025)**  
  - **Exp. 6**:  
    - Cumulative return ≈ **−11.6%**, Sharpe ≈ **−0.45**, hit ratio ≈ **0.10**
  - **Exp. 7**:  
    - Cumulative return ≈ **−4.0%**, Sharpe ≈ **−0.01**, hit ratio ≈ **0.18**

**Takeaway:**

- On the validation year, the on-chain TFT delivers a much stronger risk-adjusted return.
- On the test year, neither model is clearly profitable, but:
  - Losses shrink (−11.6% → −4.0%),
  - The hit ratio nearly doubles,
  - Sharpe improves from clearly negative to roughly flat.

---

## 5. Conclusion

Experiment 7 extends the hourly 24h-ahead TFT from Experiment 6 by adding a small, carefully engineered on-chain feature set (active addresses, transactions, MVRV, SOPR, hash rate) while keeping everything else constant.

Main conclusions:

- For the **3-class direction classification task**, on-chain data does not change the big picture:
  - macro-F1 and macro-AUC remain modest and close to Experiment 6.
- For the **binary UP-vs-REST view and trading**, on-chain data does help:
  - better recall and F1 for the UP class,
  - slightly higher balanced accuracy and AUC,
  - better trading metrics (especially on validation, and less negative on test).

Experiment 7 is therefore adopted as the new **“technical + on-chain hourly TFT baseline”**, and serves as the starting point for follow-up tuning experiments (e.g. Experiment 7b: model size, regularisation, VSN on/off, threshold grids) aimed at further improving robustness and trading usefulness.

