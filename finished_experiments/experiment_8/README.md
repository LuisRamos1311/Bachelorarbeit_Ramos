# Experiment 8 – Hourly On-Chain + Sentiment TFT with Trading-Focused Thresholding

This experiment extends the **Experiment 7b** baseline by adding **daily sentiment regime features** (Reddit + Fear & Greed) to the existing **hourly BTC OHLCV + technical indicators + on-chain** feature set, while keeping the same **24-hour-ahead 3-class direction** label and the same **trading-oriented evaluation** (Sharpe-optimized threshold selection).

A key implementation constraint is that the sentiment dataset is available only up to **2024-12-31**, so the experiment’s evaluation window is shifted accordingly. To keep training data volume comparable (despite losing 2025), we **expanded the BTCUSD hourly candle history to include 2016–2017** (where Experiment 7b originally started later).

---

## 1. Motivation

**Experiment 7b** demonstrated that:
- adding a compact set of **on-chain features** can provide incremental edge, and
- selecting the UP-vs-REST threshold τ by **validation Sharpe ratio** turns a weak directional model into a more meaningful trading signal.

**Experiment 8** tests the hypothesis that:
- market behavior is partially regime-driven and may be better captured with **sentiment state variables** (level + momentum),
- and that these sentiment regimes (especially Reddit activity/sentiment + Fear & Greed) can improve **trade selection quality** (return while in position, Sharpe), even if 3-class classification remains noisy.

---

## 2. Data, Date Ranges & Splits (Aligned for Exp7b vs Exp8)

### Why the date ranges changed
- Sentiment (final combined dataset) is only available through **2024-12-31**.
- Therefore, the test year becomes **2024** (instead of 2025).
- To compensate for the missing 2025 year and keep a sufficiently large training window, we added **BTCUSD hourly data for 2016 and 2017**.

### Data frequency
- **BTCUSD 1-hour candles** (`BTCUSD_hourly.csv`)
- **Experiment window used for modeling:** **2016-01-01 to 2024-12-31**
- Splits (used for both Exp7b rerun and Exp8):
  - **Train:** 2016-01-01 → 2022-12-31  
  - **Validation:** 2023-01-01 → 2023-12-31  
  - **Test (out-of-sample):** 2024-01-01 → 2024-12-31  

---

## 3. Features & Labels

### Past covariates (X)

#### Price & volume (MinMax-scaled)
- `open, high, low, close, volume_btc, volume_usd`

#### Technical indicators (StandardScaler)
- `roc_10, atr_14, macd, macd_signal, macd_hist, rsi_14`

#### On-chain features (daily → hourly forward-fill, StandardScaler)
- `aa_ma_ratio` – active addresses / 7-day MA  
- `tx_ma_ratio` – tx count / 7-day MA  
- `mvrv_z` – MVRV z-score  
- `sopr_z` – SOPR z-score  
- `hash_ma_ratio` – hash rate / 30-day MA  

#### Sentiment features (daily → hourly join + forward-fill, StandardScaler except binary indicator)

**Reddit (Pushshift daily aggregates)**
- `reddit_sent_mean`
- `reddit_sent_std`
- `reddit_pos_ratio`
- `reddit_neg_ratio`
- `reddit_volume_log`

**Fear & Greed (daily, engineered)**
- `fg_index_scaled` (0–1 level)
- `fg_change_1d` (1-day change / momentum)
- `fg_missing` (binary availability indicator; not scaled)

### Total past features
- **Experiment 7b rerun:** 17 features (6 price/volume + 6 indicators + 5 on-chain)
- **Experiment 8:** 25 features (17 baseline + 8 sentiment features)

### Label / target
- **Task:** 24-hour-ahead **3-class direction** (classification)
- Horizon: `FORECAST_HORIZONS = [24]` steps (next 24 hours)
- Log returns: `USE_LOG_RETURNS = True`
- Threshold: `DIRECTION_THRESHOLD = 0.005`
- `direction_3c`:
  - 0 = DOWN if future log return < –0.005  
  - 1 = FLAT if |future log return| ≤ 0.005  
  - 2 = UP if future log return > +0.005  

---

## 4. Sentiment Data Engineering (New in Experiment 8)

### 4.1 Reddit sentiment (Pushshift dumps → daily time series)
We built a complete daily Reddit sentiment series from **Pushshift dumps** for major BTC-related subreddits:

- Bitcoin, btc, CryptoCurrency, BitcoinMarkets, CryptoMarkets  
- Coverage: **2016-01-01 → 2024-12-31**

Output file (daily):
- `reddit_sentiment_daily_pushshift.csv`
- Columns:
  - `date, reddit_sent_mean, reddit_sent_std, reddit_pos_ratio, reddit_neg_ratio, reddit_volume, reddit_volume_log`

This removes any dependency on the Reddit API (access was denied) and ensures reproducibility offline.

### 4.2 Fear & Greed index (handling partial coverage without look-ahead)
Fear & Greed was sourced as a daily 0–100 index and engineered into:
- `fg_index_scaled = value / 100.0`
- `fg_change_1d = fg_index_scaled.diff()`

Challenge:
- Fear & Greed is **not available across the full 2016–2024 window** (large early missing block + sporadic gaps).
- Leaving NaNs breaks scaling/training; backfilling introduces look-ahead bias.

Solution (time-series ML standard):
- Add `fg_missing` indicator (1 if missing, else 0)
- Impute missing values with **neutral constant**:
  - missing `fg_index_scaled = 0.5` (equivalent to 50)
  - missing `fg_change_1d = 0.0`
- Forward-fill only internal gaps (no future leakage)

### 4.3 Final combined sentiment file
We produced the final model-ready sentiment dataset by joining the daily Reddit and Fear & Greed series on a **fully contiguous daily date grid**:

- **2016-01-01 → 2024-12-31**, one row per day
- Forward-fill where appropriate
- Validate: no gaps, no NaNs, correct ranges

**Final combined output:**
- `data/BTC_sentiment_daily.csv`
- Contains all engineered Reddit + Fear & Greed features per day
- This file is merged into the hourly BTC frame by date (daily → hourly mapping + forward-fill)

---

## 5. Model & Training Setup

### Temporal Fusion Transformer (same regularized design as Exp7b)
- Hidden size: 32  
- Dropout in GRNs: 0.3  
- Weight decay: 5e-4  
- Sequence length: 96 (last 96 hours ≈ 4 days)
- Loss: `CrossEntropyLoss` (3 classes)
- Model selection metric: best **validation macro-F1** (3-class)

### What changes structurally vs Exp7b
- Input projection changes from:
  - Exp7b: `Linear(17 → 32)`
  - Exp8: `Linear(25 → 32)`
- Variable Selection Network (VSN) expands to handle 25 covariates (same architecture, larger input dimension).

Everything else (LSTM encoder, attention, GRNs, future covariates) remains consistent.

---

## 6. Evaluation & Trading Logic (Same as Exp7b)

Evaluation uses `evaluate_tft.py` and follows the same procedure as Experiment 7b:

1. Compute **3-class metrics** on validation and test:
   - Accuracy, macro precision/recall/F1, AUC
2. Collapse to **UP vs REST**:
   - UP = class 2; REST = classes 0 or 1
   - Use `P(UP)` from the softmax output
3. Apply long-only trading rule:
   - Enter long if `P(UP) ≥ τ`, else stay flat
   - **Non-overlapping trades**: each decision holds for 24h (aligned to label horizon)
4. Threshold selection:
   - Sweep τ over grid: `{0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65}`
   - Select **τ\*** that maximizes **validation Sharpe**
   - Re-evaluate UP-vs-REST metrics + trading performance on test using τ\*

---

## 7. Results

### 7.1 Why Exp7b was rerun
To ensure a fair comparison, we **reran Experiment 7b** with the **same date ranges and split** used by Experiment 8:

- Train: 2016–2022
- Val: 2023
- Test: 2024

This isolates the effect of adding sentiment features (Exp8) relative to the on-chain baseline (Exp7b) under identical evaluation conditions.

---

### 7.2 3-class direction (multi-class)

**Experiment 7b (rerun, aligned ranges)**
- Validation (2023):
  - Macro-F1: **0.3626**
  - Accuracy: 0.3683
  - AUC: 0.5356
- Test (2024):
  - Macro-F1: **0.2849**
  - Accuracy: 0.3890
  - AUC: 0.4938

**Experiment 8 (sentiment added)**
- Validation (2023):
  - Macro-F1: **0.3517**
  - Accuracy: 0.3703
  - AUC: 0.5311
- Test (2024):
  - Macro-F1: **0.3358**
  - Accuracy: 0.3929
  - AUC: 0.5087

**Interpretation**
- Exp8 improves **out-of-sample (test) macro-F1** meaningfully vs Exp7b (0.3358 vs 0.2849).
- Validation macro-F1 is slightly lower in Exp8, but Exp8 generalizes better to the test year.

---

### 7.3 UP-vs-REST + trading performance (τ\* chosen by validation Sharpe)

**Experiment 7b (aligned rerun)**
- Selected threshold: **τ\* = 0.45** (max val Sharpe = 1.3058)
- Test (2024) trading:
  - Sharpe: **0.9087**
  - Cumulative return: **0.4310**
  - Avg daily return: 0.001208
  - Avg return in position: 0.002138
  - Hit ratio: 0.2920

**Experiment 8 (sentiment added)**
- Selected threshold: **τ\* = 0.50** (max val Sharpe = 1.7816)
- Test (2024) trading:
  - Sharpe: **0.9981**
  - Cumulative return: **0.4519**
  - Avg daily return: 0.001212
  - Avg return in position: 0.002529
  - Hit ratio: 0.2672

**Interpretation**
- Exp8 improves **risk-adjusted performance** and **trade efficiency**:
  - Sharpe increases (0.91 → 1.00)
  - Cumulative return increases (0.43 → 0.45)
  - Return while in position increases (0.00214 → 0.00253)
- Exp8 is more selective (higher τ\*), which is typically favorable once transaction costs/slippage are considered.

---

## 8. Conclusion

Experiment 8 upgrades the Experiment 7b on-chain TFT baseline by integrating a robust, model-ready sentiment dataset:

- Reddit sentiment regimes built offline from Pushshift dumps (2016–2024)
- Fear & Greed engineered into level + momentum features
- Missingness handled safely with `fg_missing` and neutral imputation to avoid NaNs and look-ahead bias
- Daily sentiment merged into hourly modeling data in a consistent, failure-resistant way

Under aligned date splits, Experiment 8 shows:
- improved **test macro-F1** for 3-class direction prediction, and
- improved **trading Sharpe and return-in-position**, supporting the idea that sentiment contains regime information that can improve trade selection.

In short, Exp8 becomes a stronger trading-oriented baseline than Exp7b on the 2016–2024 window, and a natural foundation for future experiments (e.g., sentiment ablations, better calibration, cost-aware evaluation, multi-horizon targets).