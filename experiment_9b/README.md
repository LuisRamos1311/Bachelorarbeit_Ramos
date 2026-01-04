# Experiment 9a – Data Integrity Hardening (Split-First Labels + Lagged Daily Features)

This experiment uses the **same modeling task, feature set, architecture, and trading-oriented evaluation** as **Experiment 8** (hourly BTC OHLCV + technical indicators + daily on-chain + daily sentiment; 24-hour-ahead 3-class direction classification with Sharpe-optimized thresholding), but introduces **critical data integrity fixes** to make the pipeline more realistic and “paper defensible”.

**Experiment 9a focus:** eliminate subtle look-ahead / split contamination by:
1) **Splitting first, then computing labels/forward returns inside each split**, and  
2) **Lagging daily on-chain + sentiment features by 1 day at merge-time** (hour *t* only sees daily aggregates from *t-1 day*).

---

## 1. Motivation

Experiment 8 showed that adding sentiment regimes (Reddit + Fear & Greed) can improve trading-oriented metrics under the existing pipeline.

However, Exp8 still had two realism risks commonly discussed in time-series forecasting / trading research:

1) **Split-boundary contamination**  
If labels (via shift(-H)) are computed before splitting, the last H rows of the train split can reference validation prices (and similarly val→test). Even though this affects only boundary rows, it is still leakage.

2) **Daily feature availability timing**  
Daily on-chain/sentiment values should not be assumed available throughout the same day at all hours. A more realistic assumption is that a daily aggregate becomes known only after the day closes (or with reporting delay). Therefore, hourly samples should use daily values from the prior day.

Experiment 9a corrects both issues while keeping the rest of the methodology unchanged, so that later experiments (9b–9e) can build on a trustworthy baseline.

---

## 2. Data, Date Ranges & Splits

### Data frequency
- **BTCUSD 1-hour candles**
- **Experiment window used for modeling:** **2016-01-01 to 2024-12-31**
- Splits:
  - **Train:** 2016-01-01 → 2022-12-31  
  - **Validation:** 2023-01-01 → 2023-12-31  
  - **Test (out-of-sample):** 2024-01-01 → 2024-12-31  

### 9a split-boundary enforcement
Horizon is **H = 24 hours**. After computing targets inside each split, the **last 24 hours of each split are dropped** so no label can reference prices outside that split.

Logged example:
- TRAIN split_end: 2022-12-31 23:00 → labeled_end: 2022-12-30 23:00  
- VAL split_end: 2023-12-31 23:00 → labeled_end: 2023-12-30 23:00  
- TEST split_end: 2024-12-31 23:00 → labeled_end: 2024-12-30 23:00  

This guarantees there is **no split-boundary contamination**.

---

## 3. Features & Labels (Same as Experiment 8)

### Past covariates (X)

#### Price & volume (MinMax-scaled)
- `open, high, low, close, volume_btc, volume_usd`

#### Technical indicators (StandardScaler)
- `roc_10, atr_14, macd, macd_signal, macd_hist, rsi_14`

#### On-chain features (daily → hourly, StandardScaler)
- `aa_ma_ratio` – active addresses / 7-day MA  
- `tx_ma_ratio` – tx count / 7-day MA  
- `mvrv_z` – MVRV z-score  
- `sopr_z` – SOPR z-score  
- `hash_ma_ratio` – hash rate / 30-day MA  

#### Sentiment features (daily → hourly, StandardScaler except binary indicator)

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
- **25 features** (6 price/volume + 6 indicators + 5 on-chain + 8 sentiment)

### Label / target
- **Task:** 24-hour-ahead **3-class direction** (classification)
- Horizon: `FORECAST_HORIZONS = [24]`
- Log returns: `USE_LOG_RETURNS = True`
- Threshold: `DIRECTION_THRESHOLD = 0.005`
- `direction_3c`:
  - 0 = DOWN if future log return < –0.005  
  - 1 = FLAT if |future log return| ≤ 0.005  
  - 2 = UP if future log return > +0.005  

---

## 4. New in Experiment 9a: Data Integrity Implementations

### 4.1 Split-first target computation (leakage fix)
**Old approach (risk):** compute `shift(-H)` forward returns on the full dataset, then split.  
**9a approach:** split by date first, then compute forward returns and labels inside each split, then drop last H rows.

Benefits:
- no label in train can reference validation prices
- no label in validation can reference test prices

### 4.2 Lag daily features by 1 day (availability realism)
Daily on-chain and sentiment features are merged using a lagged daily key:
- hourly timestamp `t` uses daily features from **floor(t) - 1 day**

This reflects a conservative “available after daily close” assumption.

### 4.3 Debug sanity checks (enabled)
Debug-only checks randomly sample timestamps and assert:
- the daily feature values match the expected lagged date (t−1 day)

This prevents accidental regressions (e.g., mistakenly joining daily features to same-day hours).

---

## 5. Model & Training Setup (Same as Experiment 8)

### Temporal Fusion Transformer (regularized design)
- Hidden size: 32  
- Dropout in GRNs: 0.3  
- Weight decay: 5e-4  
- Sequence length: 96 (last 96 hours ≈ 4 days)
- Loss: `CrossEntropyLoss` (3 classes)
- Model selection metric: best **validation macro-F1** (3-class)

---

## 6. Evaluation & Trading Logic (Same as Experiment 8)

Evaluation uses `evaluate_tft.py`:

1. Compute **3-class metrics** on validation and test:
   - Accuracy, macro precision/recall/F1, AUC
2. Collapse to **UP vs REST**:
   - UP = class 2; REST = classes 0 or 1
   - Use `P(UP)` from softmax
3. Long-only trading rule:
   - Enter long if `P(UP) ≥ τ`, else stay flat
   - **Non-overlapping trades**: each decision holds for 24h (aligned to label horizon)
4. Threshold selection:
   - Sweep τ over grid: `{0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65}`
   - Select **τ\*** that maximizes **validation Sharpe**
   - Re-evaluate on test using τ\*

---

## 7. Results

### 7.1 3-class direction (multi-class)

**Experiment 8 (reference from prior baseline)**
- Validation (2023):
  - Macro-F1: **0.3517**
  - Accuracy: 0.3703
  - AUC: 0.5311
- Test (2024):
  - Macro-F1: **0.3358**
  - Accuracy: 0.3929
  - AUC: 0.5087

**Experiment 9a (integrity hardened)**
- Validation (2023):
  - Macro-F1: **0.3651**
  - Accuracy: 0.3777
  - AUC: 0.5328
- Test (2024):
  - Macro-F1: **0.3058**
  - Accuracy: 0.3702
  - AUC: 0.5054

**Interpretation**
- Validation metrics slightly improve vs Exp8, but **test macro-F1 drops** in 9a.
- This is consistent with removing “too-optimistic” signal sources (split-boundary leakage + same-day daily feature timing).
- Importantly, Experiment 9a results are **more trustworthy** for thesis/paper reporting.

---

### 7.2 UP-vs-REST + trading performance (τ\* chosen by validation Sharpe)

**Experiment 8**
- Selected threshold: **τ\* = 0.50** (max val Sharpe = 1.7816)
- Test (2024) trading:
  - Sharpe: **0.9981**
  - Cumulative return: **0.4519**
  - Avg daily return: 0.001212
  - Avg return in position: 0.002529
  - Hit ratio: 0.2672

**Experiment 9a**
- Selected threshold: **τ\* = 0.45** (max val Sharpe = 2.3361)
- Validation (2023) trading:
  - Sharpe: **2.3361**
  - Cumulative return: **0.9716**
  - Avg daily return: 0.001970
  - Avg return in position: 0.006026
  - Hit ratio: 0.1773
- Test (2024) trading:
  - Sharpe: **1.2075**
  - Cumulative return: **0.4908**
  - Avg daily return: 0.001232
  - Avg return in position: 0.005869
  - Hit ratio: 0.1160

Additional signal behavior (9a):
- Validation positive_rate (fraction of long decisions): **0.3317**
- Test positive_rate: **0.1985**
This indicates a more selective “high-conviction” long filter than Exp8.

**Interpretation**
- Even though multi-class classification generalization worsened, 9a produces a **more selective** signal and improves:
  - test Sharpe (0.998 → 1.207)
  - test cumulative return (0.452 → 0.491)
  - return while in position (0.00253 → 0.00587)
- However, these trading metrics are still **optimistic** because transaction costs/slippage are not yet included (addressed in Experiment 9b).

---

## 8. Conclusion

Experiment 9a is the recommended **new baseline** for future experiments (9b–9e) because it introduces the two most important integrity fixes:

1) **Split-first label computation** eliminates split-boundary leakage  
2) **Lagged daily features (t−1 day)** makes daily on-chain/sentiment availability realistic

Key outcome:
- Reported directional prediction metrics become more conservative (especially out-of-sample macro-F1),
- but the pipeline is now **methodologically defensible**, and the trading signal becomes a more selective “filter” that can be evaluated more realistically next.

Next step (Experiment 9b):
- add transaction costs/slippage, buy-and-hold baseline, random exposure baseline, and drawdown metrics to verify whether the Sharpe improvement represents true alpha.
