# Experiment 1 – Temporal Fusion Transformer (TFT) on Daily BTC

## 1. Goal

Test whether a **Temporal Fusion Transformer** can forecast **short-term BTC returns** using only:

- Daily OHLCV data
- A small set of technical indicators
- Simple calendar / halving features

The model predicts **continuous 1-, 3- and 7-day returns** and is evaluated both as a **regressor** and via **up/down direction** derived from the 1-day output.

---

## 2. Data & Targets

### Data

- Source: `BTCUSD_daily.csv` (CryptoDataDownload format)
- Period: 2014-01-01 to ~2025-XX-XX
- Split by year:
  - Train: **2014–2019**
  - Val: **2020**
  - Test: **2021–2024**
- Sliding window:
  - Lookback: `SEQ_LENGTH = 30` days
  - Samples (this run):
    - Train: `1798`
    - Val: `337`
    - Test: `1432`
- 1-day up/down label (based on `future_return_1d > 0`) is close to balanced in all splits, especially on the test set (~50% UP / 50% DOWN).

### Features

**Past covariates (per day)**

- OHLCV:
  - `open`, `high`, `low`, `close`
  - `volume_btc`, `volume_usd`
- Technical indicators:
  - `roc_10`, `atr_14`
  - `macd`, `macd_signal`, `macd_hist`
  - `rsi_14`
- Calendar / halving:
  - `day_of_week`, `is_weekend`, `month`
  - `is_halving_window`, `days_to_next_halving`

**Known future covariates (t+1)**

- `dow_next`
- `is_weekend_next`
- `month_next`
- `is_halving_window_next`

These 4 values are provided as a separate “future” vector for each sample.

### Targets

- **Regression targets** (per sample):  
  `y = [future_return_1d, future_return_3d, future_return_7d]`  
  where each `future_return_hd` is defined as:
  \[
  \text{future\_return\_h d}(t) = \frac{\text{close}_{t+h} - \text{close}_t}{\text{close}_t}
  \]

- **Direction label** (for analysis only):
  - `target_up = 1` if `future_return_1d > 0`, else `0`.

---

## 3. Model & Training

### Model

Implemented in `tft_model.py` as a **simplified TFT encoder**:

- Variable Selection Network (VSN) over 12 past features
- LSTM encoder (`hidden_size = 64`)
- Multi-head self-attention (`num_heads = 4`)
- Gated Residual Networks (GRNs) for:
  - post-LSTM
  - attention output
  - temporal feed-forward
  - future covariate encoder
  - decision fusion (history + t+1 covariates)
- Final linear layer → **3 outputs** (1d, 3d, 7d returns)

### Training

- Script: `train_tft.py`
- Task: `"regression"` (multi-horizon continuous returns)
- Loss: `MSELoss`
- Optimizer: Adam (`lr = 1e-3`, `weight_decay = 1e-5`)
- Batch size: `64`
- Epochs: `20`
- Model selection: **lowest validation RMSE** (aggregated over 1d/3d/7d)
- Best model checkpoint: `models/tft_btc_best.pth`

---

## 4. Evaluation, Results & Conclusion

Evaluation is done in `evaluate_tft.py` on validation and test sets.

Metrics:

- MSE, MAE, RMSE, R² (aggregate and per horizon)
- 1-day **directional accuracy** and **F1** (from sign of predicted vs true return)
- Plots:
  - Training curves
  - 1d predicted return histogram
  - True vs predicted scatter (1d, 3d, 7d)
  - 1d direction confusion matrix and ROC curve

### Validation (2020)

Aggregate (all horizons):

- RMSE: **0.086**
- R²: **−0.29**

Per horizon (approx.):

- **1d**:
  - RMSE: **0.047**
  - R²: **−0.39**
  - Direction accuracy: **0.53**
- **3d**:
  - RMSE: **0.075**
  - R²: **−0.31**
- **7d**:
  - RMSE: **0.120**
  - R²: **−0.31**

The model already underperforms a naive “zero return” forecast (negative R²), but 1-day direction metrics are slightly better here than on the test set.

### Test (2021–2024)

Aggregate (1d + 3d + 7d):

- RMSE: **0.108**
- R²: **−2.12**
- 1-day direction accuracy: **0.49**

Per 1-day horizon:

- RMSE: **0.048**
- R²: **−1.23**
- Direction accuracy: **0.49**
- Direction F1: ~**0.49**
- ROC AUC: ~**0.49**

Per 3-day and 7-day horizons:

- 3d: RMSE ~**0.066**, R² ~**−0.48**, direction accuracy ~**0.49**
- 7d: RMSE ~**0.169**, R² ~**−2.92**, direction accuracy ~**0.47**

### Qualitative behaviour

From the diagnostic plots:

- **1d predicted returns histogram**:  
  Predictions are centered near zero, with most values between roughly −2% and +2%. The model has learned the *scale* of typical daily moves.
- **True vs predicted scatter**:  
  For all horizons, points cluster around (0,0) with no strong alignment along the `y = x` line, indicating very weak correlation between predictions and realised returns.
- **1d confusion matrix & ROC**:  
  Counts in TP/TN/FP/FN are very similar, and the ROC curve lies close to the diagonal (AUC ≈ 0.49), which is consistent with almost random up/down predictions.

### Conclusion

This experiment shows that, under the current setup:

- **Inputs**: daily BTC OHLCV, a small set of standard technical indicators, and simple calendar/halving features.
- **Objective**: regression on raw 1-, 3- and 7-day returns, with direction derived from the 1-day output.

the TFT:

- Produces **realistic return magnitudes** (no crazy outliers; most predictions live in plausible ranges).
- But **does not outperform a naive “zero return” baseline** on 2017–2024 style data (negative R² on val and test).
- And **fails to provide a robust directional signal** (≈50% accuracy and AUC ≈ 0.49 on a balanced test set).

In other words, with these features and this objective, the model is unable to extract a consistent short-horizon edge from daily BTC prices. This motivates later experiments that:

- Change the task to **classification** (UP / FLAT / DOWN),
- Use **richer feature sets** (e.g. on-chain, sentiment, macro),
- And/or move to **higher-frequency data** where more structure may be present.