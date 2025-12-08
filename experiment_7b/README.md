# Experiment 7b – Hourly On-Chain TFT with Trading-Focused Thresholding

This experiment builds on **Experiment 6** (hourly BTC TFT) and **Experiment 7** (adds on-chain data) and then **refines the model and evaluation** to focus more on *trading performance* (returns, Sharpe) rather than pure classification accuracy.

---

## 1. Motivation

- Start from the Exp-7 setup:  
  **1h BTC OHLCV + technical indicators + 5 on-chain features**, 24-hour-ahead 3-class direction label.
- The Exp-7 base model was slightly better than Exp-6 in classification, but trading performance was still weak and often negative.
- **Experiment 7b** keeps the same dataset and label, but:
  1. Simplifies and regularises the TFT (smaller hidden size, higher dropout, stronger weight decay) to reduce overfitting.
  2. Treats the TFT output more explicitly as a **trading signal**, choosing the UP-vs-REST threshold τ by **Sharpe ratio** instead of a pure classification metric.

Goal: obtain a **more robust, trading-oriented TFT baseline** on hourly BTC with on-chain features.

---

## 2. Data, Features & Labels

**Data frequency**

- BTCUSD 1-hour candles (`BTCUSD_hourly.csv`), from **2018-05-15 to 2025-12-31**.
- Splits:
  - **Train:** 2018-05-15 to 2023-12-31  
  - **Validation:** 2024-01-01 to 2024-12-31  
  - **Test (out-of-sample):** 2025-01-01 to 2025-12-31  

**Past covariates (X)**

- Price & volume (MinMax-scaled):
  - `open, high, low, close, volume_btc, volume_usd`
- Technical indicators (StandardScaler):
  - `roc_10, atr_14, macd, macd_signal, macd_hist, rsi_14`
- On-chain features (forward-filled daily → hourly, StandardScaler with indicators):
  - `aa_ma_ratio` – active addresses / 7-day MA  
  - `tx_ma_ratio` – tx count / 7-day MA  
  - `mvrv_z` – z-score of MVRV  
  - `sopr_z` – z-score of SOPR  
  - `hash_ma_ratio` – hash rate / 30-day MA  

Total **17 past features** (6 price/volume + 6 indicators + 5 on-chain).

**Label / target**

- **Task:** 24-hour-ahead **3-class direction** (classification)
- Horizon: `FORECAST_HORIZONS = [24]` steps = **next 24 hours**
- Log returns: `USE_LOG_RETURNS = True`
- Direction label: `direction_3c`  
  - 0 = **DOWN** if H-step log return < –0.005  
  - 1 = **FLAT** if |H-step log return| ≤ 0.005  
  - 2 = **UP** if H-step log return > +0.005  

---

## 3. Model & Training Setup

### Temporal Fusion Transformer (simplified/regularised)

- Hidden size: **32** (reduced from 64)
- Input projection: `Linear(17 → 32)`
- Variable Selection Network (VSN) on past covariates:
  - value projection: `17 → 17×32`  
  - weight GRN hidden size: 32
- Temporal encoder:
  - LSTM encoder: `LSTM(32, 32, batch_first=True)`
  - Multi-head attention: d_model = 32
  - GRNs after LSTM and attention (hidden size 32 → 64 → 32)
- Future covariates GRN:
  - Input: 5 known-future features (calendar + halving)  
  - GRN hidden size: 32
- Decision GRN:
  - 32 → 64 → 32
- Output layer:
  - `Linear(32 → 3)` (3 classes)

### Regularisation

- **Dropout:** increased to **0.3** in GRNs (vs. 0.2 before)
- **Weight decay:** increased to **5e-4** (previously weaker)
- Sequence length: `SEQ_LENGTH = 96` (model sees **last 96 hours ≈ 4 days**)
- Loss: `CrossEntropyLoss` on the 3-class label

### Training details

Configured in `config.py` / `train_tft.py`:

- Epochs: 12
- Optimiser: Adam
- Learning rate: `1e-3`
- Batch size: 128 (as in earlier hourly experiments)
- Model selection: **best validation macro-F1** on the 3-class task

---

## 4. Evaluation & Trading Logic

Evaluation happens in `evaluate_tft.py`:

1. **Multi-class metrics** (3-class direction, argmax over logits)
   - Accuracy, macro precision/recall/F1, AUC on **val** and **test**.
2. **UP-vs-REST view**:
   - Collapse labels into binary:  
     - `UP` = class 2  
     - `REST` = classes 0 or 1
   - Use `P(UP)` from the softmax output.
   - Consider a long-only signal: **enter BTC** if `P(UP) ≥ τ`, otherwise stay flat.
3. **Threshold grid for τ** (UP-vs-REST):
   - Grid: `τ ∈ {0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65}`
   - Non-overlapping trades: each decision holds from *t* to *t+24h* (matching the label horizon).
4. **Threshold selection based on trading metric (Sharpe)**:
   - For each τ in the grid:
     - Compute validation long-only trading metrics (non-overlapping 24h trades):
       - Average daily return
       - Cumulative return
       - Sharpe ratio
       - Hit ratio (fraction of profitable trades)
       - Average return while in position
   - **Select τ\* = argmax Sharpe on the validation set.**
   - Re-compute UP-vs-REST metrics and trading metrics on the test set using τ\*.

This shift means the model is now evaluated as a **trading strategy first**, classification model second.

---

## 5. Key Results (Final 7b Run)

Numbers below are for the final configuration (reduced TFT, dropout 0.3, weight decay 5e-4, Sharpe-based τ selection).

### 3-class direction (multi-class)

- **Validation (2024)**
  - Macro-F1 ≈ **0.366**
  - Accuracy ≈ 0.38
  - AUC ≈ 0.54
- **Test (2025)**
  - Macro-F1 ≈ **0.367**
  - Accuracy ≈ 0.37
  - AUC ≈ 0.53

Classification remains modestly above random (1/3), but consistent across years.

### UP-vs-REST + trading (τ\* chosen by Sharpe on validation)

- **Selected threshold:** τ\* = **0.35** (on validation Sharpe)

**Validation (2024)**

- Balanced accuracy ≈ **0.52**
- F1 (UP class) ≈ **0.43**
- Sharpe ≈ **0.87**
- Cumulative return ≈ **+38%**
- Avg daily return ≈ **0.11%**
- Avg return while in position ≈ 0.29%

**Test (2025)**

- Balanced accuracy ≈ **0.51**
- F1 (UP class) ≈ **0.42**
- Sharpe ≈ **0.69**
- Cumulative return ≈ **+20%**
- Avg daily return ≈ **0.065%**
- Avg return while in position ≈ 0.14%

Compared to the **Exp-6 baseline** (technical-only TFT with negative test Sharpe and negative cumulative returns), Exp-7b:
- Keeps classification performance in a similar range, and
- **Turns the long-only strategy from negative to clearly positive Sharpe and cumulative return** on the 2025 out-of-sample period.

---

## 6. Conclusion

Experiment 7b extends the original hourly TFT setup by adding a compact set of on-chain features and then carefully aligning the model and evaluation with a **trading objective** rather than pure classification accuracy. Starting from an over-parameterized TFT that overfit the training data and delivered weak, often negative out-of-sample returns, the experiment progressively:

1. **Reduced model complexity and increased dropout** to improve generalisation.
2. **Refined the UP-vs-REST threshold grid** to focus on more meaningful prediction confidence levels.
3. **Shifted threshold selection from balanced accuracy to Sharpe ratio**, explicitly optimising the decision rule for trading performance.
4. **Introduced stronger weight decay** to further stabilise the model and reduce overfitting.

The final configuration still achieves only moderate 3-class classification metrics (macro-F1 and AUC modestly above random), which is expected given the noisy nature of hourly crypto returns. However, when combined with Sharpe-based threshold selection and non-overlapping 24h trades, it yields a **profitable long-only strategy** on the 2025 out-of-sample test period, with positive cumulative return and Sharpe ratio.

In summary, Experiment 7b shows that:

- On-chain information can provide incremental edge when integrated with technical features in a TFT-style architecture.
- Careful regularisation and architecture simplification are essential to avoid overfitting in this setting.
- Most importantly, **optimising the decision threshold for trading metrics** (rather than generic classification scores) is crucial for turning a weak directional edge into a meaningful trading signal.
