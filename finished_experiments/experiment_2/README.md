# Experiment 2 – 3-Class Directional TFT on Daily BTC

## 1. Goal

Extend the TFT setup from **continuous return forecasting** (Experiment 1) to a **3-class direction classification task** on daily BTC:

- Predict whether **tomorrow’s close** is:
  - **DOWN** (meaningful drop),
  - **FLAT** (small move / noise),
  - **UP** (meaningful rise),
- Using the **same architecture, features and splits** as Experiment 1, but with:
  - a **dead-zone threshold** around zero return, and  
  - **CrossEntropyLoss** on a 3-class label instead of MSE on raw returns.

The goal is to see whether *“predicting buckets”* (DOWN / FLAT / UP) is more learnable than predicting raw returns.


## 2. Data & Targets

### Data

Same underlying dataset and time splits as Experiment 1:

- Source: `BTCUSD_daily.csv` (CryptoDataDownload format)
- Period: 2014-01-01 to ~2025-XX-XX
- Split by date:
  - **Train**: 2014–2019  
  - **Val**: 2020  
  - **Test**: 2021–2024
- Sliding window:
  - Lookback: `SEQ_LENGTH = 30` days
  - Samples (after sequence building):
    - Train: `1798`
    - Val: `337`
    - Test: `1432`

The self-test in `data_pipeline.py` prints label distributions and sample counts to confirm everything is wired correctly.

### Features

Feature engineering is unchanged from Experiment 1 – we reuse the same covariates and scaling.

**Past covariates (per day)**

- OHLCV:
  - `open`, `high`, `low`, `close`
  - `volume_btc`, `volume_usd`
- Technical indicators (from TA-Lib):
  - `roc_10`, `atr_14`
  - `macd`, `macd_signal`, `macd_hist`
  - `rsi_14`
- Calendar / halving:
  - `day_of_week`, `is_weekend`, `month`
  - `is_halving_window`, `days_to_next_halving`

Scaling:

- `PRICE_VOLUME_COLS` → `MinMaxScaler`
- `INDICATOR_COLS` → `StandardScaler`

**Known future covariates (t+1)**

Exactly as in Experiment 1, we provide a small set of **“known future”** features for the next day, created via `shift(-1)`:

- `dow_next`
- `is_weekend_next`
- `month_next`
- `is_halving_window_next`

These form a separate vector of size 4 for each sample.

### Targets

In Experiment 2, we replace the continuous regression target with a **3-class direction label** derived from the 1-day forward return.

We still compute forward returns:

- `future_return_1d = (close_{t+1} - close_t) / close_t`
- `future_return_{h}d` for `h` in `FORECAST_HORIZONS = [1, 3, 7]`

These multi-horizon returns remain available for future experiments, but the *primary* label for Experiment 2 is:

direction_3c (TRIPLE_DIRECTION_COLUMN)

0 = DOWN  if future_return_1d < -DIRECTION_THRESHOLD
1 = FLAT  if |future_return_1d| <= DIRECTION_THRESHOLD
2 = UP    if future_return_1d >  DIRECTION_THRESHOLD

Configured in `config.py` as:

- `TRIPLE_DIRECTION_COLUMN = "direction_3c"`
- `NUM_CLASSES = 3`
- `DIRECTION_THRESHOLD = 0.005` (0.5% dead-zone in the final runs)

The label is created in `data_pipeline.add_target_column`, and all rows that lack sufficient future data for `[1, 3, 7]` days are dropped.

**Label balance (example with DIRECTION_THRESHOLD = 0.005)**

- **Full dataset**:  
  - DOWN: 36.5%  
  - FLAT: 21.7%  
  - UP:   41.8%

- **Train (2014–2019)**:  
  - DOWN: 34.9%  
  - FLAT: 22.3%  
  - UP:   42.8%

- **Val (2020)**:  
  - DOWN: 35.5%  
  - FLAT: 15.8%  
  - UP:   48.6%

- **Test (2021–2024)**:  
  - DOWN: 38.8%  
  - FLAT: 21.4%  
  - UP:   39.8%

So the task is **not extremely imbalanced**, but FLAT is clearly the smallest class.

### Differences vs Experiment 1

Compared to Experiment 1’s **continuous return regression**:

- We now **discretise** the 1-day return into 3 classes with a symmetric dead-zone.
- The **objective** is classification (CrossEntropyLoss) instead of MSE.
- Direction is now **native** (DOWN/FLAT/UP) rather than derived from the sign of a regression output.


## 3. Model & Training

### Model

We keep the same **simplified Temporal Fusion Transformer** as in Experiment 1, implemented in `tft_model.py`:

- Variable Selection Network (VSN) over 12 past features
- LSTM encoder (`hidden_size = 64`)
- Multi-head self-attention (`num_heads = MODEL_CONFIG.num_heads`)
- Gated Residual Networks (GRNs) around:
  - Post-LSTM encoding
  - Attention output
  - Temporal feed-forward block
  - Future covariate encoder
  - Decision fusion (history + known-future covariates)
- Future covariate encoder for the 4 “t+1” features
- Final linear layer with:
  - `output_size = NUM_CLASSES = 3` for Experiment 2

The architecture is identical to Experiment 1, except that the last layer produces **3 logits (DOWN, FLAT, UP)** instead of 3 continuous returns.

### Training

Training is handled by `train_tft.py`. For Experiment 2, configuration and scripts are wired to **classification mode**:

- Script: `experiment_2/train_tft.py`
- Task: `TASK_TYPE = "classification"`
- Label: `TARGET_COLUMN = "direction_3c"`
- Model: `TemporalFusionTransformer` from `experiment_2.tft_model`
- Loss: `CrossEntropyLoss` over 3 classes (DOWN/FLAT/UP)
- Optimizer: Adam (learning rate, weight decay etc. taken from `TRAINING_CONFIG`)
- Batch size: `TRAINING_CONFIG.batch_size` (64 in the baseline)
- Epochs: `TRAINING_CONFIG.epochs = 20`
- Model selection:
  - **Best validation macro-F1** (averaged over all 3 classes)
  - Best checkpoint saved as `models/tft_btc_best.pth`

At the end of training, the script stores:

- A JSON history (`experiments/..._history.json`) with loss + metrics per epoch
- Training curves (`plots/..._training_curves.png`)


## 4. Evaluation, Results & Conclusion

Evaluation is performed by `experiment_2/evaluate_tft.py`.

For **classification**:

- Runs the model on validation and test sets.
- On the **validation set**, performs a **threshold search** on the **UP** class probability `P(UP)` to maximise a target metric (default: F1) for a *binary* “UP vs REST” decision:
  - REST = {DOWN, FLAT}
- Fixes this best threshold and uses it on the **test set**.
- Produces:
  - Multi-class metrics (accuracy, precision, recall, macro-F1, macro-AUC)
  - UP-vs-REST metrics at the tuned threshold
  - Plots: probability histograms, confusion matrices, ROC curve

### Validation (2020)

For a representative run with `DIRECTION_THRESHOLD = 0.005`:

- **3-class metrics**:
  - Accuracy ≈ **0.395**
  - Macro-F1 ≈ **0.378**
  - Macro-AUC ≈ **0.51**

- **UP-vs-REST** (after tuning the P(UP) threshold on val):
  - Best threshold ≈ **0.10**
  - Accuracy ≈ **0.49**
  - Precision ≈ **0.49**
  - Recall ≈ **0.99**
  - F1 ≈ **0.65**
  - AUC ≈ **0.48**

The model tends to **over-predict UP** at this threshold, giving very high recall but only moderate precision.

### Test (2021–2024)

Using the validation-tuned threshold on the held-out test set:

- **3-class metrics**:
  - Accuracy ≈ **0.41**
  - Macro-F1 ≈ **0.37**
  - Macro-AUC ≈ **0.55**

- **UP-vs-REST** (threshold ≈ 0.10):
  - Accuracy ≈ **0.42**
  - Precision ≈ **0.40**
  - Recall ≈ **0.94**
  - F1 ≈ **0.56**
  - AUC ≈ **0.50**

The 3×3 confusion matrix shows that:

- The model correctly identifies many UP and DOWN days,
- But often confuses FLAT with directional moves, which is expected given FLAT is the minority regime.

### Comparison to Experiment 1

Experiment 1 (regression on 1d/3d/7d returns) showed:

- Negative R² vs a **zero-return / mean-return baseline** on both val and test.
- 1-day **direction accuracy ~0.49** and **AUC ~0.49** – essentially random.

Experiment 2 improves the **framing**:

- It **acknowledges small moves as noise** (FLAT) via `DIRECTION_THRESHOLD`.
- It introduces a **native directional objective** and directly optimises F1.
- UP-vs-REST F1 on test reaches around **0.56**, but with AUC ≈ **0.50**, suggesting the ranking of probabilities is still close to random even if a particular operating point looks decent.

Overall, Experiment 2:

- Delivers **slightly more structured behaviour** than raw return regression.
- But **does not yet provide a robust, high-AUC signal** for BTC direction, even after introducing the FLAT class and threshold tuning.
- Serves as a **clean baseline** for:
  - trying different FLAT thresholds (`DIRECTION_THRESHOLD`),
  - richer feature sets (on-chain, sentiment, macro),
  - and more advanced training strategies (class weighting, focal loss, ablations vs the LSTM, etc.).

For the thesis, Experiment 2 is the **starting point for all future TFT direction-based experiments**, while Experiment 1 is kept as a reference showing the limitations of pure regression in this setting.