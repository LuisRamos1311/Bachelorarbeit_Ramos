# Experiment 3 – Log-Return 3-Class Directional TFT on Daily BTC

## 1. Goal

Take the **3-class directional TFT** from Experiment 2 and:

- switch the label construction from **simple 1-day returns** to **1-day log returns**, and  
- keep everything else (data, features, architecture, training) the same,

to test whether this more standard financial definition of returns makes the **next-day UP / FLAT / DOWN regime** more predictable.   

---

## 2. Data & Targets

### Data

Same dataset and splits as Experiments 1–2:

- Source: `BTCUSD_daily.csv` (CryptoDataDownload format)
- Period: 2014-01-01 to ~2025-XX-XX
- Splits:
  - **Train**: 2014–2019  
  - **Val**: 2020  
  - **Test**: 2021–2024
- Sliding window:
  - Lookback: `SEQ_LENGTH = 30` days
  - Samples (after sequence building):
    - Train: `1798`
    - Val: `337`
    - Test: `1432`   

Features (past covariates, scaling, and “known future” calendar / halving flags) are unchanged from Experiment 2. :contentReference[oaicite:2]{index=2}  

### Target: 1-Day Log-Return Direction

We construct a **single 1-day-ahead 3-class label**:

1. Compute the **1-day log return**:
   \[
   \text{log\_return\_1d}(t) = \log\left(\frac{\text{close}_{t+1}}{\text{close}_t}\right)
   \]
2. Apply a symmetric dead-zone threshold `DIRECTION_THRESHOLD = 0.005`:
   - 0 = **DOWN** if log_return_1d < −0.005  
   - 1 = **FLAT** if |log_return_1d| ≤ 0.005  
   - 2 = **UP** if log_return_1d > 0.005  

Key config:

- `USE_LOG_RETURNS = True`
- `TRIPLE_DIRECTION_COLUMN = "direction_3c"`
- `TARGET_COLUMN = "direction_3c"`
- `NUM_CLASSES = 3`
- `FORECAST_HORIZONS = [1]` (single 1-day horizon, multi-horizon ready for future work) :contentReference[oaicite:3]{index=3}  

Label balances are almost unchanged vs Experiment 2 (only a couple of borderline days flip between FLAT and UP); for example on the full dataset:

- DOWN: 36.5%  
- FLAT: 21.8%  
- UP:   41.8%

---

## 3. Model & Training

### Model

We reuse the **simplified Temporal Fusion Transformer** from Experiments 1–2:   

- Variable Selection Network over 12 past features  
- LSTM encoder (`hidden_size = 64`)  
- Multi-head self-attention (`num_heads = 4`)  
- Gated Residual Networks for:
  - post-LSTM encoding
  - attention output
  - temporal feed-forward
  - future covariate encoder
  - decision fusion (history + t+1 covariates)
- Future covariate encoder for 4 “known future” features (`dow_next`, `is_weekend_next`, `month_next`, `is_halving_window_next`)
- Final linear layer:
  - `output_size = NUM_CLASSES = 3` → logits for **DOWN / FLAT / UP**

### Training

Script: `experiment_3/train_tft.py`

- Task: `TASK_TYPE = "classification"`
- Label: `TARGET_COLUMN = "direction_3c"` (log-return based)
- Loss: `CrossEntropyLoss` over 3 classes
- Optimizer: Adam (settings from `TRAINING_CONFIG`)
- Batch size: 64
- Epochs: 20
- Model selection:
  - **Best validation macro-F1**  
  - Best checkpoint saved as `models/tft_btc_best.pth`

The script saves a JSON history and training curves plot under `experiments/` and `plots/`.

---

## 4. Evaluation & Results

Evaluation is handled by `experiment_3/evaluate_tft.py`, following the same pattern as Experiment 2: :contentReference[oaicite:5]{index=5}  

1. **Multi-class metrics** (DOWN / FLAT / UP) on val and test:
   - accuracy
   - macro precision / recall / F1
   - macro AUC (one-vs-rest)

2. **UP-vs-REST trading view**:
   - `UP` = class 2, `REST` = {DOWN, FLAT}
   - On **validation**, run a grid search over thresholds on `P(UP)` to maximise F1.
   - Apply the best threshold to the **test** set.
   - Report binary metrics (accuracy, precision, recall, F1, AUC) and plot:
     - P(UP) histograms (val/test)
     - 3-class confusion matrix
     - UP-vs-REST confusion matrix
     - ROC curve for UP-vs-REST

### Validation (2020, log-returns)

- **3-class metrics**:
  - Accuracy ≈ **0.392**
  - Macro-F1 ≈ **0.375**
  - Macro-AUC ≈ **0.51**

- **UP-vs-REST** (threshold tuned on val, ≈ 0.10):
  - Accuracy ≈ **0.487**
  - Precision ≈ **0.487**
  - Recall ≈ **0.99**
  - F1 ≈ **0.65**
  - AUC ≈ **0.47**

### Test (2021–2024, log-returns)

Using the validation-tuned threshold on the held-out test set:

- **3-class metrics**:
  - Accuracy ≈ **0.411**
  - Macro-F1 ≈ **0.367**
  - Macro-AUC ≈ **0.55**

- **UP-vs-REST**:
  - Accuracy ≈ **0.418**
  - Precision ≈ **0.40**
  - Recall ≈ **0.94**
  - F1 ≈ **0.56**
  - AUC ≈ **0.50**

These are essentially identical to the simple-return results from Experiment 2.

---

## 5. Conclusion

Experiment 3 confirms that:

- Switching from **simple 1-day returns** to **1-day log returns** for the UP / FLAT / DOWN label is **implementation-correct** and conceptually aligned with financial time-series practice.
- However, because the threshold is small (0.5%) and daily BTC moves are mostly small, the label set barely changes, and **predictive performance is unchanged**:
  - macro-F1 remains around **0.37** on the test set,
  - UP-vs-REST F1 ≈ **0.56**, but AUC ≈ **0.50**, indicating a weak ranking signal.

This experiment mainly serves to:

- put the TFT direction model on a more **standard log-return footing**, and  
- confirm that the **fundamental difficulty of 1-day BTC direction** is not an artefact of the return definition.

Future experiments will need to change **something more substantive** (horizon, threshold, feature set, or model regularisation) to move performance meaningfully beyond this baseline.
