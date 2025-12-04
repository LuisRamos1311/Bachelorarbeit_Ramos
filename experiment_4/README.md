# Experiment 4 – Binary Big-Move Directional TFT on Daily BTC

## 1. Goal

This is a **branch experiment** built on top of the 3-class directional TFT from Experiments 2–3.

The question:

> *If we only look at days where BTC moves by more than ±0.5% (in log-return terms), 
> can a TFT predict whether that move will be UP or DOWN better than random?*

To test this, we:

- Keep the same data, features, splits and architecture as Experiment 3.
- Change the **label** to a **binary “big-move direction”**.
- Drop all “small-move / noise” days from train/val/test.

---

## 2. Data & Targets

### Data

Same BTC daily dataset and splits as previous experiments:

- **Train**: 2014–2019  
- **Val**: 2020  
- **Test**: 2021–2024  
- Lookback window: `SEQ_LENGTH = 30` days  

After sequence building **and filtering out small moves** (`|log_return_1d| ≤ 0.005`):

- Train: `N = 1398` (44.7% DOWN / 55.3% UP)  
- Val:   `N = 282`  (41.8% DOWN / 58.2% UP)  
- Test:  `N = 1122` (49.4% DOWN / 50.6% UP)  

Feature engineering, scaling and known-future covariates are unchanged from Experiment 3 
(OHLCV, technicals, calendar & halving flags + 4 known-future covariates).

### Target: Binary Big-Move Direction

We keep `USE_LOG_RETURNS = True` and the same dead-zone threshold `DIRECTION_THRESHOLD = 0.005` (≈0.5%).  

For each day *t*:

1. Compute the 1-day log return  

   \[
   \text{log\_return\_1d}(t) = \log\left(\frac{\text{close}_{t+1}}{\text{close}_t}\right)
   \]

2. Define the label `direction_2c_bigmove`:

- `0 = DOWN` if log_return_1d < −0.005  
- `1 = UP`   if log_return_1d >  +0.005  
- `IGNORE_LABEL = −1` if |log_return_1d| ≤ 0.005 (these samples are **dropped**)

`direction_3c` from Experiment 3 is still computed for compatibility, but 
**Experiment 4 trains and evaluates only on `direction_2c_bigmove`**.

---

## 3. Model & Training

### Model

We reuse the same **Temporal Fusion Transformer encoder** as in Experiments 1–3: 
Variable Selection Network over 12 past features, LSTM encoder, multi-head self-attention, GRNs, 
and a GRN-based future covariate encoder.

For this experiment:

- `NUM_CLASSES = 2`
- Final linear layer outputs **2 logits**: `[DOWN, UP]`.

### Training

- Script: `experiment_4/train_tft.py`
- Task: `TASK_TYPE = "classification"`
- Label: `TARGET_COLUMN = "direction_2c_bigmove"`
- Loss: `CrossEntropyLoss` over 2 classes (optionally with class weights)
- Optimizer: Adam (same `TRAINING_CONFIG` as previous experiments)
- Batch size: `64`
- Epochs: `20`
- Model selection: **best validation F1** on the binary label
- Best checkpoint: `models/tft_btc_best.pth`

Training curves show:

- **Train loss** decreasing from ~0.76 → ~0.63  
- **Val loss** drifting upward after early epochs (overfitting)  
- **Best val F1 ≈ 0.72** at epoch 4 (noisy and not stable across epochs)

---

## 4. Evaluation & Results

Evaluation uses `experiment_4/evaluate_tft.py`:

1. Run the model on the **validation set**, extract `P(UP)` and
   perform a **threshold search** on `P(UP)` to maximise F1.
2. Apply the best threshold to the **test set**.
3. Report binary metrics and generate:
   - `P(UP)` histograms (val / test),
   - a binary confusion matrix,
   - ROC curve (UP vs DOWN).

### Validation (2020, big-move subset)

- Best `P(UP)` threshold (by F1): **0.10**  
- Metrics at this threshold:
  - Accuracy: **0.58**
  - Precision: **0.58**
  - Recall: **1.00**
  - F1: **0.74**
  - **AUC: 0.44** (worse than random)

Because almost all validation probabilities sit between ~0.55 and 0.75, a threshold of 0.10 makes the classifier 
**predict UP on essentially every sample**. These metrics therefore match an **“always-UP” baseline** on a dataset 
where ≈58% of big-move days are UP.

### Test (2021–2024, big-move subset)

Using the validation-tuned threshold `P(UP) ≥ 0.10`:

- Loss (CE): **0.72**
- Accuracy: **0.51**
- Precision: **0.51**
- Recall: **1.00**
- F1: **0.67**
- **AUC: 0.50** (random)

Confusion matrix on the test big-move subset:

- True DOWN: `554` → predicted **UP** every time  
- True UP:   `568` → predicted **UP** every time  

So the model reduces to:

> **Predict UP for every big-move day.**

On a nearly balanced test set (49.4% DOWN / 50.6% UP), this is *exactly* the trivial baseline.

### Comparison to Experiment 3

Experiment 3 (3-class log-return TFT with UP-vs-REST view) already showed a **random-like AUC (~0.50)** 
and a tendency to over-predict UP.  

Experiment 4 hoped that **conditioning on larger moves** (|log-return| > 0.5%) might make the *direction* 
of those moves more predictable. Instead, it confirms:

- The TFT **still cannot rank UP vs DOWN** on these days (AUC ≈ 0.50),
- And collapses to the same “always-UP” behaviour, now on a balanced subset.

---

## 5. Conclusion

Experiment 4 is a **targeted branch experiment** that:

- Reframes the task as **binary big-move direction** (DOWN vs UP) and
  drops small-move days as noise.
- Uses the **same TFT architecture and features** as Experiment 3.
- Finds that, even under this easier and more trading-relevant conditioning,
  the model’s **directional skill remains indistinguishable from random**.

This negative result suggests that:

- With daily OHLCV + standard technicals + simple calendar/halving features,
  **even the sign of larger BTC moves is hard to predict**.
- Further improvement will likely require **richer features (e.g. on-chain / macro / sentiment), 
  different horizons, or different modelling choices**, rather than just filtering by move size.

Experiment 4 therefore serves as a **useful diagnostic branch** documenting that “big-move days” are not obviously more 
predictable with the current TFT setup, and it complements the main 3-class direction experiments (Experiments 2–3).
