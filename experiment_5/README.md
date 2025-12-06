# Experiment 5 – TFT Robustness Tweaks on Daily BTC

Experiment 5 is a set of **small, controlled tweaks** built on top of the 3-class log-return TFT from **Experiment 3**.  
All runs keep the same:

- Dataset: `BTCUSD_daily.csv` (2014–2025, daily)
- Features: OHLCV + TA indicators + calendar/halving + 4 known-future covariates
- Model: simplified Temporal Fusion Transformer (VSN + LSTM + attention + GRNs)
- Task: 1-day-ahead **3-class log-return direction**  
  - `0 = DOWN`, `1 = FLAT`, `2 = UP` with `DIRECTION_THRESHOLD = 0.005` (unless stated otherwise)
- Training: 20 epochs, Adam, batch size 64, model selection by **best validation macro-F1**

The goal is to test whether **minor but plausible choices** (time split, label threshold, decision threshold) can unlock a useful edge, or whether the weak performance seen in Experiments 2–4 is robust.

This README is organised in three parts:

1. Date-range / time-split robustness ✔️
2. Label threshold tweaks (`DIRECTION_THRESHOLD`) ✔️
3. Decision threshold tweaks in `evaluate_tft` (P(UP) threshold, Experiment 5c) ✔️

---

## Part 1 – Date-Range / Time-Split Robustness

### 1.1 Goal

Test whether changing the **training / validation / test years** significantly alters TFT performance on next-day BTC direction:

> Is the model’s failure in Experiment 3 mainly due to training on “ancient” BTC history, or does the weak predictive power persist even when we restrict to modern regimes?

We compare three time-split configurations, all with the same label and architecture.

### 1.2 Time-Split Configurations

All splits use a 30-day rolling window (`SEQ_LENGTH = 30`).

#### Split A – Long-history baseline (Experiment 3)

- Train: **2014–2019**  
- Val: **2020**  
- Test: **2021–2024**  
- Samples:
  - Train: 1798
  - Val: 337
  - Test: 1432
- Label balance (full period):  
  DOWN 36.5%, FLAT 21.8%, UP 41.8%

This is the original baseline used for Experiments 2–3.

---

#### Split B – Modern multi-cycle (Experiment 5, 2017–2024)

- Train: **2017–2020**  
- Val: **2021**  
- Test: **2022–2024**  
- Samples:
  - Train: 1432
  - Val: 336
  - Test: 1067
- Label balance (2017–2024 full):  
  DOWN 37.6%, FLAT 19.5%, UP 42.8%

Motivation: drop pre-2017 “early BTC” years and train only on a more liquid, mature market.

---

#### Split C – Recent-only regime (Experiment 5, 2022–2025)

- Train: **2022–2023**  
- Val: **2024**  
- Test: **2025**  
- Samples:
  - Train: 701
  - Val: 337
  - Test: 304
- Label balance per split:
  - Train: 36.7% DOWN / 26.7% FLAT / 36.6% UP  
  - Val:   38.5% DOWN / 19.4% FLAT / 42.1% UP  
  - Test:  36.0% DOWN / 26.7% FLAT / 37.2% UP

Motivation: focus on the most recent ETF / post-2022-crash regime and see if short-term structure becomes easier to learn.

---

### 1.3 Training Behaviour (Split C example)

On the 2022–23 / 2024 / 2025 split:

- Train loss: **1.16 → 0.90**, train macro-F1: **0.34 → 0.56** (steady improvement)
- Val loss: **~1.07 → ~1.25**, val macro-F1 fluctuates around **0.28–0.32**

This is classic **overfitting**: with only 701 training samples, the TFT fits the training window well but generalises poorly to 2024. The other splits show the same pattern: training metrics improve steadily while validation metrics move only slightly and remain modest.

---

### 1.4 Results Overview

#### 3-Class Direction (DOWN / FLAT / UP) – Test Sets

| Split / Experiment | Train Years     | Val Year | Test Years  | Test CE | Test Accuracy | Test Macro-F1 | Test Macro AUC |
|--------------------|----------------|----------|-------------|---------|---------------|---------------|----------------|
| **A – Exp3**       | 2014–2019      | 2020     | 2021–2024   | 1.292   | **0.411**     | **0.367**     | **0.548**      |
| **B – 2017+**      | 2017–2020      | 2021     | 2022–2024   | **1.138** | 0.380       | 0.335         | 0.544          |
| **C – 2022+**      | 2022–2023      | 2024     | 2025        | 1.270   | 0.375         | 0.345         | 0.540          |

Key points:

- All splits yield **similar macro-AUC (~0.54–0.55)** and modest macro-F1.
- The **long-history baseline (Split A)** achieves the **highest test accuracy and macro-F1**, despite including “ancient” BTC data.
- Split B has the lowest CE (slightly better probability calibration) but **worse classification metrics**.
- Split C has the smallest training set and shows the strongest overfitting; performance is not better than the other splits.

---

#### UP vs REST (Binary, P(UP) Threshold Tuned on Val)

For each split, `evaluate_tft.py`:

1. Treats `UP` (class 2) as positive, `REST = {DOWN, FLAT}` as negative.  
2. Searches thresholds on `P(UP)` on the **validation set** to maximise F1.  
3. Applies the best threshold (≈ **0.10** in all cases) to the **test set**.

| Split / Experiment | Test Accuracy | Test Precision | Test Recall | Test F1 | Test AUC |
|--------------------|--------------|----------------|------------|---------|----------|
| **A – Exp3**       | **0.418**    | **0.400**      | 0.938      | **0.561** | 0.496  |
| **B – 2017+**      | 0.386        | 0.386          | 1.000      | 0.557   | 0.510  |
| **C – 2022+**      | 0.368        | 0.368          | 1.000      | 0.539   | 0.533  |

However, for all splits the F1-optimal threshold (τ ≈ 0.10) effectively yields an **“always-UP” classifier**:

- On test, **recall = 1.0** and `TN = 0`, `FN = 0`; every day is classified as UP.
- Accuracy and F1 therefore just reflect the proportion of UP days (~37–42%).
- UP-vs-REST AUC stays close to **0.5**, showing **no meaningful ranking power** in `P(UP)`.

Histograms and ROC curves confirm this: P(UP) is bunched between ~0.25 and 0.6 on val/test, and ROC curves hug the diagonal.

---

### 1.5 Conclusion – Date-Range Robustness

- **No time split produces strong predictive performance.**  
  Across all three configurations, the TFT’s out-of-sample metrics are similar:
  - 3-class macro-F1 ≈ **0.33–0.37**
  - 3-class macro-AUC ≈ **0.54–0.55**
  - UP-vs-REST AUC ≈ **0.50–0.53**

- **More data beats less data.**  
  The **long-history split (2014–2019 train)** consistently performs as well as or slightly better than the “modern only” splits, despite spanning more heterogeneous BTC regimes. Removing early years does **not** unlock a meaningful edge.

- **Recent-only windows remain negative evidence.**  
  Even when training exclusively on **2022–2023** and testing on **2025**, the model:
  - Overfits the small training set,  
  - Achieves test accuracy barely above an always-UP baseline,  
  - Cannot produce a useful UP-vs-REST trading signal.

**Decision for further work**

- For the remainder of the thesis, **Split A (Experiment 3: 2014–2019 / 2020 / 2021–2024)** is kept as the **main baseline**, because:
  - It uses the largest training set,  
  - It delivers the best overall 3-class test metrics,  
  - It aligns with earlier experiments and the original LSTM baseline.

- Splits B and C are retained as **robustness experiments** that support the conclusion:

> The difficulty of forecasting 1-day BTC direction with TFT is **not** an artefact of including old data or ignoring recent regimes; it appears robust across multiple time windows.

---

## Part 2 – Label Threshold Tweaks (`DIRECTION_THRESHOLD`)

### 2.1 Goal

Investigate how the **dead-zone threshold** used to define DOWN / FLAT / UP in the 1-day log-return label affects performance:

- Smaller thresholds (e.g. `0.003`) move more small moves into DOWN/UP (few FLAT days).  
- Larger thresholds (e.g. `0.010`) create many FLAT days and reserve DOWN/UP for bigger moves.  
- Threshold `0.0` is effectively a **binary** label with almost no FLAT samples.

All runs in this section use the **Split A baseline**:

- Train: **2014–2019**, Val: **2020**, Test: **2021–2024**
- Same TFT architecture, optimiser and training setup as Part 1.

### 2.2 Label distributions

Label balance over **2014–2025 (FULL)** for each threshold:

| `DIRECTION_THRESHOLD` | DOWN | FLAT | UP | Comment |
|-----------------------|------|------|----|---------|
| **0.000**             | 47.5% | **0.2%** | 52.3% | Almost pure 2-class (FLAT vanishes) |
| **0.003**             | 40.7% | 13.6% | 45.6% | Mild dead zone; still directional-heavy |
| **0.005** (baseline)  | 36.5% | 21.8% | 41.8% | Balanced; FLAT ≈ 1/5 of days |
| **0.010**             | 27.8% | **38.5%** | 33.6% | FLAT dominates; only large moves counted as direction |

As expected, increasing the threshold shifts probability mass from DOWN/UP into FLAT.

### 2.3 Test performance by threshold

3-class metrics on the **2021–2024** test window:

| `DIRECTION_THRESHOLD` | Test Accuracy | Test Macro-F1 | Test Macro AUC | UP-vs-REST AUC | Notes |
|-----------------------|---------------|---------------|----------------|----------------|-------|
| **0.000**             | **0.495**     | 0.330         | **0.470**      | 0.498          | Binary-like labels; high accuracy but AUC < 0.5 |
| **0.003**             | 0.426         | 0.328         | 0.548          | 0.504          | Slight bias towards DOWN; still weak |
| **0.005** (baseline)  | 0.411         | **0.367**     | 0.548          | 0.496          | Best macro-F1; moderate AUC |
| **0.010**             | 0.360         | 0.344         | **0.565**      | **0.533**      | Many FLAT days; slightly higher AUC |

Observations:

- **Threshold = 0.0**  
  - Labels become almost perfectly balanced DOWN vs UP with no FLAT.  
  - The model reaches the highest raw **accuracy** (~0.50) on test, but **macro-AUC drops below 0.5**, and the confusion matrix shows almost random guessing between DOWN and UP.  
  - This suggests the task becomes easier in a trivial sense (“just pick one of two balanced classes”) but **not more informative**.

- **Thresholds 0.003, 0.005, 0.010**  
  - All three give **similar macro-AUC (~0.55–0.57)** and macro-F1 around 0.33–0.37.  
  - `0.003` pushes many tiny moves into DOWN/UP; the model becomes slightly biased and still weak.  
  - `0.010` creates many FLAT days; AUC improves a bit but at the cost of a heavily imbalanced FLAT-heavy label that ignores a large fraction of the sample as “noise”.

- **UP-vs-REST**  
  - For `0.003` the F1-optimal threshold is ≈0.20; for `0.005` and `0.010` it is ≈0.10.  
  - In all cases, the F1-optimal threshold leads to **very high recall (~0.95–1.0)** and moderate precision, i.e. the classifier predicts UP on almost every day.  
  - UP-vs-REST AUC stays near **0.5** for all thresholds, again indicating almost no useful ranking signal in `P(UP)`.

### 2.4 Is the 3-class label helpful?

Comparing `0.0` (almost binary) to the 3-class thresholds:

- Going to `0.0` does **not** improve the underlying information content: macro-AUC actually gets **worse** than for 0.003–0.010, even though accuracy goes up.
- The 3-class setup with a small dead zone:

  - avoids forcing the model to classify **tiny, economically meaningless moves** as UP or DOWN;
  - makes interpretation more natural for trading: FLAT = “no edge / avoid trading”, DOWN/UP = “move big enough to matter after costs”.

So the FLAT class is **conceptually useful** and **empirically not harmful**: thresholds 0.003–0.010 all perform similarly (and better in AUC than 0.0), while giving more realistic labels.

### 2.5 Conclusion – Preferred `DIRECTION_THRESHOLD`

- **No threshold produces strong predictive power.**  
  All thresholds give macro-F1 around 0.33–0.37 and macro-AUC around 0.55.

- **`DIRECTION_THRESHOLD = 0.005` is kept as the canonical choice**, because:

  - it yields the **best test macro-F1** among the tested thresholds;
  - it keeps a **balanced label distribution** (FLAT ≈ 22%, DOWN/UP ≈ 40% each);
  - it matches the intuition of ignoring <0.5% daily moves as noise while still retaining enough directional samples.

- Threshold `0.010` produces slightly higher AUC but with a very FLAT-heavy label and fewer directional events, which is less appealing for downstream trading analysis.

Overall, the label threshold affects class balance and interpretation more than it affects pure predictive performance; it is **not the main bottleneck** of the model.

---

## Part 3 – Decision Threshold Tweaks in `evaluate_tft` (Experiment 5c)

### 3.1 Goal

Parts 1–2 focused on **data splits** and **label definitions**.  
Experiment 5c keeps the Split A baseline and the `DIRECTION_THRESHOLD = 0.005` label, but changes *how we turn P(UP) into a decision / trading signal*:

- Previously (`evaluate_tft.py` in Experiments 3, 5a, 5b), the script chose the UP-vs-REST **decision threshold τ** that maximised **F1 for the UP class** on the validation set.  
  - This consistently produced **τ ≈ 0.10**, leading to an almost “always-UP” classifier.
- In Experiment 5c we instead:
  1. **Sweep a small grid of thresholds** on P(UP):  
     `UP_THRESHOLD_GRID = [0.10, 0.20, 0.30, 0.40, 0.50]`.
  2. For each τ, compute **binary metrics** (accuracy, precision, recall, F1, AUC, *balanced accuracy*, macro-F1, positive_rate).  
  3. Select τ\* that maximises **balanced accuracy** on the validation set, not F1 for UP.  
  4. For each τ (and especially τ\*), evaluate a **simple long-only trading strategy**:
     - Long BTC from *t* to *t+1* if `P(UP)_t ≥ τ`, otherwise flat.

Balanced accuracy explicitly weights **true positive rate (UP days)** and **true negative rate (NOT_UP days)** equally, so it penalises trivial “predict UP every day” behaviour.

### 3.2 Setup and Implementation

All 5c runs reuse the **Split A baseline** and the TFT from Parts 1–2:

- Train: **2014–2019**, Val: **2020**, Test: **2021–2024**
- 3-class log-return label with `DIRECTION_THRESHOLD = 0.005`
- Same TFT architecture and 20-epoch training loop
- Best checkpoint chosen by **validation macro-F1 (3-class)**

For evaluation:

1. `evaluate_tft.py` runs multi-class inference on val/test and computes the usual **3-class metrics** (loss, accuracy, macro-precision/recall/F1, macro-AUC).  
   - These numbers are identical to Split A in Part 1.
2. It extracts **P(UP)** and converts the 3-class labels into **binary UP-vs-REST** labels.  
3. It obtains **daily forward returns** `future_return_1d` aligned with each sample to support trading analysis.
4. For each τ in `UP_THRESHOLD_GRID`:

   - Builds binary predictions `1{ P(UP) ≥ τ }`.  
   - Computes binary metrics using `compute_classification_metrics`.  
   - Constructs positions (1 = long, 0 = flat) and computes trading metrics via `compute_trading_metrics`, including:
     - average daily return,  
     - cumulative return,  
     - Sharpe ratio (simple, annualised),  
     - hit ratio (fraction of positive days when in position),  
     - average return when in position.

5. Among all τ, it selects **τ\*** that maximises **validation balanced accuracy**.  
6. It reports binary metrics and trading metrics at τ\* and saves the full sweep to JSON for later plotting.

### 3.3 Results

#### 3-class metrics (unchanged from Split A baseline)

On the validation and test sets, the 3-class direction performance is exactly as in Experiment 3:

- **Validation (2020)**  
  - CE loss: **1.238**  
  - Accuracy: **0.392**  
  - Macro-F1: **0.375**  
  - Macro-AUC: **0.507**

- **Test (2021–2024)**  
  - CE loss: **1.292**  
  - Accuracy: **0.411**  
  - Macro-F1: **0.367**  
  - Macro-AUC: **0.548**

The TFT has **weak but non-zero** ability to distinguish DOWN / FLAT / UP, but no high-quality signal.

---

#### Threshold sweep and τ\* (UP-vs-REST)

Balanced-accuracy tuning over the grid

> `τ ∈ {0.10, 0.20, 0.30, 0.40, 0.50}`

yields:

- **Best τ\*** on validation: **0.10**  
- **Best validation balanced accuracy** at τ\*: **0.500** (≈ random)

So even when we explicitly optimise for balanced accuracy, the optimal τ\* is still **0.10**, the same as in Experiments 3/5a/5b where F1 was used. The numerical balanced accuracy score at τ\* is essentially **0.5**, i.e. no better than always predicting a single class.

Binary metrics at **τ\* = 0.10**:

| Split | Accuracy | Balanced Acc. | Precision | Recall (TPR) | F1 (UP) | Macro-F1 | AUC   | Positive Rate |
|-------|----------|---------------|-----------|--------------|---------|----------|-------|---------------|
| **Val** | 0.487  | **0.500**     | 0.487     | **0.988**    | 0.652   | 0.337    | 0.473 | **0.988**     |
| **Test** | 0.418 | **0.507**     | 0.400     | **0.938**    | 0.561   | 0.348    | 0.496 | **0.930**     |

Interpretation:

- On **validation**, the model predicts UP on **98.8% of days** (positive_rate ≈ 0.99).  
- On **test**, it predicts UP on **93.0% of days**.  
- Balanced accuracy ≈ 0.50 on both splits; UP-vs-REST **AUC ≈ 0.50**.  
- The confusion matrix confirms this behaviour:
  - On test, most NOT_UP days are misclassified as UP (TN is tiny vs FP), while almost all UP days are predicted UP.

In other words:

> Even with balanced-accuracy tuning, the UP-vs-REST classifier degenerates into an **“almost always UP”** rule with essentially **random** ranking power.

---

#### Long-only trading metrics at τ\* = 0.10

Because the signal is UP almost every day, the trading strategy is effectively “buy and hold with occasional days out of the market”.

At τ\* = 0.10:

- **Validation (2020)** – very strong BTC bull year
  - Fraction of days in position ≈ **98.8%**
  - Average daily return (strategy): **0.0035** (~0.35%/day)
  - Cumulative return (2020): **+125%**
  - Sharpe ratio: **1.30**
  - Hit ratio (positive days when in position): **0.58**
  - Average return when in position: **0.0036**

- **Test (2021–2024)** – mix of bull and bear regimes
  - Fraction of days in position ≈ **93.0%**
  - Average daily return (strategy): **0.00063** (~0.06%/day)
  - Cumulative return: **+23%** over 2021–2024
  - Sharpe ratio: **0.32**
  - Hit ratio: **0.47**
  - Average return when in position: **0.00067**

Because the model is invested on >90% of days, these numbers are very close to the underlying BTC buy-and-hold performance over the same periods:

- The strategy’s **risk/return profile is largely driven by the unconditional BTC drift**, not by any genuine forecasting skill.
- The modest positive Sharpe on test (≈0.3) is consistent with “being long most of the time” and does not constitute strong evidence of an edge.

### 3.4 Conclusion – Impact of Decision Threshold (Experiment 5c)

Experiment 5c shows that:

1. **Changing the decision rule does not reveal hidden predictive power.**  
   - Balanced-accuracy-based tuning selects **τ\* = 0.10**, the same threshold as F1-based tuning.  
   - At τ\*, **balanced accuracy ≈ 0.5** and **AUC ≈ 0.5**, indicating **no useful binary signal** in P(UP).
   - The classifier is effectively **always UP**, as evidenced by positive rates of 93–99% and the confusion matrices.

2. **Trading results reflect market direction, not model skill.**  
   - The long-only strategy at τ\* is in the market on >90% of days.  
   - Its cumulative returns and Sharpe largely mirror BTC’s own behaviour in the respective periods (strong in 2020, modest overall in 2021–2024).  
   - There is no sign of a robust, threshold-dependent outperformance relative to a naive always-long strategy.

3. **Evaluation pipeline upgrade, not a performance upgrade.**  
   - While 5c does not improve the model’s metrics, it provides a **cleaner and more realistic evaluation framework**:
     - Threshold sweeps over P(UP),
     - Balanced-accuracy-based τ\* selection,
     - Transparent trading metrics and positive-rate reporting.
   - This framework will be reused for later experiments (e.g. alternative labels, horizons, or architectures) to ensure **consistent, trading-aware evaluation**.

Overall, Experiment 5c strengthens the negative conclusion from Experiments 3, 5a and 5b:

> For 1-day BTC direction with daily OHLCV + TA + calendar/halving features, the TFT’s P(UP) scores do **not** contain a reliable, exploitable edge. No static decision threshold on P(UP) produces a robust UP-vs-REST classifier or trading strategy beyond a trivial “always long” baseline.