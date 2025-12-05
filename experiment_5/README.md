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
3. Decision threshold tweaks in `evaluate_tft` (P(UP) threshold) ⏳

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

## Part 3 – Decision Threshold Tweaks in `evaluate_tft` *(planned)*

### 3.1 Goal

Investigate how the **decision threshold on P(UP)** used in `evaluate_tft.py` affects:

- Binary **UP-vs-REST metrics** (accuracy, precision, recall, F1, AUC),  
- And simple trading metrics (e.g. average return when in position, cumulative return, Sharpe), if added.

Currently, the script **chooses the threshold that maximises F1 on the validation set**, which tends to select τ ≈ 0.10 and yields a near “always-UP” classifier.

### 3.2 Planned changes

- Add a **manual threshold sweep** over a grid, e.g.:
  - `THRESHOLD_GRID = [0.1, 0.2, 0.3, 0.4, 0.5]`
- For each τ:
  - Compute UP-vs-REST metrics on validation and test sets.
  - (Optionally) simulate a simple long-only strategy:
    - Long BTC on day *t* if `P(UP) ≥ τ`, flat otherwise.
- Save results to a JSON/CSV for easy plotting (e.g. `threshold_sweep_up_vs_rest.json`).

### 3.3 Results (to be filled)

*(Placeholder for a compact table showing F1/precision/recall vs τ, and any trading metrics vs τ. Expectation: no threshold delivers consistent out-performance beyond buy-and-hold, reflecting AUC ≈ 0.5.)*

### 3.4 Conclusion (to be filled)

*(Short paragraph explaining that F1-optimal thresholds lead to trivial always-UP behaviour, and that no static decision threshold on P(UP) yields a robust directional edge. This would support the interpretation that the model’s probabilities contain very little exploitable information.)*