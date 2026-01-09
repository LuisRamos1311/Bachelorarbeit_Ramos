# Experiment 9b – Realistic Revenue Evaluation (Costs + Baselines + Drawdown)

This experiment keeps the **same modeling task, feature set, architecture, and signal construction** as **Experiment 9a** (hourly BTC OHLCV + technical indicators + *lagged* daily on-chain + *lagged* daily sentiment; 24-hour-ahead 3-class direction classification; Sharpe-optimized thresholding with non-overlapping 24h trades).  

**Experiment 9b focus:** make the **revenue/backtest evaluation “paper defensible”** by adding:
1) **Transaction costs + slippage (net-of-cost backtest)**  
2) **Buy & Hold baseline** (same return stream)  
3) **Random exposure baseline** (same long-rate as the strategy)  
4) **Max drawdown (MDD)** + crypto-appropriate Sharpe annualization (**365**)

> Important context: The test set is **2024**, which was a very strong year for BTC. This makes **Buy & Hold unusually hard to beat** in cumulative return, and it’s why baselines are essential to interpret “strategy alpha”.

---

## 1. Motivation

In 9a we fixed two critical realism issues:
- **Split-first labels** (no split-boundary contamination)
- **Daily feature lag (t−1 day)** for on-chain and sentiment availability

However, even with improved data integrity, any “revenue” claim is incomplete without:
- **Trading frictions** (fees + slippage)
- **Benchmarks** (Buy & Hold and random exposure)
- **Risk metrics** beyond Sharpe (e.g., Max Drawdown)

Experiment 9b adds these evaluation layers while preserving the model and prediction pipeline unchanged, so results remain comparable.

---

## 2. Data, Date Ranges & Splits (Same as 9a)

- **Frequency:** BTCUSD 1-hour candles  
- **Modeling window:** 2016-01-01 to 2024-12-31  
- Splits:
  - **Train:** 2016 → 2022  
  - **Validation:** 2023  
  - **Test (out-of-sample):** 2024  

### Split-boundary enforcement (H = 24)
Targets are computed inside each split and the **last 24 hours of each split are dropped**, ensuring no label references data beyond the split end.

---

## 3. Features & Labels (Same as 9a)

### Past covariates (25 total)
- Price/volume (MinMax scaled): `open, high, low, close, volume_btc, volume_usd`
- Technical indicators (StandardScaler): `roc_10, atr_14, macd, macd_signal, macd_hist, rsi_14`
- On-chain (daily → hourly, lagged by 1 day, StandardScaler):
  - `aa_ma_ratio, tx_ma_ratio, mvrv_z, sopr_z, hash_ma_ratio`
- Sentiment (daily → hourly, lagged by 1 day):
  - Reddit: `reddit_sent_mean, reddit_sent_std, reddit_pos_ratio, reddit_neg_ratio, reddit_volume_log`
  - Fear & Greed: `fg_index_scaled, fg_change_1d, fg_missing`

### Label / target
- **Task:** 24-hour-ahead **3-class direction** (classification)
- Log returns: `USE_LOG_RETURNS = True`
- Threshold: `DIRECTION_THRESHOLD = 0.005`
- `direction_3c`:
  - 0 = DOWN if future log return < –0.005  
  - 1 = FLAT if |future log return| ≤ 0.005  
  - 2 = UP if future log return > +0.005  

---

## 4. Model & Training Setup (Same as 9a)

- Temporal Fusion Transformer–inspired architecture
- Hidden size: 32
- Dropout: 0.3 (GRNs)
- Weight decay: 5e-4
- Sequence length: 96 (≈ last 4 days)
- Loss: CrossEntropyLoss
- Model selection: best **validation macro-F1** (3-class)

---

## 5. Evaluation & Trading Logic (Extended in 9b)

### 5.1 Prediction metrics (unchanged)
- 3-class metrics: accuracy, precision/recall/F1, AUC
- UP-vs-REST derived from softmax `P(UP)`

### 5.2 Strategy signal (unchanged)
- Long-only: enter long if `P(UP) ≥ τ`, else flat
- **Non-overlapping trades:** hold for 24h per decision (aligned to the label horizon)
- Threshold sweep grid:
  `{0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65}`
- Choose **τ\*** maximizing **validation Sharpe**

### 5.3 New in 9b: Net-of-cost backtest
We apply costs when position changes:
- **Cost = 5 bps**
- **Slippage = 2 bps**
- Total friction per entry/exit event ≈ **7 bps**

Reported metrics include:
- Net cumulative return
- Net Sharpe (annualized with **365**)
- Max drawdown (MDD)

### 5.4 New baselines (crucial for interpretation)
1) **Buy & Hold (net)** on the same return stream  
2) **Random exposure baseline**: random long/flat with the **same long-rate** as the strategy
   - report p95 Sharpe and p95 cumulative return (test)

---

## 6. Results

### 6.1 3-class direction (multi-class)

**Validation (2023)**
- CE loss: 1.4941  
- Accuracy: 0.3777  
- Macro-F1: **0.3651**  
- AUC: 0.5328  

**Test (2024)**
- CE loss: 1.6265  
- Accuracy: 0.3702  
- Macro-F1: **0.3058**  
- AUC: 0.5054  

**Interpretation**
- Multi-class generalization remains challenging (especially FLAT class).
- The main purpose of 9b is not improving predictive metrics, but improving how “revenue performance” is measured and judged.

---

### 6.2 UP-vs-REST threshold selection (τ\* by validation Sharpe)

- Selected threshold: **τ\* = 0.45**
- Validation Sharpe score (gross, annual=365): **2.8115**

**UP-vs-REST (test, τ\*)**
- Precision: 0.4703  
- Recall: 0.2214  
- Positive rate (fraction of long decisions): **0.1985**

**Signal behavior**
- The model is conservative in 2024: it goes long only ~20% of decision points.
- This tends to increase precision at the cost of recall.

---

### 6.3 Trading metrics (gross, no costs; annual=365)

**Validation**
- Avg daily return: 0.001970  
- Cumulative return: 0.9716  
- Sharpe: **2.8115**  
- Avg return in position: 0.006026  

**Test (2024)**
- Avg daily return: 0.001232  
- Cumulative return: 0.4908  
- Sharpe: **1.4532**  
- Avg return in position: 0.005869  

> Note: gross Sharpe is higher than older experiments partly because we now annualize using 365 (crypto), not 252.

---

### 6.4 Net-of-cost backtest + baselines (NEW in 9b)

**Assumptions**
- Cost = **5 bps**, slippage = **2 bps**
- Annualization = **365**

**Strategy (net)**
- Val: Sharpe=**2.4802**, MDD=**0.1138**, CumRet=**0.8118**
- Test: Sharpe=**1.2140**, MDD=**0.1896**, CumRet=**0.3845**

**Buy & Hold (net)**
- Val: Sharpe=**2.1200**, MDD=**0.2082**, CumRet=**1.2795**
- Test: Sharpe=**1.4293**, MDD=**0.3146**, CumRet=**0.8405**

**Random exposure baseline (same long-rate as strategy) – Test**
- p95 Sharpe=**1.8110**
- p95 CumRet=**0.5045**

**Interpretation (most important conclusion of 9b)**
- After costs, the strategy remains positive and has **lower drawdown than Buy & Hold** in 2024,
  but it **does not beat Buy & Hold** in cumulative return or Sharpe on the test year.
- Because 2024 was a strong BTC year, Buy & Hold is a high bar.
- The random-exposure baseline shows that some “good-looking” outcomes can occur even with random timing at the same long-rate, so beating baselines (not just having Sharpe > 1) is necessary to claim alpha.

---

## 7. Conclusion

Experiment 9b upgrades the evaluation into a **realistic revenue framework**:
- Adds costs/slippage
- Reports max drawdown
- Compares against Buy & Hold and random exposure

Key outcome:
- The 9a/9b strategy behaves like a **selective long filter** with reduced market exposure and reduced drawdown,
  but **does not yet demonstrate clear alpha** over Buy & Hold in a strong BTC year (2024).

Next steps (Experiment 9c and beyond):
- Move toward a more “true TFT” setup (multi-horizon + quantiles),
- Improve calibration and horizon modeling,
- Re-check alpha under costs across multiple years / walk-forward validation.