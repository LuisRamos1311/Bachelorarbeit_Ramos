# README.md — Final Model (TFT) Build Documentation

This document describes the **final_model / TFT model** project: a Python pipeline that trains a simplified **Temporal Fusion Transformer–style** model to forecast **24-hour-ahead BTC returns** from **hourly market data**. The model consumes configurable “feature families”: (1) **OHLCV market candles**, (2) **TA-Lib technical indicators** derived from OHLCV, and optionally (3) **daily sentiment** and (4) **daily on-chain** features merged into the hourly timeline in a leakage-aware way. The model produces **quantile forecasts** (q10/q50/q90) to express uncertainty, and the evaluation pipeline converts those forecasts into a **long / cash** trading signal using an **uncertainty-aware score** and a **validation-tuned threshold**.

---

## Repository layout

At the project root, there are four main folders:

1. `data/`  
   Contains the CSVs the model consumes (price candles + optional daily exogenous datasets).

2. `final_model/`  
   The current runnable implementation (the “final model”). This contains:
   - Core code files (`config.py`, `data_pipeline.py`, `tft_model.py`, `train_tft.py`, `evaluate_tft.py`, `utils.py`)
   - One or more run directories (commonly `standard/`; you may rename run folders as you iterate)
   - `README_model_selection.md` documenting the final selection logic

3. `finished_experiments/`  
   Archived experiments, each typically with its own README and variant code/config snapshots.

4. `miscellaneous/`  
   One-off scripts, older tests, failed experiments, and data-building scripts used to produce the CSVs in `data/`.

---

## Input data: sources and how they were built

### What is an “input” in this project?

In this project, “inputs” include:
1) **Raw input files** (CSV datasets under `data/`), and  
2) **Derived input features** computed deterministically from those files (e.g., TA-Lib indicators derived from OHLCV).

The four key feature families are configured via explicit toggles in `final_model/config.py`:

- `USE_OHLCV` — OHLCV candles (raw, required if enabled)
- `USE_TALIB_INDICATORS` — TA-Lib indicators derived from OHLCV (required if enabled)
- `USE_SENTIMENT` — daily sentiment merged onto hourly bars (required if enabled)
- `USE_ONCHAIN` — daily on-chain merged onto hourly bars (required if enabled)

A run must enable **at least one** feature family; otherwise `FEATURE_COLS` is empty and the config fails early.

### Market price candles (daily + hourly)

- Files:
  - `data/BTCUSD_daily.csv`
  - `data/BTCUSD_hourly.csv`
- Source:
  - Downloaded from CryptoDataDownload (CSV format).

Important format detail:
- The loader in `final_model/data_pipeline.py` is designed for CryptoDataDownload-style CSVs where the first line may be a URL/metadata line, so it uses `skiprows=1` and expects a `date` column.

Even if `USE_SENTIMENT` / `USE_ONCHAIN` are disabled, the system still requires the price CSV because:
- targets are forward returns computed from price,
- TA-Lib indicators (if enabled) are derived from OHLCV.

### OHLCV-derived technical indicators (TA-Lib)

These are computed inside `final_model/data_pipeline.py` using TA-Lib, based on OHLCV:

- Example indicators used (as configured in `final_model/config.py`):
  - ROC (rate of change)
  - ATR (average true range)
  - MACD (and signal/hist)
  - RSI

Why these matter (plain English):
- OHLCV is the “raw record” of market behavior.
- Technical indicators are compact summaries of **momentum**, **trend**, and **volatility** that provide structured signals the model can learn from more easily than raw candles alone.

Practical note:
- Indicators require warm-up history (rolling windows), so the pipeline drops early rows where indicators are undefined.

### On-chain daily data

- File:
  - `data/BTC_onchain_daily.csv`
- Built by:
  - `miscellaneous/download_btc_onchain_daily.py`

What the script does (high level):
- Downloads a set of daily on-chain series (e.g., active addresses, tx count, MVRV, SOPR, hash rate),
- Aligns them on a daily index,
- Writes a single merged CSV.

How it becomes model-ready features:
- The model does not require the CSV to contain engineered ratios.
- `final_model/data_pipeline.py` builds features such as:
  - `aa_ma_ratio`, `tx_ma_ratio`, `hash_ma_ratio` (series / rolling MA)
  - `mvrv_z`, `sopr_z` (then standardized during scaling)

### Sentiment daily data (Reddit + Fear & Greed)

- Final file:
  - `data/BTC_sentiment_daily.csv`

This file is assembled from two sources:

1) Reddit sentiment (Pushshift dumps, offline)
- Built by `miscellaneous/build_reddit_sentiment_from_pushshift.py`
- Input: downloaded Pushshift `.zst` dumps
- Output:
  - `data/reddit_sentiment_daily_pushshift.csv`
- Subreddits used:
  - `Bitcoin`, `btc`, `CryptoCurrency`, `BitcoinMarkets`, `CryptoMarkets`
- What’s inside:
  - Daily aggregated sentiment statistics (mean/std, pos/neg ratios, post volume, etc.)

2) Fear & Greed index (API)
- Built by `miscellaneous/build_fear_greed_daily.py`
- Output:
  - `data/fear_greed_daily_clean.csv`
- What’s inside:
  - `fg_index_scaled` (0..1), `fg_change_1d`, `fg_missing` (indicator)

3) Final join to create a contiguous daily dataset
- Built by `miscellaneous/buid_sentiment_full_daily.py` (filename typo is expected; keep as-is unless you rename)
- Output:
  - `data/BTC_sentiment_daily.csv`
- Behavior:
  - Reindexes to a contiguous daily grid (2016–2024),
  - Forward-fills where appropriate and enforces “no NaNs” in required columns.

Why data ranges typically cover 2016–2024:
- Fear & Greed provides coverage back to 2016, which anchors the long-range sentiment dataset and influenced the date ranges used in `final_model/config.py`.

---

## How the code fits together

### Core modules and their jobs

- `config.py`  
  The single place you configure:
  - Date splits (train/val/test windows)
  - Frequency (`FREQUENCY="1h"` for hourly)
  - Feature toggles (`USE_OHLCV`, `USE_TALIB_INDICATORS`, `USE_ONCHAIN`, `USE_SENTIMENT`)
  - Feature column groups and `FEATURE_COLS`
  - Model hyperparameters (hidden size, dropout, attention heads, etc.)
  - Training hyperparameters (epochs, batch size, LR, seed)
  - Trading/evaluation settings (costs, non-overlap, threshold grid mode, selection metric)
  - Output directories (run folder under `final_model/`)

- `data_pipeline.py`  
  The dataset factory. It:
  1) loads candles (hourly or daily CSV),
  2) if enabled, computes TA-Lib indicators from OHLCV,
  3) if enabled, merges daily sentiment and/or on-chain **with a daily lag**,
  4) performs date splits,
  5) computes forward return targets **inside each split**,
  6) fits scalers on train only and transforms val/test,
  7) creates sliding windows of length `SEQ_LENGTH` and targets of length `FORECAST_HORIZON`,
  8) returns PyTorch-ready datasets and metadata.

- `tft_model.py`  
  The simplified TFT model. Intuition:
  - **Variable selection**: learns which features matter most (and when).
  - **LSTM**: summarizes the recent history (sequence model).
  - **Attention**: lets the model focus on the most relevant hours in the lookback window.
  - **Quantile head**: outputs three forecasts (q10/q50/q90) for each horizon step.

- `utils.py`  
  Shared utilities:
  - seeding and device selection
  - quantile (pinball) loss
  - metric helpers (MAE on median, etc.)
  - trading/backtest utilities (costs/slippage, Sharpe, drawdown, buy&hold baseline, random exposure baseline)
  - plotting helpers (training curves, forecast bands, threshold sweeps, equity curves, signal confusion matrix)
  - JSON/CSV writing utilities

- `train_tft.py`  
  Training runner:
  1) loads config and sets seed,
  2) builds datasets via `data_pipeline`,
  3) instantiates `TemporalFusionTransformer`,
  4) trains using pinball loss,
  5) selects the best checkpoint by lowest validation pinball loss,
  6) writes checkpoint + training artifacts under `final_model/<run_folder>/`.

  Safety behavior:
  - refuses to run if the run folder is non-empty (prevents accidental overwrite).

- `evaluate_tft.py`  
  Evaluation + reporting runner:
  1) builds datasets via `data_pipeline`,
  2) loads the saved checkpoint,
  3) predicts quantiles on validation and test,
  4) computes forecast metrics,
  5) derives a long/cash signal using an uncertainty-aware score,
  6) sweeps thresholds on validation to pick `τ*` (default selection metric: Sharpe),
  7) evaluates `τ*` on test net-of-cost,
  8) writes plots + tables + JSON under `final_model/<run_folder>/`.

### Training vs evaluation (mental model)

```
TRAIN (train_tft.py)
config.py ──► data_pipeline
├─► load hourly OHLCV (price CSV)
├─► compute TA-Lib indicators (if enabled)
├─► merge daily sentiment/on-chain (if enabled, lagged)
├─► split first, then create forward-return targets
├─► scale features (fit on train only)
└─► build sliding windows (X: last 96h, y: next 24h)
└─► tft_model.TemporalFusionTransformer()
└─► optimize pinball loss
└─► save best checkpoint + training artifacts

EVALUATE (evaluate_tft.py)
config.py ──► data_pipeline (same steps)
└─► load best checkpoint
└─► predict quantiles (q10/q50/q90)
├─► compute forecast metrics
├─► score = q50 / (q90-q10)
├─► threshold sweep on val (pick τ*)
└─► net-of-cost backtest on test
└─► write plots/tables/json
````

---

## What the model predicts (plain English)

At each hourly timestamp, the model looks at the previous ~4 days of enabled inputs (default `SEQ_LENGTH=96`) and predicts the **distribution** of BTC forward returns over the next 24 hours (default `FORECAST_HORIZON=24`).

Instead of predicting a single number, it predicts three quantiles:
- **q10**: pessimistic scenario (lower bound)
- **q50**: median / “best guess”
- **q90**: optimistic scenario (upper bound)

The gap `(q90 - q10)` is used as an **uncertainty proxy**:
- narrow gap → more confident
- wide gap → less confident

This uncertainty is used directly to decide whether to be long or stay in cash.

---

## Key realism / leakage controls

1) **Split-first targets**  
Forward targets are computed *inside each split* (train/val/test), and the tail `H` rows are dropped inside each split. This prevents boundary leakage (e.g., val labels using test prices).

2) **Daily feature lagging for hourly realism**  
Daily sentiment and on-chain features are merged into hourly bars using a lagged daily key (default `DAILY_FEATURE_LAG_DAYS=1`), reflecting that daily aggregates are reliably known only after day close.

3) **Non-overlapping trade accounting**  
With `FORECAST_HORIZON=24`, evaluation can enforce one decision per 24-hour block (`NON_OVERLAPPING_TRADES=True`) to avoid inflated trade counts from overlapping positions.

4) **Net-of-cost reporting and baselines**  
Evaluation includes:
- transaction cost and slippage assumptions (`COST_BPS`, `SLIPPAGE_BPS`)
- buy & hold baseline
- random exposure baseline matched to the strategy’s long-rate

---

## Outputs: what files you get and why they matter

All outputs are written under `final_model/<run_folder>/` (commonly `final_model/standard/`).

### 1) Model checkpoint

* `models/tft_btc_best.pth`
  Best model weights (lowest validation pinball loss). Used by `evaluate_tft.py`.

### 2) Training diagnostics

* `experiments/<run_id>_history.json`
  Per-epoch training and validation metrics. Use it to:

  * confirm training is stable,
  * detect overfitting,
  * compare runs across archived folders.

* `plots/<run_id>_training_curves.png`
  Quick visual sanity check for training dynamics.

### 3) Evaluation metrics and artifacts (forecast + trading)

Evaluation produces a consistent pack, keyed by `eval_id`:

* `experiments/<eval_id>_metrics.json`
  Summary of forecast-quality metrics and trading metrics (val/test), including baselines.

* `plots/<eval_id>_threshold_sweep.png`
  Visualizes how Sharpe/return/drawdown changes with threshold; helps explain why `τ*` was chosen and whether the selection is stable.

* `plots/<eval_id>_test_equity_curve.png`
  Cumulative strategy equity (net-of-cost). Most intuitive “how it behaved over time” view.

* `plots/<eval_id>_test_signal_confusion.png`
  A 2×2 view of:
  * realized forward return sign (up/down)
  * whether the strategy was long or in cash
    Helps interpret selectivity vs missed upside vs false positives.

* `experiments/<eval_id>_forecast_table.csv`
  Row-level forecast outputs (quantiles + realized returns + derived score) for deeper analysis and custom slicing.

* `experiments/<eval_id>_trading_table.csv`
  Trade/block-level outputs (especially under non-overlap). This is the audit trail for verifying backtest correctness and reproducing metrics externally.

---

## Decisions & tradeoffs

1. **Feature families are explicit and configurable**

* Decision: treat OHLCV, TA-Lib indicators, sentiment, and on-chain as peer input families controlled via toggles.
* Benefit: enables controlled ablations (“what happens if I remove sentiment?”).
* Tradeoff: more combinations to manage; requires disciplined run-folder archiving.

2. **Quantile forecasting instead of classification**

* Decision: predict a distribution (q10/q50/q90), not just “up/down.”
* Benefit: uncertainty becomes explicit and usable for risk filtering.
* Tradeoff: evaluation is more complex than accuracy; requires the reporting pack.

3. **Risk-adjusted score drives trading**

* Decision: `score = q50 / max(q90-q10, eps)` and long-only if `(score >= τ) AND (q50 > 0)`.
* Benefit: trades only when upside is positive and uncertainty is low.
* Tradeoff: threshold calibration can be regime-dependent; mitigated by saving sweep results.

4. **Validation selects `τ*`**

* Decision: choose τ on validation using a trading objective (default: Sharpe).
* Benefit: aligns selection to end-use.
* Tradeoff: can overfit if validation is not representative; sweep artifacts help diagnose fragility.

5. **Leakage-aware daily merges**

* Decision: merge daily sentiment/on-chain into hourly bars with `DAILY_FEATURE_LAG_DAYS=1`.
* Benefit: realistic availability assumptions.
* Tradeoff: may reduce apparent signal but increases defensibility.

6. **Non-overlapping trade accounting**

* Decision: compute trading metrics on non-overlapping horizon blocks.
* Benefit: avoids inflated trade counts and overlapping exposure artifacts.
* Tradeoff: fewer trades → higher metric variance.