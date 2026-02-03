# Bachelor Thesis Project — Assessing Temporal Fusion Transformers for Cryptocurrency Forecasting Across Diverse Feature Sets

This repository contains the implementation and experimental record for my bachelor thesis project on **Bitcoin (BTC) return forecasting** and **decision-layer evaluation**. The maintained, runnable system is located in `final_model/`. Archived intermediate experiments are kept under `finished_experiments/` for traceability.

**Repository URL:** https://github.com/LuisRamos1311/Bachelorarbeit_Ramos

---

## Project summary

The final system trains a **simplified Temporal Fusion Transformer (TFT)–style** model to forecast **multi-horizon forward returns** and expresses uncertainty via **quantile outputs** (e.g., q10 / q50 / q90). The evaluation pipeline converts these forecasts into a **long / cash** trading decision using an **uncertainty-aware score** and a **threshold tuned on the validation period**, and it writes a compact reporting pack (metrics + plots) to disk for later comparison across runs.

The thesis treats `config.py` as an **executable contract**: a run is defined by (i) the input files under `data/` and (ii) the configuration settings that determine splits, feature families, horizon, and evaluation assumptions.

---

## Repository layout

At the project root:

- `data/`  
  Input boundary. Contains the CSV datasets consumed by the pipeline (price candles + optional daily exogenous datasets).

- `final_model/`  
  **The only maintained execution path**. Contains:
  - `config.py` — run definition (data paths, splits, feature toggles, horizon, hyperparameters, evaluation settings)
  - `data_pipeline.py` — leakage-aware dataset construction + scaling + sliding windows
  - `tft_model.py` — TFT-style model implementation (VSN/GRN/LSTM/attention; quantile head)
  - `train_tft.py` — training stage (pinball loss; checkpoint selection)
  - `evaluate_tft.py` — evaluation stage (forecast metrics + decision-layer metrics; artifacts)
  - `utils.py` — shared metrics, backtest utilities, plotting, and reporting writers
  - Run folders (default: `standard/`)

- `finished_experiments/`  
  Archived snapshots of earlier experiments (each experiment should include a short README recording: goal → change → observation).

- `miscellaneous/`  
  One-off scripts used to build or sanity-check datasets. Not part of the maintained execution path.

---

## Implementation guide:

### 1) Environment

Recommended: Python 3.10+.

Minimum libraries used by the final pipeline include:
- PyTorch
- NumPy, pandas
- scikit-learn
- matplotlib
- TA-Lib (used for technical indicators; may require platform-specific installation)

Create an environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install --upgrade pip
pip install torch numpy pandas scikit-learn matplotlib ta-lib
```

---

## 2) Input files needed under `data/`

The default configuration expects these filenames:
- `data/BTCUSD_hourly.csv` (CryptoDataDownload-style; default run uses hourly)
- `data/BTCUSD_daily.csv` (optional if you switch `FREQUENCY` to daily)
- `data/BTC_onchain_daily.csv` (required if `USE_ONCHAIN=True`)
- `data/BTC_sentiment_daily.csv` (required if `USE_SENTIMENT=True`)

### Price CSV format (hourly or daily)

The loader is designed for CryptoDataDownload-style CSVs:
- the first line may be a URL/metadata line (the loader uses `skiprows=1`)
- it must contain a `date` column
- required columns: `open`, `high`, `low`, `close`
- volume columns are expected as `volume btc` / `volume usd` (renamed internally), but if missing they will be created as zeros

### On-chain daily CSV format

Expected columns (one row per calendar day):
- `date`
- `active_addresses`
- `tx_count`
- `mvrv`
- `sopr`
- `hash_rate`
The pipeline engineers the model-ready on-chain features internally (ratios / z-style columns used later).

### Sentiment daily CSV format

Must contain a `date` column and all columns configured in `final_model/config.py` under `SENTIMENT_COLS` (e.g., Reddit aggregates + Fear & Greed engineered features). The loader expects a contiguous daily grid and applies safety forward-filling.

---

### 3) What training does

- Builds train/validation datasets via `data_pipeline.py` using the active configuration in `final_model/config.py`.
- Trains the TFT-style model with **pinball loss** for multi-horizon **quantile** forecasting.
- Selects the best checkpoint by **validation pinball loss** and saves it to:
  - `final_model/<run_folder>/models/tft_btc_best.pth`
- Writes training artifacts (logs and diagnostics) under:
  - `final_model/<run_folder>/experiments/`
  - `final_model/<run_folder>/plots/`

---

## 4) What evaluation does
- Reloads the best checkpoint (weights-only)
- Rebuilds validation + test datasets using the same dataset factory (prevents training/eval mismatch)
- Computes forecast metrics (pinball + median-based summaries) and decision-layer metrics
- Tunes a threshold on the validation set (according to the active selection settings)
- Produces a reporting pack (JSON/CSV + plots) under `final_model/<run_folder>/`

---

## 5) Configuration: what defines a run

All run-defining choices live in `final_model/config.py`. The most relevant knobs:

### Data coverage and splits
- `TRAIN_START_DATE`, `TRAIN_END_DATE`
- `VAL_START_DATE`, `VAL_END_DATE`
- `TEST_START_DATE`, `TEST_END_DATE`
- `FREQUENCY`: `"1h"` (hourly) or `"D"` (daily)

### Feature families (past covariates)
- `USE_OHLCV`
- `USE_TALIB_INDICATORS`
- `USE_ONCHAIN`
- `USE_SENTIMENT`

---

## 6) Output artifacts (reporting pack)

All artifacts are written under:

`final_model/<run_folder>/`

The folder schema is stable so that runs can be compared without manual file hunting:

### Produced by training
- `models/tft_btc_best.pth` — selected checkpoint
- `experiments/*history*.json` — epoch-by-epoch training log
- `plots/*training_curves*.png` — training/validation curves diagnostic

### Produced by evaluation
- `experiments/*metrics*.json` — main summary (forecast + decision-layer metrics; includes selected threshold)
- `plots/*threshold_sweep*.png` — validation sweep + selected threshold highlighted
- `experiments/*forecast_table*.csv` — compact forecast metrics table
- `experiments/*trading_table*.csv` — compact trading metrics table
- `plots/*test_equity_curve*.png` — net-of-cost equity curve diagnostic
- `plots/*test_signal_confusion*.png` — participation/outcome diagnostic

---

## 7) Reproducing results

1) Place the required CSVs under `data/`  
2) Set `final_model/config.py` to the exact run settings you want to reproduce  
3) Ensure the target run folder is empty (or rename it)  
4) Run training:
   ```bash
   python final_model/train_tft.py
5) Run evaluation:
   ```bash
   python final_model/evaluate_tft.py
6) Use the artifact pack under `final_model/<run_folder>/` as the interface for reporting/comparison