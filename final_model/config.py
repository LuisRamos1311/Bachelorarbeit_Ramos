"""
config.py

Central configuration for the BTC Temporal Fusion Transformer (TFT) pipeline.

The current codebase supports ONLY:
  - TASK_TYPE = "quantile_forecast"  (multi-horizon quantile regression)

Key knobs you typically change:
- Date windows: TRAIN_* / VAL_* / TEST_*
- Price data frequency / source:
  - FREQUENCY controls whether the pipeline loads BTC_DAILY_CSV_PATH ("D") or BTC_HOURLY_CSV_PATH ("1h")
- Optional auxiliary data (merged onto the price timeline):
  - USE_ONCHAIN adds ONCHAIN_COLS from BTC_ONCHAIN_DAILY_CSV_PATH (with DAILY_FEATURE_LAG_DAYS)
  - USE_SENTIMENT adds SENTIMENT_COLS from BTC_SENTIMENT_DAILY_CSV_PATH (with DAILY_FEATURE_LAG_DAYS)

This file also defines:
- Feature lists (past covariates + known-future covariates at t+H)
- Forecast horizon + quantiles (FORECAST_HORIZON, QUANTILES)
- Model / training hyperparameters
- Evaluation & trading signal-threshold defaults
"""

import os
from dataclasses import dataclass
from typing import List

# ============================
# 1. DATA PATHS
# ============================

# Folder that contains this final_model code (…/project_root/final_model)
EXPERIMENT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Top-level project folder one level above (…/project_root)
PROJECT_ROOT = os.path.dirname(EXPERIMENT_ROOT)

# Shared data directory at project root
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Output directories (routed under EXPERIMENT_ROOT/standard/)
STANDARD_RUN_DIRNAME = "standard"
STANDARD_RUN_DIR = os.path.join(EXPERIMENT_ROOT, STANDARD_RUN_DIRNAME)

MODELS_DIR = os.path.join(STANDARD_RUN_DIR, "models")
EXPERIMENTS_DIR = os.path.join(STANDARD_RUN_DIR, "experiments")
PLOTS_DIR = os.path.join(STANDARD_RUN_DIR, "plots")

# Main BTC daily CSV (CryptoDataDownload-style)
BTC_DAILY_CSV_PATH = os.path.join(DATA_DIR, "BTCUSD_daily.csv")

# BTC hourly CSV (used when FREQUENCY is hourly)
BTC_HOURLY_CSV_PATH = os.path.join(DATA_DIR, "BTCUSD_hourly.csv")

# Data frequency for the BTC price CSV used in this run
FREQUENCY: str = "1h"  # "D" for daily bars, "1h" for hourly bars, etc.

# On-chain daily CSV (used when USE_ONCHAIN is True)
BTC_ONCHAIN_DAILY_CSV_PATH = os.path.join(DATA_DIR, "BTC_onchain_daily.csv")

# Sentiment daily CSV (combined Reddit + Fear & Greed)
# Combined daily sentiment (Reddit Pushshift + Fear & Greed engineered features), fully contiguous daily grid; no NaNs.
BTC_SENTIMENT_DAILY_CSV_PATH = os.path.join(DATA_DIR, "BTC_sentiment_daily.csv")


# ============================
# 2. DATE RANGES
# ============================

# Adjust these if your BTCUSD_hourly.csv covers a different period.
TRAIN_START_DATE = "2016-01-01"
TRAIN_END_DATE   = "2020-12-31"

VAL_START_DATE   = "2021-01-01"
VAL_END_DATE     = "2021-10-31"

TEST_START_DATE  = "2021-11-01"
TEST_END_DATE    = "2024-12-31"


# ============================
# 3. FEATURES & TASK (quantile-only)
# ============================

# Whether to construct forward returns as log returns instead of simple percentage returns.
USE_LOG_RETURNS: bool = True

# This codebase supports ONLY quantile multi-horizon forecasting.
TASK_TYPE: str = "quantile_forecast"


# ============================
# 4. SEQUENCE / HORIZON SETUP
# ============================

# Number of past steps (encoder length) fed into the model.
# For 1h data, 96 = 96 hourly bars of history (~4 days)
SEQ_LENGTH: int = 96  # 96 hourly bars of history (~4 days)

# -------- Forecast horizon configuration --------
# Horizon is expressed in time steps (e.g. hours when FREQUENCY="1h").
FORECAST_HORIZON: int = 24

# Quantiles predicted at each step 1..H
QUANTILES: List[float] = [0.1, 0.5, 0.9]

# Multi-step return targets (y_ret_1 ... y_ret_H)
TARGET_RET_PREFIX: str = "y_ret_"
TARGET_RET_COLS: List[str] = [f"{TARGET_RET_PREFIX}{h}" for h in range(1, FORECAST_HORIZON + 1)]

# Daily feature availability control:
# When merging daily on-chain/sentiment into hourly bars at timestamp t, use day (t - lag_days)
# to reflect that daily aggregates are only known after day close.
DAILY_FEATURE_LAG_DAYS: int = 1

# Integrity checks inside the data pipeline (split boundary + daily-lag sanity).
DEBUG_DATA_INTEGRITY: bool = True

# (Recommended) Strict split-boundary enforcement:
# If True, drop the final H rows *inside each split* after target creation so that
# no label / forward return can reference prices beyond that split's end.
DROP_LAST_H_IN_EACH_SPLIT: bool = True

# -------- Core price & indicator features (past covariates) --------

# Price & volume columns that will be scaled with MinMaxScaler
PRICE_VOLUME_COLS: List[str] = [
    "open",
    "high",
    "low",
    "close",
    "volume_btc",
    "volume_usd",
]

# Indicator columns that will be scaled with StandardScaler
INDICATOR_COLS: List[str] = [
    "roc_10",
    "atr_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "rsi_14",
]

# On-chain feature columns
ONCHAIN_COLS: List[str] = [
    "aa_ma_ratio",
    "tx_ma_ratio",
    "mvrv_z",
    "sopr_z",
    "hash_ma_ratio",
]

USE_ONCHAIN: bool = False

# Sentiment feature columns
USE_SENTIMENT: bool = False
SENTIMENT_COLS: List[str] = [
    # Reddit (Pushshift daily)
    "reddit_sent_mean",
    "reddit_sent_std",
    "reddit_pos_ratio",
    "reddit_neg_ratio",
    "reddit_volume_log",

    # Fear & Greed (engineered daily)
    "fg_index_scaled",
    "fg_change_1d",
    "fg_missing",
]

# Columns that are binary indicators and should NOT be standardized.
# Keep them as 0/1 so the model can interpret them cleanly.
BINARY_COLS: List[str] = [
    "fg_missing",
]

# Full list of feature columns fed into the model as PAST covariates
FEATURE_COLS: List[str] = (
    PRICE_VOLUME_COLS
    + INDICATOR_COLS
    + (ONCHAIN_COLS if USE_ONCHAIN else [])
    + (SENTIMENT_COLS if USE_SENTIMENT else [])
)

# Future (t+H) versions that the model will use as known future covariates.
# For hourly data, this includes the hour of day at the horizon endpoint,
# which is fully known in advance (calendar structure).
FUTURE_COVARIATE_COLS: List[str] = [
    "day_of_week_fut",
    "is_weekend_fut",
    "month_fut",
    "hour_of_day_fut",
    "is_halving_window_fut",
    "days_to_next_halving_fut",
]


# ============================
# 4. MODEL HYPERPARAMETERS
# ============================


@dataclass
class ModelConfig:
    """
    Hyperparameters for the Temporal Fusion Transformer model.
    """
    # --- Core sizes ---
    hidden_size: int = 32

    # LSTM depth (tft_model.py reads this)
    lstm_layers: int = 1

    # Dropout applied in LSTM/attention/GRNs
    dropout: float = 0.3

    # Attention heads
    num_heads: int = 4

    # Feed-forward hidden size (tft_model.py reads this)
    ff_hidden_size: int = 64

    # --- TFT-style switches (tft_model.py reads these) ---
    use_gating: bool = True
    use_variable_selection: bool = True
    use_future_covariates: bool = True

    # Hidden size used inside VSNs/GRNs (tft_model.py reads this)
    variable_selection_hidden_size: int = 32

    # Input sizes
    input_size: int = len(FEATURE_COLS)

    # IMPORTANT: data_pipeline uses FUTURE_COVARIATE_COLS as the official list
    future_input_size: int = len(FUTURE_COVARIATE_COLS)

    # Quantile-only output head: H * Q
    output_size: int = FORECAST_HORIZON * len(QUANTILES)

MODEL_CONFIG = ModelConfig()


# ============================
# 5. TRAINING HYPERPARAMETERS
# ============================


@dataclass
class TrainingConfig:
    """
    Hyperparameters for training the TFT model.
    """

    batch_size: int = 64
    num_epochs: int = 12
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4

    # Seed for reproducibility (torch, numpy, etc.)
    seed: int = 42

    # Gradient clipping (by norm)
    grad_clip: float = 1.0

TRAINING_CONFIG = TrainingConfig()


# ============================
# 6. EVALUATION / ARTIFACT DEFAULTS
# ============================

# Name of the file where the best model checkpoint is saved.
BEST_MODEL_NAME: str = "tft_btc_best.pth"

# Full path to the best-model checkpoint (used by both training & evaluation).
BEST_MODEL_PATH: str = os.path.join(MODELS_DIR, BEST_MODEL_NAME)

# ----------------------------
# Threshold / signal configuration (quantile_forecast only)
# ----------------------------

# Which horizon step you trade on (typically equal to FORECAST_HORIZON)
SIGNAL_HORIZON: int = FORECAST_HORIZON

# For score = mu / (iqr + eps)
SCORE_EPS: float = 1e-6

# ----------------------------
# ACTIVE signal-threshold settings (quantile-only)
# ----------------------------
ACTIVE_SIGNAL_NAME: str = "score"
ACTIVE_SELECTION_METRIC: str = "sharpe"
ACTIVE_AUTO_TUNE: bool = True

# Thresholds are interpreted as percentiles of validation scores (quantile-only).
ACTIVE_THRESHOLD_GRID_MODE: str = "percentile"
ACTIVE_THRESHOLD_GRID: List[float] = [
    0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95
]

# ----------------------------
# Trading / evaluation options
# ----------------------------
# For hourly data with H>1 (e.g. 24h horizon), use non-overlapping trades
# when computing trading metrics (one trade per H-step block).
NON_OVERLAPPING_TRADES: bool = True

# ----------------------------
# Backtest assumptions (costs + annualization + baselines)
# ----------------------------

# Crypto trades 365 days/year, so Sharpe annualization should use 365 (not 252).
TRADING_DAYS_PER_YEAR: int = 365

# Transaction cost assumptions in basis points (bps). 1 bp = 0.01% = 0.0001.
COST_BPS: float = 5.0
SLIPPAGE_BPS: float = 2.0

# Baseline sampling for random strategy comparisons.
RANDOM_BASELINE_RUNS: int = 1000
RANDOM_SEED: int = 42