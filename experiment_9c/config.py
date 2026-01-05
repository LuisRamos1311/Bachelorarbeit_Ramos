"""
config.py

Central configuration for the TFT BTC up/down / return project.

This file groups together:
- Data paths and date ranges
- Feature definitions (past & known-future covariates)
- Labeling rules (e.g. what counts as "up")
- Task configuration (classification vs regression)
- Model and training hyperparameters

Other modules (data_pipeline.py, train_tft.py, tft_model.py, etc.)
should import from here instead of hardcoding values.
"""

import os
from dataclasses import dataclass, field
from typing import List


# ============================
# 1. DATA PATHS
# ============================

# Folder that contains this experiment (…/project_root/experiment_9c)
EXPERIMENT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Top-level project folder one level above (…/project_root)
PROJECT_ROOT = os.path.dirname(EXPERIMENT_ROOT)

# Shared data directory at project root
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Experiment-specific output directories
MODELS_DIR = os.path.join(EXPERIMENT_ROOT, "models")
EXPERIMENTS_DIR = os.path.join(EXPERIMENT_ROOT, "experiments")
PLOTS_DIR = os.path.join(EXPERIMENT_ROOT, "plots")

# Main BTC daily CSV (CryptoDataDownload-style)
BTC_DAILY_CSV_PATH = os.path.join(DATA_DIR, "BTCUSD_daily.csv")

# NEW: BTC hourly CSV (for Experiment 6)
BTC_HOURLY_CSV_PATH = os.path.join(DATA_DIR, "BTCUSD_hourly.csv")

# Optional flag describing the data frequency used in this experiment
FREQUENCY: str = "1h"  # "D" for daily, "1h" for hourly, etc.

# On-chain daily CSV for Experiment 7
BTC_ONCHAIN_DAILY_CSV_PATH = os.path.join(DATA_DIR, "BTC_onchain_daily.csv")

# Sentiment daily CSV for Experiment 8 (combined Reddit + Fear & Greed)
# Combined daily sentiment (Reddit Pushshift + Fear & Greed engineered features), fully contiguous daily grid; no NaNs.
BTC_SENTIMENT_DAILY_CSV_PATH = os.path.join(DATA_DIR, "BTC_sentiment_daily.csv")


# ============================
# 2. DATE RANGES
# ============================

# Adjust these if your BTCUSD_hourly.csv covers a different period.
TRAIN_START_DATE = "2016-01-01"   # first available hourly bar
TRAIN_END_DATE   = "2020-12-31"   # covers 2018 bear, 2019 recovery, 2020–21 bull, 2022 bear

VAL_START_DATE   = "2021-01-01"   # recent but separate for tuning / threshold selection
VAL_END_DATE     = "2021-12-31"

TEST_START_DATE  = "2022-01-01"   # most recent, fully out-of-sample regime
TEST_END_DATE    = "2022-12-31"   # or last available 2024 timestamp


# ============================
# 3. FEATURES, LABELS & TASK
# ============================

# We keep the same label name for now ("direction_3c") so the rest of the pipeline
# continues to work. In data_pipeline.add_target_column(), we will redefine it to mean:
#   0 = DOWN  (H-step return < -DIRECTION_THRESHOLD)
#   1 = FLAT  (|H-step return| <= DIRECTION_THRESHOLD)
#   2 = UP    (H-step return >  DIRECTION_THRESHOLD)
# where H = FORECAST_HORIZONS[0] is now measured in *hourly steps* (e.g. 24).
TRIPLE_DIRECTION_COLUMN: str = "direction_3c"

# For classification tasks, TARGET_COLUMN is the class label column.
# For quantile forecasting (9c), targets are multi-horizon return columns (TARGET_RET_COLS).
TARGET_COLUMN: str = TRIPLE_DIRECTION_COLUMN
DIRECTION_LABEL_COLUMN: str = TRIPLE_DIRECTION_COLUMN

# Number of classes for the direction classification task
NUM_CLASSES: int = 3

# Symmetric threshold around zero for deciding DOWN / FLAT / UP.
# When USE_LOG_RETURNS = True this is applied to the H-step *log* return
# log(close_{t+H} / close_t); for small moves it is still ~equal to a % move.
# Example: 0.003 ≈ 0.3% absolute move over the horizon.
DIRECTION_THRESHOLD: float = 0.005

# Whether to construct forward returns as log returns instead of simple
# percentage returns when building future_return_* and the direction_3c label
USE_LOG_RETURNS: bool = True

# Task type: "classification" uses direction_3c, "regression" predicts a continuous return/price
TASK_TYPE: str = "quantile_forecast"


# ============================
# 4. SEQUENCE / HORIZON SETUP
# ============================

# Number of past steps (encoder length) fed into the model.
# For 1h data, 96 = 96 hourly bars of history (~4 days)
SEQ_LENGTH: int = 96  # 96 hourly bars of history (~4 days)

# -------- Multi-horizon forecasting configuration --------
#
# FORECAST_HORIZONS is now interpreted in *time steps*.
# For hourly data with FORECAST_HORIZONS = [24],
# this means "next 24 hours" as the prediction horizon.
FORECAST_HORIZONS: List[int] = [24]

QUANTILES: List[float] = [0.1, 0.5, 0.9]
N_QUANTILES: int = len(QUANTILES)

# Canonical single-horizon convenience variable (useful when you want one H everywhere).
# For now, this equals FORECAST_HORIZONS[0].
if len(FORECAST_HORIZONS) < 1:
    raise ValueError("FORECAST_HORIZONS must contain at least one horizon step.")
FORECAST_HORIZON: int = FORECAST_HORIZONS[0]

# Multi-horizon regression targets (Experiment 9c)
TARGET_RET_PREFIX: str = "y_ret_"
TARGET_RET_COLS: List[str] = [f"{TARGET_RET_PREFIX}{h}" for h in range(1, FORECAST_HORIZON + 1)]

# Daily feature availability control:
# When merging daily on-chain/sentiment into hourly bars at timestamp t, use day (t - lag_days)
# to reflect that daily aggregates are only known after day close.
DAILY_FEATURE_LAG_DAYS: int = 1

# Debug-only integrity checks inside the data pipeline (split boundary + daily-lag sanity).
DEBUG_DATA_INTEGRITY: bool = True

# (Recommended) Strict split-boundary enforcement:
# If True, drop the final H rows *inside each split* after target creation so that
# no label / forward return can reference prices beyond that split's end.
DROP_LAST_H_IN_EACH_SPLIT: bool = True

# Convenience flag: True if we are in a genuine multi-horizon setup.
USE_MULTI_HORIZON: bool = len(FORECAST_HORIZONS) > 1

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

USE_ONCHAIN: bool = True  # Experiment 7/8: include on-chain features

# Sentiment feature columns (Experiment 8)
USE_SENTIMENT: bool = True
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

# -------- Calendar & halving features (base + future) --------
# Base calendar features – attached to each timestamp t.
# For hourly data, hour_of_day captures intraday patterns (0–23).
CALENDAR_COLS: List[str] = [
    "day_of_week",   # 0=Monday, ..., 6=Sunday  (for t)
    "is_weekend",    # 1 if Saturday/Sunday, else 0 (for t)
    "month",         # 1–12 (for t)
    "hour_of_day",   # 0–23 (for t, important for intraday data)
]

# Base halving-related features – attached to each timestamp t.
HALVING_COLS: List[str] = [
    "is_halving_window",     # 1 if within ±N days of a halving, else 0
    "days_to_next_halving",  # integer days from t to the next halving
]

# Future (t+H) versions that the model will use as known future covariates.
# For hourly data, this includes the hour of day at the horizon endpoint,
# which is fully known in advance (calendar structure).
FUTURE_KNOWN_COLS: List[str] = [
    "day_of_week_fut",
    "is_weekend_fut",
    "month_fut",
    "hour_of_day_fut",
    "is_halving_window_fut",
    "days_to_next_halving_fut",
]

# Full known-future feature list
KNOWN_FUTURE_COLS: List[str] = FUTURE_KNOWN_COLS

# Alias used by data_pipeline.py
FUTURE_COVARIATE_COLS: List[str] = KNOWN_FUTURE_COLS


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

    # Output size (overridden below for classification)
    output_size: int = len(FORECAST_HORIZONS)

MODEL_CONFIG = ModelConfig()

# Make sure the final layer has the right size for the active task
if TASK_TYPE == "classification":
    MODEL_CONFIG.output_size = NUM_CLASSES
elif TASK_TYPE == "quantile_forecast":
    MODEL_CONFIG.output_size = FORECAST_HORIZON * N_QUANTILES


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

    # If pos_weight == 1 -> no reweighting (standard CE).
    # Only used when TASK_TYPE == "classification".
    pos_weight: float = 1.0

    # Seed for reproducibility (torch, numpy, etc.)
    seed: int = 42

    # Gradient clipping (by norm)
    grad_clip: float = 1.0

    # Initial classification threshold (used as a default / fallback).
    # Final threshold for UP-vs-REST is tuned in evaluate_tft.py.
    threshold: float = 0.55


TRAINING_CONFIG = TrainingConfig()


# ============================
# 6. EVALUATION / ARTIFACT DEFAULTS
# ============================

# Name of the file where the best model checkpoint is saved.
BEST_MODEL_NAME: str = "tft_btc_best.pth"

# Full path to the best-model checkpoint (used by both training & evaluation).
BEST_MODEL_PATH: str = os.path.join(MODELS_DIR, BEST_MODEL_NAME)

# Default threshold to use during final evaluation on the test set.
# Only meaningful for classification tasks.
EVAL_THRESHOLD: float = TRAINING_CONFIG.threshold

# ----------------------------
# Threshold tuning configuration
# ----------------------------
# These settings are only relevant when TASK_TYPE == "classification".

# If True, evaluate_tft will:
#   1) Run the model on the *validation* set,
#   2) Grid-search over a range of thresholds,
#   3) Pick the threshold that maximizes THRESHOLD_TARGET_METRIC
#      on the validation set,
#   4) Use that threshold for computing test metrics.
AUTO_TUNE_THRESHOLD: bool = True

# Range of thresholds to search over [min, max] (inclusive, via linspace).
THRESHOLD_SEARCH_MIN: float = 0.10
THRESHOLD_SEARCH_MAX: float = 0.90
THRESHOLD_SEARCH_STEPS: int = 17  # e.g. 0.10, 0.15, ..., 0.90

# Which metric to maximize during tuning: "f1", "accuracy",
# "precision", or "recall".
THRESHOLD_TARGET_METRIC: str = "f1"

# ----------------------------
# Experiment 5c-style: UP-vs-REST threshold grid
# ----------------------------
UP_THRESHOLD_GRID = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]

# Metric to use when selecting τ* from UP_THRESHOLD_GRID on the validation set.
THRESHOLD_SELECTION_METRIC: str = "sharpe"

SIGNAL_HORIZON: int = FORECAST_HORIZON     # which step you trade on (use 24 first)
SCORE_EPS: float = 1e-6                    # for mu/(iqr+eps)
SCORE_GRID = [0.055, 0.060, 0.0625, 0.065, 0.0675, 0.070, 0.0725, 0.075, 0.080]

# ----------------------------
# ACTIVE signal-threshold settings (foolproof)
# ----------------------------
# These are the ONLY variables evaluation should use going forward.
# They select the correct signal + grid depending on TASK_TYPE.

if TASK_TYPE == "classification":
    ACTIVE_SIGNAL_NAME: str = "p_up"
    ACTIVE_THRESHOLD_GRID = UP_THRESHOLD_GRID
    ACTIVE_SELECTION_METRIC: str = THRESHOLD_SELECTION_METRIC
    ACTIVE_AUTO_TUNE: bool = AUTO_TUNE_THRESHOLD
elif TASK_TYPE == "quantile_forecast":
    ACTIVE_SIGNAL_NAME: str = "score"
    ACTIVE_THRESHOLD_GRID = SCORE_GRID
    ACTIVE_SELECTION_METRIC: str = "sharpe"  # keep consistent with your outline
    ACTIVE_AUTO_TUNE: bool = True
else:
    raise ValueError(f"Unsupported TASK_TYPE: {TASK_TYPE}")

if TASK_TYPE == "quantile_forecast":
    # Guardrail: if someone accidentally tries to use classification-only thresholding
    # in quantile mode, they should notice immediately.
    assert ACTIVE_SIGNAL_NAME == "score"
    assert ACTIVE_THRESHOLD_GRID == SCORE_GRID

# ----------------------------
# Trading / evaluation options
# ----------------------------
# For hourly data with H>1 (e.g. 24h horizon), use non-overlapping trades
# when computing trading metrics (one trade per H-step block).
NON_OVERLAPPING_TRADES: bool = True

# ----------------------------
# Experiment 9b: backtest assumptions (costs + annualization + baselines)
# ----------------------------

# Crypto trades 365 days/year, so Sharpe annualization should use 365 (not 252).
TRADING_DAYS_PER_YEAR: int = 365

# Transaction cost assumptions in basis points (bps). 1 bp = 0.01% = 0.0001.
COST_BPS: float = 5.0
SLIPPAGE_BPS: float = 2.0

# Baseline sampling for random strategy comparisons.
RANDOM_BASELINE_RUNS: int = 1000
RANDOM_SEED: int = 42
