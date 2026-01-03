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

# Folder that contains this experiment (…/project_root/experiment_9)
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
TRAIN_END_DATE   = "2022-12-31"   # covers 2018 bear, 2019 recovery, 2020–21 bull, 2022 bear

VAL_START_DATE   = "2023-01-01"   # recent but separate for tuning / threshold selection
VAL_END_DATE     = "2023-12-31"

TEST_START_DATE  = "2024-01-01"   # most recent, fully out-of-sample regime
TEST_END_DATE    = "2024-12-31"   # or last available 2024 timestamp


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

# Number of classes for the direction classification task
NUM_CLASSES: int = 3

# Symmetric threshold around zero for deciding DOWN / FLAT / UP.
# When USE_LOG_RETURNS = True this is applied to the H-step *log* return
# log(close_{t+H} / close_t); for small moves it is still ~equal to a % move.
# Example: 0.003 ≈ 0.3% absolute move over the horizon.
DIRECTION_THRESHOLD: float = 0.005

# Whether to construct forward returns as log returns instead of simple
# percentage returns when building future_return_* and the direction_3c label.
# True  -> use log(close_{t+H} / close_t)
# False -> use close_{t+H} / close_t - 1.0
USE_LOG_RETURNS: bool = True


# -------- Task-level configuration --------
#
# Experiment 6: H-step-ahead 3-class direction classification based on direction_3c.
TASK_TYPE: str = "classification"
TARGET_COLUMN: str = TRIPLE_DIRECTION_COLUMN
DIRECTION_LABEL_COLUMN: str = TRIPLE_DIRECTION_COLUMN

# Sequence length (number of past time steps the model sees).
# For hourly data: 96 = 4 days of history.
SEQ_LENGTH = 96  # 96 hourly bars of history (~4 days)

# -------- Multi-horizon forecasting configuration --------
#
# FORECAST_HORIZONS is now interpreted in *time steps*.
# For hourly data with FORECAST_HORIZONS = [24],
# this means "next 24 hours" as the prediction horizon.
FORECAST_HORIZONS: List[int] = [24]

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
FUTURE_CALENDAR_COLS: List[str] = [
    "dow_next",           # day_of_week for t+H
    "is_weekend_next",    # is_weekend for t+H
    "month_next",         # month for t+H
    "hour_next",          # hour_of_day for t+H
]

FUTURE_HALVING_COLS: List[str] = [
    "is_halving_window_next",  # halving-window flag for t+H
]

# All known future covariates (for now: calendar + halving for t+H).
FUTURE_COVARIATE_COLS: List[str] = FUTURE_CALENDAR_COLS + FUTURE_HALVING_COLS

# -------- Advanced TFT-style feature grouping --------

# For now, we treat all existing FEATURE_COLS as past time-varying covariates.
PAST_COVARIATE_COLS: List[str] = FEATURE_COLS.copy()

# FUTURE_COVARIATE_COLS is defined above and already used in data_pipeline.py
# to build per-sample future covariate vectors for t+H.

# Static covariates (e.g. asset ID, regime label) – not used yet for single BTC.
STATIC_COLS: List[str] = []


# ============================
# 4. MODEL HYPERPARAMETERS
# ============================


@dataclass
class ModelConfig:
    """
    Hyperparameters for the Temporal Fusion Transformer-style model.

    Current implementation:
    - Variable Selection Network (VSN) or linear projection from raw features
      to hidden_size
    - LSTM encoder over the past sequence
    - multi-head self-attention over the encoded sequence
    - temporal feed-forward / gating blocks (GRNs with GLU)
    - optional known future covariate encoder (calendar + halving for t+H)
    - final linear layer that outputs one or more values per sample:
        * classification: logits
        * regression:     predicted return(s) for one or more horizons
    """

    # Number of input features per time step (must match FEATURE_COLS length)
    input_size: int = len(FEATURE_COLS)

    # Shared hidden size for LSTM, attention and feed-forward blocks
    hidden_size: int = 32

    # Number of stacked LSTM layers
    lstm_layers: int = 1

    # Dropout applied in LSTM, attention and feed-forward blocks
    dropout: float = 0.3

    # Number of attention heads in the multi-head self-attention block
    num_heads: int = 4

    # Hidden size of the position-wise feed-forward network
    ff_hidden_size: int = 64

    # -------- Advanced TFT-style options --------

    # Use GRN + GLU gating around key submodules.
    use_gating: bool = True

    # Use Variable Selection Networks over past covariates.
    use_variable_selection: bool = True

    # Use known future covariates (e.g. calendar & halving info for t+H).
    use_future_covariates: bool = True

    # Hidden size used inside variable selection networks (VSNs) and GRNs.
    variable_selection_hidden_size: int = 32

    # Hidden size for potential static covariate encoders (multi-asset extension).
    static_hidden_size: int = 16

    # Number of future-covariate features for t+H.
    # Computed from FUTURE_COVARIATE_COLS; currently 4.
    future_input_size: int = len(FUTURE_COVARIATE_COLS)

    # Placeholder for future extension with static covariates
    use_static_covariates: bool = False

    # Forecast horizons in time steps (e.g. [24] for next 24 hours at 1h frequency).
    forecast_horizons: List[int] = field(
        default_factory=lambda: FORECAST_HORIZONS
    )

    # Output dimension: one value per forecast horizon for regression,
    # or logits if you extend classification to multi-horizon later.
    # For Experiment 6 (classification), this will be overridden to NUM_CLASSES.
    output_size: int = len(FORECAST_HORIZONS)


MODEL_CONFIG = ModelConfig()

# Make sure the final layer has the right size for the active task
if TASK_TYPE == "classification":
    # 3-class UP / FLAT / DOWN for the H-step-ahead direction.
    MODEL_CONFIG.output_size = NUM_CLASSES


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

# ----------------------------
# Trading / evaluation options
# ----------------------------
# For hourly data with H>1 (e.g. 24h horizon), use non-overlapping trades
# when computing trading metrics (one trade per H-step block).
NON_OVERLAPPING_TRADES: bool = True