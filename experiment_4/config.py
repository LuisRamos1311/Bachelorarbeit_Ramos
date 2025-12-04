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

Experiment 4:
-------------
We extend the original Experiment 3 setup by introducing a binary
"big-move" direction label:

    direction_2c_bigmove:
        1 = UP   if future_return_1d >  +DIRECTION_THRESHOLD
        0 = DOWN if future_return_1d <  -DIRECTION_THRESHOLD
       -1 = IGNORE_LABEL for |future_return_1d| <= DIRECTION_THRESHOLD

For Experiment 4 this binary label becomes the *active* target, while
the original 3-class label (direction_3c) is still computed for
comparison but not used for training.
"""

import os
from dataclasses import dataclass, field
from typing import List


# ============================
# 1. DATA PATHS
# ============================

# Folder that contains this experiment (…/project_root/experiment_4)
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


# ============================
# 2. DATE RANGES
# ============================

# You can adjust these for your final thesis experiments
TRAIN_START_DATE = "2014-01-01"
TRAIN_END_DATE = "2019-12-31"

VAL_START_DATE = "2020-01-01"
VAL_END_DATE = "2020-12-31"

TEST_START_DATE = "2021-01-01"
TEST_END_DATE = "2024-12-31"


# ============================
# 3. FEATURES, LABELS & TASK
# ============================

# === Experiment 3: 3-class 1-day-ahead direction label configuration ===
#
# We create a column (in data_pipeline.add_target_column) with:
#   0 = DOWN  (return < -DIRECTION_THRESHOLD)
#   1 = FLAT  (|return| <= DIRECTION_THRESHOLD)
#   2 = UP    (return >  DIRECTION_THRESHOLD)
# for the 1-day ahead return.
# In Experiment 4 this label is still computed for reference but is
# not the primary training target.
TRIPLE_DIRECTION_COLUMN: str = "direction_3c"

# === Experiment 4: 2-class big-move direction label ====================
#
# We additionally create a binary label:
#   1 = UP   if future_return_1d >  +DIRECTION_THRESHOLD
#   0 = DOWN if future_return_1d <  -DIRECTION_THRESHOLD
#  -1 = IGNORE_LABEL when |future_return_1d| <= DIRECTION_THRESHOLD
#
# During sequence construction we will drop all samples whose label is
# IGNORE_LABEL so the model only sees "big-move" days.
BINARY_DIRECTION_COLUMN: str = "direction_2c_bigmove"
IGNORE_LABEL: int = -1

# Number of classes for the *active* direction classification task
# (Experiment 4: binary UP/DOWN).
NUM_CLASSES: int = 2

# Symmetric threshold around zero for deciding DOWN / FLAT / UP / big move.
# When USE_LOG_RETURNS = True this is applied to the 1-day *log* return
# log(close_{t+1} / close_t); for small moves it is still ~equal to a % move.
# Example: 0.003 ≈ 0.3% absolute daily move.
DIRECTION_THRESHOLD: float = 0.005

# Whether to construct forward returns as log returns instead of simple
# percentage returns when building future_return_*d and the labels.
# True  -> use log(close_{t+1} / close_t)
# False -> use close_{t+1} / close_t - 1.0
USE_LOG_RETURNS: bool = True


# -------- Task-level configuration --------
#
# Experiment 4: 1-day-ahead binary big-move direction classification
# based on direction_2c_bigmove (0=DOWN, 1=UP) with intermediate
# "ignore" label for small moves.
TASK_TYPE: str = "classification"
TARGET_COLUMN: str = BINARY_DIRECTION_COLUMN
DIRECTION_LABEL_COLUMN: str = BINARY_DIRECTION_COLUMN

# Sequence length (number of past days the model sees)
SEQ_LENGTH = 30  # 30 days of history

# -------- Multi-horizon forecasting configuration --------
#
# We keep a general FORECAST_HORIZONS list so we can easily switch to
# multi-horizon experiments later. For Experiment 4, we still focus on a
# single 1-day horizon.
FORECAST_HORIZONS: List[int] = [1]

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

# Full list of feature columns fed into the model as PAST covariates
FEATURE_COLS: List[str] = PRICE_VOLUME_COLS + INDICATOR_COLS

# (For the future) additional sentiment feature columns
USE_SENTIMENT: bool = False
SENTIMENT_COLS: List[str] = [
    # Example placeholders:
    # "sentiment_mean",
    # "sentiment_pos",
    # "sentiment_neg",
    # Fill these in when you implement sentiment_features.py
]

# -------- Calendar & halving features (base + future) --------
# Base (per-day) calendar features – attached to each date t.
CALENDAR_COLS: List[str] = [
    "day_of_week",   # 0=Monday, ..., 6=Sunday  (for day t)
    "is_weekend",    # 1 if Saturday/Sunday, else 0 (for day t)
    "month",         # 1–12 (for day t)
]

# Base halving-related features – attached to each date t.
# These are computed from known / approximate halving dates.
HALVING_COLS: List[str] = [
    "is_halving_window",     # 1 if within ±N days of a halving, else 0
    "days_to_next_halving",  # integer days from t to the next halving
]

# Future (t+1) versions that the model will use as known future covariates.
# We create these from the base columns using shift(-1) in data_pipeline.py.
FUTURE_CALENDAR_COLS: List[str] = [
    "dow_next",           # day_of_week for t+1
    "is_weekend_next",    # is_weekend for t+1
    "month_next",         # month for t+1
]

FUTURE_HALVING_COLS: List[str] = [
    "is_halving_window_next",  # halving-window flag for t+1
    # You could also add "days_to_next_halving_next" if you want it later.
]

# All known future covariates (for now: calendar + halving for t+1).
FUTURE_COVARIATE_COLS: List[str] = FUTURE_CALENDAR_COLS + FUTURE_HALVING_COLS

# -------- Advanced TFT-style feature grouping --------
# In the original TFT, inputs are split into:
#   - static covariates       (do not change over time)
#   - past time-varying       (observed up to "now")
#   - known future covariates (known into the future, e.g. calendar features)

# For now, we treat all existing FEATURE_COLS as past time-varying covariates.
PAST_COVARIATE_COLS: List[str] = FEATURE_COLS.copy()

# FUTURE_COVARIATE_COLS is defined above and already used in data_pipeline.py
# to build per-sample future covariate vectors for t+1.

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
    - optional known future covariate encoder (calendar + halving for t+1)
    - final linear layer that outputs one or more values per sample:
        * classification: logits
        * regression:     predicted return(s) for one or more horizons
    """

    # Number of input features per time step (must match FEATURE_COLS length)
    input_size: int = len(FEATURE_COLS)

    # Shared hidden size for LSTM, attention and feed-forward blocks
    hidden_size: int = 64

    # Number of stacked LSTM layers
    lstm_layers: int = 1

    # Dropout applied in LSTM, attention and feed-forward blocks
    dropout: float = 0.1

    # Number of attention heads in the multi-head self-attention block
    num_heads: int = 4

    # Hidden size of the position-wise feed-forward network
    ff_hidden_size: int = 128

    # -------- Advanced TFT-style options --------

    # Use GRN + GLU gating around key submodules.
    use_gating: bool = True

    # Use Variable Selection Networks over past covariates.
    use_variable_selection: bool = True

    # Use known future covariates (e.g. calendar & halving info for t+1).
    use_future_covariates: bool = True

    # Hidden size used inside variable selection networks (VSNs) and GRNs.
    variable_selection_hidden_size: int = 64

    # Hidden size for potential static covariate encoders (multi-asset extension).
    static_hidden_size: int = 16

    # Number of future-covariate features for t+1.
    # Computed from FUTURE_COVARIATE_COLS; currently 4.
    future_input_size: int = len(FUTURE_COVARIATE_COLS)

    # Placeholder for future extension with static covariates
    use_static_covariates: bool = False

    # Forecast horizons in days (e.g. [1] for Exp.3/4, later maybe [1, 3, 7]).
    forecast_horizons: List[int] = field(
        default_factory=lambda: FORECAST_HORIZONS
    )

    # Output dimension: one value per forecast horizon for regression,
    # or logits for classification.
    # For classification experiments, this will be overridden to NUM_CLASSES.
    output_size: int = len(FORECAST_HORIZONS)


MODEL_CONFIG = ModelConfig()

# Make sure the final layer has the right size for the active task
if TASK_TYPE == "classification":
    # Experiment 4: binary UP / DOWN for the 1-day-ahead big-move direction.
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
    num_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    # If pos_weight == 1 -> no reweighting (standard CE).
    # For Experiment 4 (roughly balanced 0/1 on big-move days),
    # we typically keep pos_weight = 1.0.
    pos_weight: float = 1.0

    # Seed for reproducibility (torch, numpy, etc.)
    seed: int = 42

    # Gradient clipping threshold
    grad_clip: float = 1.0

    # Initial classification threshold (used as a default / fallback).
    # Final threshold for UP-vs-DOWN is tuned in evaluate_tft.py.
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