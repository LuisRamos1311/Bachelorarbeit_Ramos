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
from dataclasses import dataclass
from typing import List


# ============================
# 1. DATA PATHS
# ============================


# Root directories (you can change these if needed)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

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

# -------- Task-level configuration --------
#
# TASK_TYPE controls how the rest of the pipeline interprets the target:
#   - "classification": binary up/down, using target_up
#   - "regression":    continuous return, using target_return (or similar)
#
# TARGET_COLUMN is the main label column used when building sequences.
# DIRECTION_LABEL_COLUMN is the binary up/down label derived from returns
# and used for counts, confusion matrices, etc.
TASK_TYPE: str = "regression"          # "regression" or "classification"
TARGET_COLUMN: str = "target_return"   # will be created in data_pipeline.py
DIRECTION_LABEL_COLUMN: str = "target_up"

# Sequence length (number of past days the model sees)
SEQ_LENGTH = 30  # 30 days of history

# Threshold for calling a move "up".
# If future_return > UP_THRESHOLD -> label = 1 (UP), else 0 (DOWN/FLAT)
# This is used when constructing target_up from future_return_1d.
UP_THRESHOLD = 0.0

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
    - final linear layer that outputs one value per sample:
        * classification: logit (before sigmoid)
        * regression:     predicted return
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

    # Single output dimension: either logit (classification) or return (regression)
    output_size: int = 1


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
    num_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    # If pos_weight == 1 -> no reweighting (standard BCE).
    # Only used when TASK_TYPE == "classification".
    pos_weight: float = 1.0

    # Seed for reproducibility (torch, numpy, etc.)
    seed: int = 42

    # Initial classification threshold on the sigmoid output.
    # Only relevant when TASK_TYPE == "classification".
    threshold: float = 0.55

    # gradient clipping (L2 norm)
    # If > 0: clip gradients to this max norm.
    # If <= 0: no clipping applied.
    grad_clip: float = 1.0

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