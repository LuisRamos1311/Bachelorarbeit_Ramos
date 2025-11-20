"""
config.py

Central configuration for the TFT BTC up/down project.

This file groups together:
- Data paths and date ranges
- Feature definitions
- Labeling rules (e.g. what counts as "up")
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

DATA_DIR      = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models")
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
PLOTS_DIR     = os.path.join(PROJECT_ROOT, "plots")

# Main BTC daily CSV (CryptoDataDownload-style)
BTC_DAILY_CSV_PATH = os.path.join(DATA_DIR, "BTCUSD_daily.csv")


# ============================
# 2. DATE RANGES
# ============================

# You can adjust these for your final thesis experiments
TRAIN_START_DATE = "2014-01-01"
TRAIN_END_DATE   = "2019-12-31"

VAL_START_DATE   = "2020-01-01"
VAL_END_DATE     = "2020-12-31"

TEST_START_DATE  = "2021-01-01"
TEST_END_DATE    = "2024-12-31"


# ============================
# 3. FEATURES & LABELS
# ============================

# Sequence length (number of past days the model sees)
SEQ_LENGTH = 30  # 30 days of history

# Threshold for calling a move "up".
# If future_return > UP_THRESHOLD -> label = 1 (UP), else 0 (DOWN/FLAT)
UP_THRESHOLD = 0.0

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

# Full list of feature columns fed into the model
FEATURE_COLS: List[str] = PRICE_VOLUME_COLS + INDICATOR_COLS

# (For the future) additional sentiment feature columns
USE_SENTIMENT: bool = False
SENTIMENT_COLS: List[str] = [
    # Example placeholders: "sentiment_mean", "sentiment_pos", ...
    # Fill these in when you implement sentiment_features.py
]


# ============================
# 4. MODEL HYPERPARAMETERS
# ============================

@dataclass
class ModelConfig:
    """Hyperparameters for the Temporal Fusion Transformer model."""
    input_size: int = len(FEATURE_COLS)  # number of input features per time step
    hidden_size: int = 64
    lstm_layers: int = 1
    dropout: float = 0.1
    num_heads: int = 4  # for multi-head attention
    use_static_covariates: bool = False  # you can keep this False at first
    output_size: int = 1  # 1 logit for binary classification


MODEL_CONFIG = ModelConfig()


# ============================
# 5. TRAINING HYPERPARAMETERS
# ============================

@dataclass
class TrainingConfig:
    """Hyperparameters for training the TFT model."""
    batch_size: int = 64
    num_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    pos_weight: float = 1.0   # can adjust if classes are imbalanced
    seed: int = 42
    threshold: float = 0.5    # initial classification threshold, can be tuned later


TRAINING_CONFIG = TrainingConfig()