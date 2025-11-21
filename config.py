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
    # Example placeholders:
    # "sentiment_mean",
    # "sentiment_pos",
    # "sentiment_neg",
    # Fill these in when you implement sentiment_features.py
]

# ============================
# 4. MODEL HYPERPARAMETERS
# ============================


@dataclass
class ModelConfig:
    """
    Hyperparameters for the Temporal Fusion Transformer-style model.

    This *simplified* TFT implementation uses:
    - an input projection from raw features to a hidden_size
    - an LSTM encoder over the past sequence
    - multi-head self-attention over the encoded sequence
    - a small position-wise feed-forward network
    - a final linear layer that outputs one logit for binary up/down

    We do NOT yet include all advanced TFT features (static covariates,
    known future inputs, etc.) to keep the project manageable for a
    bachelor thesis. Those can be added later if needed.
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

    # Placeholder for future extension with static covariates
    use_static_covariates: bool = False

    # 1 logit for binary classification (up vs down)
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

    # Can be used to rebalance the loss if up/down labels are imbalanced.
    # This is passed to BCEWithLogitsLoss(pos_weight=...).
    pos_weight: float = 1.0

    # Seed for reproducibility (torch, numpy, etc.)
    seed: int = 42

    # Initial classification threshold on the sigmoid output.
    # You can tune this later on the validation set.
    threshold: float = 0.5


TRAINING_CONFIG = TrainingConfig()