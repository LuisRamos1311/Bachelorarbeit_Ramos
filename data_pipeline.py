"""
data_pipeline.py

Turns raw daily BTC price data into PyTorch Datasets for the TFT model.

Main steps:
1. Load daily BTC OHLCV data from CSV.
2. Add technical indicators (ROC, ATR, MACD, RSI).
3. Create binary up/down label for next day.
4. Split into train / validation / test sets by date.
5. Scale features (prices & volume with MinMax, indicators with StandardScaler).
6. Build sliding window sequences for the TFT model.
7. Provide a BTCTFTDataset class to be used in train_tft.py and evaluate_tft.py.
"""

import os
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import talib # Technical Analysis library (C + Python wrapper)
import config

from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# ============================
# 2. LOADING & PREPROCESSING
# ============================

def load_btc_daily(csv_path: str | None = None) -> pd.DataFrame:
    """
    Load daily BTC data from a CryptoDataDownload CSV and return a cleaned DataFrame.

    This function:
        1. Skips the first URL line so pandas uses line 2 as the header.
        2. Parses the 'date' column to datetime and uses it as the index.
        3. Sorts rows chronologically (oldest → newest).
        4. Renames volume columns: 'volume_btc', 'volume_usd'.
        5. Ensures there is exactly one row per calendar day and forward-fills
           any missing days.
    """
    if csv_path is None:
        csv_path = config.BTC_DAILY_CSV_PATH

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    # Read the CSV, skipping the first line with the URL
    df = pd.read_csv(csv_path, skiprows=1)

    # Parse the date column and set it as the index
    if "date" not in df.columns:
        raise ValueError(
            "Expected a 'date' column but did not find one. "
            "Check if the CSV structure matches the CryptoDataDownload format."
        )

    # Convert text to real datetime objects
    df["date"] = pd.to_datetime(df["date"])

    # Sort by date (CryptoDataDownload is newest → oldest)
    df = df.sort_values("date")

    # Set the date as index so time-based operations are easy
    df = df.set_index("date")

    # Make everything lowercase for consistency
    df.columns = [c.lower() for c in df.columns]

    # Rename volume columns to snake_case
    rename_map = {}
    if "volume btc" in df.columns:
        rename_map["volume btc"] = "volume_btc"
    if "volume usd" in df.columns:
        rename_map["volume usd"] = "volume_usd"

    if rename_map:
        df = df.rename(columns=rename_map)

    # We don't  need 'unix' or 'symbol' for modelling
    for col in ["unix", "symbol"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Check essential price columns exist
    required_cols = ["open", "high", "low", "close"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in CSV after cleaning.")

    # If volume columns are missing, create them with zeros
    if "volume_btc" not in df.columns:
        df["volume_btc"] = 0.0
    if "volume_usd" not in df.columns:
        df["volume_usd"] = 0.0

    # Ensure a continuous daily index
    # Reindex to daily frequency: if some days are missing, they will appear as NaN.
    df = df.asfreq("D")

    # Forward-fill missing values (e.g. weekends/holidays) so that the model
    # still sees a continuous time series.
    df = df.ffill()

    return df


# ============================
# 3. TECHNICAL INDICATORS
# ============================

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators using TA-Lib.

    Indicators:
        - ROC(10)        : 10-day Rate of Change (%)
        - ATR(14)        : 14-day Average True Range
        - MACD(12,26,9)  : MACD line, signal line, histogram
        - RSI(14)        : 14-day Relative Strength Index

    After computing indicators, we drop the initial rows that contain NaNs
    caused by the warm-up periods.
    """
    # Work on a copy so we don't modify the original DataFrame
    df = df.copy()

    # TA-Lib functions expect numpy arrays, not pandas Series
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    # --------- ROC(10) ---------
    # Rate of Change over 10 days, in percent
    df["roc_10"] = talib.ROC(close, timeperiod=10)

    # --------- ATR(14) ---------
    # Average True Range over 14 days
    df["atr_14"] = talib.ATR(high, low, close, timeperiod=14)

    # --------- MACD (12, 26, 9) ---------
    # MACD line, signal line and histogram
    macd, macd_signal, macd_hist = talib.MACD(
        close,
        fastperiod=12,
        slowperiod=26,
        signalperiod=9,
    )
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist

    # --------- RSI(14) ---------
    df["rsi_14"] = talib.RSI(close, timeperiod=14)

    # First rows (roughly first 30 days) will be NaN because indicators
    # need enough history → drop them so later code doesn't see NaNs.
    df = df.dropna()

    return df


# ============================
# 4. TARGET (UP / DOWN LABEL)
# ============================

def add_target_column(df: pd.DataFrame, up_threshold: float = config.UP_THRESHOLD) -> pd.DataFrame:
    """
    Add a binary target column 'target_up' to the DataFrame.

    Steps:
    1. Compute the future 1-day return:
         future_return_t = (close_{t+1} - close_t) / close_t
       aligned with day t.
    2. Label:
         target_up = 1 if future_return_t > up_threshold else 0

    We drop the last row because it has no "next day" to compare to.
    """
    df = df.copy()

    # pct_change() gives (close_t - close_{t-1}) / close_{t-1}
    # Shift -1 so that at index t we have (close_{t+1} - close_t) / close_t
    df["future_return_1d"] = df["close"].pct_change().shift(-1)

    # Binary label: 1 if future return > threshold, else 0
    df["target_up"] = (df["future_return_1d"] > up_threshold).astype(int)

    # Last row will have NaN future_return_1d, so drop it
    df = df.dropna(subset=["future_return_1d"])

    return df


# ============================
# 5. TRAIN / VAL / TEST SPLIT
# ============================

def split_by_date(
    df: pd.DataFrame,
    train_start: str = config.TRAIN_START_DATE,
    train_end: str   = config.TRAIN_END_DATE,
    val_start: str   = config.VAL_START_DATE,
    val_end: str     = config.VAL_END_DATE,
    test_start: str  = config.TEST_START_DATE,
    test_end: str    = config.TEST_END_DATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the full DataFrame into train / validation / test sets by date range.

    Assumes df has a DatetimeIndex.
    """
    train_df = df.loc[train_start:train_end].copy()
    val_df   = df.loc[val_start:val_end].copy()
    test_df  = df.loc[test_start:test_end].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("One of the splits is empty. Check your date ranges.")

    return train_df, val_df, test_df


# ============================
# 6. FEATURE SCALING
# ============================

def scale_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """
    Scale features using:
        - MinMaxScaler for price & volume columns
        - StandardScaler for indicator columns

    Returns:
        scaled_train_df, scaled_val_df, scaled_test_df, scalers_dict

    scalers_dict contains:
        {
            "price_volume_scaler": ...,
            "indicator_scaler": ...,
            "feature_cols": [...],
            "price_volume_cols": [...],
            "indicator_cols": [...],
        }
    """

    # Use the definitions from config so that scaling is consistent everywhere.
    price_volume_cols = [col for col in feature_cols if col in config.PRICE_VOLUME_COLS]
    indicator_cols    = [col for col in feature_cols if col in config.INDICATOR_COLS]

    # Make copies so we don't overwrite originals by accident
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    # --- Fit scalers on TRAIN only ---
    pv_scaler = MinMaxScaler()
    ind_scaler = StandardScaler()

    if price_volume_cols:
        pv_scaler.fit(train_df[price_volume_cols])
    if indicator_cols:
        ind_scaler.fit(train_df[indicator_cols])

    # --- Transform all splits ---
    for df in (train_df, val_df, test_df):
        if price_volume_cols:
            df[price_volume_cols] = pv_scaler.transform(df[price_volume_cols])
        if indicator_cols:
            df[indicator_cols] = ind_scaler.transform(df[indicator_cols])

    scalers = {
        "price_volume_scaler": pv_scaler,
        "indicator_scaler": ind_scaler,
        "feature_cols": feature_cols,
        "price_volume_cols": price_volume_cols,
        "indicator_cols": indicator_cols,
    }

    return train_df, val_df, test_df, scalers


# ============================
# 7. SEQUENCE CREATION
# ============================

def build_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_length: int = config.SEQ_LENGTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window sequences and labels from a DataFrame.

    For each sample:
        - X: sequence of length `seq_length` over feature_cols
        - y: 'target_up' value of the LAST day in the sequence

    Example:
        If seq_length = 30, we use days [t-29, ..., t] as input,
        and y = target_up at day t (which encodes up/down from t -> t+1).
    """
    if "target_up" not in df.columns:
        raise ValueError("DataFrame must contain 'target_up' column before building sequences.")

    df = df.copy()
    df = df.sort_index()  # ensure chronological order

    feature_array = df[feature_cols].values.astype(np.float32)
    target_array = df["target_up"].values.astype(np.float32)

    sequences = []
    labels = []

    # We start at index seq_length-1 because we need seq_length days for the first sample
    for end_idx in range(seq_length - 1, len(df)):
        start_idx = end_idx - seq_length + 1

        seq_x = feature_array[start_idx : end_idx + 1]   # shape: (seq_length, n_features)
        y = target_array[end_idx]                        # scalar label

        sequences.append(seq_x)
        labels.append(y)

    sequences = np.stack(sequences)   # shape: (num_samples, seq_length, n_features)
    labels = np.array(labels)         # shape: (num_samples,)

    return sequences, labels


# ============================
# 8. PYTORCH DATASET
# ============================

class BTCTFTDataset(Dataset):
    """
    Simple PyTorch Dataset for BTC TFT model.

    Holds:
        - sequences: numpy array of shape (N, seq_length, n_features)
        - labels:    numpy array of shape (N,)

    __getitem__ returns:
        X: torch.FloatTensor (seq_length, n_features)
        y: torch.FloatTensor scalar (0.0 or 1.0)
    """

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        assert len(sequences) == len(labels), "Sequences and labels must have same length."
        self.sequences = torch.from_numpy(sequences)  # float32
        self.labels = torch.from_numpy(labels).float()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.labels[idx]


# ============================
# 9. HIGH-LEVEL PIPELINE
# ============================

def prepare_datasets(
    csv_path: str | None = None,
    seq_length: int = config.SEQ_LENGTH,
) -> Tuple[BTCTFTDataset, BTCTFTDataset, BTCTFTDataset, Dict[str, object]]:
    """
    High-level function that runs the whole pipeline:

    1. Load BTC daily data.
    2. Add technical indicators.
    3. Add target_up column.
    4. Split by date into train / val / test.
    5. Scale features.
    6. Build sequences.
    7. Wrap them into BTCTFTDataset objects.

    Returns:
        train_dataset, val_dataset, test_dataset, scalers_dict
    """
    # 1. Load
    df = load_btc_daily(csv_path)

    # 2. Add indicators
    df = add_technical_indicators(df)

    # 3. Add up/down label
    df = add_target_column(df, up_threshold=config.UP_THRESHOLD)

    # Which columns we will feed into the model (from config)
    feature_cols = config.FEATURE_COLS

    # Make sure all required feature columns exist
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Feature column '{col}' is missing. Check indicator generation.")

    # 4. Train/Val/Test split
    train_df, val_df, test_df = split_by_date(df)

    # 5. Scale features
    train_df_scaled, val_df_scaled, test_df_scaled, scalers = scale_features(
        train_df, val_df, test_df, feature_cols
    )

    # 6. Build sequences
    train_seq, train_labels = build_sequences(train_df_scaled, feature_cols, seq_length)
    val_seq, val_labels     = build_sequences(val_df_scaled, feature_cols, seq_length)
    test_seq, test_labels   = build_sequences(test_df_scaled, feature_cols, seq_length)

    # 7. Wrap into Datasets
    train_dataset = BTCTFTDataset(train_seq, train_labels)
    val_dataset   = BTCTFTDataset(val_seq, val_labels)
    test_dataset  = BTCTFTDataset(test_seq, test_labels)

    return train_dataset, val_dataset, test_dataset, scalers


# ============================
# 10. QUICK SELF-TEST
# ============================

if __name__ == "__main__":
    """
    Run this file to check that the pipeline works end-to-end with the CSV.

    This self-test will:
      - Build train/val/test datasets
      - Print how many samples each split has
      - Inspect the first training sample:
          * its shape
          * its label (0.0 or 1.0)
          * first 3 timesteps of the sequence
          * last 3 timesteps of the sequence
          * last timestep with feature names
    """
    print("[data_pipeline] Running self-test...")

    # Run the full pipeline
    train_ds, val_ds, test_ds, scalers = prepare_datasets()

    # Print how many samples we have
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")
    print(f"Test samples:  {len(test_ds)}")

    # Take the first sample from the training set
    x, y = train_ds[0]

    # x shape: (seq_length, n_features)
    print(f"One sample X shape: {x.shape}  (seq_length, n_features)")
    print(f"One sample y: {y.item()} (0.0 or 1.0)")

    # Convert to numpy for nicer printing (still scaled values!)
    x_np = x.numpy()
    seq_len, n_features = x_np.shape

    # Get feature names from the scalers dict
    feature_names = scalers.get("feature_cols", [])

    print("\nFirst training sample (scaled feature values):")

    # ---------- First 3 timesteps ----------
    print("\n  First 3 timesteps of the sequence:")
    num_first = min(3, seq_len)
    for t in range(num_first):
        row_values = ", ".join(f"{v:.4f}" for v in x_np[t])
        print(f"    t={t:02d}: [{row_values}]")

    # ---------- Last 3 timesteps ----------
    print("\n  Last 3 timesteps of the sequence:")
    num_last = min(3, seq_len)
    # start index so we don't overlap weirdly for very short sequences
    start_last = seq_len - num_last
    for t in range(start_last, seq_len):
        row_values = ", ".join(f"{v:.4f}" for v in x_np[t])
        print(f"    t={t:02d}: [{row_values}]")

    # ---------- Last timestep with feature names ----------
    print("\n  Last timestep (most recent day in the window):")
    last_step = x_np[-1]
    for i in range(n_features):
        fname = feature_names[i] if i < len(feature_names) else f"feat_{i}"
        print(f"    {fname:>12}: {last_step[i]: .4f}")

    print("\n[data_pipeline] Self-test finished.")