"""
Turns raw daily BTC price data into PyTorch Datasets for the TFT model
used in Experiment 4 (1-day-ahead binary big-move direction classification).

Main steps:
1. Load daily BTC price data from CSV.
2. Add technical indicators (ROC, ATR, MACD, RSI).
3. Add calendar and halving-related features (per day).
4. Create "next-day" future covariate columns via shift(-1).
5. Create the 1-day forward return and both:
     - a 3-class direction label (DOWN / FLAT / UP), and
     - a binary "big-move" label with an IGNORE state for small moves.
6. Split into train / validation / test sets by date.
7. Scale features (prices & volume with MinMax, indicators with StandardScaler).
8. Build sliding window sequences for the TFT model (past inputs)
   and per-sample future covariate vectors for t+1.
9. Filter out samples whose label equals config.IGNORE_LABEL so that
   Experiment 4 only trains on big-move days.
10. Provide a BTCTFTDataset class to be used in train_tft.py and
    evaluate_tft.py.
"""

import os
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import talib  # Technical Analysis library (C + Python wrapper)
from experiment_4 import config

from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# ============================
# 1. LOADING & PREPROCESSING
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

    # We don't need 'unix' or 'symbol' for modelling
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
    df = df.asfreq("D")

    # Forward-fill missing values (e.g. weekends/holidays) so that the model
    # still sees a continuous time series.
    df = df.ffill()

    return df


# ============================
# 2. TECHNICAL INDICATORS
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
    df = df.copy()

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    # ROC(10)
    df["roc_10"] = talib.ROC(close, timeperiod=10)

    # ATR(14)
    df["atr_14"] = talib.ATR(high, low, close, timeperiod=14)

    # MACD (12, 26, 9)
    macd, macd_signal, macd_hist = talib.MACD(
        close,
        fastperiod=12,
        slowperiod=26,
        signalperiod=9,
    )
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist

    # RSI(14)
    df["rsi_14"] = talib.RSI(close, timeperiod=14)

    # Drop warm-up rows
    df = df.dropna()

    return df


# ============================
# 3. CALENDAR & HALVING FEATURES
# ============================

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-day calendar features:

        - day_of_week : 0=Monday, ..., 6=Sunday
        - is_weekend  : 1 if Saturday/Sunday, else 0
        - month       : 1–12

    These are base features for day t, used later to derive t+1 covariates.
    """
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex to add calendar features.")

    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df.index.month

    return df


def add_halving_features(df: pd.DataFrame, window_days: int = 90) -> pd.DataFrame:
    """
    Add per-day halving-related features:

        - days_to_next_halving : integer days from t to the next halving date
        - is_halving_window    : 1 if within ±window_days of any halving date
    """
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex to add halving features.")

    halving_dates = [
        pd.Timestamp("2012-11-28"),
        pd.Timestamp("2016-07-09"),
        pd.Timestamp("2020-05-11"),
        pd.Timestamp("2024-04-19"),
        pd.Timestamp("2028-04-20"),
    ]

    dates = df.index.to_pydatetime()
    num_rows = len(df)

    days_to_next_halving = np.zeros(num_rows, dtype=np.float32)
    is_halving_window = np.zeros(num_rows, dtype=np.int64)

    for i, current_date in enumerate(dates):
        day_diffs = np.array(
            [(hd - current_date).days for hd in halving_dates],
            dtype=np.int32,
        )

        non_negative = day_diffs[day_diffs >= 0]
        if len(non_negative) > 0:
            days_next = int(non_negative.min())
        else:
            days_next = int(day_diffs[np.argmin(np.abs(day_diffs))])

        days_to_next_halving[i] = days_next

        nearest_days = int(day_diffs[np.argmin(np.abs(day_diffs))])
        if abs(nearest_days) <= window_days:
            is_halving_window[i] = 1

    df["days_to_next_halving"] = days_to_next_halving
    df["is_halving_window"] = is_halving_window

    return df


def add_future_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create next-day (t+1) future covariate columns by shifting base features.

    We use:
        - day_of_week       -> dow_next
        - is_weekend        -> is_weekend_next
        - month             -> month_next
        - is_halving_window -> is_halving_window_next

    After this transformation, at index t we have information about day t+1
    in these *_next columns. The last row will have NaNs for these columns
    (because there is no t+1), but it will be dropped later together with
    the NaN target in add_target_column().
    """
    df = df.copy()

    required_base_cols = [
        "day_of_week",
        "is_weekend",
        "month",
        "is_halving_window",
    ]
    missing = [c for c in required_base_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"add_future_covariates expected base columns {required_base_cols}, "
            f"but missing: {missing}. Make sure to call add_calendar_features() "
            f"and add_halving_features() first."
        )

    df["dow_next"] = df["day_of_week"].shift(-1)
    df["is_weekend_next"] = df["is_weekend"].shift(-1)
    df["month_next"] = df["month"].shift(-1)
    df["is_halving_window_next"] = df["is_halving_window"].shift(-1)

    return df


# ============================
# 4. TARGETS (RETURN + LABELS)
# ============================

def add_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add forward return columns for all horizons in config.FORECAST_HORIZONS,
    plus both the 3-class and binary direction labels.

    Created columns:
      - future_return_{h}d for each h in config.FORECAST_HORIZONS
      - future_return_1d   (always created) as base for labels

      - direction_3c (config.TRIPLE_DIRECTION_COLUMN):
            0 = DOWN  if future_return_1d < -DIRECTION_THRESHOLD
            1 = FLAT  if |future_return_1d| <= DIRECTION_THRESHOLD
            2 = UP    if future_return_1d >  DIRECTION_THRESHOLD

      - direction_2c_bigmove (config.BINARY_DIRECTION_COLUMN):
            1 = UP    if future_return_1d >  +DIRECTION_THRESHOLD
            0 = DOWN  if future_return_1d <  -DIRECTION_THRESHOLD
           -1 = IGNORE_LABEL when |future_return_1d| <= DIRECTION_THRESHOLD

    After creating these columns, rows with NaNs in any forward-return
    column are dropped (this removes the last max(h) rows).
    """
    df = df.copy()

    use_log_returns = getattr(config, "USE_LOG_RETURNS", False)
    close = df["close"].astype(float)

    # Collect horizons and ensure 1-day is always present
    horizons = sorted(set(config.FORECAST_HORIZONS))
    if not horizons:
        horizons = [1]
    if 1 not in horizons:
        horizons = [1] + [h for h in horizons if h != 1]

    # --- 1-day forward return (base for labels) ---
    if use_log_returns:
        df["future_return_1d"] = np.log(close.shift(-1) / close)
    else:
        df["future_return_1d"] = close.shift(-1) / close - 1.0

    # --- Multi-horizon forward returns from config.FORECAST_HORIZONS ---
    for h in horizons:
        if h == 1:
            continue
        col_name = f"future_return_{h}d"
        if use_log_returns:
            df[col_name] = np.log(close.shift(-h) / close)
        else:
            df[col_name] = close.shift(-h) / close - 1.0

    fr1 = df["future_return_1d"].values
    thr = config.DIRECTION_THRESHOLD

    # --- 3-class direction label (DOWN / FLAT / UP) ---
    direction_3c = np.ones(len(df), dtype=np.int64)  # start as FLAT
    direction_3c[fr1 > thr] = 2
    direction_3c[fr1 < -thr] = 0
    df[config.TRIPLE_DIRECTION_COLUMN] = direction_3c

    # --- Binary big-move label with IGNORE for small moves ---
    direction_2c = np.full(len(df), config.IGNORE_LABEL, dtype=np.int64)
    direction_2c[fr1 > thr] = 1  # UP
    direction_2c[fr1 < -thr] = 0  # DOWN
    df[config.BINARY_DIRECTION_COLUMN] = direction_2c

    # --- Drop rows with NaNs in any forward-return column we care about ---
    drop_cols = ["future_return_1d"]
    drop_cols += [f"future_return_{h}d" for h in horizons if h != 1]
    drop_cols = list(dict.fromkeys(drop_cols))

    df = df.dropna(subset=drop_cols)

    return df


def print_label_distribution(
    df: pd.DataFrame,
    name: str = "FULL",
    freq: str = "Y",
    direction_col: str | None = None,
) -> None:
    """
    Print label counts (and percentages) aggregated over time.

    The behaviour depends on which label column is passed:

      - If direction_col == config.TRIPLE_DIRECTION_COLUMN or if the
        values are in {0,1,2}, interpret as the 3-class label:
            0 = DOWN, 1 = FLAT, 2 = UP

      - Otherwise interpret as the binary big-move label:
            0 = DOWN, 1 = UP, IGNORE_LABEL (e.g. -1) = small move

    By default direction_col = config.DIRECTION_LABEL_COLUMN, which in
    Experiment 4 points to the binary big-move label.
    """
    if direction_col is None:
        direction_col = config.DIRECTION_LABEL_COLUMN

    if direction_col not in df.columns:
        raise ValueError(
            f"print_label_distribution expects a '{direction_col}' column in df."
        )
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("print_label_distribution expects a DatetimeIndex on df.")

    if len(df) == 0:
        print(f"[data_pipeline] [{name}] No rows -> cannot compute label distribution.")
        return

    labels = df[direction_col].astype(int)
    unique_vals = set(labels.unique())

    is_triple = (
        direction_col == config.TRIPLE_DIRECTION_COLUMN
        or unique_vals.issubset({0, 1, 2})
    )

    print(f"[data_pipeline] Label distribution for {name} (grouped by {freq}):")
    grouped = df.groupby(df.index.to_period(freq))[direction_col]

    if is_triple:
        # 3-class case: 0=DOWN,1=FLAT,2=UP
        total_downs = int((labels == 0).sum())
        total_flats = int((labels == 1).sum())
        total_ups = int((labels == 2).sum())
        total_count = total_downs + total_flats + total_ups

        for period, group in grouped:
            g = group.astype(int)
            downs = int((g == 0).sum())
            flats = int((g == 1).sum())
            ups = int((g == 2).sum())
            count = downs + flats + ups
            if count == 0:
                continue

            print(
                f"  [{name}] {period} -> "
                f"DOWN: {downs:4d} ({downs / count * 100:5.1f}%), "
                f"FLAT: {flats:4d} ({flats / count * 100:5.1f}%), "
                f"UP: {ups:4d} ({ups / count * 100:5.1f}%)  (N={count})"
            )

        print(
            f"  [{name}] TOTAL -> "
            f"DOWN: {total_downs} ({total_downs / total_count * 100:.1f}%), "
            f"FLAT: {total_flats} ({total_flats / total_count * 100:.1f}%), "
            f"UP: {total_ups} ({total_ups / total_count * 100:.1f}%)  (N={total_count})"
        )
    else:
        # Binary big-move case: 0=DOWN,1=UP,IGNORE_LABEL=small moves
        ignore_label = getattr(config, "IGNORE_LABEL", None)

        total_downs = int((labels == 0).sum())
        total_ups = int((labels == 1).sum())
        total_ignored = int((labels == ignore_label).sum()) if ignore_label is not None else 0
        total_count = len(labels)

        for period, group in grouped:
            g = group.astype(int)
            downs = int((g == 0).sum())
            ups = int((g == 1).sum())
            ignored = int((g == ignore_label).sum()) if ignore_label is not None else 0
            count = len(g)
            if count == 0:
                continue

            print(
                f"  [{name}] {period} -> "
                f"DOWN: {downs:4d} ({downs / count * 100:5.1f}%), "
                f"UP:   {ups:4d} ({ups / count * 100:5.1f}%), "
                f"IGN:  {ignored:4d} ({ignored / count * 100:5.1f}%)  (N={count})"
            )

        print(
            f"  [{name}] TOTAL -> "
            f"DOWN: {total_downs} ({total_downs / total_count * 100:.1f}%), "
            f"UP:   {total_ups} ({total_ups / total_count * 100:.1f}%), "
            f"IGN:  {total_ignored} ({total_ignored / total_count * 100:.1f}%)  (N={total_count})"
        )

    print()


# ============================
# 5. TRAIN / VAL / TEST SPLIT
# ============================

def split_by_date(
    df: pd.DataFrame,
    train_start: str = config.TRAIN_START_DATE,
    train_end: str = config.TRAIN_END_DATE,
    val_start: str = config.VAL_START_DATE,
    val_end: str = config.VAL_END_DATE,
    test_start: str = config.TEST_START_DATE,
    test_end: str = config.TEST_END_DATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the full DataFrame into train / validation / test sets by date range.

    Assumes df has a DatetimeIndex.
    """
    train_df = df.loc[train_start:train_end].copy()
    val_df = df.loc[val_start:val_end].copy()
    test_df = df.loc[test_start:test_end].copy()

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
    """
    price_volume_cols = [col for col in feature_cols if col in config.PRICE_VOLUME_COLS]
    indicator_cols = [col for col in feature_cols if col in config.INDICATOR_COLS]

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    pv_scaler = MinMaxScaler()
    ind_scaler = StandardScaler()

    if price_volume_cols:
        pv_scaler.fit(train_df[price_volume_cols])
    if indicator_cols:
        ind_scaler.fit(train_df[indicator_cols])

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
    future_cols: List[str] | None = None,
    label_col: str | None = config.TARGET_COLUMN,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Build sliding-window sequences and labels from a DataFrame for classification.

    For each sample, we create:
      - X_past: sequence of length `seq_length` over `feature_cols`
      - y:      scalar integer label taken from `label_col` at the LAST day
                in the sequence (e.g. the binary big-move label in Exp.4)
      - f:      (optional) vector of future covariates for t+1 taken from
                `future_cols` at the LAST day in the sequence
    """
    df = df.copy()
    df = df.sort_index()  # ensure chronological order

    missing_feat = [c for c in feature_cols if c not in df.columns]
    if missing_feat:
        raise ValueError(
            f"Missing feature columns in df: {missing_feat}. "
            "Check indicator and feature generation."
        )
    feature_array = df[feature_cols].values.astype(np.float32)

    if label_col is None:
        raise ValueError("label_col must be provided for classification.")
    if label_col not in df.columns:
        raise ValueError(
            f"DataFrame must contain '{label_col}' column before building sequences."
        )

    target_array = df[label_col].values.astype(np.int64)

    if future_cols is not None:
        for col in future_cols:
            if col not in df.columns:
                raise ValueError(
                    f"Future covariate column '{col}' is missing. "
                    f"Make sure add_future_covariates() was called."
                )
        future_array = df[future_cols].values.astype(np.float32)
    else:
        future_array = None

    sequences: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    future_covariates: List[np.ndarray] = []

    for end_idx in range(seq_length - 1, len(df)):
        start_idx = end_idx - seq_length + 1

        seq_x = feature_array[start_idx: end_idx + 1]
        y = target_array[end_idx]

        sequences.append(seq_x)
        labels.append(y)

        if future_array is not None:
            f = future_array[end_idx]
            future_covariates.append(f)

    sequences_arr = np.stack(sequences)
    labels_arr = np.array(labels)

    if future_array is not None:
        future_covariates_arr: np.ndarray | None = np.stack(future_covariates)
    else:
        future_covariates_arr = None

    return sequences_arr, labels_arr, future_covariates_arr


# ============================
# 8. FILTER IGNORE LABEL
# ============================

def filter_ignore_label(
    sequences: np.ndarray,
    labels: np.ndarray,
    future_covariates: np.ndarray | None,
    ignore_label: int = config.IGNORE_LABEL,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Remove samples whose label equals `ignore_label`.

    Used in Experiment 4 so that the model only sees big-move days
    (DOWN / UP) and drops small-move days entirely.
    """
    if ignore_label is None:
        return sequences, labels, future_covariates

    mask = labels != ignore_label
    sequences = sequences[mask]
    labels = labels[mask]
    if future_covariates is not None:
        future_covariates = future_covariates[mask]

    return sequences, labels, future_covariates


# ============================
# 9. PYTORCH DATASET
# ============================

class BTCTFTDataset(Dataset):
    """
    Simple PyTorch Dataset for the BTC TFT model.

    Each item represents a single training sample:
      - sequences: past covariates of shape (seq_length, num_features)
      - labels:    scalar integer class
                   (Experiment 4: 0 = DOWN, 1 = UP)
      - future_covariates (optional): known future covariates for t+1
        of shape (num_future_features,)
    """

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        future_covariates: np.ndarray | None = None,
    ):
        assert len(sequences) == len(labels), "Sequences and labels must have same length."

        if future_covariates is not None:
            assert len(future_covariates) == len(labels), (
                "Future covariates and labels must have same length."
            )
            self.future_covariates = torch.from_numpy(future_covariates).float()
        else:
            self.future_covariates = None

        self.sequences = torch.from_numpy(sequences).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        if self.future_covariates is None:
            return self.sequences[idx], self.labels[idx]
        else:
            return self.sequences[idx], self.future_covariates[idx], self.labels[idx]


# ============================
# 10. HIGH-LEVEL PIPELINE
# ============================

def prepare_datasets(
    csv_path: str | None = None,
    seq_length: int = config.SEQ_LENGTH,
) -> Tuple[BTCTFTDataset, BTCTFTDataset, BTCTFTDataset, Dict[str, object]]:
    """
    High-level function that runs the full data pipeline for Experiment 4.

    Experiment 4 setup:
      - 1-day-ahead binary big-move direction classification
        (0 = DOWN, 1 = UP, IGNORE_LABEL = small moves)
      - The original 3-class label (direction_3c) is still created for
        analysis but is not used as the training target.
    """
    # 1. Load
    df = load_btc_daily(csv_path)

    # 2. Add indicators
    df = add_technical_indicators(df)

    # 3. Add calendar and halving features (per day)
    df = add_calendar_features(df)
    df = add_halving_features(df)

    # 4. Add next-day future covariate columns
    df = add_future_covariates(df)

    # 5. Add forward returns + labels
    df = add_target_column(df)

    # 6. Print global label distribution (yearly) using active label
    print_label_distribution(df, name="FULL", freq="Y")

    feature_cols = config.FEATURE_COLS
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Feature column '{col}' is missing. Check indicator generation.")

    future_cols = config.FUTURE_COVARIATE_COLS

    # 7. Train/Val/Test split
    train_df, val_df, test_df = split_by_date(df)

    # 8. Print label distribution for each split (before filtering ignore)
    print_label_distribution(train_df, name="TRAIN", freq="Y")
    print_label_distribution(val_df, name="VAL", freq="Y")
    print_label_distribution(test_df, name="TEST", freq="Y")

    # 9. Scale features (for past covariates)
    train_df_scaled, val_df_scaled, test_df_scaled, scalers = scale_features(
        train_df, val_df, test_df, feature_cols
    )

    # 10. Build sequences + future covariate vectors (using binary label)
    label_col = config.TARGET_COLUMN

    train_seq, train_labels, train_future = build_sequences(
        train_df_scaled,
        feature_cols,
        seq_length,
        future_cols=future_cols,
        label_col=label_col,
    )
    val_seq, val_labels, val_future = build_sequences(
        val_df_scaled,
        feature_cols,
        seq_length,
        future_cols=future_cols,
        label_col=label_col,
    )
    test_seq, test_labels, test_future = build_sequences(
        test_df_scaled,
        feature_cols,
        seq_length,
        future_cols=future_cols,
        label_col=label_col,
    )

    # 11. Filter out IGNORE_LABEL samples (small moves)
    ignore = config.IGNORE_LABEL
    train_seq, train_labels, train_future = filter_ignore_label(
        train_seq, train_labels, train_future, ignore
    )
    val_seq, val_labels, val_future = filter_ignore_label(
        val_seq, val_labels, val_future, ignore
    )
    test_seq, test_labels, test_future = filter_ignore_label(
        test_seq, test_labels, test_future, ignore
    )

    # Optional: quick summary of post-filtering label balance
    for split_name, labels_arr in [
        ("TRAIN", train_labels),
        ("VAL", val_labels),
        ("TEST", test_labels),
    ]:
        num = len(labels_arr)
        if num == 0:
            print(f"[data_pipeline] {split_name}: 0 samples after filtering IGNORE_LABEL.")
            continue
        downs = int((labels_arr == 0).sum())
        ups = int((labels_arr == 1).sum())
        print(
            f"[data_pipeline] {split_name} after filtering IGNORE_LABEL: "
            f"N={num}, DOWN={downs} ({downs/num:.1%}), UP={ups} ({ups/num:.1%})"
        )

    # 12. Wrap into Datasets
    train_dataset = BTCTFTDataset(train_seq, train_labels, train_future)
    val_dataset = BTCTFTDataset(val_seq, val_labels, val_future)
    test_dataset = BTCTFTDataset(test_seq, test_labels, test_future)

    return train_dataset, val_dataset, test_dataset, scalers


# ============================
# 11. QUICK SELF-TEST
# ============================

if __name__ == "__main__":
    print("[data_pipeline] Running self-test for Experiment 4.")

    train_ds, val_ds, test_ds, scalers = prepare_datasets()

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")
    print(f"Test samples:  {len(test_ds)}")

    sample = train_ds[0]

    if len(sample) == 2:
        x, y = sample
        future = None
    else:
        x, future, y = sample

    print(f"One sample X shape:       {x.shape}  (seq_length, n_features)")
    if future is not None:
        print(f"One sample future shape:  {future.shape}  (n_future_features,)")

    print(f"One sample y shape:       {tuple(y.shape)}")
    print(f"One sample y values:      {y.numpy()}")

    x_np = x.numpy()
    seq_len, n_features = x_np.shape
    feature_names = scalers.get("feature_cols", [])

    print("\nFirst training sample (scaled feature values):")

    print("\n  First 3 timesteps of the sequence:")
    num_first = min(3, seq_len)
    for t in range(num_first):
        row_values = ", ".join(f"{v:.4f}" for v in x_np[t])
        print(f"    t={t:02d}: [{row_values}]")

    print("\n  Last 3 timesteps of the sequence:")
    num_last = min(3, seq_len)
    start_last = seq_len - num_last
    for t in range(start_last, seq_len):
        row_values = ", ".join(f"{v:.4f}" for v in x_np[t])
        print(f"    t={t:02d}: [{row_values}]")

    print("\n  Last timestep (most recent day in the window):")
    last_step = x_np[-1]
    for i in range(n_features):
        fname = feature_names[i] if i < len(feature_names) else f"feat_{i}"
        print(f"    {fname:>12}: {last_step[i]: .4f}")

    if future is not None:
        print("\n  Future covariates for t+1 (raw values):")
        print("    Values:", ", ".join(f"{v:.4f}" for v in future.numpy()))
        print("    Columns:", ", ".join(config.FUTURE_COVARIATE_COLS))

    print("\n[data_pipeline] Self-test finished.")