"""
Turns raw BTC price data (daily or hourly) into PyTorch Datasets for the TFT
model used in the experiment_8 setup.

Main steps:
1. Load BTC price data from CSV (daily or hourly, depending on config).
2. Add technical indicators (ROC, ATR, MACD, RSI).
3. Add calendar and halving-related features.
4. Create future covariate columns via shift to t+H (aligned with the
   main forecast horizon FORECAST_HORIZONS[0]).
5. Create forward return(s) and a 3-class direction label.
6. Split into train / validation / test sets by date.
7. Scale features (prices & volume with MinMax, indicators with StandardScaler).
8. Build sliding window sequences for the TFT model (past inputs)
   and per-sample future covariate vectors.
9. Provide a BTCTFTDataset class to be used in train_tft.py and evaluate_tft.py.
"""

import os
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import talib  # Technical Analysis library (C + Python wrapper)
from experiment_8 import config

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
    df = df.ffill()

    return df


def load_btc_hourly(csv_path: str | None = None) -> pd.DataFrame:
    """
    Load hourly BTC data from a CryptoDataDownload-style CSV and return a cleaned
    DataFrame with a continuous hourly DatetimeIndex.

    This mirrors load_btc_daily but uses hourly frequency ("H") instead of daily.
    """
    if csv_path is None:
        csv_path = config.BTC_HOURLY_CSV_PATH

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

    df["date"] = pd.to_datetime(df["date"])

    # Sort oldest → newest
    df = df.sort_values("date")
    df = df.set_index("date")

    # Lowercase all column names for consistency
    df.columns = [c.lower() for c in df.columns]

    # Rename volume columns
    rename_map = {}
    if "volume btc" in df.columns:
        rename_map["volume btc"] = "volume_btc"
    if "volume usd" in df.columns:
        rename_map["volume usd"] = "volume_usd"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Drop columns we don't use
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

    # Ensure continuous hourly index
    df = df.asfreq("h")
    df = df.ffill()

    return df


def load_onchain_daily(csv_path: str | None = None) -> pd.DataFrame:
    """
    Load daily BTC on-chain metrics from CSV and return a cleaned DataFrame.

    Expected columns in the CSV (one row per calendar day):
        - date
        - active_addresses
        - tx_count
        - mvrv
        - sopr
        - hash_rate

    This matches the output of download_btc_onchain_daily.py.
    """
    if csv_path is None:
        csv_path = config.BTC_ONCHAIN_DAILY_CSV_PATH

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"On-chain CSV file not found at: {csv_path}")

    # Read the CSV
    df = pd.read_csv(csv_path)

    # Ensure we have a 'date' column
    if "date" not in df.columns:
        raise ValueError(
            "Expected a 'date' column in on-chain CSV but did not find one. "
            "Check BTC_onchain_daily.csv structure."
        )

    # Parse date and set as index
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df = df.set_index("date")

    # Columns coming from the downloader are already lowercase, but normalize just in case
    df.columns = [c.lower() for c in df.columns]

    # Enforce daily frequency and forward-fill gaps
    df = df.asfreq("D")
    df = df.ffill()

    # Drop any leading rows where everything is NaN (should be rare)
    df = df.dropna(how="all")

    return df


def add_onchain_features(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Load daily on-chain data.
    2. Compute aa_ma_ratio, tx_ma_ratio, hash_ma_ratio, mvrv_z, sopr_z
       on the DAILY index.
    3. Merge those daily features into the hourly BTC dataframe
       by calendar day.
    """
    if not isinstance(hourly_df.index, pd.DatetimeIndex):
        raise TypeError("add_onchain_features expects hourly_df with a DatetimeIndex.")

    df = hourly_df.copy()

    # --- avoid ambiguity between index name and 'date' column ---
    # After load_btc_hourly(), the index name is usually 'date'.
    # We rename it so we can safely create a separate 'date' column for merging.
    if df.index.name == "date":
        df.index = df.index.rename("timestamp")

    # ---- 1) Load daily on-chain data ----
    onchain_df = load_onchain_daily()  # index: daily dates

    # ---- 2) Compute derived features on the DAILY frame ----
    onchain_df["aa_ma_ratio"] = onchain_df["active_addresses"] / (
        onchain_df["active_addresses"].rolling(window=7, min_periods=1).mean()
    )
    onchain_df["tx_ma_ratio"] = onchain_df["tx_count"] / (
        onchain_df["tx_count"].rolling(window=7, min_periods=1).mean()
    )
    onchain_df["hash_ma_ratio"] = onchain_df["hash_rate"] / (
        onchain_df["hash_rate"].rolling(window=30, min_periods=1).mean()
    )
    onchain_df["mvrv_z"] = onchain_df["mvrv"]
    onchain_df["sopr_z"] = onchain_df["sopr"]

    onchain_df = onchain_df[config.ONCHAIN_COLS]

    # ---- 3) Merge DAILY features into HOURLY BTC df ----
    # Create a daily key from the hourly timestamps
    df["date"] = df.index.floor("D")

    df = df.merge(
        onchain_df,
        how="left",
        left_on="date",
        right_index=True,
    )

    # OPTIONAL: if you want to be safe against missing on-chain days, you can:
    # df[config.ONCHAIN_COLS] = df[config.ONCHAIN_COLS].ffill()

    # Drop temporary merge key
    df = df.drop(columns=["date"])

    return df


# ============================
# 2. TECHNICAL INDICATORS
# ============================

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators using TA-Lib.

    Indicators:
        - ROC(10)        : 10-period Rate of Change (%)
        - ATR(14)        : 14-period Average True Range
        - MACD(12,26,9)  : MACD line, signal line, histogram
        - RSI(14)        : 14-period Relative Strength Index

    'Period' means days for daily data, hours for hourly data, etc.

    After computing indicators, we drop the initial rows that contain NaNs
    caused by the warm-up periods.
    """
    df = df.copy()

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    # --------- ROC(10) ---------
    df["roc_10"] = talib.ROC(close, timeperiod=10)

    # --------- ATR(14) ---------
    df["atr_14"] = talib.ATR(high, low, close, timeperiod=14)

    # --------- MACD (12, 26, 9) ---------
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

    # Drop warm-up rows with NaNs
    df = df.dropna()

    return df


# ============================
# 3. CALENDAR & HALVING FEATURES
# ============================

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-timestep calendar features:

        - day_of_week : 0=Monday, ..., 6=Sunday
        - is_weekend  : 1 if Saturday/Sunday, else 0
        - month       : 1–12
        - hour_of_day : 0–23 (for intraday data; will be 0 for pure daily data)

    These are base features for time t, used later to derive t+1 (or t+H)
    covariates in add_future_covariates().
    """
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex to add calendar features.")

    # Day of week: Monday=0, Sunday=6
    df["day_of_week"] = df.index.dayofweek

    # Weekend flag: 1 if Saturday (5) or Sunday (6), else 0
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Month number: 1–12
    df["month"] = df.index.month

    # Hour of day: 0–23. For daily data (freq='D'), this is typically 0.
    df["hour_of_day"] = df.index.hour

    return df


def add_halving_features(df: pd.DataFrame, window_days: int = 90) -> pd.DataFrame:
    """
    Add halving-related features:

        - days_to_next_halving : integer days from t to the next halving date
        - is_halving_window    : 1 if within ±window_days of any halving date

    Halving dates are known (or well-approximated) in advance, so they are
    valid sources of "known future" information for crypto.
    """
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex to add halving features.")

    # Known / approximate Bitcoin halving dates.
    halving_dates = [
        pd.Timestamp("2012-11-28"),
        pd.Timestamp("2016-07-09"),
        pd.Timestamp("2020-05-11"),
        pd.Timestamp("2024-04-19"),  # recent halving (approx / actual)
        pd.Timestamp("2028-04-20"),  # approximate next halving
    ]

    dates = df.index.to_pydatetime()
    num_rows = len(df)

    days_to_next_halving = np.zeros(num_rows, dtype=np.float32)
    is_halving_window = np.zeros(num_rows, dtype=np.int64)

    for i, current_date in enumerate(dates):
        # Compute day differences to each halving date
        day_diffs = np.array(
            [(hd - current_date).days for hd in halving_dates],
            dtype=np.int32,
        )

        # Days to the next halving (smallest non-negative diff if exists,
        # otherwise the closest halving in absolute terms).
        non_negative = day_diffs[day_diffs >= 0]
        if len(non_negative) > 0:
            days_next = int(non_negative.min())
        else:
            # After last halving: just take the closest halving in absolute value
            days_next = int(day_diffs[np.argmin(np.abs(day_diffs))])

        days_to_next_halving[i] = days_next

        # Halving window flag: within ±window_days of ANY halving
        nearest_days = int(day_diffs[np.argmin(np.abs(day_diffs))])
        if abs(nearest_days) <= window_days:
            is_halving_window[i] = 1

    df["days_to_next_halving"] = days_to_next_halving
    df["is_halving_window"] = is_halving_window

    return df


def add_future_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create future covariate columns by shifting base features.

    Step 5 (Experiment 6, hourly data):

      We now align "known future" covariates with the same horizon H that
      is used for the label in add_target_column(), where:

          H = config.FORECAST_HORIZONS[0]  (in time steps, e.g. 24 hours)

      Concretely, we take base features at time t+H:

          - day_of_week       -> dow_next
          - is_weekend        -> is_weekend_next
          - month             -> month_next
          - hour_of_day       -> hour_next
          - is_halving_window -> is_halving_window_next

      and shift them back by H steps so that at index t we store the
      information about time t+H in these *_next columns.

      The last H rows will have NaNs in these *_next columns (because there
      is no t+H); they will later be dropped together with the NaN targets
      in add_target_column().
    """
    df = df.copy()

    required_base_cols = [
        "day_of_week",
        "is_weekend",
        "month",
        "hour_of_day",
        "is_halving_window",
    ]
    missing = [c for c in required_base_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"add_future_covariates expected base columns {required_base_cols}, "
            f"but missing: {missing}. Make sure to call add_calendar_features() "
            f"and add_halving_features() first."
        )

    # Horizon H used for the direction label / future return.
    horizons_cfg = list(getattr(config, "FORECAST_HORIZONS", []))
    if not horizons_cfg:
        horizon = 1
    else:
        horizon = int(horizons_cfg[0])

    if horizon < 1:
        raise ValueError(
            f"FORECAST_HORIZONS[0] must be a positive integer, got {horizon}."
        )

    # Shift base features by -H so that at index t we store information
    # corresponding to t+H.
    df["dow_next"] = df["day_of_week"].shift(-horizon)
    df["is_weekend_next"] = df["is_weekend"].shift(-horizon)
    df["month_next"] = df["month"].shift(-horizon)
    df["hour_next"] = df["hour_of_day"].shift(-horizon)
    df["is_halving_window_next"] = df["is_halving_window"].shift(-horizon)

    return df


# ============================
# 4. TARGETS (RETURN + UP/DOWN)
# ============================

def add_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add forward return columns for all horizons in config.FORECAST_HORIZONS
    and the 3-class direction label.

    Step 3 (Experiment 6, hourly data):

      - We always create a 1-step forward return column 'future_return_1d'
        (for potential baselines / diagnostics).

      - For each horizon h in config.FORECAST_HORIZONS (e.g. [24]) we create
        a column:

            future_return_{h}d

        which is either a simple return:

            (close_{t+h} - close_t) / close_t

        or a log return:

            log(close_{t+h} / close_t)

        depending on config.USE_LOG_RETURNS.

      - The 3-class direction label (config.TRIPLE_DIRECTION_COLUMN, i.e.
        'direction_3c') is now defined on the H-step return where:

            H = config.FORECAST_HORIZONS[0]

        using a symmetric threshold tau = config.DIRECTION_THRESHOLD:

            0 = DOWN  if r_H < -tau
            1 = FLAT  if |r_H| <= tau
            2 = UP    if r_H >  tau

    After creating these columns, rows with NaNs in any forward-return column
    are dropped (this removes the last max(h) rows).
    """
    df = df.copy()

    # Decide whether to use log-returns or simple returns.
    use_log_returns = getattr(config, "USE_LOG_RETURNS", False)

    # Make sure we have a clean, float-valued close series
    close = df["close"].astype(float)

    # Horizons defined in config (e.g. [24]); fallback to [1] if empty.
    horizons_cfg = list(getattr(config, "FORECAST_HORIZONS", []))
    if not horizons_cfg:
        horizons_cfg = [1]

    # Horizon used for the classification label (H-step).
    label_horizon = int(horizons_cfg[0])

    # Collect all horizons for which we want forward-return columns,
    # and ensure 1-step is always present.
    horizons = sorted(set(horizons_cfg + [1]))

    # --- 1-step forward return (kept for diagnostics / baselines) ---
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

    # --- 3-class direction label (based on H-step return) ---
    if label_horizon == 1:
        label_return_col = "future_return_1d"
    else:
        label_return_col = f"future_return_{label_horizon}d"
        if label_return_col not in df.columns:
            raise ValueError(
                f"Expected column '{label_return_col}' for label horizon H={label_horizon}, "
                f"but it was not created. Check config.FORECAST_HORIZONS."
            )

    thr = config.DIRECTION_THRESHOLD

    r_H = df[label_return_col].values
    direction_3c = np.ones(len(df), dtype=np.int64)  # start as FLAT

    # UP if strictly above +thr
    direction_3c[r_H > thr] = 2
    # DOWN if strictly below -thr
    direction_3c[r_H < -thr] = 0

    df[config.TRIPLE_DIRECTION_COLUMN] = direction_3c

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
) -> None:
    """
    Print DOWN / FLAT / UP counts (and percentages) aggregated over time.

    Uses the column specified in config.DIRECTION_LABEL_COLUMN:

        0 = DOWN
        1 = FLAT
        2 = UP

    The DataFrame is grouped by a time frequency (default: yearly) and the
    distribution of labels is printed for each period and overall.
    """
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

    total_downs = int((labels == 0).sum())
    total_flats = int((labels == 1).sum())
    total_ups   = int((labels == 2).sum())
    total_count = total_downs + total_flats + total_ups

    down_ratio = total_downs / total_count
    flat_ratio = total_flats / total_count
    up_ratio   = total_ups   / total_count

    print(f"[data_pipeline] Label distribution for {name} (grouped by {freq}):")

    # Group by the requested frequency (default: yearly)
    grouped = df.groupby(df.index.to_period(freq))[direction_col]

    for period, group in grouped:
        g = group.astype(int)
        downs = int((g == 0).sum())
        flats = int((g == 1).sum())
        ups   = int((g == 2).sum())
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
        f"DOWN: {total_downs} ({down_ratio*100:.1f}%), "
        f"FLAT: {total_flats} ({flat_ratio*100:.1f}%), "
        f"UP: {total_ups} ({up_ratio*100:.1f}%)  (N={total_count})"
    )
    print()


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
    """
    price_volume_cols = [c for c in feature_cols if c in config.PRICE_VOLUME_COLS]

    indicator_like_cols = [
        c for c in feature_cols
        if c not in price_volume_cols  # everything else -> StandardScaler
    ]
    # or explicit union of INDICATOR_COLS + ONCHAIN_COLS

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    # --- Fit scalers on TRAIN only ---
    pv_scaler = MinMaxScaler()
    ind_scaler = StandardScaler()

    if price_volume_cols:
        pv_scaler.fit(train_df[price_volume_cols])
    if indicator_like_cols:
        ind_scaler.fit(train_df[indicator_like_cols])

    # --- Transform all splits ---
    for df in (train_df, val_df, test_df):
        if price_volume_cols:
            df[price_volume_cols] = pv_scaler.transform(df[price_volume_cols])
        if indicator_like_cols:
            df[indicator_like_cols] = ind_scaler.transform(df[indicator_like_cols])

    scalers = {
        "price_volume_scaler": pv_scaler,
        "indicator_scaler": ind_scaler,
        "feature_cols": feature_cols,
        "price_volume_cols": price_volume_cols,
        "indicator_cols": indicator_like_cols,
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
      - y:      scalar integer label taken from `label_col` at the LAST timestep
      - f:      (optional) vector of future covariates for the next timestep
                future covariates aligned with the main horizon H (t+H) for each sample.

    Indexing convention:
      - The DataFrame is assumed to be sorted in chronological order.
      - If seq_length = 30 and the current end index is t, then
          X_past uses rows [t-29, ..., t]
          y      is df[label_col] at index t
          f      is df[future_cols] at index t
    """
    df = df.copy()
    df = df.sort_index()  # ensure chronological order

    # Past covariates
    missing_feat = [c for c in feature_cols if c not in df.columns]
    if missing_feat:
        raise ValueError(
            f"Missing feature columns in df: {missing_feat}. "
            "Check indicator and feature generation."
        )
    feature_array = df[feature_cols].values.astype(np.float32)

    # Targets: scalar classification label
    if label_col is None:
        raise ValueError("label_col must be provided for classification.")
    if label_col not in df.columns:
        raise ValueError(
            f"DataFrame must contain '{label_col}' column before building sequences."
        )

    target_array = df[label_col].values.astype(np.int64)

    # Future covariates
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

    # We start at index seq_length-1 because we need seq_length timesteps
    for end_idx in range(seq_length - 1, len(df)):
        start_idx = end_idx - seq_length + 1

        # Past sequence for [start_idx, ... end_idx]
        seq_x = feature_array[start_idx: end_idx + 1]   # (seq_length, n_features)
        y = target_array[end_idx]                       # scalar at end_idx

        sequences.append(seq_x)
        labels.append(y)

        # Future covariates
        if future_array is not None:
            f = future_array[end_idx]                   # (n_future_features,)
            future_covariates.append(f)

    sequences_arr = np.stack(sequences)  # (N, seq_length, n_features)
    labels_arr = np.array(labels)        # (N,)

    if future_array is not None:
        future_covariates_arr: np.ndarray | None = np.stack(
            future_covariates
        )  # (N, n_future_features)
    else:
        future_covariates_arr = None

    return sequences_arr, labels_arr, future_covariates_arr


# ============================
# 8. PYTORCH DATASET
# ============================

class BTCTFTDataset(Dataset):
    """
    Simple PyTorch Dataset for the BTC TFT model.

    Each item represents a single training sample:
      - sequences: past covariates of shape (seq_length, num_features)
      - labels:    scalar integer class in {0, 1, 2}
      - future_covariates (optional): known future covariates
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
        # Long dtype for CrossEntropyLoss (class indices)
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        if self.future_covariates is None:
            return self.sequences[idx], self.labels[idx]
        else:
            return self.sequences[idx], self.future_covariates[idx], self.labels[idx]


# ============================
# 9. HIGH-LEVEL PIPELINE
# ============================

def prepare_datasets(
    csv_path: str | None = None,
    seq_length: int = config.SEQ_LENGTH,
    return_forward_returns: bool = False,
) -> (
    Tuple[BTCTFTDataset, BTCTFTDataset, BTCTFTDataset, Dict[str, object]]
    | Tuple[
        BTCTFTDataset,
        BTCTFTDataset,
        BTCTFTDataset,
        Dict[str, object],
        Dict[str, np.ndarray],
    ]
):
    """
    High-level function that runs the full data pipeline for the current experiment.

    It now supports both daily and hourly BTC data:

      - If csv_path is None:
          * Uses config.FREQUENCY to decide:
              "1h"  -> load_btc_hourly(config.BTC_HOURLY_CSV_PATH)
              other -> load_btc_daily(config.BTC_DAILY_CSV_PATH)

      - If csv_path is provided:
          * Heuristically picks hourly vs daily based on the path name
            ("hour" or "1h" -> hourly; otherwise daily).

    Steps (unchanged conceptually from the daily setup):
      1. Load BTC OHLCV data from CSV.
      2. Add technical indicators (ROC, ATR, MACD, RSI).
      2.5 (Exp-7) Optionally add on-chain features.
      3. Add calendar and halving-related base features.
      4. Add future covariate columns via shift to t+H (H = FORECAST_HORIZONS[0]).
      5. Add forward return columns and the 3-class direction label
         based on the H-step return where H = FORECAST_HORIZONS[0].
      6. Print label distribution (per year + overall) using the direction label.
      7. Split by date into train / val / test using config date ranges.
      8. Print label distribution for each split.
      9. Scale past covariates.
     10. Build sliding-window sequences and future covariate vectors.
     11. Wrap them into BTCTFTDataset objects.
     12. Optionally return aligned H-step forward returns for each sample.
    """
    # 1. Load
    if csv_path is not None:
        lower = csv_path.lower()
        if "hour" in lower or "1h" in lower:
            df = load_btc_hourly(csv_path)
        else:
            df = load_btc_daily(csv_path)
    else:
        freq = getattr(config, "FREQUENCY", "D")
        if freq.lower() in ("1h", "h", "hourly"):
            df = load_btc_hourly()
        else:
            df = load_btc_daily()

    # 2. Add indicators
    df = add_technical_indicators(df)

    # 2.5 Add on-chain features (Experiment 7)
    # Only do this when USE_ONCHAIN is True in config.py
    if getattr(config, "USE_ONCHAIN", False):
        df = add_onchain_features(df)

    # 3. Add calendar and halving features
    df = add_calendar_features(df)
    df = add_halving_features(df)

    # 4. Add future covariate columns (shift to t+H, where H = FORECAST_HORIZONS[0])
    df = add_future_covariates(df)

    # 5. Add forward return(s) + 3-class direction label (H-step)
    df = add_target_column(df)

    # 6. Print global label distribution (grouped yearly)
    print_label_distribution(df, name="FULL", freq="Y")

    # Which columns we feed into the model as past covariates
    feature_cols = config.FEATURE_COLS

    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Feature column '{col}' is missing. Check indicator/on-chain generation.")

    # Future covariate columns
    future_cols = config.FUTURE_COVARIATE_COLS

    # 7. Train/Val/Test split
    train_df, val_df, test_df = split_by_date(df)

    # 8. Print label distribution for each split
    print_label_distribution(train_df, name="TRAIN", freq="Y")
    print_label_distribution(val_df,   name="VAL",   freq="Y")
    print_label_distribution(test_df,  name="TEST",  freq="Y")

    # 9. Scale features (for past covariates)
    train_df_scaled, val_df_scaled, test_df_scaled, scalers = scale_features(
        train_df, val_df, test_df, feature_cols
    )

    # 9.5 Extract H-step forward returns aligned with samples (optional).
    #
    # For Experiment 6 with hourly data and FORECAST_HORIZONS = [24], this
    # is the "next 24 hours" return (log or simple), i.e. the same horizon
    # used for the direction_3c label.
    def extract_forward_returns(df_scaled: pd.DataFrame, seq_len: int) -> np.ndarray:
        horizons_cfg = list(getattr(config, "FORECAST_HORIZONS", []))
        if not horizons_cfg:
            label_horizon = 1
        else:
            label_horizon = int(horizons_cfg[0])

        if label_horizon == 1:
            col_name = "future_return_1d"
        else:
            col_name = f"future_return_{label_horizon}d"

        if col_name not in df_scaled.columns:
            raise ValueError(
                f"Column '{col_name}' not found. "
                f"Make sure add_target_column() was called before scaling."
            )

        returns_full = df_scaled[col_name].astype(float).values
        if len(returns_full) < seq_len:
            raise ValueError(
                f"Not enough rows ({len(returns_full)}) to build sequences "
                f"of length {seq_len}."
            )
        # Each sequence uses rows [t-seq_len+1 ... t], so the first label/return
        # corresponds to index t = seq_len-1.
        return returns_full[seq_len - 1 :]

    train_forward = extract_forward_returns(train_df_scaled, seq_length)
    val_forward   = extract_forward_returns(val_df_scaled, seq_length)
    test_forward  = extract_forward_returns(test_df_scaled, seq_length)

    # 10. Build sequences + future covariate vectors
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

    # Sanity check: returns must align with number of samples
    if (
        train_forward.shape[0] != train_seq.shape[0]
        or val_forward.shape[0] != val_seq.shape[0]
        or test_forward.shape[0] != test_seq.shape[0]
    ):
        raise RuntimeError(
            "Mismatch between number of sequences and forward returns. "
            f"Train: seq={train_seq.shape[0]}, ret={train_forward.shape[0]}; "
            f"Val: seq={val_seq.shape[0]}, ret={val_forward.shape[0]}; "
            f"Test: seq={test_seq.shape[0]}, ret={test_forward.shape[0]}"
        )

    # 11. Wrap into Datasets
    train_dataset = BTCTFTDataset(train_seq, train_labels, train_future)
    val_dataset   = BTCTFTDataset(val_seq, val_labels, val_future)
    test_dataset  = BTCTFTDataset(test_seq, test_labels, test_future)

    if return_forward_returns:
        forward_returns = {
            "train": train_forward,
            "val": val_forward,
            "test": test_forward,
        }
        return train_dataset, val_dataset, test_dataset, scalers, forward_returns

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
      - Print label distributions per period (full + splits)
      - Inspect the first training sample.
    """
    print("[data_pipeline] Running self-test.")

    # Run the full pipeline
    train_ds, val_ds, test_ds, scalers = prepare_datasets()

    # Print how many samples we have
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")
    print(f"Test samples:  {len(test_ds)}")

    # Take the first sample from the training set
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

    print("\n  Last timestep (most recent in the window):")
    last_step = x_np[-1]
    for i in range(n_features):
        fname = feature_names[i] if i < len(feature_names) else f"feat_{i}"
        print(f"    {fname:>12}: {last_step[i]: .4f}")

    if future is not None:
        print("\n  Future covariates for next step (raw values):")
        print("    Values:", ", ".join(f"{v:.4f}" for v in future.numpy()))
        print("    Columns:", ", ".join(config.FUTURE_COVARIATE_COLS))

    print("\n[data_pipeline] Self-test finished.")
