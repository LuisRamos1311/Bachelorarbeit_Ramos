"""
Build train/val/test PyTorch Datasets for the final_model Temporal Fusion Transformer (TFT)
from BTC OHLCV (daily or hourly), with optional external daily features.

Main steps:
1. Load BTC OHLCV from CSV (daily or hourly, depending on config.FREQUENCY / csv_path).
2. Add technical indicators (ROC, ATR, MACD, RSI).
3. Optionally merge DAILY on-chain and sentiment features into the time-indexed BTC frame
   using a lagged daily key (t joins day=t-config.DAILY_FEATURE_LAG_DAYS) to avoid same-day leakage.
4. Add calendar and halving-related features.
5. Create known-future covariate columns via shift to t+H
   (H = config.FORECAST_HORIZON; the future vector corresponds to the forecast end).
6. Split into train / validation / test sets by date.
7. Create multi-horizon return targets y_ret_1..y_ret_H *inside each split* and drop unlabeled tail rows.
8. Scale past covariates (fit scalers on train only; MinMax for PRICE_VOLUME_COLS, StandardScaler for other continuous cols).
   Binary indicator columns are kept as-is (not scaled).
9. Build sliding-window sequences (past inputs) and per-sample future covariate vectors.
10. Provide a BTCTFTDataset used by train_tft.py and evaluate_tft.py.
"""

import os
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import talib  # Technical Analysis library (C + Python wrapper)
from final_model import config
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# Public API (used by training/evaluation scripts):
# - prepare_datasets(...): builds train/val/test BTCTFTDataset objects
# - BTCTFTDataset: the PyTorch Dataset wrapper returned by prepare_datasets()

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
    Merge derived DAILY on-chain features into a time-indexed BTC dataframe (hourly or daily).

    - Loads daily on-chain data and engineers: aa_ma_ratio, tx_ma_ratio, hash_ma_ratio, mvrv_z, sopr_z.
    - Builds a lagged daily merge key:
        date_lagged = floor(timestamp to day) - config.DAILY_FEATURE_LAG_DAYS
      (this avoids same-day leakage when training on intraday data).
    - Left-joins the daily features onto each timestamp, forward-fills from past values,
      and drops any remaining NaNs at the very beginning (no look-ahead).
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

    # ---- 3) Merge DAILY features into the time-indexed BTC df ----
    # IMPORTANT: lag DAILY features so we don't use same-day values intraday.
    lag_days = int(config.DAILY_FEATURE_LAG_DAYS)
    if lag_days < 0:
        raise ValueError("config.DAILY_FEATURE_LAG_DAYS must be >= 0")

    # Merge key: for each hour on day D, attach daily features from day (D - lag_days)
    df["date_lagged"] = df.index.floor("D") - pd.Timedelta(days=lag_days)

    df = df.merge(
        onchain_df,
        how="left",
        left_on="date_lagged",
        right_index=True,
    )

    # Safe imputation (no look-ahead): forward-fill from past on-chain values.
    df[config.ONCHAIN_COLS] = df[config.ONCHAIN_COLS].ffill()

    # If there are still NaNs (typically only at the very beginning), drop them safely.
    df = df.dropna(subset=config.ONCHAIN_COLS)

    # Drop temporary merge key
    df = df.drop(columns=["date_lagged"])

    return df


def load_sentiment_daily(csv_path: str | None = None) -> pd.DataFrame:
    """
    Load the combined DAILY sentiment dataset (Reddit Pushshift + Fear & Greed)
    and return a DataFrame indexed by daily dates.

    Expected columns (besides "date") are config.SENTIMENT_COLS, e.g.:
        - reddit_sent_mean, reddit_sent_std, reddit_pos_ratio, reddit_neg_ratio,
          reddit_volume, reddit_volume_log,
        - fg_index_scaled, fg_change_1d, fg_missing

    The file is expected to be on a contiguous daily date grid. As a safety net,
    this loader reindexes to daily frequency (asfreq("D")) and forward-fills any internal gaps.
    """
    if csv_path is None:
        csv_path = config.BTC_SENTIMENT_DAILY_CSV_PATH

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Sentiment CSV not found at: {csv_path}\n"
            f"Expected it under your data folder as BTC_sentiment_daily.csv."
        )

    df = pd.read_csv(csv_path)

    if "date" not in df.columns:
        raise ValueError("Sentiment CSV must contain a 'date' column.")

    # Parse date, sort, set as index
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    # Make sure it is daily-indexed; forward fill internal gaps if any
    df = df.asfreq("D").ffill()

    # Validate required columns exist
    missing_cols = [c for c in config.SENTIMENT_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Sentiment CSV is missing required columns: {missing_cols}\n"
            f"Check BTC_sentiment_daily.csv header vs config.SENTIMENT_COLS."
        )

    return df


def add_sentiment_features(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge DAILY sentiment features into a time-indexed BTC dataframe (hourly or daily).

    Strategy:
      1) Load daily sentiment features (index = daily dates).
      2) Build a lagged daily merge key:
           date_lagged = floor(timestamp to day) - config.DAILY_FEATURE_LAG_DAYS
         (prevents same-day leakage when using intraday data).
      3) Left-join sentiment features onto each timestamp.
      4) Drop the temporary merge key and fill any remaining missing values with neutral constants
         (to prevent NaNs from crashing scaling/training if the sentiment series has gaps).

    Note: Filling uses constants only (no look-ahead), so it is safe.
    """
    if not isinstance(hourly_df.index, pd.DatetimeIndex):
        raise TypeError("add_sentiment_features expects hourly_df with a DatetimeIndex.")

    df = hourly_df.copy()

    # Avoid ambiguity between index name and 'date' column (same pattern as on-chain)
    if df.index.name == "date":
        df.index.name = "timestamp"

    sentiment_df = load_sentiment_daily()  # daily index

    # Merge daily → hourly using a LAGGED daily key (prevents same-day leakage intraday)
    lag_days = int(config.DAILY_FEATURE_LAG_DAYS)
    if lag_days < 0:
        raise ValueError("config.DAILY_FEATURE_LAG_DAYS must be >= 0")

    df["date_lagged"] = df.index.floor("D") - pd.Timedelta(days=lag_days)

    df = df.merge(
        sentiment_df,
        how="left",
        left_on="date_lagged",
        right_index=True,
    )

    df = df.drop(columns=["date_lagged"])

    # --- Safety: fill any missing values with neutral constants ---
    # (This prevents scaler/training crashes if there are any unexpected gaps.)
    # Reddit defaults: "no activity / neutral"
    reddit_defaults = {
        "reddit_sent_mean": 0.0,
        "reddit_sent_std": 0.0,
        "reddit_pos_ratio": 0.0,
        "reddit_neg_ratio": 0.0,
        "reddit_volume": 0.0,
        "reddit_volume_log": 0.0,
    }

    # Fear & Greed defaults: neutral level + explicit missingness
    fg_defaults = {
        "fg_index_scaled": 0.5,   # 50 scaled to 0.5
        "fg_change_1d": 0.0,
        "fg_missing": 1.0,
    }

    # Apply defaults only for columns that exist in your config
    for col, default in {**reddit_defaults, **fg_defaults}.items():
        if col in config.SENTIMENT_COLS and col in df.columns:
            df[col] = df[col].fillna(default)

    # If fg_missing exists, ensure it is 1 when fg_index_scaled was missing originally
    # (Optional safety; harmless even if already correct.)
    if "fg_missing" in config.SENTIMENT_COLS and "fg_missing" in df.columns:
        df["fg_missing"] = df["fg_missing"].astype(float)

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

    These are base features for time t, used later to derive t+H covariates in add_future_covariates().
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
    Create known-future covariate columns by shifting base calendar/halving features.

    For the main horizon H = config.FORECAST_HORIZON, we take base features
    at time t+H and shift them back so that at index t we store information
    about the horizon endpoint (t+H) using the *_fut naming:

        day_of_week        -> day_of_week_fut
        is_weekend         -> is_weekend_fut
        month              -> month_fut
        hour_of_day         -> hour_of_day_fut
        is_halving_window  -> is_halving_window_fut
        days_to_next_halving -> days_to_next_halving_fut

    Note:
      The last H rows will have NaNs in these *_fut columns (no t+H exists).
      These rows are later removed alongside target NaNs during target creation / split-safe labeling.
    """

    df = df.copy()

    required_base_cols = [
        "day_of_week",
        "is_weekend",
        "month",
        "hour_of_day",
        "is_halving_window",
        "days_to_next_halving",
    ]

    missing = [c for c in required_base_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"add_future_covariates expected base columns {required_base_cols}, "
            f"but missing: {missing}. Make sure to call add_calendar_features() "
            f"and add_halving_features() first."
        )

    # Horizon H used to align known-future covariates with the forecast endpoint (t+H).
    horizon = int(config.FORECAST_HORIZON)
    if horizon <= 0:
        raise ValueError(f"Expected horizon > 0, got {horizon}")

    # Shift base features by -H so that at index t we store information corresponding to t+H.
    df["day_of_week_fut"] = df["day_of_week"].shift(-horizon)
    df["is_weekend_fut"] = df["is_weekend"].shift(-horizon)
    df["month_fut"] = df["month"].shift(-horizon)
    df["hour_of_day_fut"] = df["hour_of_day"].shift(-horizon)
    df["is_halving_window_fut"] = df["is_halving_window"].shift(-horizon)
    df["days_to_next_halving_fut"] = df["days_to_next_halving"].shift(-horizon)

    return df


# ============================
# 4. TARGETS
# ============================
def compute_future_return(close: pd.Series, h: int, use_log_returns: bool) -> pd.Series:
    """
    Compute the forward return from time t to time t+h.

    Args:
        close: Close price series indexed by time.
        h: Horizon step (1..H). Uses shift(-h), so the last h rows become NaN.
        use_log_returns: If True, use log returns; otherwise simple returns.

    Returns:
        A pd.Series of forward returns aligned to time t.
    """
    if use_log_returns:
        return np.log(close.shift(-h) / close)
    else:
        return close.shift(-h) / close - 1.0


def add_multi_horizon_targets(df: pd.DataFrame, H: int, use_log_returns: bool) -> pd.DataFrame:
    """
    Adds y_ret_1..y_ret_H columns where y_ret_h is the return from t -> t+h.
    """

    prefix = config.TARGET_RET_PREFIX

    close = df["close"]
    for h in range(1, H + 1):
        df[f"{prefix}{h}"] = compute_future_return(close, h, use_log_returns)

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
        - MinMaxScaler for price & volume columns (config.PRICE_VOLUME_COLS)
        - StandardScaler for all other continuous columns
        - NO scaling for binary indicator columns (config.BINARY_COLS), e.g. fg_missing

    Returns:
        scaled_train_df, scaled_val_df, scaled_test_df, scalers_dict
    """
    # 1) Split columns into groups
    price_volume_cols = [c for c in feature_cols if c in config.PRICE_VOLUME_COLS]

    binary_cols_cfg = config.BINARY_COLS
    binary_cols = [c for c in feature_cols if c in binary_cols_cfg]

    # Everything that is not price/volume and not binary gets standardized
    indicator_like_cols = [
        c for c in feature_cols
        if c not in price_volume_cols and c not in binary_cols
    ]

    # 2) Work on copies
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    # 3) Safety checks: fail early if NaN/Inf exists
    def _assert_finite(df: pd.DataFrame, cols: List[str], split_name: str, stage: str) -> None:
        if not cols:
            return

        # NaN check
        nan_mask = df[cols].isna()
        if nan_mask.any().any():
            nan_cols = nan_mask.columns[nan_mask.any()].tolist()
            raise ValueError(
                f"[scale_features] Found NaNs in {split_name} {stage} for columns: {nan_cols}\n"
                f"Common causes:\n"
                f"  - Sentiment CSV does not cover the split date range\n"
                f"  - On-chain merge left early years empty\n"
                f"  - A feature column name mismatch between config and data\n"
            )

        # Inf check (also catches non-numeric values when converting)
        arr = df[cols].to_numpy(dtype=float)
        if not np.isfinite(arr).all():
            raise ValueError(
                f"[scale_features] Found Inf or non-finite values in {split_name} {stage}.\n"
                f"Check your merged feature columns for extreme values or parsing issues."
            )

    _assert_finite(train_df, feature_cols, "TRAIN", "(before scaling)")
    _assert_finite(val_df, feature_cols, "VAL", "(before scaling)")
    _assert_finite(test_df, feature_cols, "TEST", "(before scaling)")

    # 4) Fit scalers on TRAIN only
    pv_scaler = MinMaxScaler()
    ind_scaler = StandardScaler()

    if price_volume_cols:
        pv_scaler.fit(train_df[price_volume_cols])
    if indicator_like_cols:
        ind_scaler.fit(train_df[indicator_like_cols])

    # 5) Transform all splits
    for df in (train_df, val_df, test_df):
        if price_volume_cols:
            df[price_volume_cols] = pv_scaler.transform(df[price_volume_cols])
        if indicator_like_cols:
            df[indicator_like_cols] = ind_scaler.transform(df[indicator_like_cols])
        # binary_cols remain untouched (0/1)

    # 6) Final checks after scaling
    _assert_finite(train_df, feature_cols, "TRAIN", "(after scaling)")
    _assert_finite(val_df, feature_cols, "VAL", "(after scaling)")
    _assert_finite(test_df, feature_cols, "TEST", "(after scaling)")

    scalers = {
        "price_volume_scaler": pv_scaler,
        "indicator_scaler": ind_scaler,
        "feature_cols": feature_cols,
        "price_volume_cols": price_volume_cols,
        "indicator_cols": indicator_like_cols,
        "binary_cols": binary_cols,
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
    target_cols: List[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Build sliding-window sequences for quantile multi-horizon forecasting only.

    For each sample ending at index t (the last timestep of the window):
      - X_past:  df[feature_cols] from [t-seq_length+1 .. t]   -> (seq_length, n_features)
      - y:       df[target_cols] at t                          -> (H,)
      - X_future (optional): df[future_cols] at t              -> (n_future_features,)

    Shape glossary:
      - N = number of samples (windows)
      - H = config.FORECAST_HORIZON (targets per sample)
      - n_features = len(feature_cols)
      - n_future_features = len(future_cols) if provided

    Returns:
      sequences_arr: (N, seq_length, n_features) float32
      labels_arr:    (N, H) float32
      future_arr:    (N, n_future_features) float32 OR None
    """
    if seq_length < 1:
        raise ValueError(f"seq_length must be >= 1, got {seq_length}")
    if len(df) < seq_length:
        raise ValueError(f"Not enough rows ({len(df)}) for seq_length={seq_length}")

    # Past features
    for c in feature_cols:
        if c not in df.columns:
            raise ValueError(f"Missing feature column '{c}' in df.")
    feature_array = df[feature_cols].values.astype(np.float32)

    # Targets (quantile-only)
    if not target_cols:
        raise ValueError("target_cols must be provided (quantile_forecast only).")
    for c in target_cols:
        if c not in df.columns:
            raise ValueError(f"Missing target column '{c}' in df.")
    target_array = df[target_cols].values.astype(np.float32)  # (len(df), H)

    # Future covariates (optional)
    if future_cols:
        for c in future_cols:
            if c not in df.columns:
                raise ValueError(
                    f"Future covariate column '{c}' not in df. "
                    "Make sure add_future_covariates() was called."
                )
        future_array = df[future_cols].values.astype(np.float32)
    else:
        future_array = None

    sequences: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    future_covariates: List[np.ndarray] = []

    for end_idx in range(seq_length - 1, len(df)):
        start_idx = end_idx - seq_length + 1

        seq_x = feature_array[start_idx : end_idx + 1]  # (seq_length, n_features)
        y = target_array[end_idx]                       # (H,)

        if np.isnan(y).any():
            raise ValueError(
                f"NaNs found in multi-horizon target at end_idx={end_idx}. "
                "Check split-safe labeling + tail dropping."
            )

        sequences.append(seq_x)
        labels.append(y)

        if future_array is not None:
            future_covariates.append(future_array[end_idx])

    sequences_arr = np.stack(sequences).astype(np.float32)  # (N, seq_length, n_features)
    labels_arr = np.stack(labels).astype(np.float32)        # (N, H)

    if future_array is not None:
        future_arr = np.stack(future_covariates).astype(np.float32)  # (N, n_future_features)
    else:
        future_arr = None

    return sequences_arr, labels_arr, future_arr


# ============================
# 8. PYTORCH DATASET
# ============================
class BTCTFTDataset(Dataset):
    """
    Simple PyTorch Dataset for the BTC TFT model.

    Each item represents one training sample:
      - sequences: past covariates of shape (seq_length, num_features)
      - future_covariates (optional): known-future covariates of shape (num_future_features,)
      - labels: multi-horizon return targets of shape (H,) where H = config.FORECAST_HORIZON

    __getitem__ returns either (sequences, labels) or (sequences, future_covariates, labels)
    depending on whether future covariates are enabled in the pipeline.
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

        # Quantile-only: (N, H) float targets for pinball loss
        self.labels = torch.from_numpy(labels).float()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        if self.future_covariates is None:
            return self.sequences[idx], self.labels[idx]
        else:
            return self.sequences[idx], self.future_covariates[idx], self.labels[idx]


# Compute targets *inside a split*
def label_split_safely(df_split: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    Quantile-forecast only:
    Compute multi-horizon return targets *inside a split* so targets never cross split boundaries.
    """
    if not isinstance(df_split.index, pd.DatetimeIndex):
        raise TypeError(f"[{split_name}] Expected DatetimeIndex.")
    if len(df_split) == 0:
        raise ValueError(f"[{split_name}] Split is empty.")
    if not df_split.index.is_monotonic_increasing:
        df_split = df_split.sort_index()

    end_ts_before = df_split.index[-1]

    H = int(config.FORECAST_HORIZON)

    df_labeled = add_multi_horizon_targets(
        df_split.copy(),
        H=H,
        use_log_returns=config.USE_LOG_RETURNS,
    )
    df_labeled = df_labeled.dropna(subset=config.TARGET_RET_COLS)

    if len(df_labeled) == 0:
        raise ValueError(
            f"[{split_name}] No labeled rows after target creation (split too small for H={H}?)."
        )

    # ---- Integrity debug (kept) ----
    if config.DEBUG_DATA_INTEGRITY:
        if len(df_split) > H:
            last_allowed_ts = df_split.index[-H - 1]
            if df_labeled.index[-1] > last_allowed_ts:
                raise AssertionError(
                    f"[{split_name}] Leakage risk: last labeled ts={df_labeled.index[-1]} "
                    f"> last_allowed_ts={last_allowed_ts} (split_end={end_ts_before}, H={H})"
                )
    return df_labeled


def _debug_check_daily_lag_alignment(
    split_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    cols: List[str],
    lag_days: int,
    split_name: str,
    n_samples: int = 5,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """
    Debug-only integrity check:
    For random hourly timestamps t in split_df, verify that daily feature columns
    match daily_df at day = floor(t) - lag_days (not floor(t)).
    """
    if len(split_df) == 0:
        return

    if not isinstance(split_df.index, pd.DatetimeIndex):
        raise TypeError(f"[data_pipeline][DEBUG] {split_name}: split_df must have DatetimeIndex.")
    if not isinstance(daily_df.index, pd.DatetimeIndex):
        raise TypeError(f"[data_pipeline][DEBUG] {split_name}: daily_df must have DatetimeIndex.")
    if lag_days < 0:
        raise ValueError("[data_pipeline][DEBUG] lag_days must be >= 0.")

    # Only check columns that exist on BOTH sides
    cols_to_check = [c for c in cols if c in split_df.columns and c in daily_df.columns]
    if not cols_to_check:
        return

    rng = np.random.default_rng(42)
    idx = split_df.index
    k = min(int(n_samples), len(idx))

    # pick random timestamps from the split index
    sampled = rng.choice(idx.values, size=k, replace=False)
    sampled = [pd.Timestamp(t) for t in sampled]
    sampled.sort()

    for ts in sampled:
        day_lagged = ts.floor("D") - pd.Timedelta(days=int(lag_days))

        if day_lagged not in daily_df.index:
            continue

        for col in cols_to_check:
            expected = daily_df.at[day_lagged, col]
            observed = split_df.at[ts, col]

            # If source daily value is NaN (rare), skip that comparison
            if pd.isna(expected):
                continue

            # numeric compare (most of your features are numeric)
            try:
                if not np.isclose(float(observed), float(expected), rtol=rtol, atol=atol):
                    raise AssertionError(
                        f"[data_pipeline][DEBUG] Daily-lag mismatch in {split_name}\n"
                        f"  ts={ts}  day_lagged={day_lagged.date()}  col={col}\n"
                        f"  observed(split)={observed}  expected(daily)={expected}\n"
                        f"  (This suggests the merge used t instead of t-lag_days, or daily preprocessing differs.)"
                    )
            except (TypeError, ValueError):
                # fallback for non-numeric (shouldn't happen, but keep it safe)
                if observed != expected:
                    raise AssertionError(
                        f"[data_pipeline][DEBUG] Daily-lag mismatch (non-numeric) in {split_name}\n"
                        f"  ts={ts}  day_lagged={day_lagged.date()}  col={col}\n"
                        f"  observed(split)={observed}  expected(daily)={expected}"
                    )


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
    Run the full data pipeline for the current configuration and return train/val/test datasets.

    Input frequency:
      - If csv_path is None: uses config.FREQUENCY to choose BTC hourly vs daily CSV.
      - If csv_path is provided: heuristically chooses hourly vs daily based on the path name
        ("hour" or "1h" -> hourly; otherwise daily).

    Pipeline outline:
      1) Load BTC OHLCV data from CSV.
      2) Add technical indicators (ROC, ATR, MACD, RSI).
      3) Optionally merge external DAILY features (on-chain, sentiment) into the time-indexed BTC frame.
         - Merge uses a lagged daily key controlled by config.DAILY_FEATURE_LAG_DAYS.
      4) Add calendar and halving-related base features.
      5) Add known-future covariate columns via shift to t+H (H = config.FORECAST_HORIZON).
      6) Split by date into train / val / test using config date ranges.
      7) Create multi-horizon return targets y_ret_1..y_ret_H *inside each split* (no boundary leakage).
      8) Scale past covariates (fit on train only).
      9) Build sliding-window sequences + per-sample future covariate vectors.
     10) Wrap into BTCTFTDataset objects.

    Optional extra return:
      - If return_forward_returns=True, also returns per-sample forward returns aligned with the built sequences
        at config.SIGNAL_HORIZON (i.e., y_ret_{SIGNAL_HORIZON}).
    """
    # 1. Load
    if csv_path is not None:
        lower = csv_path.lower()
        if "hour" in lower or "1h" in lower:
            df = load_btc_hourly(csv_path)
        else:
            df = load_btc_daily(csv_path)
    else:
        freq = config.FREQUENCY
        if freq.lower() in ("1h", "h", "hourly"):
            df = load_btc_hourly()
        else:
            df = load_btc_daily()

    # 2. Add indicators
    df = add_technical_indicators(df)

    # 2.5 Add on-chain features
    if config.USE_ONCHAIN:
        df = add_onchain_features(df)

    # 2.6 Add sentiment features
    if config.USE_SENTIMENT:
        df = add_sentiment_features(df)

    # 3. Add calendar and halving features
    df = add_calendar_features(df)
    df = add_halving_features(df)

    # 4. Add future covariate columns (shift to t+H, where H = FORECAST_HORIZON)
    df = add_future_covariates(df)

    # 5. Split FIRST
    train_df, val_df, test_df = split_by_date(df)

    # 6. Create targets INSIDE each split (no split-boundary contamination)
    train_df = label_split_safely(train_df, "TRAIN")
    val_df = label_split_safely(val_df, "VAL")
    test_df = label_split_safely(test_df, "TEST")

    # ----------------------------
    # Debug-only integrity checks
    # ----------------------------
    # Verifies that daily features were merged using the intended lag rule:
    # daily_key = floor(t) - DAILY_FEATURE_LAG_DAYS
    if config.DEBUG_DATA_INTEGRITY:
        lag_days = int(config.DAILY_FEATURE_LAG_DAYS)

        # --- On-chain daily-lag check ---
        if config.USE_ONCHAIN:
            # Rebuild the SAME daily on-chain feature frame used for merge (includes derived cols)
            onchain_daily = load_onchain_daily()
            onchain_daily["aa_ma_ratio"] = onchain_daily["active_addresses"] / (
                onchain_daily["active_addresses"].rolling(window=7, min_periods=1).mean()
            )
            onchain_daily["tx_ma_ratio"] = onchain_daily["tx_count"] / (
                onchain_daily["tx_count"].rolling(window=7, min_periods=1).mean()
            )
            onchain_daily["hash_ma_ratio"] = onchain_daily["hash_rate"] / (
                onchain_daily["hash_rate"].rolling(window=30, min_periods=1).mean()
            )
            onchain_daily["mvrv_z"] = onchain_daily["mvrv"]
            onchain_daily["sopr_z"] = onchain_daily["sopr"]
            onchain_daily = onchain_daily[config.ONCHAIN_COLS]

            _debug_check_daily_lag_alignment(train_df, onchain_daily, config.ONCHAIN_COLS, lag_days, "TRAIN/onchain")
            _debug_check_daily_lag_alignment(val_df, onchain_daily, config.ONCHAIN_COLS, lag_days, "VAL/onchain")
            _debug_check_daily_lag_alignment(test_df, onchain_daily, config.ONCHAIN_COLS, lag_days, "TEST/onchain")

        # --- Sentiment daily-lag check ---
        if config.USE_SENTIMENT:
            sentiment_daily = load_sentiment_daily()
            _debug_check_daily_lag_alignment(train_df, sentiment_daily, config.SENTIMENT_COLS, lag_days,
                                             "TRAIN/sentiment")
            _debug_check_daily_lag_alignment(val_df, sentiment_daily, config.SENTIMENT_COLS, lag_days, "VAL/sentiment")
            _debug_check_daily_lag_alignment(test_df, sentiment_daily, config.SENTIMENT_COLS, lag_days,
                                             "TEST/sentiment")

    # Which columns we feed into the model as past covariates
    feature_cols = config.FEATURE_COLS

    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Feature column '{col}' is missing. Check indicator/on-chain generation.")

    # Future covariate columns
    future_cols = config.FUTURE_COVARIATE_COLS
    for col in future_cols:
        if col not in df.columns:
            raise ValueError(f"Future covariate column '{col}' is missing. Check add_future_covariates().")

    # 9. Scale features (for past covariates)
    train_df_scaled, val_df_scaled, test_df_scaled, scalers = scale_features(
        train_df, val_df, test_df, feature_cols
    )

    # Extract signal-horizon forward returns aligned with built samples (optional).
    def extract_forward_returns(df_scaled: pd.DataFrame, seq_len: int) -> np.ndarray:
        """
        Returns aligned "trade horizon" forward returns for each built sample.
        Uses y_ret_{config.SIGNAL_HORIZON} (must be <= config.FORECAST_HORIZON).
        """
        H_main = int(config.FORECAST_HORIZON)
        H_trade = int(config.SIGNAL_HORIZON)  # usually equals H_main

        if H_trade > H_main:
            raise ValueError(
                f"SIGNAL_HORIZON ({H_trade}) cannot exceed FORECAST_HORIZON ({H_main}). "
                f"Targets only exist up to y_ret_{H_main}."
            )

        col_name = f"{config.TARGET_RET_PREFIX}{H_trade}"

        if col_name not in df_scaled.columns:
            raise ValueError(
                f"Column '{col_name}' not found for forward returns extraction "
                f"(col='{col_name}', H_main={H_main}). "
                "Make sure targets were created inside each split."
            )

        returns_full = df_scaled[col_name].astype(float).values
        if len(returns_full) < seq_len:
            raise ValueError(
                f"Not enough rows ({len(returns_full)}) to build sequences of length {seq_len}."
            )
        return returns_full[seq_len - 1:]

    train_forward = extract_forward_returns(train_df_scaled, seq_length)
    val_forward   = extract_forward_returns(val_df_scaled, seq_length)
    test_forward  = extract_forward_returns(test_df_scaled, seq_length)

    # 10. Build sequences + future covariate vectors (quantile-only)
    target_cols = config.TARGET_RET_COLS

    train_seq, train_labels, train_future = build_sequences(
        train_df_scaled,
        feature_cols,
        seq_length,
        future_cols=future_cols,
        target_cols=target_cols,
    )
    val_seq, val_labels, val_future = build_sequences(
        val_df_scaled,
        feature_cols,
        seq_length,
        future_cols=future_cols,
        target_cols=target_cols,
    )
    test_seq, test_labels, test_future = build_sequences(
        test_df_scaled,
        feature_cols,
        seq_length,
        future_cols=future_cols,
        target_cols=target_cols,
    )

    H = int(config.FORECAST_HORIZON)
    assert train_labels.ndim == 2 and train_labels.shape[1] == H
    assert val_labels.ndim == 2 and val_labels.shape[1] == H
    assert test_labels.ndim == 2 and test_labels.shape[1] == H
    assert not np.isnan(train_labels).any()
    assert train_labels.dtype in (np.float32, np.float64)

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