"""
Turns raw BTC price data (daily or hourly) into PyTorch Datasets for the TFT model.

Main steps:
1. Load BTC price data from CSV (daily or hourly, depending on config).
2. Add technical indicators (ROC, ATR, MACD, RSI).
3. Add calendar and halving-related features.
4. Create future covariate columns via shift to t+H (aligned with the
   main forecast horizon FORECAST_HORIZONS[0]).
5. Split into train / validation / test sets by date.
6. Create forward return(s) and a 3-class direction label inside each split (split-safe; no boundary contamination).
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
from final_model import config
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
    # IMPORTANT: lag DAILY features so we don't use same-day values intraday.
    lag_days = int(getattr(config, "DAILY_FEATURE_LAG_DAYS", 1))
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
    Load the combined daily sentiment dataset (Reddit Pushshift + Fear & Greed)
    and return a DataFrame indexed by daily dates.

    Expected columns (besides 'date') are config.SENTIMENT_COLS, e.g.:
        - reddit_sent_mean, reddit_sent_std, reddit_pos_ratio, reddit_neg_ratio,
          reddit_volume, reddit_volume_log,
        - fg_index_scaled, fg_change_1d, fg_missing

    The file should be on a fully contiguous daily date grid and should not
    contain NaNs inside the intended experiment window.
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
    Merge DAILY sentiment features into an HOURLY BTC dataframe.

    Strategy:
      1) Load daily sentiment features (index = daily dates).
      2) Create a daily merge key for hourly timestamps: df.index.floor('D').
      3) Left-join sentiment onto each hour.
      4) Fill any remaining missing values with safe constants to prevent NaNs.

    Note: Filling is done with neutral constants (no look-ahead), so it is safe.
    """
    if not isinstance(hourly_df.index, pd.DatetimeIndex):
        raise TypeError("add_sentiment_features expects hourly_df with a DatetimeIndex.")

    df = hourly_df.copy()

    # Avoid ambiguity between index name and 'date' column (same pattern as on-chain)
    if df.index.name == "date":
        df.index.name = "timestamp"

    sentiment_df = load_sentiment_daily()  # daily index

    # Merge daily → hourly using a LAGGED daily key (prevents same-day leakage intraday)
    lag_days = int(getattr(config, "DAILY_FEATURE_LAG_DAYS", 1))
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
# 2B. STATIC REGIME FEATURES (Experiment 9d)
# ============================

def add_static_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add leakage-safe "static regime" features at each timestamp t.

    Important:
    - These are computed per timestamp, but later you will take them ONCE per sample
      (e.g., at the window end_idx) as x_static.
    - This function MUST NOT use shift(-1) or any future-looking operation.
    - Rolling windows are causal by default (center=False), so they are safe.

    Features (as specified in the Experiment 9d guide):
      - reg_vol_24         : rolling std of log returns over 24 periods
      - reg_vol_168        : rolling std of log returns over 168 periods
      - reg_trend_168      : close / rolling_mean_168(close) - 1
      - reg_drawdown_168   : close / rolling_max_168(close) - 1
      - reg_volume_z_168   : z-score of volume_usd over 168 periods
      - reg_vol_ratio      : reg_vol_24 / (reg_vol_168 + eps)

    NaNs:
    - Rolling windows will create NaNs at the beginning (warmup).
    - Do NOT drop them here. The guide’s recommended approach is to drop warmup once
      later via df.dropna(subset=config.STATIC_COLS).
    """
    df = df.copy()

    # Basic column requirements
    required_cols = {"close", "volume_usd"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(
            f"add_static_regime_features() missing required columns: {sorted(missing)}. "
            f"Available columns: {sorted(df.columns)}"
        )

    close = df["close"].astype(float)
    volume_usd = df["volume_usd"].astype(float)

    eps = 1e-9

    # Log returns: uses t and t-1 only (no future leakage)
    ret = np.log(close / close.shift(1))

    # Rolling volatilities
    df["reg_vol_24"] = ret.rolling(window=24, min_periods=24).std()
    df["reg_vol_168"] = ret.rolling(window=168, min_periods=168).std()

    # Trend vs rolling mean (168)
    ma_168 = close.rolling(window=168, min_periods=168).mean()
    df["reg_trend_168"] = close / ma_168 - 1.0

    # Drawdown vs rolling max (168)
    max_168 = close.rolling(window=168, min_periods=168).max()
    df["reg_drawdown_168"] = close / max_168 - 1.0

    # Volume regime z-score (168)
    vol_mean_168 = volume_usd.rolling(window=168, min_periods=168).mean()
    vol_std_168 = volume_usd.rolling(window=168, min_periods=168).std()
    df["reg_volume_z_168"] = (volume_usd - vol_mean_168) / (vol_std_168 + eps)

    # Vol ratio
    df["reg_vol_ratio"] = df["reg_vol_24"] / (df["reg_vol_168"] + eps)

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
        "days_to_next_halving",
    ]

    missing = [c for c in required_base_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"add_future_covariates expected base columns {required_base_cols}, "
            f"but missing: {missing}. Make sure to call add_calendar_features() "
            f"and add_halving_features() first."
        )

    # Horizon H used for the direction label / future return.
    horizon = int(config.FORECAST_HORIZONS[0])
    if horizon <= 0:
        raise ValueError(f"Expected horizon > 0, got {horizon}")

    # Shift base features by -H so that at index t we store information
    # corresponding to t+H, using the Experiment 9a naming (*_fut).
    df["day_of_week_fut"] = df["day_of_week"].shift(-horizon)
    df["is_weekend_fut"] = df["is_weekend"].shift(-horizon)
    df["month_fut"] = df["month"].shift(-horizon)
    df["hour_of_day_fut"] = df["hour_of_day"].shift(-horizon)
    df["is_halving_window_fut"] = df["is_halving_window"].shift(-horizon)
    df["days_to_next_halving_fut"] = df["days_to_next_halving"].shift(-horizon)

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

def compute_future_return(close: pd.Series, h: int, use_log_returns: bool) -> pd.Series:
    if use_log_returns:
        return np.log(close.shift(-h) / close)
    else:
        return close.shift(-h) / close - 1.0


def add_multi_horizon_targets(df: pd.DataFrame, H: int, use_log_returns: bool) -> pd.DataFrame:
    """
    Adds y_ret_1..y_ret_H columns where y_ret_h is the return from t -> t+h.
    """

    prefix = getattr(config, "TARGET_RET_PREFIX", "y_ret_")

    close = df["close"]
    for h in range(1, H + 1):
        df[f"{prefix}{h}"] = compute_future_return(close, h, use_log_returns)

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
        - MinMaxScaler for price & volume columns (config.PRICE_VOLUME_COLS)
        - StandardScaler for all other continuous columns
        - NO scaling for binary indicator columns (config.BINARY_COLS), e.g. fg_missing

    Returns:
        scaled_train_df, scaled_val_df, scaled_test_df, scalers_dict
    """
    # 1) Split columns into groups
    price_volume_cols = [c for c in feature_cols if c in config.PRICE_VOLUME_COLS]

    binary_cols_cfg = getattr(config, "BINARY_COLS", [])
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

def scale_static_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    static_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler | None]:
    """
    Standardize static regime features using StandardScaler.

    Leakage-safe rule:
      - Fit scaler on TRAIN only
      - Transform TRAIN / VAL / TEST with same scaler

    Returns:
      (train_df_scaled, val_df_scaled, test_df_scaled, static_scaler)
      If static_cols is empty, returns dfs unchanged and static_scaler=None.
    """
    static_cols = list(static_cols) if static_cols is not None else []
    if len(static_cols) == 0:
        return train_df, val_df, test_df, None

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    # Validate columns exist
    missing = [c for c in static_cols if c not in train_df.columns]
    if missing:
        raise ValueError(
            f"[scale_static_features] Missing static columns in df: {missing}. "
            "Make sure add_static_regime_features() ran and you did df.dropna(subset=config.STATIC_COLS) "
            "before splitting."
        )

    # Safety checks: NaN / Inf
    def _assert_finite(df: pd.DataFrame, cols: List[str], split_name: str, stage: str) -> None:
        if not cols:
            return

        nan_mask = df[cols].isna()
        if nan_mask.any().any():
            nan_cols = nan_mask.columns[nan_mask.any()].tolist()
            raise ValueError(
                f"[scale_static_features] NaNs in {split_name} {stage} for columns: {nan_cols}. "
                "Likely missing warmup dropna(subset=STATIC_COLS) or static features not computed for early rows."
            )

        arr = df[cols].to_numpy(dtype=float)
        if not np.isfinite(arr).all():
            raise ValueError(
                f"[scale_static_features] Inf/non-finite values in {split_name} {stage}. "
                "Check static feature generation for division-by-zero (std=0) or extreme values."
            )

    _assert_finite(train_df, static_cols, "TRAIN", "(before scaling)")
    _assert_finite(val_df, static_cols, "VAL", "(before scaling)")
    _assert_finite(test_df, static_cols, "TEST", "(before scaling)")

    static_scaler = StandardScaler()
    static_scaler.fit(train_df[static_cols])  # <-- TRAIN ONLY

    train_df.loc[:, static_cols] = static_scaler.transform(train_df[static_cols])
    val_df.loc[:, static_cols] = static_scaler.transform(val_df[static_cols])
    test_df.loc[:, static_cols] = static_scaler.transform(test_df[static_cols])

    _assert_finite(train_df, static_cols, "TRAIN", "(after scaling)")
    _assert_finite(val_df, static_cols, "VAL", "(after scaling)")
    _assert_finite(test_df, static_cols, "TEST", "(after scaling)")

    return train_df, val_df, test_df, static_scaler

# ============================
# 7. SEQUENCE CREATION
# ============================

def build_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_length: int = config.SEQ_LENGTH,
    future_cols: List[str] | None = None,
    label_col: str | None = config.TARGET_COLUMN,
        target_cols: List[str] | None = None,  # multi-horizon targets for quantile_forecast
        static_cols: List[str] | None = None,  # static/regime covariates
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Build sliding-window sequences and labels.

    Returns (in order):
      - sequences_arr: (N, seq_length, n_features)
      - labels_arr:    (N,) for classification OR (N, H) for quantile forecast
      - future_covariates_arr: (N, n_future_features) or None
      - static_covariates_arr: (N, n_static_features) or None  <-- NEW
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

    # Targets
    task_type = getattr(config, "TASK_TYPE", "classification")

    if task_type == "quantile_forecast":
        if target_cols is None:
            raise ValueError(
                "target_cols must be provided for quantile_forecast "
                "(e.g., config.TARGET_RET_COLS = ['y_ret_1', ... 'y_ret_24'])."
            )
        missing_tgt = [c for c in target_cols if c not in df.columns]
        if missing_tgt:
            raise ValueError(
                f"Missing target columns in df: {missing_tgt}. "
                "Make sure multi-horizon targets were created before build_sequences()."
            )
        target_array = df[target_cols].values.astype(np.float32)  # (len(df), H)
    else:
        if label_col is None:
            raise ValueError("label_col must be provided for classification.")
        if label_col not in df.columns:
            raise ValueError(
                f"DataFrame must contain '{label_col}' column before building sequences."
            )
        target_array = df[label_col].values.astype(np.int64)  # (len(df),)

    # Future covariates
    if future_cols is not None:
        for col in future_cols:
            if col not in df.columns:
                raise ValueError(
                    f"Future covariate column '{col}' is missing. "
                    "Make sure add_future_covariates() was called."
                )
        future_array = df[future_cols].values.astype(np.float32)
    else:
        future_array = None

    # Static covariates (regime features)
    if static_cols is not None and len(static_cols) > 0:
        for col in static_cols:
            if col not in df.columns:
                raise ValueError(
                    f"Static covariate column '{col}' is missing. "
                    "Make sure add_static_regime_features() was called and df.dropna(subset=STATIC_COLS) ran."
                )
        static_array = df[static_cols].values.astype(np.float32)
    else:
        static_array = None

    sequences: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    future_covariates: List[np.ndarray] = []
    static_covariates: List[np.ndarray] = []

    for end_idx in range(seq_length - 1, len(df)):
        start_idx = end_idx - seq_length + 1

        seq_x = feature_array[start_idx: end_idx + 1]  # (seq_length, n_features)

        if task_type == "quantile_forecast":
            y = target_array[end_idx]  # (H,)
            if np.isnan(y).any():
                raise ValueError(
                    f"NaNs found in multi-horizon target at end_idx={end_idx}. "
                    "Check tail-dropping (DROP_LAST_H_IN_EACH_SPLIT) and target creation."
                )
        else:
            y = target_array[end_idx]  # scalar

        sequences.append(seq_x)
        labels.append(y)

        if future_array is not None:
            future_covariates.append(future_array[end_idx])

        if static_array is not None:
            static_covariates.append(static_array[end_idx])

    sequences_arr = np.stack(sequences)

    if task_type == "quantile_forecast":
        labels_arr = np.stack(labels).astype(np.float32)
    else:
        labels_arr = np.array(labels, dtype=np.int64)

    future_covariates_arr = np.stack(future_covariates) if future_array is not None else None
    static_covariates_arr = np.stack(static_covariates) if static_array is not None else None

    return sequences_arr, labels_arr, future_covariates_arr, static_covariates_arr

# ============================
# 8. PYTORCH DATASET
# ============================

class BTCTFTDataset(Dataset):
    """
    Simple PyTorch Dataset for the BTC TFT model.

    Each item represents one sample window ending at time t:
      - x_past:   past covariate sequence, shape (seq_length, n_past_features)
      - x_future: (optional) known-ahead features aligned at time t, shape (n_future_features,)
      - x_static: (optional) static/regime/context features aligned at time t, shape (n_static_features,)
      - y:
          * TASK_TYPE=="classification": scalar class index in {0,1,2}
          * TASK_TYPE=="quantile_forecast": multi-horizon forward returns, shape (H,)
    """

    def __init__(
            self,
            sequences: np.ndarray,
            labels: np.ndarray,
            future_covariates: np.ndarray | None = None,
            static_covariates: np.ndarray | None = None,
    ):
        assert len(sequences) == len(labels), "Sequences and labels must have same length."

        # --- future covariates (known-ahead features) ---
        if future_covariates is not None:
            assert len(future_covariates) == len(labels), (
                "Future covariates and labels must have same length."
            )
            self.future_covariates = torch.from_numpy(future_covariates).float()
        else:
            self.future_covariates = None

        # --- static covariates (regime/context features) ---
        if static_covariates is not None:
            assert len(static_covariates) == len(labels), (
                "Static covariates and labels must have same length."
            )
            self.static_covariates = torch.from_numpy(static_covariates).float()
        else:
            self.static_covariates = None

        # --- past sequence (observed inputs) ---
        self.sequences = torch.from_numpy(sequences).float()

        # --- labels dtype depends on task ---
        task_type = getattr(config, "TASK_TYPE", "classification")

        if task_type == "quantile_forecast":
            # (N, H) float targets for pinball loss
            self.labels = torch.from_numpy(labels).float()
        else:
            # (N,) integer class indices for CrossEntropyLoss
            self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x_past = self.sequences[idx]
        y = self.labels[idx]

        # 1) no future, no static
        if self.future_covariates is None and self.static_covariates is None:
            return x_past, y

        # 2) future only (legacy behavior)
        if self.future_covariates is not None and self.static_covariates is None:
            return x_past, self.future_covariates[idx], y

        # 3) static only
        if self.future_covariates is None and self.static_covariates is not None:
            return x_past, self.static_covariates[idx], y

        # 4) both future + static (new behavior for 9d)
        return x_past, self.future_covariates[idx], self.static_covariates[idx], y

# Compute targets *inside a split*
def label_split_safely(df_split: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    Compute targets *inside a split* so future returns never cross split boundaries.
    Works for both classification and quantile forecasting.
    """
    if not isinstance(df_split.index, pd.DatetimeIndex):
        raise TypeError(f"[{split_name}] Expected DatetimeIndex.")
    if len(df_split) == 0:
        raise ValueError(f"[{split_name}] Split is empty.")

    if not df_split.index.is_monotonic_increasing:
        df_split = df_split.sort_index()

    end_ts_before = df_split.index[-1]

    horizons = list(getattr(config, "FORECAST_HORIZONS", [1])) or [1]
    max_h = max(int(h) for h in horizons)

    # ---- multi-horizon quantile returns ----
    if getattr(config, "TASK_TYPE", "") == "quantile_forecast":
        df_labeled = add_multi_horizon_targets(df_split.copy(), H=max_h, use_log_returns=config.USE_LOG_RETURNS)

        prefix = getattr(config, "TARGET_RET_PREFIX", "y_ret_")
        target_cols = [f"{prefix}{h}" for h in range(1, max_h + 1)]
        df_labeled = df_labeled.dropna(subset=target_cols)

    # ---- legacy path: single-horizon classification ----
    else:
        df_labeled = add_target_column(df_split.copy())  # existing behavior

    if len(df_labeled) == 0:
        raise ValueError(f"[{split_name}] No labeled rows after target creation (split too small for H={max_h}?).")

    # ---- Integrity debug ----
    if getattr(config, "DEBUG_DATA_INTEGRITY", False):
        if len(df_split) > max_h:
            last_allowed_ts = df_split.index[-max_h - 1]
            if df_labeled.index[-1] > last_allowed_ts:
                raise AssertionError(
                    f"[{split_name}] Leakage risk: last labeled ts={df_labeled.index[-1]} "
                    f"> last_allowed_ts={last_allowed_ts} (split_end={end_ts_before}, max_h={max_h})"
                )

        print(
            f"[data_pipeline] [{split_name}] split_end={end_ts_before} -> "
            f"labeled_end={df_labeled.index[-1]} (rows {len(df_split)} -> {len(df_labeled)})"
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
        print(f"[data_pipeline][DEBUG] {split_name}: empty split -> skip daily-lag check.")
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
        print(f"[data_pipeline][DEBUG] {split_name}: no overlapping cols to check -> skip.")
        return

    rng = np.random.default_rng(42)
    idx = split_df.index
    k = min(int(n_samples), len(idx))

    # pick random timestamps from the split index
    sampled = rng.choice(idx.values, size=k, replace=False)
    sampled = [pd.Timestamp(t) for t in sampled]
    sampled.sort()

    checked = 0
    skipped = 0

    for ts in sampled:
        day_lagged = ts.floor("D") - pd.Timedelta(days=int(lag_days))

        if day_lagged not in daily_df.index:
            skipped += 1
            continue

        for col in cols_to_check:
            expected = daily_df.at[day_lagged, col]
            observed = split_df.at[ts, col]

            # If source daily value is NaN (rare), skip that comparison
            if pd.isna(expected):
                skipped += 1
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

            checked += 1

    # Print one “acceptance check” style example (hourly ts, lagged date, joined date)
    example_ts = sampled[0]
    example_day_lagged = example_ts.floor("D") - pd.Timedelta(days=int(lag_days))
    example_col = cols_to_check[0]
    if example_day_lagged in daily_df.index:
        print(
            f"[data_pipeline][DEBUG] Daily lag example ({split_name}): "
            f"ts={example_ts} -> day_lagged={example_day_lagged.date()} "
            f"(col='{example_col}', split={split_df.at[example_ts, example_col]}, daily={daily_df.at[example_day_lagged, example_col]})"
        )

    print(
        f"[data_pipeline][DEBUG] Daily-lag check PASSED for {split_name} "
        f"(checked={checked}, skipped={skipped}, lag_days={lag_days})."
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
      5. Split by date into train / val / test using config date ranges.
      6. Add forward returns + direction label inside each split (shift(-H) happens only within that split).
      7. Print label distribution (FULL = concatenation of labeled splits) + each split.
      8. Scale past covariates (fit on train only).
      9. Build sliding-window sequences and future covariate vectors.
     10. Wrap them into BTCTFTDataset objects.
     11. Optionally return aligned H-step forward returns for each sample.
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

    # 2.6 Add sentiment features (Experiment 8)
    if getattr(config, "USE_SENTIMENT", False):
        df = add_sentiment_features(df)

    # 3. Add calendar and halving features
    df = add_calendar_features(df)
    df = add_halving_features(df)

    # 4. Add future covariate columns (shift to t+H, where H = FORECAST_HORIZONS[0])
    df = add_future_covariates(df)

    # 4.5 Add static regime features (Experiment 9d) + drop warmup once
    if getattr(config, "STATIC_COLS", None):
        df = add_static_regime_features(df)

        # Drop the initial warmup rows created by rolling windows (e.g., 168h)
        before = len(df)
        df = df.dropna(subset=config.STATIC_COLS)
        after = len(df)
        if getattr(config, "DEBUG_DATA_INTEGRITY", False):
            print(f"[data_pipeline][DEBUG] Dropped {before - after} warmup rows due to STATIC_COLS.")

    # 5. Split FIRST (Experiment 9a Part A)
    train_df, val_df, test_df = split_by_date(df)

    # 6. Create targets INSIDE each split (no split-boundary contamination)
    train_df = label_split_safely(train_df, "TRAIN")
    val_df = label_split_safely(val_df, "VAL")
    test_df = label_split_safely(test_df, "TEST")

    # 6.1 Debug-only: verify daily feature availability lagging (Experiment 9a)
    if getattr(config, "DEBUG_DATA_INTEGRITY", False):
        lag_days = int(getattr(config, "DAILY_FEATURE_LAG_DAYS", 1))

        # --- On-chain daily-lag check ---
        if getattr(config, "USE_ONCHAIN", False):
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
        if getattr(config, "USE_SENTIMENT", False):
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

    task_type = getattr(config, "TASK_TYPE", "classification")

    if task_type != "quantile_forecast":
        full_df = pd.concat([train_df, val_df, test_df]).sort_index()
        print_label_distribution(full_df, name="FULL", freq="Y")
        print_label_distribution(train_df, name="TRAIN", freq="Y")
        print_label_distribution(val_df, name="VAL", freq="Y")
        print_label_distribution(test_df, name="TEST", freq="Y")

    # 9. Scale features (for past covariates)
    train_df_scaled, val_df_scaled, test_df_scaled, scalers = scale_features(
        train_df, val_df, test_df, feature_cols
    )

    # Scale static regime features (train-fit only)
    train_df_scaled, val_df_scaled, test_df_scaled, static_scaler = scale_static_features(
        train_df_scaled, val_df_scaled, test_df_scaled, getattr(config, "STATIC_COLS", [])
    )
    scalers["static_scaler"] = static_scaler
    scalers["static_cols"] = list(getattr(config, "STATIC_COLS", []))

    # Extract H-step forward returns aligned with samples (optional).
    def extract_forward_returns(df_scaled: pd.DataFrame, seq_len: int) -> np.ndarray:
        """
        Returns aligned "trade horizon" forward returns for each built sample.
        - classification: uses future_return_{H}d
        - quantile_forecast: uses y_ret_{H}
        """
        horizons_cfg = list(getattr(config, "FORECAST_HORIZONS", [])) or [1]
        H = int(horizons_cfg[0])

        task_type = getattr(config, "TASK_TYPE", "classification")

        if task_type == "quantile_forecast":
            prefix = getattr(config, "TARGET_RET_PREFIX", "y_ret_")
            col_name = f"{prefix}{H}"
        else:
            col_name = "future_return_1d" if H == 1 else f"future_return_{H}d"

        if col_name not in df_scaled.columns:
            raise ValueError(
                f"Column '{col_name}' not found for forward returns extraction. "
                f"Task={task_type}, H={H}. "
                f"Make sure targets were created inside each split."
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

    # 10. Build sequences + future covariate vectors
    task_type = getattr(config, "TASK_TYPE", "classification")

    if task_type == "quantile_forecast":
        target_cols = list(getattr(config, "TARGET_RET_COLS", []))
        if not target_cols:
            # fallback: build from horizon if you haven't defined TARGET_RET_COLS in config yet
            H = int(getattr(config, "FORECAST_HORIZONS", [24])[0])
            prefix = getattr(config, "TARGET_RET_PREFIX", "y_ret_")
            target_cols = [f"{prefix}{h}" for h in range(1, H + 1)]

        train_seq, train_labels, train_future, train_static = build_sequences(
            train_df_scaled,
            feature_cols,
            seq_length,
            future_cols=future_cols,
            label_col=None,
            target_cols=target_cols,
            static_cols=getattr(config, "STATIC_COLS", []),
        )

        val_seq, val_labels, val_future, val_static = build_sequences(
            val_df_scaled,
            feature_cols,
            seq_length,
            future_cols=future_cols,
            label_col=None,
            target_cols=target_cols,
            static_cols=getattr(config, "STATIC_COLS", []),
        )
        test_seq, test_labels, test_future, test_static = build_sequences(
            test_df_scaled,
            feature_cols,
            seq_length,
            future_cols=future_cols,
            label_col=None,
            target_cols=target_cols,
            static_cols=getattr(config, "STATIC_COLS", []),
        )

    else:
        label_col = config.TARGET_COLUMN

        train_seq, train_labels, train_future, train_static = build_sequences(
            train_df_scaled,
            feature_cols,
            seq_length,
            future_cols=future_cols,
            label_col=label_col,
            static_cols=getattr(config, "STATIC_COLS", []),
        )
        val_seq, val_labels, val_future, val_static = build_sequences(
            val_df_scaled,
            feature_cols,
            seq_length,
            future_cols=future_cols,
            label_col=label_col,
            static_cols=getattr(config, "STATIC_COLS", []),
        )
        test_seq, test_labels, test_future, test_static = build_sequences(
            test_df_scaled,
            feature_cols,
            seq_length,
            future_cols=future_cols,
            label_col=label_col,
            static_cols=getattr(config, "STATIC_COLS", []),
        )

    if task_type == "quantile_forecast":
        H = int(getattr(config, "FORECAST_HORIZON", 24))
        assert train_labels.ndim == 2 and train_labels.shape[1] == H
        assert val_labels.ndim == 2 and val_labels.shape[1] == H
        assert test_labels.ndim == 2 and test_labels.shape[1] == H
        assert not np.isnan(train_labels).any()
        assert train_labels.dtype == np.float32 or train_labels.dtype == np.float64

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
    train_dataset = BTCTFTDataset(
        sequences=train_seq,
        labels=train_labels,
        future_covariates=train_future,
        static_covariates=train_static,
    )
    val_dataset = BTCTFTDataset(
        sequences=val_seq,
        labels=val_labels,
        future_covariates=val_future,
        static_covariates=val_static,
    )
    test_dataset = BTCTFTDataset(
        sequences=test_seq,
        labels=test_labels,
        future_covariates=test_future,
        static_covariates=test_static,
    )

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

    future = None
    x_static = None

    if len(sample) == 2:
        x, y = sample

    elif len(sample) == 3:
        x, x2, y = sample

        # Disambiguate 3-tuple case based on which covariates are enabled
        if config.MODEL_CONFIG.use_future_covariates and not config.MODEL_CONFIG.use_static_covariates:
            future = x2
        elif config.MODEL_CONFIG.use_static_covariates and not config.MODEL_CONFIG.use_future_covariates:
            x_static = x2
        else:
            raise ValueError(
                "Got a 3-item sample while BOTH future and static covariates are enabled. "
                "Expected (x_past, x_future, x_static, y)."
            )

    elif len(sample) == 4:
        x, future, x_static, y = sample

    else:
        raise ValueError(f"Unexpected sample length: {len(sample)}")

    print(f"One sample X shape:       {x.shape}  (seq_length, n_features)")
    if future is not None:
        print(f"One sample future shape:  {future.shape}  (n_future_features,)")
    if x_static is not None:
        print(f"One sample static shape:  {x_static.shape}  (n_static_features,)")

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
