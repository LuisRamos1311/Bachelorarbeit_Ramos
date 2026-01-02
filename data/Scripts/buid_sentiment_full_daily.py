import os
import pandas as pd

# -----------------------
# PATH SETUP (FIXED)
# -----------------------
# This script lives in: .../Bachelorarbeit_Ramos/data/Scripts
SCRIPT_DIR = os.path.dirname(__file__)        # .../data/Scripts
DATA_DIR = os.path.dirname(SCRIPT_DIR)        # .../data

REDDIT_PATH = os.path.join(DATA_DIR, "reddit_sentiment_daily_pushshift.csv")
FG_PATH     = os.path.join(DATA_DIR, "fear_greed_daily_clean.csv")
OUT_PATH    = os.path.join(DATA_DIR, "BTC_sentiment_daily.csv")

# Date range (inclusive)
MIN_DATE = "2016-01-01"
MAX_DATE = "2024-12-31"


def _fail(msg: str):
    raise ValueError(msg)


def load_reddit() -> pd.DataFrame:
    """Load and standardize the daily Reddit sentiment dataframe."""
    if not os.path.exists(REDDIT_PATH):
        raise FileNotFoundError(f"Reddit sentiment file not found at {REDDIT_PATH}")

    df = pd.read_csv(REDDIT_PATH)

    if "date" not in df.columns:
        _fail("reddit_sentiment_daily_pushshift.csv must have a 'date' column.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        _fail("Reddit sentiment CSV contains invalid dates that could not be parsed.")

    # Drop duplicates (keep first) to avoid weird joins later
    if df["date"].duplicated().any():
        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="first")

    df = df.sort_values("date")
    df = df[(df["date"] >= MIN_DATE) & (df["date"] <= MAX_DATE)]

    # Daily index, forward fill across missing days
    df = df.set_index("date").asfreq("D").ffill()

    # Validate expected columns
    expected_cols = [
        "reddit_sent_mean",
        "reddit_sent_std",
        "reddit_pos_ratio",
        "reddit_neg_ratio",
        "reddit_volume",
        "reddit_volume_log",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        _fail(
            f"Reddit sentiment is missing expected columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    return df


def load_fear_greed() -> pd.DataFrame:
    """Load and standardize the daily Fear & Greed dataframe (already imputed/engineered in step 2.3)."""
    if not os.path.exists(FG_PATH):
        raise FileNotFoundError(f"Fear & Greed file not found at {FG_PATH}")

    df = pd.read_csv(FG_PATH)

    if "date" not in df.columns:
        _fail("fear_greed_daily_clean.csv must have a 'date' column.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        _fail("Fear & Greed CSV contains invalid dates that could not be parsed.")

    # Drop duplicates (keep first)
    if df["date"].duplicated().any():
        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="first")

    df = df.sort_values("date")
    df = df[(df["date"] >= MIN_DATE) & (df["date"] <= MAX_DATE)]

    # Ensure daily index
    df = df.set_index("date").asfreq("D")

    expected_cols = ["fg_index_scaled", "fg_change_1d", "fg_missing"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        _fail(
            f"Fear & Greed is missing expected columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    return df


def sanity_checks(df: pd.DataFrame):
    """
    Foolproof checks to ensure the output is model-ready:
    - one row per day, contiguous date grid
    - no NaNs
    - expected columns present
    - basic value sanity for engineered features
    """
    # 1) Date range and continuity
    if "date" not in df.columns:
        _fail("Sanity check failed: 'date' column missing in output dataframe.")

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        _fail("Sanity check failed: 'date' must be datetime dtype.")

    expected_index = pd.date_range(MIN_DATE, MAX_DATE, freq="D")
    if len(df) != len(expected_index):
        _fail(
            f"Sanity check failed: expected {len(expected_index)} rows "
            f"(one per day from {MIN_DATE} to {MAX_DATE}) but got {len(df)}."
        )

    # Ensure sorted
    if not df["date"].is_monotonic_increasing:
        _fail("Sanity check failed: dates are not sorted ascending.")

    # Exact continuity check
    if not df["date"].reset_index(drop=True).equals(pd.Series(expected_index, name="date")):
        # Provide helpful diagnostics
        actual = df["date"].reset_index(drop=True)
        missing_days = expected_index.difference(actual)
        extra_days = actual[~actual.isin(expected_index)]
        _fail(
            "Sanity check failed: date continuity mismatch.\n"
            f"Missing days count: {len(missing_days)}\n"
            f"Extra/unexpected days count: {len(extra_days)}"
        )

    # 2) No NaNs anywhere
    if df.isna().any().any():
        nan_counts = df.isna().sum()
        _fail(
            "Sanity check failed: NaNs detected in output dataframe.\n"
            f"NaN counts per column:\n{nan_counts}"
        )

    # 3) Required columns
    required = [
        "reddit_sent_mean",
        "reddit_sent_std",
        "reddit_pos_ratio",
        "reddit_neg_ratio",
        "reddit_volume_log",
        "fg_index_scaled",
        "fg_change_1d",
        "fg_missing",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        _fail(f"Sanity check failed: missing required columns: {missing}")

    # 4) Basic value sanity (non-fatal thresholds can be turned into warnings, but here we fail loudly)
    if not ((df["fg_index_scaled"] >= 0.0) & (df["fg_index_scaled"] <= 1.0)).all():
        _fail("Sanity check failed: fg_index_scaled has values outside [0, 1].")

    # fg_missing should be 0/1 only
    unique_missing = set(df["fg_missing"].unique().tolist())
    if not unique_missing.issubset({0, 1}):
        _fail(f"Sanity check failed: fg_missing contains values outside {{0,1}}: {sorted(unique_missing)}")

    # reddit pos/neg ratios should be in [0,1]
    for col in ["reddit_pos_ratio", "reddit_neg_ratio"]:
        if not ((df[col] >= 0.0) & (df[col] <= 1.0)).all():
            _fail(f"Sanity check failed: {col} has values outside [0, 1].")

    # reddit_volume_log should be >= 0 typically (log(1) = 0). If you used log(volume+1).
    if (df["reddit_volume_log"] < 0.0).any():
        _fail("Sanity check failed: reddit_volume_log contains negative values.")

    # If we get here, all checks passed
    return


def main():
    reddit = load_reddit()
    fg = load_fear_greed()

    # Create full daily index
    full_index = pd.date_range(MIN_DATE, MAX_DATE, freq="D")

    # Align to full index
    reddit = reddit.reindex(full_index).ffill()
    fg = fg.reindex(full_index)

    # Join (fg is already imputed/engineered; we do not add new imputation here)
    combined = reddit.join(fg, how="left")

    # Reset index to a proper date column
    combined = combined.reset_index().rename(columns={"index": "date"})
    combined["date"] = combined["date"].dt.normalize()

    # Keep only model-needed columns
    out_cols = [
        "date",
        "reddit_sent_mean",
        "reddit_sent_std",
        "reddit_pos_ratio",
        "reddit_neg_ratio",
        "reddit_volume_log",
        "fg_index_scaled",
        "fg_change_1d",
        "fg_missing",
    ]
    combined = combined[out_cols]

    # Run sanity checks (will raise if anything is wrong)
    combined["date"] = pd.to_datetime(combined["date"])
    sanity_checks(combined)

    # Save output
    combined.to_csv(OUT_PATH, index=False)

    # Minimal success prints (same style you already saw)
    print(combined.head(5))
    print(combined.tail(5))
    print(f"Rows: {len(combined)} (should be one row per day from {MIN_DATE} to {MAX_DATE})")
    print(f"Saved combined sentiment dataframe to: {OUT_PATH}")


if __name__ == "__main__":
    main()