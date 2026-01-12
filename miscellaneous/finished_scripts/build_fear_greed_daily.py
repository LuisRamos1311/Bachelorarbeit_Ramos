import os
import requests
import pandas as pd


# --------- SETTINGS (student-friendly) ----------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

OUT_PATH = os.path.join(DATA_DIR, "fear_greed_daily_clean.csv")

# Your preferred date range
MIN_DATE = "2016-01-01"
MAX_DATE = "2024-12-31"

# How to handle missing Fear & Greed values (especially pre-2018):
# - Add a missingness flag (1 if missing, else 0)
# - Fill missing values with a neutral constant so downstream scaling / tensors don't see NaNs
NEUTRAL_FG_VALUE = 50.0  # 0–100 scale; corresponds to 0.5 when scaled

API_URL = "https://api.alternative.me/fng/"  # documented endpoint
PARAMS = {
    "limit": 0,         # 0 means: return all available data
    "format": "json",   # easiest to parse robustly
}
# -----------------------------------------------


def fetch_fng_json() -> dict:
    r = requests.get(API_URL, params=PARAMS, timeout=60)
    r.raise_for_status()
    return r.json()


def build_daily_df(payload: dict) -> pd.DataFrame:
    if "data" not in payload:
        raise ValueError("Unexpected response: missing 'data' field")

    rows = payload["data"]
    df = pd.DataFrame(rows)

    # Expected fields in API response: value (string), timestamp (string unix seconds)
    if "value" not in df.columns or "timestamp" not in df.columns:
        raise ValueError(f"Unexpected columns from API: {df.columns.tolist()}")

    # Convert types
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    # Convert timestamp -> date (UTC)
    df["date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.normalize()
    df = df.dropna(subset=["date", "value"])

    # Remove duplicates by date (keep last)
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    # Create full daily date range so we have one row per calendar day
    full_idx = pd.date_range(MIN_DATE, MAX_DATE, freq="D", tz="UTC")
    df = df.set_index("date").reindex(full_idx).sort_index()

    # Missingness flag BEFORE we fill anything:
    # 1 = missing in the source data for that date, 0 = observed
    df["fg_missing"] = df["value"].isna().astype(int)

    # Fill strategy:
    # - forward-fill INTERNAL gaps (if the provider skips a day)
    # - fill LEADING NaNs (pre-index-era) with a neutral constant
    df["value"] = df["value"].ffill()
    df["value"] = df["value"].fillna(NEUTRAL_FG_VALUE)

    # Create features (no NaNs from this point on)
    df["fg_index_scaled"] = df["value"] / 100.0
    df["fg_change_1d"] = df["fg_index_scaled"].diff().fillna(0.0)

    # Final formatting
    df = df[["fg_index_scaled", "fg_change_1d", "fg_missing"]].reset_index()
    df = df.rename(columns={"index": "date"})
    df["date"] = df["date"].dt.tz_localize(None)  # drop timezone for CSV clarity
    return df


def main():
    print("Fetching Fear & Greed data from Alternative.me ...")
    payload = fetch_fng_json()

    print("Building clean daily dataframe ...")
    df = build_daily_df(payload)

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH}")
    print(df.head(5))
    print(df.tail(5))
    print(f"Rows: {len(df)} (should be one row per day in range)")

    # Quick sanity checks
    print("Sanity checks:")
    print("  fg_index_scaled min/max:", df["fg_index_scaled"].min(), df["fg_index_scaled"].max())
    print("  fg_change_1d min/max:", df["fg_change_1d"].min(), df["fg_change_1d"].max())
    print("  fg_missing count:", int(df["fg_missing"].sum()))
    print("  any NaNs:", df.isna().any().to_dict())


if __name__ == "__main__":
    main()
