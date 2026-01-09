import chaindl
import pandas as pd

# Define the column mappings explicitly for each metric
COLUMN_MAP = {
    "active_addresses": "Active Addresses",
    "tx_count": "Transaction Count",
    "mvrv": "MVRV Ratio",
    "sopr": "Spent Output Profit Ratio (SOPR)",
    "hash_rate": "Hashrate (EH/s)"
}

# URLs for CheckOnChain charts
URL_ACTIVE = "https://charts.checkonchain.com/btconchain/adoption/actaddress_momentum/actaddress_momentum_light.html"
URL_TX = "https://charts.checkonchain.com/btconchain/adoption/txcount_momentum/txcount_momentum_light.html"
URL_MVRV = "https://charts.checkonchain.com/btconchain/pricing/pricing_mvrv_bands/pricing_mvrv_bands_light.html"
URL_SOPR = "https://charts.checkonchain.com/btconchain/realised/sopr/sopr_light.html"
URL_HASH = "https://charts.checkonchain.com/btconchain/mining/hashribbons/hashribbons_light.html"

def normalize_column_names(df):
    """
    Normalizes column names to handle encoded characters and escape sequences.
    """
    df.columns = df.columns.str.replace(r'\\u002f', '/', regex=True)  # Normalize \u002f to /
    df.columns = df.columns.str.replace(r'\\u003d', '=', regex=True)  # Normalize \u003d to '=' if needed
    return df

def download_checkonchain_series(url: str, value_col_name: str) -> pd.DataFrame:
    """
    Downloads a CheckOnChain chart with chaindl and returns a
    DataFrame with two columns: ['date', value_col_name].
    """
    print(f"Downloading: {url}")
    df = chaindl.download(url)  # returns a pandas DataFrame with Date index

    # Normalize column names to ensure they match expected names
    df = normalize_column_names(df)

    # Show the actual headers from the chart
    print("Actual headers (columns) from the chart:")
    print(df.columns)  # This prints the column names in the downloaded DataFrame
    print("-" * 60)

    # Check if the required column exists in the chart
    if COLUMN_MAP[value_col_name] in df.columns:
        print(f"Found the column '{COLUMN_MAP[value_col_name]}' for {value_col_name}.")
        series = df[COLUMN_MAP[value_col_name]].copy()
    else:
        # If the exact column doesn't match, you could add logic to select by regex or fallback
        print(f"Warning: Column '{COLUMN_MAP[value_col_name]}' not found for {value_col_name}, checking alternative columns...")
        series = df[df.columns[1]].copy()  # Default to the second column if not found (or handle differently)

    series.name = value_col_name

    out = series.to_frame()
    out["date"] = out.index.normalize()  # drop time, keep just the date

    # Keep only 'date' + our metric name
    return out[["date", value_col_name]]

def main():
    # Download each metric as a separate DataFrame
    df_active = download_checkonchain_series(URL_ACTIVE, "active_addresses")
    df_tx = download_checkonchain_series(URL_TX, "tx_count")
    df_mvrv = download_checkonchain_series(URL_MVRV, "mvrv")
    df_sopr = download_checkonchain_series(URL_SOPR, "sopr")
    df_hash = download_checkonchain_series(URL_HASH, "hash_rate")

    # Merge them together on 'date'
    df = (
        df_active
        .merge(df_tx, on="date", how="outer")
        .merge(df_mvrv, on="date", how="outer")
        .merge(df_sopr, on="date", how="outer")
        .merge(df_hash, on="date", how="outer")
    )

    # Sort by date
    df = df.sort_values("date")

    # Remove duplicate dates if any
    df = df[~df["date"].duplicated(keep="first")]

    # Optional: enforce exactly one row per calendar day
    df = df.set_index("date").asfreq("1D")

    # Make sure all values are numeric (floats); bad strings become NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Move 'date' back to a column
    df = df.reset_index()

    print("Final merged head:")
    print(df.head())
    print("Final dtypes:")
    print(df.dtypes)

    # Save to CSV (adjust path if needed; this matches your Data folder style)
    output_path = "data/BTC_onchain_daily.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved daily on-chain metrics to: {output_path}")

if __name__ == "__main__":
    main()