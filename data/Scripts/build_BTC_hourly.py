import os
import pandas as pd


def main():
    # Script is located in the `data/` folder, so use that as data_dir directly
    try:
        data_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        data_dir = os.getcwd()

    in_path = os.path.join(data_dir, "XBTUSD_60.csv")
    out_path = os.path.join(data_dir, "BTCUSD_hourly.csv")

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input not found: {in_path}")

    if os.path.exists(out_path):
        raise FileExistsError(
            f"Output already exists: {out_path}\n"
            "Rename/delete it first, or change out_path to a new filename."
        )

    print(f"Reading Kraken file:\n  {in_path}")

    # Kraken format: timestamp,open,high,low,close,volume,trades (no header)
    df = pd.read_csv(
        in_path,
        header=None,
        names=["timestamp", "open", "high", "low", "close", "volume_btc", "trades"],
    )

    # Coerce numeric
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64")
    for c in ["open", "high", "low", "close", "volume_btc"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with invalid timestamp or close (close is essential)
    df = df.dropna(subset=["timestamp", "close"]).copy()

    # Convert to UTC datetime index
    df["date"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="s", utc=True)
    df = df.drop(columns=["timestamp"], errors="ignore")
    df = df.set_index("date").sort_index()

    # Remove duplicates on index (keep first)
    df = df[~df.index.duplicated(keep="first")]

    # Build continuous hourly index (UTC)
    start = df.index.min().floor("h")
    end = df.index.max().floor("h")
    full_idx = pd.date_range(start=start, end=end, freq="h", tz="UTC")

    # Mark original rows BEFORE reindex, using nullable boolean dtype to avoid warnings
    df["is_original"] = pd.Series(True, index=df.index, dtype="boolean")

    # Reindex to continuous grid
    df = df.reindex(full_idx)

    # Identify filled rows (still nullable boolean; fill missing with False)
    df["is_original"] = df["is_original"].fillna(False)

    # Fill prices: for missing hours, use previous close
    prev_close = df["close"].ffill()
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].fillna(prev_close)

    # Volume handling:
    # - original rows keep Kraken's volume_btc
    # - inserted rows => 0
    df["volume_btc"] = df["volume_btc"].fillna(0.0)

    # Create quote volume proxy in USD
    df["volume_usd"] = df["volume_btc"] * df["close"]

    # Prepare CryptoDataDownload-style columns
    out = pd.DataFrame(index=df.index)
    out["unix"] = (out.index.view("int64") // 10**9).astype("int64")
    out["date"] = out.index.strftime("%Y-%m-%d %H:%M:%S")
    out["symbol"] = "BTC/USD"
    out["open"] = df["open"].astype("float64")
    out["high"] = df["high"].astype("float64")
    out["low"] = df["low"].astype("float64")
    out["close"] = df["close"].astype("float64")
    out["Volume BTC"] = df["volume_btc"].astype("float64")
    out["Volume USD"] = df["volume_usd"].astype("float64")

    # Sort newest-first to mimic CryptoDataDownload typical ordering
    out = out.sort_values("unix", ascending=False)

    # Write with the extra top URL line (your loader uses skiprows=1)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write("https://www.CryptoDataDownload.com\n")
    out.to_csv(tmp_path, mode="a", index=False)

    os.replace(tmp_path, out_path)

    # Summary
    earliest = out["date"].iloc[-1]
    latest = out["date"].iloc[0]
    total_rows = len(out)
    inserted_rows = int((~df["is_original"].astype(bool)).sum())

    print("\nWrote converted file:")
    print(f"  {out_path}")
    print(f"Rows (hourly candles): {total_rows}")
    print(f"Coverage (UTC): {earliest}  ->  {latest}")
    print(f"Inserted (filled) hours: {inserted_rows}")
    print("\nDone.")


if __name__ == "__main__":
    main()
