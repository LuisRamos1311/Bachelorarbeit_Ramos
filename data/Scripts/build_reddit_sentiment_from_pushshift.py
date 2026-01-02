import io
import math
import json
from collections import defaultdict
from datetime import datetime, timezone, date
from pathlib import Path

import pandas as pd
import ujson
import zstandard as zstd
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ========= PATHS & CONFIG =========

# This script is inside the data folder: Bachelorarbeit_Ramos/data
DATA_DIR = Path(__file__).resolve().parent

# Project root is the parent of data:
ROOT_DIR = DATA_DIR.parent

# Folder where the subreddit dumps are stored
# (data/reddit/subreddits24)
PUSHSHIFT_DIR = DATA_DIR / "reddit" / "subreddits24"

# Output CSV we want to create in the data folder
OUT_PATH = DATA_DIR / "reddit_sentiment_daily_pushshift.csv"

# Names of the .zst files we want to process
# (these should match what you see in data/reddit/subreddits24)
ZST_FILES = [
    "Bitcoin_comments.zst",
    "Bitcoin_submissions.zst",
    "btc_comments.zst",
    "btc_submissions.zst",
    "CryptoCurrency_comments.zst",
    "CryptoCurrency_submissions.zst",
    "BitcoinMarkets_comments.zst",
    "BitcoinMarkets_submissions.zst",
    "CryptoMarkets_comments.zst",
    "CryptoMarkets_submissions.zst",
]

# Date range of interest (inclusive)
MIN_DATE = date(2016, 1, 1)    # start collecting from 2016
MAX_DATE = date(2024, 12, 31)  # keep the same end date


# ========= HELPERS TO OPEN COMPRESSED FILES =========

def open_zst(path: Path):
    """Open a .zst/.zstd file and return a text stream (utf-8)."""
    fh = path.open("rb")
    dctx = zstd.ZstdDecompressor(max_window_size=2**31)
    reader = dctx.stream_reader(fh)
    text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    return text_stream


def open_xz(path: Path):
    import lzma
    return io.TextIOWrapper(
        lzma.open(path, mode="rb"),
        encoding="utf-8",
        errors="replace",
    )


def open_bz2(path: Path):
    import bz2
    return io.TextIOWrapper(
        bz2.open(path, mode="rb"),
        encoding="utf-8",
        errors="replace",
    )


def open_gz(path: Path):
    import gzip
    return io.TextIOWrapper(
        gzip.open(path, mode="rb"),
        encoding="utf-8",
        errors="replace",
    )


def iter_dump_files():
    """
    Yield paths to each subreddit-wide .zst file under PUSHSHIFT_DIR.

    We use the filenames listed in ZST_FILES and look for them directly
    in the subreddits24 folder (not inside the extracted folders).
    """
    if not PUSHSHIFT_DIR.exists():
        raise RuntimeError(f"PUSHSHIFT_DIR does not exist: {PUSHSHIFT_DIR}")

    print(f"Looking for subreddit .zst files in: {PUSHSHIFT_DIR}")

    paths = []
    for fname in ZST_FILES:
        p = PUSHSHIFT_DIR / fname
        if p.exists():
            paths.append(p)
        else:
            print(f"  !! Warning: {fname} not found, skipping")

    if not paths:
        raise RuntimeError(f"No .zst files found in {PUSHSHIFT_DIR}")

    return paths


# ========= CORE PROCESSING FUNCTIONS =========

def parse_pushshift_file(path: Path,
                         analyzer: SentimentIntensityAnalyzer,
                         stats: dict):
    """
    Stream one compressed NDJSON file and update 'stats' dict.

    stats[date] = {
        "n": count,
        "sum": sum_scores,
        "sum_sq": sum_of_squares,
        "pos": count_pos,
        "neg": count_neg,
    }
    """
    print(f"Processing file: {path}")

    # Choose correct opener
    ext = path.suffix.lower()
    if ext in [".zst", ".zstd"]:
        opener = open_zst
    elif ext == ".xz":
        opener = open_xz
    elif ext == ".bz2":
        opener = open_bz2
    elif ext == ".gz":
        opener = open_gz
    else:
        # Assume it's already plain-text JSON (no compression)
        def opener(p: Path):
            return p.open("rt", encoding="utf-8", errors="replace")

    with opener(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Each line is a JSON object
            try:
                obj = ujson.loads(line)
            except ValueError:
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue

            created_utc = obj.get("created_utc")
            if created_utc is None:
                continue

            # created_utc is Unix seconds
            try:
                ts = int(created_utc)
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            except (TypeError, ValueError, OSError):
                continue

            d = dt.date()
            if d < MIN_DATE or d > MAX_DATE:
                continue

            # Extract text
            # comments → "body"; submissions → "title" + "selftext"
            text = ""
            if "body" in obj:
                text = obj.get("body") or ""
            else:
                title = obj.get("title") or ""
                selftext = obj.get("selftext") or ""
                text = f"{title} {selftext}"

            text = text.strip()
            if not text:
                continue
            if text in ("[deleted]", "[removed]"):
                continue

            # VADER compound sentiment
            score = analyzer.polarity_scores(text)["compound"]

            s = stats[d]
            s["n"] += 1
            s["sum"] += score
            s["sum_sq"] += score * score
            if score > 0:
                s["pos"] += 1
            if score < 0:
                s["neg"] += 1


def build_daily_stats() -> pd.DataFrame:
    """
    Iterate over all dump files, aggregate daily stats into a DataFrame.
    """
    analyzer = SentimentIntensityAnalyzer()

    # stats[date] = {"n": ..., "sum": ..., "sum_sq": ..., "pos": ..., "neg": ...}
    stats = defaultdict(lambda: {"n": 0, "sum": 0.0, "sum_sq": 0.0, "pos": 0, "neg": 0})

    paths = list(iter_dump_files())
    print(f"Found {len(paths)} .zst files to process.\n")

    for path in paths:
        parse_pushshift_file(path, analyzer, stats)

    # Convert stats dict → DataFrame
    rows = []
    for d, s in stats.items():
        n = s["n"]
        if n == 0:
            continue

        mean = s["sum"] / n
        if n > 1:
            var = (s["sum_sq"] - n * mean * mean) / (n - 1)
            var = max(var, 0.0)
            std = var ** 0.5
        else:
            std = 0.0

        pos_ratio = s["pos"] / n
        neg_ratio = s["neg"] / n
        volume = n
        volume_log = math.log(volume + 1)

        rows.append(
            {
                "date": d,
                "reddit_sent_mean": mean,
                "reddit_sent_std": std,
                "reddit_pos_ratio": pos_ratio,
                "reddit_neg_ratio": neg_ratio,
                "reddit_volume": volume,
                "reddit_volume_log": volume_log,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("date")
    return df


# ========= MAIN ENTRY =========

def main():
    print("Project root:", ROOT_DIR)
    print("Pushshift dir:", PUSHSHIFT_DIR)
    print("Output CSV  :", OUT_PATH)
    print()

    df = build_daily_stats()
    df.to_csv(OUT_PATH, index=False)
    print("\nDone.")
    print(f"Saved daily sentiment to: {OUT_PATH}")
    print(df.head())
    print(df.tail())


if __name__ == "__main__":
    main()
