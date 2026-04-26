"""
Bulk daily close-price fetcher for the SP500 universe.

Each Alpha Vantage TIME_SERIES_DAILY call returns the full daily history
for one ticker (with outputsize=full). For ~500 SP500 tickers that's
~500 calls = ~7 min on the 75 req/min premium tier.

Why we need this:
  The "hold to expiration" backtest needs the underlying's close on each
  short-straddle's actual expiration date — typically T+3 to T+30 from
  the announcement. We have option chains on T-1 and T+1 only, so we
  can't read expiry-date intrinsic directly from chain data.

Cache layout:
  unified_strategy/cache/daily_prices/{TICKER}.json   (Alpha Vantage raw)

The features module reads these and joins close-on-expiration onto each
event using the cached `atm_expiration_pre` column.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from unified_strategy import UNIFIED_DIR
from unified_strategy.features import load_sp500_tickers

DAILY_CACHE_DIR = UNIFIED_DIR / "cache" / "daily_prices"


def fetch_one_ticker(
    symbol: str,
    api_key: str,
    cache_dir: Path = DAILY_CACHE_DIR,
    delay: float = 0.85,
    max_retries: int = 3,
) -> str:
    """Return 'cached' | 'fetched' | 'rate_limited' | 'empty' | 'error_after_retry'."""
    cache_path = cache_dir / f"{symbol}.json"
    if cache_path.exists():
        return "cached"

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={api_key}"
    )

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=60)
            data = resp.json()
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [net-err] {symbol} attempt {attempt + 1}: {e}, sleeping {wait}s")
            time.sleep(wait)
            continue

        if "Information" in data or "Note" in data:
            msg = (data.get("Information") or data.get("Note"))[:120]
            print(f"  [rate-limit] {msg}  sleeping 60s")
            time.sleep(60)
            continue

        ts = data.get("Time Series (Daily)")
        if not ts:
            cache_path.write_text(json.dumps(data))
            return "empty"

        cache_path.write_text(json.dumps(data))
        time.sleep(delay)
        return "fetched"

    return "error_after_retry"


def load_daily_prices(ticker: str, cache_dir: Path = DAILY_CACHE_DIR) -> pd.Series | None:
    """Load one ticker's daily close prices from cache. Returns None if missing/empty."""
    p = cache_dir / f"{ticker}.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except Exception:
        return None
    ts = data.get("Time Series (Daily)")
    if not ts:
        return None

    rows = []
    for date, fields in ts.items():
        close = fields.get("4. close") or fields.get("close")
        if close is None:
            continue
        rows.append((date, float(close)))
    df = pd.DataFrame(rows, columns=["date", "close"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--delay", type=float, default=0.85)
    args = ap.parse_args()

    api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        print("ERROR: set ALPHAVANTAGE_API_KEY env var", file=sys.stderr)
        return 1

    tickers = load_sp500_tickers()
    cached_at_start = sum(1 for t in tickers if (DAILY_CACHE_DIR / f"{t}.json").exists())
    todo = len(tickers) - cached_at_start
    print(f"Tickers in SP500 list: {len(tickers)}")
    print(f"Already cached:        {cached_at_start}")
    print(f"To fetch:              {todo}")
    print(f"ETA:                   {todo * args.delay / 60:.1f} min")

    if args.dry_run:
        return 0

    counts = {"cached": 0, "fetched": 0, "empty": 0, "error_after_retry": 0}
    t0 = time.time()
    for i, t in enumerate(tickers, 1):
        s = fetch_one_ticker(t, api_key, delay=args.delay)
        counts[s] += 1
        if i % 25 == 0 or i == len(tickers):
            elapsed = time.time() - t0
            print(f"  [{i:4d}/{len(tickers)}] {t}: {s}  | rate {i / max(elapsed, 1):.1f}/s")

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
