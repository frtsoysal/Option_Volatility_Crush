"""
Bulk fetcher for the full S&P 500 vol-crush dataset.

Fetches pre + post option chains for every earnings event in the
2021-06-23 to 2026-04-21 window (~8,005 events × 2 dates = ~16K calls).

Robust against:
  - Rate-limit responses (does NOT cache; sleeps 60s; retries forever)
  - Transient network errors (3x exponential backoff)
  - Existing cache (skips instantly)

Usage:
    export ALPHAVANTAGE_API_KEY=...
    python3 run_bulk_fetch.py [--dry-run]

Logs progress every 50 events. Saves tracking CSV every 250 events so a
crashed run leaves a partial audit trail you can resume from.
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

# Make package importable when running from inside unified_strategy/
HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from unified_strategy import CACHE_DIR, WINDOW_END, WINDOW_START
from unified_strategy.features import load_all_stock_events, load_sp500_tickers
from unified_strategy.fetch_event_options import event_dates


def fetch_with_retry(
    symbol: str,
    date: str,
    api_key: str,
    cache_dir: Path,
    delay: float = 0.85,
    max_retries: int = 3,
) -> str:
    """
    Fetch one (symbol, date). Returns one of:
        'cached'           cache hit, no API call
        'fetched'          API success, written to cache
        'empty'            API returned no data (e.g. holiday); written to cache
        'error_after_retry' all retries exhausted
    """
    cache_path = cache_dir / symbol / f"{date}.json"
    if cache_path.exists():
        return "cached"

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://www.alphavantage.co/query"
        f"?function=HISTORICAL_OPTIONS&symbol={symbol}&date={date}&apikey={api_key}"
    )

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=60)
            data = resp.json()
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [net-err] {symbol} {date} attempt {attempt + 1}: {e}, sleeping {wait}s")
            time.sleep(wait)
            continue

        # Rate limit — DON'T cache, wait 60s, count this as a retry
        if "Information" in data or "Note" in data:
            msg = (data.get("Information") or data.get("Note"))[:120]
            print(f"  [rate-limit] {msg}  sleeping 60s then retrying ({symbol} {date})")
            time.sleep(60)
            continue

        # Genuine API response — cache and return
        cache_path.write_text(json.dumps(data))
        time.sleep(delay)
        return "fetched" if data.get("data") else "empty"

    return "error_after_retry"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="just count work, don't fetch")
    ap.add_argument("--start", default=WINDOW_START)
    ap.add_argument("--end", default=WINDOW_END)
    ap.add_argument("--delay", type=float, default=0.85)
    args = ap.parse_args()

    api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        print("ERROR: set ALPHAVANTAGE_API_KEY env var", file=sys.stderr)
        return 1

    print(f"window:       {args.start} → {args.end}")
    print(f"cache_dir:    {CACHE_DIR}")
    print(f"sleep/call:   {args.delay}s  (effective ~{60 / args.delay:.0f} calls/min)")

    tickers = load_sp500_tickers()
    print(f"\nLoading SP500 events from {len(tickers)} tickers...")
    events = load_all_stock_events(tickers, window_start=args.start, window_end=args.end)
    print(f"events: {len(events):,}")

    # Build the (symbol, date) call list
    calls = []
    for row in events.itertuples(index=False):
        pre, post = event_dates(pd.Timestamp(row.announcement_date))
        calls.append((row.ticker.upper(), pre))
        calls.append((row.ticker.upper(), post))

    # Already-cached?
    cached_at_start = sum(1 for s, d in calls if (CACHE_DIR / s / f"{d}.json").exists())
    todo = len(calls) - cached_at_start
    minutes_estimate = todo * args.delay / 60
    print(f"\ncalls total:    {len(calls):,}")
    print(f"already cached: {cached_at_start:,}")
    print(f"to fetch:       {todo:,}")
    print(f"estimated wall: {minutes_estimate:.0f} min  ≈ {minutes_estimate / 60:.1f} hours")

    if args.dry_run:
        return 0

    # Fetch loop
    counts = {"cached": 0, "fetched": 0, "empty": 0, "error_after_retry": 0}
    tracking_rows = []
    t0 = time.time()
    tracking_path = HERE / "data" / "bulk_fetch_tracking.csv"
    tracking_path.parent.mkdir(exist_ok=True)

    for i, (sym, dt) in enumerate(calls, 1):
        status = fetch_with_retry(sym, dt, api_key, CACHE_DIR, args.delay)
        counts[status] += 1
        tracking_rows.append({"symbol": sym, "date": dt, "status": status})

        if i % 50 == 0 or i == len(calls):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            new = counts["fetched"] + counts["empty"] + counts["error_after_retry"]
            remaining = (len(calls) - i) / rate if rate > 0 else 0
            print(
                f"[{i:5d}/{len(calls):,}]  "
                f"cached={counts['cached']}  fetched={counts['fetched']}  "
                f"empty={counts['empty']}  err={counts['error_after_retry']}  "
                f"|  {rate:.1f} call/s  ETA {remaining / 60:.0f} min"
            )

        if i % 250 == 0:
            pd.DataFrame(tracking_rows).to_csv(tracking_path, index=False)

    # Final save
    pd.DataFrame(tracking_rows).to_csv(tracking_path, index=False)

    elapsed = time.time() - t0
    print(f"\n=== bulk fetch complete in {elapsed / 60:.1f} min ===")
    for k, v in counts.items():
        print(f"  {k}: {v:,}")
    print(f"  tracking → {tracking_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
