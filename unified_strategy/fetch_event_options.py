"""
Bulk per-event option-chain fetcher.

Wraps `vol_crush_utils.fetch_historical_options` (which is idempotent + caches
per-date JSON) for many (ticker, date) pairs at the Alpha Vantage premium
75/min rate. Resumable: cached files are skipped instantly on re-run.

Each event needs two calls — `pre_date` (1 BD before announcement) and
`post_date` (1 BD after). For ~9,500 events that's ~19,000 calls = ~4.2h.

Usage:
    from unified_strategy.fetch_event_options import fetch_for_events
    fetch_for_events(events_df, api_key=os.environ["ALPHAVANTAGE_API_KEY"])

events_df must have columns: ticker, announcement_date.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import CACHE_DIR, VOL_CRUSH_UTILS_DIR

# Add option_volatility_crush.ipynb/ to sys.path so we can import vol_crush_utils
# despite the directory's .ipynb suffix breaking normal package import semantics.
if str(VOL_CRUSH_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(VOL_CRUSH_UTILS_DIR))

import vol_crush_utils as vcu  # noqa: E402

# Reuse the holiday-aware shifter from features.py (single source of truth).
from .features import trading_day_offset


def event_dates(announcement_date: pd.Timestamp) -> tuple[str, str]:
    """Return (pre_date, post_date) as YYYY-MM-DD strings — 1 US trading day each side, holiday-aware."""
    pre = trading_day_offset(announcement_date, -1)
    post = trading_day_offset(announcement_date, 1)
    return pre, post


def fetch_one_event(
    ticker: str,
    announcement_date: pd.Timestamp,
    api_key: str,
    cache_dir: Path = CACHE_DIR,
    delay: float = 0.85,
) -> dict[str, pd.DataFrame | None]:
    """Fetch pre and post chains for one event. Returns {'pre': df, 'post': df}."""
    pre, post = event_dates(announcement_date)
    return {
        "pre": vcu.fetch_historical_options(ticker, pre, api_key, cache_dir, delay=delay),
        "post": vcu.fetch_historical_options(ticker, post, api_key, cache_dir, delay=delay),
    }


def fetch_for_events(
    events: pd.DataFrame,
    api_key: str | None = None,
    cache_dir: Path = CACHE_DIR,
    delay: float = 0.85,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Fetch pre+post option chains for every (ticker, announcement_date) row.

    Returns a tracking frame with columns:
        ticker, announcement_date, pre_date, post_date,
        pre_status (cached|fetched|empty|error), post_status, pre_path, post_path
    """
    api_key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set ALPHAVANTAGE_API_KEY env var or pass api_key=... to fetch_for_events()"
        )

    rows = []
    n = len(events)
    for i, row in enumerate(events.itertuples(index=False), 1):
        ticker = str(row.ticker).upper()
        ann = pd.Timestamp(row.announcement_date)
        pre, post = event_dates(ann)

        pre_path = cache_dir / ticker / f"{pre}.json"
        post_path = cache_dir / ticker / f"{post}.json"

        pre_existed = pre_path.exists()
        post_existed = post_path.exists()

        # Fetcher writes to cache_dir / {symbol} / {date}.json regardless
        try:
            vcu.fetch_historical_options(ticker, pre, api_key, cache_dir, delay=delay)
            pre_status = "cached" if pre_existed else "fetched"
        except Exception as e:
            pre_status = f"error:{type(e).__name__}"

        try:
            vcu.fetch_historical_options(ticker, post, api_key, cache_dir, delay=delay)
            post_status = "cached" if post_existed else "fetched"
        except Exception as e:
            post_status = f"error:{type(e).__name__}"

        rows.append(
            {
                "ticker": ticker,
                "announcement_date": ann,
                "pre_date": pre,
                "post_date": post,
                "pre_status": pre_status,
                "post_status": post_status,
                "pre_path": str(pre_path),
                "post_path": str(post_path),
            }
        )

        if progress and (i % 25 == 0 or i == n):
            cached = sum(1 for r in rows if r["pre_status"] == "cached" and r["post_status"] == "cached")
            print(f"  [{i:5d}/{n}] {ticker} {ann.date()}  cached_so_far={cached}")

    return pd.DataFrame(rows)
