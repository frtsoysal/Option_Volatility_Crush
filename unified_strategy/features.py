"""
Feature engineering for the unified vol-crush pipeline.

Three blocks per event row:
  A. Stock fundamentals (33 features)  — port of /ML/scripts/with_estimates/prepare_data.py
  B. Stock options metrics (~10 features) — newly fetched + vol_crush_utils
  C. SPY market regime (30 features) — joined from spy_strategy/data/spy_daily_features.csv

Block A is implemented here. Blocks B and C live alongside in 02_features.ipynb
(the notebook orchestrates fetching + merging; this module supplies pure data
transforms with no I/O side effects beyond reading the raw earnings CSVs).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from . import ML_RAW_DIR, SP500_TICKERS_PATH, WINDOW_END, WINDOW_START

# ---------------------------------------------------------------------------
# Block A: Stock fundamentals
# Verbatim copy of the whitelist at:
#   /Users/ibrahimfiratsoysal/Documents/ML/scripts/with_estimates/prepare_data.py:71-100
# Do NOT add anything from the leakage list at line 66 of that file:
#   actual_eps, eps_beat, eps_delta, elo_after, elo_change, K_adaptive,
#   date, horizon, reported_date, price_at_report
# ---------------------------------------------------------------------------

STOCK_FEATURE_COLS: list[str] = [
    # Price momentum (4)
    "price_1m_before",
    "price_3m_before",
    "price_change_1m_pct",
    "price_change_3m_pct",
    # EPS estimates (8)
    "eps_estimate_average",
    "eps_estimate_high",
    "eps_estimate_low",
    "eps_estimate_analyst_count",
    "eps_estimate_average_7_days_ago",
    "eps_estimate_average_30_days_ago",
    "eps_estimate_average_60_days_ago",
    "eps_estimate_average_90_days_ago",
    # Analyst revisions (4)
    "eps_estimate_revision_up_trailing_7_days",
    "eps_estimate_revision_down_trailing_7_days",
    "eps_estimate_revision_up_trailing_30_days",
    "eps_estimate_revision_down_trailing_30_days",
    # Revenue estimates (4)
    "revenue_estimate_average",
    "revenue_estimate_high",
    "revenue_estimate_low",
    "revenue_estimate_analyst_count",
    # Elo (4) — historical only, NOT elo_after / elo_change / K_adaptive
    "elo_before",
    "elo_decay",
    "elo_momentum",
    "elo_vol_4q",
    # Historical growth, lag-1 = previous quarter's published number (9)
    "total_revenue_yoy_growth_lag1",
    "total_revenue_qoq_growth_lag1",
    "total_revenue_ttm_yoy_growth_lag1",
    "actual_eps_yoy_growth_lag1",
    "actual_eps_qoq_growth_lag1",
    "ebitda_yoy_growth_lag1",
    "operating_income_yoy_growth_lag1",
    "gross_margin_yoy_change_lag1",
    "operating_margin_yoy_change_lag1",
]

# Forbidden — assert these never make it into the feature matrix
LEAKAGE_COLS: list[str] = [
    "actual_eps",
    "eps_beat",
    "eps_delta",
    "elo_after",
    "elo_change",
    "K_adaptive",
    "price_at_report",
]


def load_sp500_tickers() -> list[str]:
    """Return the SP500 ticker list from the user's ML repo."""
    df = pd.read_csv(SP500_TICKERS_PATH)
    # The CSV has a "Symbol" column (or similar); be flexible
    col = next((c for c in df.columns if c.lower() in ("symbol", "ticker")), df.columns[0])
    return df[col].dropna().astype(str).str.upper().tolist()


def load_stock_events(
    ticker: str,
    raw_dir: Path = ML_RAW_DIR,
    window_start: str = WINDOW_START,
    window_end: str = WINDOW_END,
) -> pd.DataFrame:
    """
    Load one ticker's earnings events with the same filtering as
    `prepare_data.py:42-53`, restricted to our [window_start, window_end] window.

    Returns columns: ticker, fiscal_quarter_end, announcement_date, eps_beat,
                     and the 33 features in STOCK_FEATURE_COLS.
    """
    path = raw_dir / f"{ticker}_earnings_with_q4.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    # Same filters as prepare_data.py
    df = df[~df["horizon"].str.contains("fiscal year", case=False, na=False)].copy()
    df = df[df["actual_eps"].notna()].copy()

    # Date window: announcement (`reported_date`) inside our trading window
    df["reported_date"] = pd.to_datetime(df["reported_date"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[(df["reported_date"] >= window_start) & (df["reported_date"] <= window_end)].copy()
    if df.empty:
        return df

    # Pick out the columns we care about
    keep = ["date", "reported_date", "eps_beat"] + STOCK_FEATURE_COLS
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    out.insert(0, "ticker", ticker.upper())
    out = out.rename(columns={"date": "fiscal_quarter_end", "reported_date": "announcement_date"})

    # Revision NaNs → 0 (per prepare_data.py:127-129)
    for col in [c for c in STOCK_FEATURE_COLS if "revision" in c]:
        if col in out.columns:
            out[col] = out[col].fillna(0)

    # Some tickers (AAPL, MSFT) carry both calendar-quarter and fiscal-quarter
    # rows that share the same announcement_date. Dedupe — keep the row whose
    # fiscal_quarter_end is closest to the announcement (most recent quarter).
    out = out.sort_values(["ticker", "announcement_date", "fiscal_quarter_end"])
    out = out.drop_duplicates(subset=["ticker", "announcement_date"], keep="last")

    return out.reset_index(drop=True)


def load_all_stock_events(
    tickers: Iterable[str] | None = None,
    raw_dir: Path = ML_RAW_DIR,
    window_start: str = WINDOW_START,
    window_end: str = WINDOW_END,
    progress: bool = False,
) -> pd.DataFrame:
    """Concat every available ticker's events into one DataFrame."""
    if tickers is None:
        tickers = load_sp500_tickers()

    frames = []
    missing = []
    for i, t in enumerate(tickers, 1):
        df = load_stock_events(t, raw_dir, window_start, window_end)
        if df.empty:
            missing.append(t)
        else:
            frames.append(df)
        if progress and i % 50 == 0:
            print(f"  [{i}/{len(list(tickers)) if hasattr(tickers, '__len__') else '?'}] tickers loaded")

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["announcement_date", "ticker"]).reset_index(drop=True)
    if missing and progress:
        print(f"  {len(missing)} tickers had no events in window: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    return out


def assert_no_leakage(df: pd.DataFrame) -> None:
    """Fail loudly if any leakage column is present in a feature matrix."""
    bad = [c for c in LEAKAGE_COLS if c in df.columns]
    if bad:
        raise AssertionError(f"Leakage columns present in feature matrix: {bad}")
