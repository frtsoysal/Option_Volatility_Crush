"""
Feature engineering for the unified vol-crush pipeline.

Three blocks per event row:
  A. Stock fundamentals (33 features)  — port of /ML/scripts/with_estimates/prepare_data.py
  B. Stock options metrics (~10 features) — derived from cached Alpha Vantage chains
  C. SPY market regime (30 features) — joined from spy_strategy/data/spy_daily_features.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import (
    CACHE_DIR,
    ML_RAW_DIR,
    SPY_FEATURES_PATH,
    SP500_TICKERS_PATH,
    VOL_CRUSH_UTILS_DIR,
    WINDOW_END,
    WINDOW_START,
)

# vol_crush_utils lives in a directory ending in .ipynb; the dot breaks
# normal package import semantics, so we add the dir to sys.path.
if str(VOL_CRUSH_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(VOL_CRUSH_UTILS_DIR))

import vol_crush_utils as vcu  # noqa: E402

# Holiday-aware US trading-day calendar. CustomBusinessDay with USFederalHolidayCalendar
# correctly skips MLK Day, Presidents Day, Good Friday, Memorial Day, etc. — which
# the plain pd.bdate_range() / pd.tseries.offsets.BDay used in vol_crush_utils does not.
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

_US_BD = CustomBusinessDay(calendar=USFederalHolidayCalendar())


def trading_day_offset(date: str | pd.Timestamp, offset: int) -> str:
    """Shift `date` by `offset` US trading days (skipping federal holidays)."""
    dt = pd.Timestamp(date) + offset * _US_BD
    return str(dt.date())

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


# ---------------------------------------------------------------------------
# Block B: Stock options metrics
# Per event, compute ~10 features from the cached pre/post chain JSONs.
# ---------------------------------------------------------------------------

OPTION_FEATURE_COLS: list[str] = [
    "stock_price_pre",
    "stock_price_post",
    "atm_strike_pre",
    "atm_expiration_pre",  # the contract we sold, looked up again in post chain
    "atm_call_mid_pre",
    "atm_put_mid_pre",
    "straddle_price_pre",
    "straddle_pct_pre",
    # Real post-event mids on the SAME (strike, expiration) we sold.
    # Used by the backtest as the realistic close-out price (not theoretical
    # intrinsic). NaN if the contract isn't quoted on the post date (rare).
    "atm_call_mid_post",
    "atm_put_mid_post",
    "exit_premium",  # = atm_call_mid_post + atm_put_mid_post
    # Hold-to-expiration close: the underlying's close on the contract's
    # expiration date, joined from cached TIME_SERIES_DAILY. Used by the
    # backtest's hold_to_expiry mode — at expiration the option's value
    # is purely intrinsic, so we don't pay back any time value.
    "expiry_close",
    "exit_intrinsic_at_expiry",  # = max(0, |expiry_close - atm_strike_pre|)
    "iv_call_pre",
    "iv_put_pre",
    "iv_avg_pre",
    "iv_avg_post",
    "iv_crush_pct",
    "iv_term_slope",
    "atm_open_interest_pre",
    "atm_volume_pre",
    "chain_pc_volume_ratio",
    "dte_pre",  # days to expiration of the chosen ATM expiration
]


def _load_chain_json(path: Path) -> pd.DataFrame | None:
    """Load one cached HISTORICAL_OPTIONS JSON into a DataFrame, or None if empty."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    rows = data.get("data") or []
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["fetch_date"] = path.stem
    return df


def compute_options_features(
    ticker: str,
    announcement_date: pd.Timestamp,
    cache_dir: Path = CACHE_DIR,
) -> dict[str, float]:
    """
    Compute Block-B options features for one event.

    Returns a dict keyed by OPTION_FEATURE_COLS. Missing values become NaN.
    """
    out = {col: np.nan for col in OPTION_FEATURE_COLS}

    pre_str = trading_day_offset(announcement_date, -1)
    post_str = trading_day_offset(announcement_date, 1)

    pre_path = cache_dir / ticker / f"{pre_str}.json"
    post_path = cache_dir / ticker / f"{post_str}.json"

    pre_chain = _load_chain_json(pre_path)
    post_chain = _load_chain_json(post_path)

    if pre_chain is None:
        return out

    # Estimate spot via put-call parity
    spot_pre = vcu.estimate_stock_price_from_chain(pre_chain)
    if spot_pre is None or spot_pre <= 0:
        return out
    out["stock_price_pre"] = float(spot_pre)

    if post_chain is not None:
        spot_post = vcu.estimate_stock_price_from_chain(post_chain)
        if spot_post is not None and spot_post > 0:
            out["stock_price_post"] = float(spot_post)

    # ATM extraction (nearest-strike, near-expiration)
    atm_pre = vcu.extract_atm_options(pre_chain, spot_pre)
    if atm_pre is None:
        return out

    call_pre = atm_pre["call"]
    put_pre = atm_pre["put"]
    out["atm_strike_pre"] = float(call_pre["strike"])
    pre_strike = float(call_pre["strike"])
    pre_expiration = str(call_pre["expiration"]) if "expiration" in call_pre.index else None
    out["atm_expiration_pre"] = pre_expiration

    straddle = vcu.compute_straddle_metrics(call_pre, put_pre, spot_pre)
    out["atm_call_mid_pre"] = straddle["call_price"]
    out["atm_put_mid_pre"] = straddle["put_price"]
    out["straddle_price_pre"] = straddle["straddle_price"]
    out["straddle_pct_pre"] = straddle["straddle_pct"]

    # DTE for the chosen ATM expiration
    if "expiration" in call_pre.index:
        try:
            exp = pd.to_datetime(call_pre["expiration"])
            ref = pd.to_datetime(pre_str)
            out["dte_pre"] = float((exp - ref).days)
        except Exception:
            pass

    # ── Real post-event mids on the SAME contract we sold.
    # Look up (pre_strike, pre_expiration) in the post chain. This is the
    # realistic price you'd pay to close the short straddle the day after
    # earnings — NOT theoretical intrinsic, NOT the post-chain ATM (because
    # spot moved and a different strike would now be ATM). We need the
    # SAME (strike, expiration) pair, otherwise we're closing a different
    # contract than the one we opened.
    if post_chain is not None and pre_expiration is not None:
        post_match = post_chain[
            (post_chain["expiration"] == pre_expiration)
            & (pd.to_numeric(post_chain["strike"], errors="coerce") == pre_strike)
        ]
        if not post_match.empty:
            types = post_match["type"].astype(str).str.lower()
            calls = post_match[types == "call"]
            puts = post_match[types == "put"]
            if not calls.empty:
                out["atm_call_mid_post"] = vcu._mid_price(calls.iloc[0])
            if not puts.empty:
                out["atm_put_mid_post"] = vcu._mid_price(puts.iloc[0])
            if pd.notna(out["atm_call_mid_post"]) and pd.notna(out["atm_put_mid_post"]):
                out["exit_premium"] = out["atm_call_mid_post"] + out["atm_put_mid_post"]

    # IVs at pre — invert BS for both legs.
    # Use the chain's reported implied_volatility column if present (Alpha Vantage
    # already provides Greeks/IV in the chain), else invert via vol_crush_utils.
    def _safe_iv(row, ot):
        # 1) trust AV's own field if it exists and is numeric
        for col in ("implied_volatility", "iv"):
            v = row.get(col)
            try:
                v = float(v)
                if 0 < v < 5.0:
                    return v
            except Exception:
                continue
        # 2) fallback: invert BS
        if out["dte_pre"] is None or np.isnan(out["dte_pre"]):
            return np.nan
        T = float(out["dte_pre"]) / 365.0
        if T <= 0:
            return np.nan
        try:
            mid = vcu._mid_price(row)
            return vcu.implied_vol(mid, spot_pre, float(row["strike"]), T, r=0.04, option_type=ot)
        except Exception:
            return np.nan

    out["iv_call_pre"] = _safe_iv(call_pre, "call")
    out["iv_put_pre"] = _safe_iv(put_pre, "put")
    if pd.notna(out["iv_call_pre"]) and pd.notna(out["iv_put_pre"]):
        out["iv_avg_pre"] = (out["iv_call_pre"] + out["iv_put_pre"]) / 2

    # Post-event IV (for crush calculation)
    if post_chain is not None:
        spot_post_local = out.get("stock_price_post") or spot_pre
        atm_post = vcu.extract_atm_options(post_chain, spot_post_local)
        if atm_post is not None:
            iv_call_post = _safe_iv(atm_post["call"], "call")
            iv_put_post = _safe_iv(atm_post["put"], "put")
            if pd.notna(iv_call_post) and pd.notna(iv_put_post):
                out["iv_avg_post"] = (iv_call_post + iv_put_post) / 2

    if pd.notna(out["iv_avg_pre"]) and pd.notna(out["iv_avg_post"]) and out["iv_avg_pre"] > 0:
        out["iv_crush_pct"] = (out["iv_avg_pre"] - out["iv_avg_post"]) / out["iv_avg_pre"] * 100

    # IV term structure: ATM IV at near expiry vs ~60-90 DTE
    out["iv_term_slope"] = _term_structure_slope(pre_chain, spot_pre, near_dte=out["dte_pre"])

    # Liquidity at the ATM strike
    out["atm_open_interest_pre"] = pd.to_numeric(call_pre.get("open_interest", np.nan), errors="coerce")
    out["atm_volume_pre"] = pd.to_numeric(call_pre.get("volume", np.nan), errors="coerce")

    # Chain-level put/call volume ratio
    out["chain_pc_volume_ratio"] = _chain_pc_ratio(pre_chain)

    return out


def _term_structure_slope(
    chain: pd.DataFrame, spot: float, near_dte: float | None = None
) -> float:
    """Far-expiry ATM IV minus near-expiry ATM IV (positive = contango)."""
    if "expiration" not in chain.columns or "implied_volatility" not in chain.columns:
        return np.nan

    df = chain.copy()
    df["dte"] = (pd.to_datetime(df["expiration"], errors="coerce")
                 - pd.to_datetime(df["fetch_date"], errors="coerce")).dt.days
    df["iv"] = pd.to_numeric(df["implied_volatility"], errors="coerce")
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["dist"] = (df["strike"] - spot).abs()

    if df.empty or df["iv"].isna().all():
        return np.nan

    # ATM only: take the strike closest to spot per (expiration, type) and average call+put
    df["dist_rank"] = df.groupby(["expiration", "type"])["dist"].rank(method="first")
    atm = df[df["dist_rank"] == 1]

    iv_by_exp = atm.groupby(["expiration", "dte"])["iv"].mean().reset_index()
    iv_by_exp = iv_by_exp[iv_by_exp["dte"] > 0].sort_values("dte")

    if len(iv_by_exp) < 2:
        return np.nan

    near = iv_by_exp.iloc[0]
    # Far = first expiry with DTE >= near + 30, else use the last expiry
    far_candidates = iv_by_exp[iv_by_exp["dte"] >= near["dte"] + 30]
    far = far_candidates.iloc[0] if not far_candidates.empty else iv_by_exp.iloc[-1]

    return float(far["iv"] - near["iv"])


def _chain_pc_ratio(chain: pd.DataFrame) -> float:
    """Total put volume / total call volume across the whole chain."""
    if "type" not in chain.columns or "volume" not in chain.columns:
        return np.nan
    df = chain.copy()
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    types = df["type"].astype(str).str.lower()
    call_v = df.loc[types == "call", "volume"].sum()
    put_v = df.loc[types == "put", "volume"].sum()
    return float(put_v / call_v) if call_v > 0 else np.nan


def add_options_features(
    events: pd.DataFrame,
    cache_dir: Path = CACHE_DIR,
    progress: bool = False,
) -> pd.DataFrame:
    """Compute Block-B features for every event row. Returns a new frame."""
    rows = []
    for i, ev in enumerate(events.itertuples(index=False), 1):
        feats = compute_options_features(
            str(ev.ticker), pd.Timestamp(ev.announcement_date), cache_dir
        )
        rows.append(feats)
        if progress and i % 50 == 0:
            print(f"  options features: {i}/{len(events)}")
    out = pd.concat([events.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    return out


# ---------------------------------------------------------------------------
# Block B+: hold-to-expiration close prices
# Joined separately because daily prices come from a different cache
# (TIME_SERIES_DAILY per ticker) than the per-event option chains.
# ---------------------------------------------------------------------------

def add_expiry_intrinsic(
    events: pd.DataFrame,
    daily_cache_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Join the underlying's close on each contract's expiration date and
    compute exit_intrinsic_at_expiry for the hold-to-expiry backtest.

    Requires: events must have `ticker`, `atm_strike_pre`, `atm_expiration_pre`.
    Reads from `unified_strategy/cache/daily_prices/{TICKER}.json` (one
    Alpha Vantage TIME_SERIES_DAILY response per ticker).

    NaN where:
      - daily price file missing
      - expiration date is missing or post-event (after our data window)
      - market was closed on the expiration date and no nearby trading day
    """
    from .fetch_daily_prices import DAILY_CACHE_DIR, load_daily_prices

    if daily_cache_dir is None:
        daily_cache_dir = DAILY_CACHE_DIR

    out = events.copy()
    expiry_close = []
    for ticker, exp in zip(out["ticker"], out["atm_expiration_pre"]):
        if pd.isna(exp) or pd.isna(ticker):
            expiry_close.append(np.nan)
            continue
        prices = load_daily_prices(str(ticker), daily_cache_dir)
        if prices is None or prices.empty:
            expiry_close.append(np.nan)
            continue
        exp_dt = pd.Timestamp(exp).normalize()
        # Try exact match first; if expiration is on a non-trading day
        # (rare — most options expire on a Friday) take the prior trading day.
        if exp_dt in prices.index:
            expiry_close.append(float(prices.loc[exp_dt]))
        else:
            prior = prices.index[prices.index <= exp_dt]
            expiry_close.append(float(prices.loc[prior[-1]]) if len(prior) else np.nan)

    out["expiry_close"] = expiry_close
    out["exit_intrinsic_at_expiry"] = (
        out["expiry_close"] - out["atm_strike_pre"]
    ).abs().clip(lower=0)
    return out


# ---------------------------------------------------------------------------
# Block C: SPY market-regime context
# ---------------------------------------------------------------------------

def add_spy_context(
    events: pd.DataFrame,
    spy_path: Path = SPY_FEATURES_PATH,
    date_col: str = "announcement_date",
) -> pd.DataFrame:
    """
    Join Joshua's SPY daily features onto each event using merge_asof
    (most recent SPY trading day STRICTLY BEFORE announcement). Robust
    to US market holidays.
    """
    spy = pd.read_csv(spy_path, parse_dates=["date"])
    spy = spy.sort_values("date").rename(columns={"date": "spy_join_date"})

    # Prefix every SPY column except the join key for clarity in the merged frame
    spy = spy.add_prefix("spy_").rename(columns={"spy_spy_join_date": "spy_join_date"})

    events_sorted = events.sort_values(date_col).reset_index(drop=True)
    merged = pd.merge_asof(
        events_sorted,
        spy,
        left_on=date_col,
        right_on="spy_join_date",
        direction="backward",
        allow_exact_matches=False,
    )
    return merged
