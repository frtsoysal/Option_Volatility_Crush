"""
vol_crush_utils.py — Utility module for the NVDA Volatility Crush Pilot
=========================================================================
Sections:
  0. Visualization Theme
  1. API & Data Fetching
  2. Implied Volatility Computation
  3. Feature Engineering Helpers
  4. Strategy & Evaluation Metrics
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from scipy.optimize import brentq
from scipy.stats import norm

# ═══════════════════════════════════════════════════════════════════════════════
# Section 0 — Visualization Theme
# ═══════════════════════════════════════════════════════════════════════════════

THEME = {
    "primary": "#1B2A4A",
    "secondary": "#E8792B",
    "tertiary": "#2E86AB",
    "positive": "#2A9D8F",
    "negative": "#E63946",
    "neutral": "#8D99AE",
    "bg": "#FAFAFA",
    "text": "#2B2D42",
    "palette": ["#1B2A4A", "#E8792B", "#2E86AB", "#6C757D", "#2A9D8F"],
}


def setup_theme():
    """Apply the consulting-grade theme globally."""
    sns.set_theme(style="whitegrid", palette=THEME["palette"])
    plt.rcParams.update(
        {
            "figure.facecolor": THEME["bg"],
            "axes.facecolor": THEME["bg"],
            "axes.edgecolor": THEME["neutral"],
            "axes.labelcolor": THEME["text"],
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.titlepad": 12,
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.color": "#E0E0E0",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.7,
            "xtick.color": THEME["text"],
            "ytick.color": THEME["text"],
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "text.color": THEME["text"],
            "font.family": "sans-serif",
            "font.size": 11,
            "legend.fontsize": 10,
            "legend.framealpha": 0.9,
            "legend.edgecolor": THEME["neutral"],
            "figure.figsize": (10, 6),
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def style_axis(ax, title=None, xlabel=None, ylabel=None):
    """Remove top/right spines and set labels."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.5, alpha=0.7)
    ax.xaxis.grid(False)
    if title:
        ax.set_title(title, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — API & Data Fetching
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_dedup_earnings(csv_path: str | Path) -> pd.DataFrame:
    """Load raw earnings CSV and deduplicate on (date, announcement_date)."""
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=["date", "announcement_date"], keep="first")
    df = df.sort_values("announcement_date").reset_index(drop=True)
    return df


def get_trading_day_offset(date_str: str, offset: int) -> str:
    """Shift a date by N business days. Positive = forward, negative = back."""
    dt = pd.Timestamp(date_str)
    if offset >= 0:
        dates = pd.bdate_range(start=dt, periods=offset + 1)
        return str(dates[-1].date())
    else:
        dates = pd.bdate_range(end=dt, periods=abs(offset) + 1)
        return str(dates[0].date())


def fetch_historical_options(
    symbol: str,
    date: str,
    api_key: str,
    cache_dir: Path,
    delay: float = 1.0,
) -> pd.DataFrame | None:
    """
    Fetch historical option chain from Alpha Vantage HISTORICAL_OPTIONS.
    Caches per-date JSON in cache_dir/{symbol}/{date}.json.
    Returns a DataFrame or None on failure.
    """
    cache_path = cache_dir / symbol / f"{date}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        with open(cache_path, "r") as f:
            data = json.load(f)
    else:
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=HISTORICAL_OPTIONS&symbol={symbol}&date={date}"
            f"&apikey={api_key}"
        )
        resp = requests.get(url, timeout=30)
        data = resp.json()
        with open(cache_path, "w") as f:
            json.dump(data, f)
        time.sleep(delay)

    if "data" not in data or len(data["data"]) == 0:
        return None

    df = pd.DataFrame(data["data"])
    df["fetch_date"] = date
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — Implied Volatility Computation
# ═══════════════════════════════════════════════════════════════════════════════

def black_scholes_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> float:
    """Standard Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if option_type == "call" else max(0, K - S)
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
) -> float | None:
    """Compute IV by inverting Black-Scholes using Brent's method."""
    if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None
    intrinsic = max(0, S - K) if option_type == "call" else max(0, K - S)
    if market_price < intrinsic:
        return None
    try:
        iv = brentq(
            lambda sigma: black_scholes_price(S, K, T, r, sigma, option_type)
            - market_price,
            0.01,
            10.0,
            xtol=1e-6,
            maxiter=200,
        )
        return iv
    except (ValueError, RuntimeError):
        return None


def estimate_stock_price_from_chain(chain_df: pd.DataFrame) -> float | None:
    """
    Estimate the underlying stock price from option chain using put-call parity.
    Finds the strike where |call_mid - put_mid| is minimized (ATM strike ≈ stock price).
    """
    df = chain_df.copy()
    for col in ["strike", "bid", "ask", "last", "mark"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    calls = df[df["type"].str.lower() == "call"].copy()
    puts = df[df["type"].str.lower() == "put"].copy()

    if calls.empty or puts.empty:
        return None

    calls["mid"] = calls.apply(_mid_price, axis=1)
    puts["mid"] = puts.apply(_mid_price, axis=1)

    calls_by_strike = calls.groupby("strike")["mid"].first()
    puts_by_strike = puts.groupby("strike")["mid"].first()

    common = calls_by_strike.index.intersection(puts_by_strike.index)
    if len(common) == 0:
        return None

    diffs = (calls_by_strike[common] - puts_by_strike[common]).abs()
    atm_strike = diffs.idxmin()
    return float(atm_strike)


def extract_atm_options(chain_df: pd.DataFrame, stock_price: float) -> dict | None:
    """
    Find the nearest-strike ATM call and put from an option chain DataFrame.
    Filters to the nearest weekly/monthly expiration for tighter pricing.
    """
    if chain_df is None or chain_df.empty:
        return None

    df = chain_df.copy()
    for col in ["strike", "last", "bid", "ask", "mark"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["strike"])

    # Pick the nearest expiration with both calls and puts
    if "expiration" in df.columns:
        df["expiration_dt"] = pd.to_datetime(df["expiration"], errors="coerce")
        df["fetch_dt"] = pd.to_datetime(df["fetch_date"], errors="coerce")
        df["dte"] = (df["expiration_dt"] - df["fetch_dt"]).dt.days
        # Target 2-30 DTE for earnings-week expirations
        near = df[(df["dte"] >= 1) & (df["dte"] <= 30)]
        if near.empty:
            near = df[df["dte"] >= 1]
        if not near.empty:
            best_exp = near.groupby("expiration")["dte"].first().idxmin()
            df = df[df["expiration"] == best_exp]

    df["dist"] = (df["strike"] - stock_price).abs()

    calls = df[df["type"].str.lower() == "call"]
    puts = df[df["type"].str.lower() == "put"]

    if calls.empty or puts.empty:
        return None

    atm_call = calls.loc[calls["dist"].idxmin()]
    atm_put = puts.loc[puts["dist"].idxmin()]

    return {"call": atm_call, "put": atm_put}


def compute_straddle_metrics(
    atm_call: pd.Series, atm_put: pd.Series, stock_price: float
) -> dict:
    """Compute straddle price and expected move from ATM options."""
    call_mid = _mid_price(atm_call)
    put_mid = _mid_price(atm_put)
    straddle_price = call_mid + put_mid
    straddle_pct = (straddle_price / stock_price) * 100 if stock_price > 0 else np.nan
    return {
        "call_price": call_mid,
        "put_price": put_mid,
        "straddle_price": straddle_price,
        "straddle_pct": straddle_pct,
    }


def _mid_price(option_row: pd.Series) -> float:
    """Get mid price from bid/ask, falling back to last."""
    bid = pd.to_numeric(option_row.get("bid", np.nan), errors="coerce")
    ask = pd.to_numeric(option_row.get("ask", np.nan), errors="coerce")
    last = pd.to_numeric(option_row.get("last", np.nan), errors="coerce")
    if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0:
        return (bid + ask) / 2
    if pd.notna(last) and last > 0:
        return last
    return np.nan


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — Feature Engineering Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def compute_realized_vol(price_series: pd.Series, window: int) -> pd.Series:
    """Annualized realized volatility from log returns over a rolling window."""
    log_ret = np.log(price_series / price_series.shift(1))
    return log_ret.rolling(window, min_periods=window).std() * np.sqrt(252)


def build_vol_features(
    events_df: pd.DataFrame, price_history: pd.DataFrame
) -> pd.DataFrame:
    """
    For each event (by announcement_date), compute realized-vol features
    from daily price history.

    price_history must have columns: Date, Close (or Adj Close).
    """
    ph = price_history.copy()
    ph["Date"] = pd.to_datetime(ph["Date"])
    ph = ph.sort_values("Date").set_index("Date")

    close_col = "Adj Close" if "Adj Close" in ph.columns else "Close"
    close = ph[close_col]

    rv5 = compute_realized_vol(close, 5)
    rv21 = compute_realized_vol(close, 21)
    rv63 = compute_realized_vol(close, 63)
    log_ret = np.log(close / close.shift(1))
    vol_of_vol = log_ret.rolling(21).std().rolling(21).std() * np.sqrt(252)

    rows = []
    for _, ev in events_df.iterrows():
        ann_date = pd.Timestamp(ev["announcement_date"])
        pre_dates = ph.index[ph.index < ann_date]
        if len(pre_dates) == 0:
            rows.append({})
            continue

        pre_date = pre_dates[-1]

        post_dates = ph.index[ph.index > ann_date]
        post_date = post_dates[0] if len(post_dates) > 0 else None

        pre_close = close.get(pre_date, np.nan)
        post_close = close.get(post_date, np.nan) if post_date else np.nan
        actual_move_pct = (
            abs(post_close / pre_close - 1) * 100
            if pd.notna(pre_close) and pd.notna(post_close) and pre_close > 0
            else np.nan
        )

        past_moves = []
        for prev_ann in events_df["announcement_date"]:
            if pd.Timestamp(prev_ann) < ann_date:
                prev_ts = pd.Timestamp(prev_ann)
                prev_pre = ph.index[ph.index < prev_ts]
                prev_post = ph.index[ph.index > prev_ts]
                if len(prev_pre) > 0 and len(prev_post) > 0:
                    pp = close.get(prev_pre[-1], np.nan)
                    po = close.get(prev_post[0], np.nan)
                    if pd.notna(pp) and pd.notna(po) and pp > 0:
                        past_moves.append(abs(po / pp - 1) * 100)
        historical_move_avg = np.mean(past_moves) if past_moves else np.nan

        rows.append(
            {
                "date": ev["date"],
                "announcement_date": ev["announcement_date"],
                "stock_price_pre": pre_close,
                "stock_price_post": post_close,
                "actual_move_pct": actual_move_pct,
                "realized_vol_5d": rv5.get(pre_date, np.nan),
                "realized_vol_21d": rv21.get(pre_date, np.nan),
                "realized_vol_63d": rv63.get(pre_date, np.nan),
                "vol_of_vol": vol_of_vol.get(pre_date, np.nan),
                "vol_expansion_ratio": (
                    rv5.get(pre_date, np.nan) / rv21.get(pre_date, np.nan)
                    if pd.notna(rv21.get(pre_date, np.nan))
                    and rv21.get(pre_date, np.nan) > 0
                    else np.nan
                ),
                "historical_move_avg": historical_move_avg,
            }
        )

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — Strategy & Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def sharpe_ratio(
    returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252
) -> float:
    excess = returns - risk_free / periods_per_year
    if excess.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std())


def max_drawdown(equity_curve: pd.Series) -> float:
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return float(drawdown.min())


def calmar_ratio(
    returns: pd.Series, periods_per_year: int = 252
) -> float:
    ann_return = returns.mean() * periods_per_year
    mdd = abs(max_drawdown((1 + returns).cumprod()))
    if mdd == 0:
        return 0.0
    return float(ann_return / mdd)


def profit_factor(pnls: pd.Series) -> float:
    gains = pnls[pnls > 0].sum()
    losses = abs(pnls[pnls < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def win_rate(pnls: pd.Series) -> float:
    if len(pnls) == 0:
        return 0.0
    return float((pnls > 0).sum() / len(pnls))
