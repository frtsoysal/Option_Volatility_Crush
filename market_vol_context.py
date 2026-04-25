"""
market_vol_context.py — Market-Wide Volatility Context Features

This module adds regime-aware, market-level volatility features to the
Option Volatility Crush strategy pipeline. It provides two categories
of features:

1. VIX-Based Features (7 features)
   - Raw VIX and VIX3M levels
   - Term structure ratio and regime detection (contango vs backwardation)
   - VIX percentile rank over trailing year
   - Short-term and medium-term VIX momentum

2. Sector-Level IV Features (8 features)
   - ATM implied volatility of sector ETFs (call, put, average)
   - Sector straddle pricing as % of ETF price
   - Stock-vs-sector relative IV metrics (spread, ratio, binary flag)

All features use only data available before each earnings event.
No forward-looking data leakage.

Requirements:
    numpy, pandas, scipy, yfinance, requests

Usage:
    from market_vol_context import add_market_wide_vol_features
    df = add_market_wide_vol_features(earnings_df, api_key='YOUR_KEY')

Tests:
    python market_vol_context.py
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import brentq
from scipy.stats import norm


# =============================================================================
# CONFIGURATION
# =============================================================================

# API Keys — set via environment variables or hardcode for development
ALPHA_VANTAGE_API_KEY = (
    os.environ.get('ALPHA_VANTAGE_API')
    or os.environ.get('ALPHA_VANTAGE_API_KEY')
    or os.environ.get('AV_API_KEY')
    or 'YOUR_KEY_HERE'
)
FMP_API_KEY = os.environ.get('FMP_API_KEY', 'YOUR_KEY_HERE')

# Data Paths
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_REPO_ROOT, 'data')
VOL_CRUSH_DIR = os.path.join(DATA_DIR, 'vol_crush')
VIX_CACHE_DIR = os.path.join(VOL_CRUSH_DIR, 'vix_cache')
SECTOR_IV_CACHE_DIR = os.path.join(VOL_CRUSH_DIR, 'sector_iv_cache')
ML_READY_DIR = os.path.join(DATA_DIR, 'ml_ready')

# Pilot data (NVDA single-stock proof-of-concept)
PILOT_DIR = os.path.join(_REPO_ROOT, 'option_volatility_crush.ipynb')
PILOT_DATA_DIR = os.path.join(PILOT_DIR, 'pilot_data')

# Pilot Data Column Names
PILOT_STOCK_IV_COL = 'iv_avg_pre'
PILOT_STOCK_STRADDLE_COL = 'straddle_pct_pre'
PILOT_DATE_COL = 'announcement_date'

# Sector ETF Mapping — GICS sectors to liquid sector ETFs
SECTOR_ETF_MAP = {
    'Technology':              'XLK',
    'Information Technology':  'XLK',
    'Communication Services':  'XLC',
    'Consumer Discretionary':  'XLY',
    'Consumer Staples':        'XLP',
    'Energy':                  'XLE',
    'Financials':              'XLF',
    'Health Care':             'XLV',
    'Healthcare':              'XLV',
    'Industrials':             'XLI',
    'Materials':               'XLB',
    'Real Estate':             'XLRE',
    'Utilities':               'XLU',
}

# Sub-sector ETFs for more granular IV context
SUBSECTOR_ETF_MAP = {
    'Semiconductors':          'SMH',
    'Semiconductor':           'SMH',
    'Biotech':                 'XBI',
    'Biotechnology':           'XBI',
    'Retail':                  'XRT',
    'Homebuilders':            'XHB',
    'Banks':                   'KBE',
    'Regional Banks':          'KRE',
    'Oil & Gas E&P':           'XOP',
    'Oil & Gas':               'XOP',
    'Software':                'IGV',
    'Internet':                'FDN',
}


def get_sector_etf(sector=None, sub_sector=None):
    """Resolve a sector/sub-sector label to the best sector ETF ticker."""
    if sub_sector and sub_sector in SUBSECTOR_ETF_MAP:
        return SUBSECTOR_ETF_MAP[sub_sector]
    if sector and sector in SECTOR_ETF_MAP:
        return SECTOR_ETF_MAP[sector]
    return None


# VIX Configuration
VIX_TICKER = '^VIX'
VIX3M_TICKER = '^VIX3M'
VIX_LOOKBACK_START = '2016-01-01'
VIX_PERCENTILE_WINDOW_DAYS = 365
VIX_SHORT_MOMENTUM_DAYS = 5
VIX_MEDIUM_MOMENTUM_DAYS = 21
VIX_BACKWARDATION_THRESHOLD = 1.0

# Sector IV Configuration
SECTOR_IV_MIN_DTE = 2
SECTOR_IV_MAX_DTE = 45
STRADDLE_PCT_MULTIPLIER = 100
DEFAULT_RISK_FREE_RATE = 0.05
AV_REQUEST_DELAY_SECONDS = 0.5
AV_FREE_TIER_DAILY_LIMIT = 25
AV_PREMIUM_CALLS_PER_MINUTE = 75

# Relative IV Feature Thresholds
IV_ELEVATED_VS_SECTOR_THRESHOLD = 1.5

# Feature Names
VIX_FEATURE_NAMES = [
    'vix_level', 'vix3m_level', 'vix_term_structure_ratio',
    'vix_term_structure_regime', 'vix_percentile_252d',
    'vix_change_5d', 'vix_change_21d',
]

SECTOR_IV_RAW_FEATURE_NAMES = [
    'sector_etf', 'sector_atm_iv_call', 'sector_atm_iv_put',
    'sector_atm_iv_avg', 'sector_straddle_pct',
]

SECTOR_IV_RELATIVE_FEATURE_NAMES = [
    'iv_stock_minus_sector', 'iv_stock_sector_ratio',
    'straddle_stock_minus_sector', 'iv_elevated_vs_sector',
]

ALL_MARKET_VOL_FEATURE_NAMES = (
    VIX_FEATURE_NAMES + SECTOR_IV_RAW_FEATURE_NAMES + SECTOR_IV_RELATIVE_FEATURE_NAMES
)

# Validation Ranges
FEATURE_VALIDATION_RANGES = {
    'vix_level':                (5, 90),
    'vix3m_level':              (5, 70),
    'vix_term_structure_ratio': (0.5, 2.5),
    'vix_percentile_252d':      (0.0, 1.0),
    'vix_change_5d':            (-0.8, 3.0),
    'vix_change_21d':           (-0.8, 5.0),
    'sector_atm_iv_avg':        (0.01, 2.0),
    'iv_stock_sector_ratio':    (0.1, 50.0),
}


# =============================================================================
# PART A: VIX DATA FETCHING
# =============================================================================

def fetch_vix_data(start_date=None, end_date=None, cache=True):
    """
    Fetch VIX and VIX3M daily close prices from Yahoo Finance.

    Uses file-based caching to avoid re-downloading on every run.
    VIX data is free and doesn't consume any Alpha Vantage API quota.

    Parameters:
        start_date: str 'YYYY-MM-DD', defaults to VIX_LOOKBACK_START
        end_date: str 'YYYY-MM-DD', defaults to today
        cache: bool, if True, saves/loads from disk cache

    Returns:
        pd.DataFrame indexed by date with columns:
            - vix_close (float): VIX daily close
            - vix3m_close (float): VIX3M daily close
    """
    start_date = start_date or VIX_LOOKBACK_START
    end_date = end_date or datetime.now().strftime('%Y-%m-%d')

    # Check cache
    if cache:
        os.makedirs(VIX_CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(VIX_CACHE_DIR, 'vix_daily.csv')
        if os.path.exists(cache_file):
            cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if (cached.index.min() <= pd.Timestamp(start_date) and
                    cached.index.max() >= pd.Timestamp(end_date) - pd.Timedelta(days=5)):
                print(f"  Loaded VIX data from cache ({len(cached)} rows)")
                return cached

    print(f"  Downloading VIX data from Yahoo Finance ({start_date} to {end_date})...")

    vix_raw = yf.download(VIX_TICKER, start=start_date, end=end_date, progress=False)
    vix3m_raw = yf.download(VIX3M_TICKER, start=start_date, end=end_date, progress=False)

    if vix_raw.empty:
        raise ValueError(f"Failed to download {VIX_TICKER} data from Yahoo Finance")

    def _extract_close(df, ticker_label):
        if df.empty:
            return pd.Series(dtype=float)
        cols = df.columns
        if isinstance(cols, pd.MultiIndex):
            if 'Close' in cols.get_level_values(0):
                return df['Close'].iloc[:, 0] if df['Close'].ndim > 1 else df['Close']
        if 'Close' in cols:
            s = df['Close']
            return s.iloc[:, 0] if hasattr(s, 'iloc') and s.ndim > 1 else s
        return df.iloc[:, 0]

    vix_close = _extract_close(vix_raw, VIX_TICKER).rename('vix_close')
    vix3m_close = _extract_close(vix3m_raw, VIX3M_TICKER).rename('vix3m_close')

    combined = pd.DataFrame(vix_close).join(pd.DataFrame(vix3m_close), how='outer')
    combined = combined.ffill()
    combined = combined.dropna(subset=['vix_close'])
    combined['vix_close'] = pd.to_numeric(combined['vix_close'], errors='coerce')
    combined['vix3m_close'] = pd.to_numeric(combined['vix3m_close'], errors='coerce')

    print(f"  Downloaded {len(combined)} rows of VIX data "
          f"({combined.index.min().date()} to {combined.index.max().date()})")

    if cache:
        combined.to_csv(cache_file)
        print(f"  Cached VIX data to {cache_file}")

    return combined


# =============================================================================
# PART B: VIX FEATURE COMPUTATION
# =============================================================================

def compute_vix_features(earnings_df, vix_df, date_col='announcement_date'):
    """
    For each earnings event, compute 7 VIX-based market regime features.

    All features use data from the event date or earlier — no leakage.

    Parameters:
        earnings_df: DataFrame containing earnings events.
        vix_df: DataFrame from fetch_vix_data() with vix_close, vix3m_close.
        date_col: name of the date column in earnings_df.

    Returns:
        earnings_df with 7 new VIX feature columns appended.
    """
    print(f"  Computing VIX features for {len(earnings_df)} events...")

    vix_sorted = vix_df.sort_index()
    results = {name: [] for name in VIX_FEATURE_NAMES}

    for idx, row in earnings_df.iterrows():
        event_date = pd.Timestamp(row[date_col])
        mask = vix_sorted.index <= event_date
        available = vix_sorted.loc[mask]

        if len(available) == 0:
            for name in VIX_FEATURE_NAMES:
                results[name].append(np.nan)
            continue

        latest = available.iloc[-1]
        latest_date = available.index[-1]
        vix_level = float(latest['vix_close'])
        vix3m_level = float(latest['vix3m_close']) if pd.notna(latest['vix3m_close']) else np.nan

        results['vix_level'].append(vix_level)
        results['vix3m_level'].append(vix3m_level)

        if pd.notna(vix3m_level) and vix3m_level > 0:
            ts_ratio = vix_level / vix3m_level
        else:
            ts_ratio = np.nan
        results['vix_term_structure_ratio'].append(ts_ratio)

        if pd.notna(ts_ratio):
            regime = 1 if ts_ratio > VIX_BACKWARDATION_THRESHOLD else 0
        else:
            regime = np.nan
        results['vix_term_structure_regime'].append(regime)

        lookback_start = latest_date - pd.Timedelta(days=VIX_PERCENTILE_WINDOW_DAYS)
        trailing = available.loc[available.index >= lookback_start, 'vix_close']

        if len(trailing) >= 50:
            pctl = float((trailing < vix_level).sum()) / len(trailing)
        else:
            pctl = np.nan
        results['vix_percentile_252d'].append(pctl)

        n_short = VIX_SHORT_MOMENTUM_DAYS + 1
        if len(available) >= n_short:
            vix_prev = float(available.iloc[-n_short]['vix_close'])
            change_5d = (vix_level - vix_prev) / vix_prev if vix_prev > 0 else np.nan
        else:
            change_5d = np.nan
        results['vix_change_5d'].append(change_5d)

        n_med = VIX_MEDIUM_MOMENTUM_DAYS + 1
        if len(available) >= n_med:
            vix_prev_21 = float(available.iloc[-n_med]['vix_close'])
            change_21d = (vix_level - vix_prev_21) / vix_prev_21 if vix_prev_21 > 0 else np.nan
        else:
            change_21d = np.nan
        results['vix_change_21d'].append(change_21d)

    vix_features = pd.DataFrame(results, index=earnings_df.index)

    print(f"  VIX features computed. Coverage: "
          f"{vix_features['vix_level'].notna().sum()}/{len(earnings_df)} events")

    return pd.concat([earnings_df, vix_features], axis=1)


# =============================================================================
# PART C: BLACK-SCHOLES IV INVERSION
# =============================================================================

def implied_vol(price, S, K, T, r, option_type='call'):
    """
    Compute implied volatility via Black-Scholes inversion using Brent's method.

    Parameters:
        price: observed market price of the option (mid-price preferred)
        S: current underlying price
        K: strike price
        T: time to expiration in years (must be > 0)
        r: annualized risk-free rate
        option_type: 'call' or 'put'

    Returns:
        float: annualized implied volatility, or np.nan if solver fails
    """
    if price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return np.nan

    def bs_price(sigma):
        d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    if option_type == 'call':
        intrinsic = max(S - K * np.exp(-r * T), 0)
    else:
        intrinsic = max(K * np.exp(-r * T) - S, 0)

    if price <= intrinsic:
        return np.nan

    try:
        return brentq(lambda s: bs_price(s) - price, 0.01, 5.0, xtol=1e-6)
    except (ValueError, RuntimeError):
        return np.nan


# =============================================================================
# PART D: SECTOR ETF OPTION CHAIN FETCHING
# =============================================================================

def fetch_option_chain_av(symbol, date_str, api_key):
    """
    Fetch a historical option chain snapshot from Alpha Vantage.

    Parameters:
        symbol: ticker string (e.g., 'XLK', 'SMH')
        date_str: date string 'YYYY-MM-DD'
        api_key: Alpha Vantage API key

    Returns:
        list of option contract dicts, or empty list on failure
    """
    import requests

    url = (
        f'https://www.alphavantage.co/query'
        f'?function=HISTORICAL_OPTIONS'
        f'&symbol={symbol}'
        f'&date={date_str}'
        f'&apikey={api_key}'
    )

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if 'Error Message' in data or 'Note' in data:
            note = data.get('Error Message') or data.get('Note', '')
            if 'rate limit' in note.lower() or 'call frequency' in note.lower():
                warnings.warn(f"Rate limited on {symbol} {date_str}: {note}")
            return []

        return data.get('data', [])

    except Exception as e:
        warnings.warn(f"Error fetching option chain for {symbol} on {date_str}: {e}")
        return []


def _get_etf_price(symbol, date_str):
    """Get the closing price of an ETF on a specific date via Yahoo Finance."""
    target = pd.Timestamp(date_str)
    start = target - pd.Timedelta(days=10)

    try:
        data = yf.download(
            symbol,
            start=start.strftime('%Y-%m-%d'),
            end=(target + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
            progress=False
        )
        if data.empty:
            return np.nan

        if isinstance(data.columns, pd.MultiIndex):
            close_col = data['Close']
            if close_col.ndim > 1:
                close_col = close_col.iloc[:, 0]
        else:
            close_col = data['Close']

        valid = close_col.loc[close_col.index <= target]
        if valid.empty:
            return np.nan

        return float(valid.iloc[-1])

    except Exception:
        return np.nan


_etf_price_cache = {}


def get_etf_price_cached(symbol, date_str):
    """Cached wrapper around _get_etf_price."""
    cache_key = f'{symbol}_{date_str}'
    if cache_key not in _etf_price_cache:
        _etf_price_cache[cache_key] = _get_etf_price(symbol, date_str)
    return _etf_price_cache[cache_key]


# =============================================================================
# PART E: SECTOR ETF ATM IV COMPUTATION
# =============================================================================

def compute_sector_etf_iv(chain, etf_price, chain_date_str,
                          risk_free_rate=DEFAULT_RISK_FREE_RATE):
    """
    From a raw option chain, find ATM options and compute implied volatility.

    Parameters:
        chain: list of contract dicts from Alpha Vantage HISTORICAL_OPTIONS
        etf_price: current price of the sector ETF
        chain_date_str: 'YYYY-MM-DD' date the chain was fetched for
        risk_free_rate: annualized risk-free rate for BS inversion

    Returns:
        dict with keys:
            sector_atm_iv_call, sector_atm_iv_put, sector_atm_iv_avg,
            sector_straddle_pct
    """
    empty_result = {
        'sector_atm_iv_call':  np.nan,
        'sector_atm_iv_put':   np.nan,
        'sector_atm_iv_avg':   np.nan,
        'sector_straddle_pct': np.nan,
    }

    if not chain or not etf_price or np.isnan(etf_price) or etf_price <= 0:
        return empty_result

    chain_date = datetime.strptime(chain_date_str, '%Y-%m-%d')

    valid_contracts = []
    for c in chain:
        try:
            exp = datetime.strptime(c['expiration'], '%Y-%m-%d')
            dte = (exp - chain_date).days
            if SECTOR_IV_MIN_DTE <= dte <= SECTOR_IV_MAX_DTE:
                valid_contracts.append(c)
        except (KeyError, ValueError):
            continue

    if not valid_contracts:
        return empty_result

    strikes = set()
    for c in valid_contracts:
        try:
            strikes.add(float(c['strike']))
        except (KeyError, ValueError):
            continue

    if not strikes:
        return empty_result

    atm_strike = min(strikes, key=lambda k: abs(k - etf_price))

    atm_call_price = np.nan
    atm_put_price = np.nan
    expiration_str = None

    for c in valid_contracts:
        try:
            if float(c['strike']) != atm_strike:
                continue

            bid = float(c.get('bid', 0) or 0)
            ask = float(c.get('ask', 0) or 0)
            if bid > 0 and ask > 0:
                mid = (bid + ask) / 2
            else:
                mid = float(c.get('last', 0) or 0)

            if mid <= 0:
                continue

            contract_type = c.get('type', '').lower()
            if contract_type == 'call':
                atm_call_price = mid
                expiration_str = c.get('expiration')
            elif contract_type == 'put':
                atm_put_price = mid
                expiration_str = expiration_str or c.get('expiration')

        except (KeyError, ValueError):
            continue

    if expiration_str:
        exp_dt = datetime.strptime(expiration_str, '%Y-%m-%d')
        T = max((exp_dt - chain_date).days / 365.0, 1 / 365)
    else:
        T = 30 / 365

    iv_call = np.nan
    iv_put = np.nan

    if not np.isnan(atm_call_price) and atm_call_price > 0:
        iv_call = implied_vol(
            atm_call_price, etf_price, atm_strike, T, risk_free_rate, 'call'
        )

    if not np.isnan(atm_put_price) and atm_put_price > 0:
        iv_put = implied_vol(
            atm_put_price, etf_price, atm_strike, T, risk_free_rate, 'put'
        )

    ivs = [v for v in [iv_call, iv_put] if not np.isnan(v)]
    iv_avg = float(np.mean(ivs)) if ivs else np.nan

    straddle = 0
    n_legs = 0
    if not np.isnan(atm_call_price) and atm_call_price > 0:
        straddle += atm_call_price
        n_legs += 1
    if not np.isnan(atm_put_price) and atm_put_price > 0:
        straddle += atm_put_price
        n_legs += 1

    if n_legs == 2:
        straddle_pct = (straddle / etf_price) * STRADDLE_PCT_MULTIPLIER
    elif n_legs == 1:
        straddle_pct = ((straddle * 2) / etf_price) * STRADDLE_PCT_MULTIPLIER
    else:
        straddle_pct = np.nan

    return {
        'sector_atm_iv_call':  iv_call,
        'sector_atm_iv_put':   iv_put,
        'sector_atm_iv_avg':   iv_avg,
        'sector_straddle_pct': straddle_pct,
    }


# =============================================================================
# PART F: SECTOR IV PIPELINE (with caching)
# =============================================================================

def get_sector_iv_for_event(sector, event_date, api_key,
                            sub_sector=None, risk_free_rate=DEFAULT_RISK_FREE_RATE):
    """
    Get sector ETF ATM IV for a given sector and date.

    Uses file-based caching to avoid redundant API calls.
    """
    etf_symbol = get_sector_etf(sector=sector, sub_sector=sub_sector)

    empty_result = {
        'sector_etf':            None,
        'sector_atm_iv_call':    np.nan,
        'sector_atm_iv_put':     np.nan,
        'sector_atm_iv_avg':     np.nan,
        'sector_straddle_pct':   np.nan,
    }

    if not etf_symbol:
        warnings.warn(f"No ETF mapping for sector='{sector}', sub_sector='{sub_sector}'")
        return empty_result

    date_str = pd.Timestamp(event_date).strftime('%Y-%m-%d')

    os.makedirs(SECTOR_IV_CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(SECTOR_IV_CACHE_DIR, f'{etf_symbol}_{date_str}.json')

    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cached = json.load(f)
        for key in cached:
            if cached[key] is None and key != 'sector_etf':
                cached[key] = np.nan
        cached['sector_etf'] = etf_symbol
        return cached

    chain = fetch_option_chain_av(etf_symbol, date_str, api_key)
    etf_price = get_etf_price_cached(etf_symbol, date_str)
    result = compute_sector_etf_iv(chain, etf_price, date_str, risk_free_rate)
    result['sector_etf'] = etf_symbol

    cache_data = {}
    for key, val in result.items():
        if isinstance(val, float) and np.isnan(val):
            cache_data[key] = None
        else:
            cache_data[key] = val

    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)

    time.sleep(AV_REQUEST_DELAY_SECONDS)

    return result


def add_sector_iv_features(earnings_df, api_key,
                           sector_col='sector', sub_sector_col=None,
                           date_col='announcement_date'):
    """Add sector-level IV features to every earnings event."""
    print(f"  Computing sector IV features for {len(earnings_df)} events...")

    unique_pairs = set()
    for _, row in earnings_df.iterrows():
        sector = row.get(sector_col, '')
        sub = row.get(sub_sector_col, '') if sub_sector_col else None
        date = pd.Timestamp(row[date_col]).strftime('%Y-%m-%d')
        etf = get_sector_etf(sector=sector, sub_sector=sub)
        if etf:
            unique_pairs.add((etf, date))

    uncached = 0
    for etf, date in unique_pairs:
        cache_file = os.path.join(SECTOR_IV_CACHE_DIR, f'{etf}_{date}.json')
        if not os.path.exists(cache_file):
            uncached += 1

    print(f"  Unique (ETF, date) pairs: {len(unique_pairs)} "
          f"({uncached} uncached, will need API calls)")

    sector_features = []
    for i, (idx, row) in enumerate(earnings_df.iterrows()):
        sector = row.get(sector_col, '')
        sub = row.get(sub_sector_col, '') if sub_sector_col else None
        event_date = row[date_col]

        result = get_sector_iv_for_event(
            sector=sector, event_date=event_date,
            api_key=api_key, sub_sector=sub,
        )
        sector_features.append(result)

        if (i + 1) % 100 == 0:
            print(f"    Processed {i + 1}/{len(earnings_df)} events...")

    sector_df = pd.DataFrame(sector_features, index=earnings_df.index)

    coverage = sector_df['sector_atm_iv_avg'].notna().sum()
    print(f"  Sector IV features computed. Coverage: {coverage}/{len(earnings_df)} events")

    return pd.concat([earnings_df, sector_df], axis=1)


# =============================================================================
# PART G: RELATIVE IV FEATURES (Stock vs. Sector)
# =============================================================================

def compute_relative_iv_features(df,
                                 stock_iv_col=PILOT_STOCK_IV_COL,
                                 stock_straddle_col=PILOT_STOCK_STRADDLE_COL,
                                 sector_iv_col='sector_atm_iv_avg',
                                 sector_straddle_col='sector_straddle_pct'):
    """
    Compute features that compare stock-level IV to sector-level IV.

    NOTE: Mutates df in-place and also returns it for chaining.
    """
    print("  Computing relative IV features (stock vs. sector)...")

    if stock_iv_col in df.columns and sector_iv_col in df.columns:
        df['iv_stock_minus_sector'] = df[stock_iv_col] - df[sector_iv_col]
    else:
        df['iv_stock_minus_sector'] = np.nan
        warnings.warn(f"Missing columns for IV spread: need '{stock_iv_col}' and '{sector_iv_col}'")

    if stock_iv_col in df.columns and sector_iv_col in df.columns:
        df['iv_stock_sector_ratio'] = np.where(
            df[sector_iv_col] > 0,
            df[stock_iv_col] / df[sector_iv_col],
            np.nan
        )
    else:
        df['iv_stock_sector_ratio'] = np.nan

    if stock_straddle_col in df.columns and sector_straddle_col in df.columns:
        df['straddle_stock_minus_sector'] = (
            df[stock_straddle_col] - df[sector_straddle_col]
        )
    else:
        df['straddle_stock_minus_sector'] = np.nan

    if 'iv_stock_sector_ratio' in df.columns:
        df['iv_elevated_vs_sector'] = np.where(
            df['iv_stock_sector_ratio'].notna(),
            (df['iv_stock_sector_ratio'] > IV_ELEVATED_VS_SECTOR_THRESHOLD).astype(int),
            np.nan
        )
    else:
        df['iv_elevated_vs_sector'] = np.nan

    return df


# =============================================================================
# PART H: MISSING DATA HANDLING
# =============================================================================

def handle_market_vol_missing_data(df, model_type='tree'):
    """
    Handle NaN values in market-wide volatility features.

    Strategy depends on model type:
    - Tree-based models (LightGBM, XGBoost): leave sector IV NaN as-is
    - Linear models (LR, RF): median-impute sector IV columns
    """
    print(f"  Handling missing data (strategy: {model_type})...")

    vix_numeric_cols = [
        'vix_level', 'vix3m_level', 'vix_term_structure_ratio',
        'vix_percentile_252d', 'vix_change_5d', 'vix_change_21d'
    ]
    for col in vix_numeric_cols:
        if col in df.columns:
            before_na = df[col].isna().sum()
            df[col] = df[col].ffill()
            after_na = df[col].isna().sum()
            if before_na > after_na:
                print(f"    {col}: filled {before_na - after_na} NaN via forward-fill")

    if 'vix_term_structure_regime' in df.columns:
        df['vix_term_structure_regime'] = df['vix_term_structure_regime'].fillna(0)

    sector_cols = [
        'sector_atm_iv_call', 'sector_atm_iv_put', 'sector_atm_iv_avg',
        'sector_straddle_pct', 'iv_stock_minus_sector', 'iv_stock_sector_ratio',
        'straddle_stock_minus_sector', 'iv_elevated_vs_sector'
    ]

    if model_type == 'linear':
        for col in sector_cols:
            if col in df.columns:
                before_na = df[col].isna().sum()
                if before_na > 0:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    print(f"    {col}: filled {before_na} NaN with median ({median_val:.4f})")
    else:
        for col in sector_cols:
            if col in df.columns:
                na_count = df[col].isna().sum()
                if na_count > 0:
                    print(f"    {col}: {na_count} NaN (left as-is for tree model)")

    return df


# =============================================================================
# PART I: VALIDATION
# =============================================================================

def validate_market_vol_features(df):
    """Sanity-check the market-wide volatility features."""
    print("\n  === Market-Wide Vol Feature Validation ===\n")

    summary = {}
    for col, (low, high) in FEATURE_VALIDATION_RANGES.items():
        if col not in df.columns:
            print(f"  WARNING: {col} not found in DataFrame")
            summary[col] = {'status': 'missing'}
            continue

        valid = df[col].dropna()
        total = len(df)
        missing = total - len(valid)
        out_low = (valid < low).sum()
        out_high = (valid > high).sum()

        status = 'OK'
        if missing / total > 0.5:
            status = 'HIGH_MISSING'
        elif (out_low + out_high) / max(len(valid), 1) > 0.05:
            status = 'OUT_OF_RANGE'

        summary[col] = {
            'status': status,
            'min': float(valid.min()) if len(valid) > 0 else None,
            'max': float(valid.max()) if len(valid) > 0 else None,
            'mean': float(valid.mean()) if len(valid) > 0 else None,
            'missing': missing,
            'missing_pct': 100 * missing / total,
            'out_of_range': int(out_low + out_high),
        }

        print(f"  {col}:")
        if summary[col]['min'] is not None:
            print(f"    Range:  {summary[col]['min']:.4f} to {summary[col]['max']:.4f}")
        else:
            print(f"    Range: N/A")
        print(f"    Missing: {missing}/{total} ({summary[col]['missing_pct']:.1f}%)")
        print(f"    Out of expected [{low}, {high}]: {summary[col]['out_of_range']}")
        print(f"    Status: {status}")
        print()

    return summary


# =============================================================================
# PART J: MASTER FUNCTION — ties everything together
# =============================================================================

def add_market_wide_vol_features(earnings_df, api_key=None,
                                 date_col='announcement_date',
                                 sector_col='sector',
                                 sub_sector_col=None,
                                 stock_iv_col=PILOT_STOCK_IV_COL,
                                 stock_straddle_col=PILOT_STOCK_STRADDLE_COL,
                                 model_type='tree',
                                 validate=True):
    """
    Master function: adds all market-wide volatility context features.

    Pipeline:
        1. Fetch VIX/VIX3M data (Yahoo Finance, free)
        2. Compute 7 VIX-based regime features
        3. Fetch sector ETF option chains (Alpha Vantage, uses API quota)
        4. Compute 4 raw sector IV features
        5. Compute 4 relative stock-vs-sector features
        6. Handle missing data
        7. Validate feature ranges
    """
    print("=" * 60)
    print("MARKET-WIDE VOLATILITY CONTEXT PIPELINE")
    print("=" * 60)

    n_events = len(earnings_df)
    print(f"\nInput: {n_events} earnings events")

    has_api_key = (api_key and api_key != 'YOUR_KEY_HERE')
    has_sector_col = (sector_col and sector_col in earnings_df.columns)

    if not has_api_key:
        print("\n  NOTE: No API key provided — sector IV steps will be skipped.")
    if not has_sector_col:
        print(f"\n  NOTE: Column '{sector_col}' not found — sector IV steps will be skipped.")

    print("\n[Step 1/6] Fetching VIX data...")
    vix_df = fetch_vix_data()

    print("\n[Step 2/6] Computing VIX features...")
    df = compute_vix_features(earnings_df, vix_df, date_col=date_col)

    if has_api_key and has_sector_col:
        print("\n[Step 3/6] Computing sector IV features...")
        df = add_sector_iv_features(
            df, api_key,
            sector_col=sector_col,
            sub_sector_col=sub_sector_col,
            date_col=date_col
        )

        print("\n[Step 4/6] Computing relative IV features...")
        df = compute_relative_iv_features(
            df,
            stock_iv_col=stock_iv_col,
            stock_straddle_col=stock_straddle_col,
        )
    else:
        print("\n[Step 3/6] Skipping sector IV (no API key or sector column)")
        print("[Step 4/6] Skipping relative IV features")

    print("\n[Step 5/6] Handling missing data...")
    df = handle_market_vol_missing_data(df, model_type=model_type)

    if validate:
        print("\n[Step 6/6] Validating features...")
        validate_market_vol_features(df)
    else:
        print("\n[Step 6/6] Skipping validation (validate=False)")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"Output: {len(df)} events x {len(df.columns)} features")
    print("=" * 60)

    return df


# =============================================================================
# TESTS — run with: python market_vol_context.py
# =============================================================================

def create_sample_earnings():
    """Create a small synthetic earnings DataFrame for testing."""
    data = {
        'symbol': ['NVDA', 'AAPL', 'JPM', 'JNJ', 'XOM',
                    'MSFT', 'AMZN', 'META', 'GOOGL', 'TSLA'],
        'announcement_date': [
            '2024-02-21', '2024-02-01', '2024-01-12', '2024-01-23', '2024-02-02',
            '2024-01-30', '2024-02-01', '2024-02-01', '2024-01-30', '2024-01-24',
        ],
        'sector': [
            'Technology', 'Technology', 'Financials', 'Health Care', 'Energy',
            'Technology', 'Consumer Discretionary', 'Communication Services',
            'Communication Services', 'Consumer Discretionary',
        ],
        'sub_sector': [
            'Semiconductors', 'Technology', 'Banks', 'Healthcare', 'Oil & Gas',
            'Software', 'Internet', 'Internet', 'Internet', 'Consumer Discretionary',
        ],
        'iv_avg_pre': [0.65, 0.32, 0.22, 0.18, 0.28,
                       0.35, 0.42, 0.48, 0.38, 0.72],
        'straddle_pct_pre': [8.0, 4.0, 3.0, 2.0, 3.5,
                             4.5, 5.0, 6.0, 4.8, 9.0],
    }

    df = pd.DataFrame(data)
    df['announcement_date'] = pd.to_datetime(df['announcement_date'])
    return df


def test_sector_etf_mapping():
    """Test that sector-to-ETF mapping works correctly."""
    print("\n" + "=" * 50)
    print("TEST: Sector ETF Mapping")
    print("=" * 50)

    assert get_sector_etf(sector='Technology') == 'XLK'
    assert get_sector_etf(sector='Financials') == 'XLF'
    assert get_sector_etf(sector='Energy') == 'XLE'
    print("  Basic sector mapping: PASS")

    assert get_sector_etf(sector='Technology', sub_sector='Semiconductors') == 'SMH'
    assert get_sector_etf(sector='Financials', sub_sector='Banks') == 'KBE'
    assert get_sector_etf(sector='Health Care', sub_sector='Biotech') == 'XBI'
    print("  Sub-sector override: PASS")

    assert get_sector_etf(sector='Technology', sub_sector='Unknown') == 'XLK'
    print("  Fallback to sector: PASS")

    assert get_sector_etf(sector='Aliens') is None
    print("  Unknown sector returns None: PASS")

    print("\n  ALL MAPPING TESTS PASSED")


def test_vix_fetch_and_features():
    """Test VIX data fetching and feature computation."""
    print("\n" + "=" * 50)
    print("TEST: VIX Data Fetch & Feature Computation")
    print("=" * 50)

    vix_df = fetch_vix_data(start_date='2023-01-01', end_date='2024-12-31')

    assert not vix_df.empty, "VIX data is empty"
    assert 'vix_close' in vix_df.columns, "Missing vix_close column"
    assert 'vix3m_close' in vix_df.columns, "Missing vix3m_close column"
    print(f"\n  Fetched {len(vix_df)} rows of VIX data")
    print(f"  VIX range: {vix_df['vix_close'].min():.2f} to {vix_df['vix_close'].max():.2f}")
    print(f"  VIX3M range: {vix_df['vix3m_close'].min():.2f} to {vix_df['vix3m_close'].max():.2f}")

    sample = create_sample_earnings()
    result = compute_vix_features(sample, vix_df)

    expected_cols = [
        'vix_level', 'vix3m_level', 'vix_term_structure_ratio',
        'vix_term_structure_regime', 'vix_percentile_252d',
        'vix_change_5d', 'vix_change_21d'
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing feature column: {col}"
    print(f"\n  All 7 VIX feature columns present: PASS")

    vix_levels = result['vix_level'].dropna()
    assert (vix_levels > 0).all(), "VIX levels should be positive"
    assert (vix_levels < 100).all(), "VIX levels should be < 100"
    print(f"  VIX levels range: {vix_levels.min():.2f} to {vix_levels.max():.2f}: PASS")

    ts_ratios = result['vix_term_structure_ratio'].dropna()
    assert (ts_ratios > 0).all(), "Term structure ratio should be positive"
    print(f"  Term structure ratios: {ts_ratios.min():.3f} to {ts_ratios.max():.3f}: PASS")

    percentiles = result['vix_percentile_252d'].dropna()
    assert (percentiles >= 0).all() and (percentiles <= 1).all(), "Percentile should be [0, 1]"
    print(f"  VIX percentiles: {percentiles.min():.3f} to {percentiles.max():.3f}: PASS")

    print("\n  ALL VIX TESTS PASSED")
    return result


def test_relative_iv_features():
    """Test relative IV feature computation."""
    print("\n" + "=" * 50)
    print("TEST: Relative IV Features")
    print("=" * 50)

    sample = create_sample_earnings()
    sample['sector_atm_iv_avg'] = [0.18, 0.15, 0.14, 0.12, 0.20,
                                    0.16, 0.19, 0.21, 0.17, 0.25]
    sample['sector_straddle_pct'] = [2.5, 2.0, 1.8, 1.5, 2.8,
                                      2.2, 2.6, 2.9, 2.3, 3.3]

    result = compute_relative_iv_features(sample)

    expected = ['iv_stock_minus_sector', 'iv_stock_sector_ratio',
                'straddle_stock_minus_sector', 'iv_elevated_vs_sector']
    for col in expected:
        assert col in result.columns, f"Missing: {col}"
    print(f"  All 4 relative feature columns present: PASS")

    nvda_ratio = result.iloc[0]['iv_stock_sector_ratio']
    assert 3.5 < nvda_ratio < 3.7, f"NVDA IV ratio should be ~3.6, got {nvda_ratio}"
    print(f"  NVDA iv_stock_sector_ratio = {nvda_ratio:.2f} (expected ~3.61): PASS")

    assert result.iloc[0]['iv_elevated_vs_sector'] == 1
    print(f"  NVDA iv_elevated_vs_sector = 1: PASS")

    jnj_ratio = result.iloc[3]['iv_stock_sector_ratio']
    print(f"  JNJ iv_stock_sector_ratio = {jnj_ratio:.2f}")

    spreads = result['iv_stock_minus_sector'].dropna()
    print(f"  IV spread range: {spreads.min():.3f} to {spreads.max():.3f}")

    print("\n  ALL RELATIVE IV TESTS PASSED")


def test_missing_data_handling():
    """Test missing data imputation strategies."""
    print("\n" + "=" * 50)
    print("TEST: Missing Data Handling")
    print("=" * 50)

    sample = create_sample_earnings()
    sample['vix_level'] = [13.5, np.nan, 14.2, 13.8, np.nan,
                           12.9, 13.1, np.nan, 14.0, 13.7]
    sample['vix_term_structure_regime'] = [0, np.nan, 1, 0, np.nan,
                                            0, 0, np.nan, 1, 0]
    sample['sector_atm_iv_avg'] = [0.18, np.nan, 0.14, np.nan, 0.20,
                                    0.16, np.nan, 0.21, 0.17, np.nan]

    tree_result = handle_market_vol_missing_data(sample.copy(), model_type='tree')
    assert tree_result['vix_level'].iloc[1] == 13.5
    print("  Tree model forward-fill: PASS")

    assert tree_result['vix_term_structure_regime'].iloc[1] == 0
    print("  Regime fill with 0: PASS")

    assert pd.isna(tree_result['sector_atm_iv_avg'].iloc[1])
    print("  Sector IV NaN preserved for tree model: PASS")

    linear_result = handle_market_vol_missing_data(sample.copy(), model_type='linear')
    assert not pd.isna(linear_result['sector_atm_iv_avg'].iloc[1])
    print("  Sector IV median-imputed for linear model: PASS")

    print("\n  ALL MISSING DATA TESTS PASSED")


def test_validation():
    """Test the validation function catches issues."""
    print("\n" + "=" * 50)
    print("TEST: Feature Validation")
    print("=" * 50)

    sample = create_sample_earnings()
    sample['vix_level'] = [13.5, 14.2, 13.8, 95.0, 12.9,
                           13.1, 14.5, 14.0, 13.7, 15.1]
    sample['vix_percentile_252d'] = [0.45, 0.52, 0.61, 0.38, 0.72,
                                      0.55, 0.48, 0.66, 0.59, 0.41]

    summary = validate_market_vol_features(sample)

    assert summary['vix_level']['out_of_range'] >= 1, "Should flag out-of-range VIX"
    print("  Out-of-range detection: PASS")

    print("\n  ALL VALIDATION TESTS PASSED")


def run_full_pipeline_demo():
    """Run the complete pipeline on sample data."""
    print("\n" + "=" * 60)
    print("FULL PIPELINE DEMO")
    print("=" * 60)

    sample = create_sample_earnings()
    print(f"\nSample data: {len(sample)} earnings events")
    print(sample[['symbol', 'announcement_date', 'sector', 'iv_avg_pre']].to_string())

    has_api_key = (ALPHA_VANTAGE_API_KEY != 'YOUR_KEY_HERE'
                   and ALPHA_VANTAGE_API_KEY is not None)

    if has_api_key:
        print("\n  Alpha Vantage API key detected — running full pipeline")
        result = add_market_wide_vol_features(
            sample,
            api_key=ALPHA_VANTAGE_API_KEY,
            sector_col='sector',
            sub_sector_col='sub_sector',
            model_type='tree',
            validate=True,
        )
    else:
        print("\n  No API key — running VIX-only pipeline with mock sector IV")

        vix_df = fetch_vix_data(start_date='2023-01-01')
        result = compute_vix_features(sample, vix_df)

        np.random.seed(42)
        result['sector_etf'] = result['sector'].map(SECTOR_ETF_MAP)
        result['sector_atm_iv_call'] = np.random.uniform(0.12, 0.25, len(result))
        result['sector_atm_iv_put'] = result['sector_atm_iv_call'] + np.random.uniform(-0.02, 0.02, len(result))
        result['sector_atm_iv_avg'] = (result['sector_atm_iv_call'] + result['sector_atm_iv_put']) / 2
        result['sector_straddle_pct'] = result['sector_atm_iv_avg'] * 15.0

        result = compute_relative_iv_features(result)
        result = handle_market_vol_missing_data(result)
        validate_market_vol_features(result)

    print("\n  FINAL FEATURE SUMMARY:")
    market_cols = [c for c in result.columns if c.startswith(('vix_', 'sector_', 'iv_stock', 'iv_elevated', 'straddle_stock'))]
    print(f"  New market-wide features: {len(market_cols)}")
    for col in market_cols:
        na = result[col].isna().sum()
        print(f"    {col}: {na}/{len(result)} NaN")

    print(f"\n  Total features: {len(result.columns)}")
    return result


if __name__ == '__main__':
    print("=" * 60)
    print("MARKET-WIDE VOLATILITY CONTEXT — TEST SUITE")
    print("=" * 60)

    test_sector_etf_mapping()
    result = test_vix_fetch_and_features()
    test_relative_iv_features()
    test_missing_data_handling()
    test_validation()
    final = run_full_pipeline_demo()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
