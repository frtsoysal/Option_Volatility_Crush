"""Daily feature engineering for the SPY volatility-crush strategy.

Inputs
    option_volatility_crush.ipynb/pilot_data/spy_chains.csv.gz
    option_volatility_crush.ipynb/pilot_data/spy_pc_ratio.csv.gz
    option_volatility_crush.ipynb/pilot_data/spy_vol_oi_ratio.csv.gz
    yfinance: SPY (prices), ^VIX / ^VIX3M (via market_vol_context.fetch_vix_data)

Output
    spy_strategy/data/spy_daily_features.csv  — one row per NYSE trading day

Intermediate caches (pickle) live in spy_strategy/data/cache/ so repeated runs
skip the ~5 minute per-day IV-surface loop. Pass force_rebuild=True to ignore.

Reuses from market_vol_context.py:
    implied_vol()     — brentq Black-Scholes IV solver (fallback path)
    fetch_vix_data()  — VIX / VIX3M via yfinance with file-based caching
"""
from __future__ import annotations

import os
import sys
import time
import pickle
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

# Project paths / module imports ---------------------------------------------
HERE          = Path(__file__).resolve().parent                   # .../spy_strategy
PROJECT_ROOT  = HERE.parent                                       # .../Option_Volatility_Crush
PILOT_DIR     = PROJECT_ROOT / 'option_volatility_crush.ipynb' / 'pilot_data'
DATA_DIR      = HERE / 'data'
CACHE_DIR     = DATA_DIR / 'cache'
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from market_vol_context import implied_vol, fetch_vix_data  # noqa: E402

# Tunables --------------------------------------------------------------------
CHAIN_PATH    = PILOT_DIR / 'spy_chains.csv.gz'
PC_PATH       = PILOT_DIR / 'spy_pc_ratio.csv.gz'
VOI_PATH      = PILOT_DIR / 'spy_vol_oi_ratio.csv.gz'

TARGET_DTES   = (30, 60, 90)
RISK_FREE     = 0.04                           # fallback r for Brent IV when needed
SPY_START_PAD = 380                            # days of SPY history before chain start (>252 + 60)

OUT_PATH      = DATA_DIR / 'spy_daily_features.csv'

NUM_COLS_CHAIN = [
    'strike','last','mark','bid','ask','volume','open_interest',
    'implied_volatility','delta','gamma','theta','vega','rho',
    'bid_size','ask_size',
]


# ============================================================================
# LOADERS
# ============================================================================

def load_chain(path: Path = CHAIN_PATH, use_cache: bool = True) -> pd.DataFrame:
    """Load spy_chains.csv.gz; coerce Alpha Vantage '-' sentinels to NaN; cache."""
    cache = CACHE_DIR / 'chain.pkl'
    if use_cache and cache.exists() and cache.stat().st_mtime > path.stat().st_mtime:
        print(f'  [cache hit] chain: {cache}')
        return pd.read_pickle(cache)

    print(f'  Loading {path} ...')
    chain = pd.read_csv(path, parse_dates=['date', 'expiration', 'fetch_date'], low_memory=False)
    for c in NUM_COLS_CHAIN:
        if c in chain.columns:
            chain[c] = pd.to_numeric(chain[c], errors='coerce')
    chain['dte'] = (chain['expiration'] - chain['date']).dt.days
    chain.to_pickle(cache)
    print(f'  Cached chain to {cache}  ({len(chain):,} rows)')
    return chain


def load_pc_ratio(path: Path = PC_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df['expiration'] == 'FULL_CHAIN'][['fetch_date', 'pc_ratio']].copy()
    df['date'] = pd.to_datetime(df['fetch_date'])
    df['pc_ratio'] = pd.to_numeric(df['pc_ratio'], errors='coerce')
    return df[['date', 'pc_ratio']].sort_values('date').reset_index(drop=True)


def load_voi_ratio(path: Path = VOI_PATH, use_cache: bool = True) -> pd.DataFrame:
    """Daily mean of per-contract volume/open_interest ratios."""
    cache = CACHE_DIR / 'voi_daily.pkl'
    if use_cache and cache.exists() and cache.stat().st_mtime > path.stat().st_mtime:
        print(f'  [cache hit] voi_daily: {cache}')
        return pd.read_pickle(cache)
    print(f'  Loading {path} and aggregating ...')
    df = pd.read_csv(path, parse_dates=['date'])
    df['volume_open_interest_ratio'] = pd.to_numeric(df['volume_open_interest_ratio'], errors='coerce')
    daily = (df.groupby('date')['volume_open_interest_ratio']
               .mean()
               .rename('voi_ratio')
               .reset_index())
    daily.to_pickle(cache)
    print(f'  Cached voi_daily to {cache}  ({len(daily)} days)')
    return daily


def load_spy_prices(start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """SPY OHLC from yfinance with file cache. Returns DataFrame indexed by date."""
    cache = CACHE_DIR / 'spy_prices.pkl'
    if use_cache and cache.exists():
        cached = pd.read_pickle(cache)
        if cached.index.min() <= pd.Timestamp(start) and cached.index.max() >= pd.Timestamp(end) - pd.Timedelta(days=5):
            print(f'  [cache hit] SPY prices: {cache}')
            return cached
    print(f'  Downloading SPY prices {start} -> {end} ...')
    raw = yf.download('SPY', start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=str.lower)
    raw.index = pd.to_datetime(raw.index).normalize()
    raw.index.name = 'date'
    spy = raw[['open', 'high', 'low', 'close']].copy()
    spy['log_ret'] = np.log(spy['close'] / spy['close'].shift(1))
    spy.to_pickle(cache)
    print(f'  Cached SPY prices to {cache}  ({len(spy)} days)')
    return spy


# ============================================================================
# IV SURFACE — per-day computation
# ============================================================================

def _infer_spot_from_parity(day_df: pd.DataFrame) -> float:
    """Use the nearest-dated expiry: spot ≈ strike where |call_mark - put_mark| is minimized.

    For a given expiry, call - put = S - K*e^{-rT}, so the strike where the
    absolute difference is smallest is closest to the discounted forward —
    accurate enough for ATM-strike selection.
    """
    dfe = day_df[(day_df['dte'] > 0) & day_df['mark'].notna()]
    if dfe.empty:
        return np.nan
    near_exp = dfe.groupby('expiration').size().idxmax()
    sub = dfe[dfe['expiration'] == near_exp]
    wide = sub.pivot_table(index='strike', columns='type', values='mark', aggfunc='first')
    if 'call' not in wide.columns or 'put' not in wide.columns:
        return np.nan
    wide = wide.dropna(subset=['call', 'put'])
    if wide.empty:
        return np.nan
    return float((wide['call'] - wide['put']).abs().idxmin())


def _iv_from_prices(row: pd.Series, spot: float) -> float:
    """Fallback: invert Black-Scholes from the mid price if IV column is NaN."""
    price = row.get('mark')
    if not (isinstance(price, (int, float)) and price > 0):
        price = row.get('last')
    T = max(row['dte'], 1) / 365.0
    return implied_vol(price, spot, row['strike'], T, RISK_FREE, row['type'])


def _atm_iv_one_expiry(sub: pd.DataFrame, spot: float) -> float:
    """Average of call and put IV at the strike closest to spot."""
    s = sub.dropna(subset=['implied_volatility'])
    if s.empty or not np.isfinite(spot):
        return np.nan
    atm_strike = s.iloc[(s['strike'] - spot).abs().values.argsort()]['strike'].iloc[0]
    return float(s.loc[s['strike'] == atm_strike, 'implied_volatility'].mean())


def _delta_iv_one_expiry(sub: pd.DataFrame, target_delta: float, option_type: str) -> float:
    """IV of the contract whose delta is closest to target_delta among the given side."""
    s = sub[(sub['type'] == option_type) &
            sub['delta'].notna() &
            sub['implied_volatility'].notna()]
    if s.empty:
        return np.nan
    idx = (s['delta'] - target_delta).abs().values.argsort()[0]
    return float(s.iloc[idx]['implied_volatility'])


def _interp_by_dte(points: list[tuple[int, float]], target_dte: int) -> float:
    """Linear interpolation across (dte, value) pairs. Nearest-extrapolation outside range."""
    pts = sorted((d, v) for d, v in points if np.isfinite(v) and d > 0)
    if not pts:
        return np.nan
    if target_dte <= pts[0][0]:
        return pts[0][1]
    if target_dte >= pts[-1][0]:
        return pts[-1][1]
    for (d0, v0), (d1, v1) in zip(pts[:-1], pts[1:]):
        if d0 <= target_dte <= d1:
            if d1 == d0:
                return 0.5 * (v0 + v1)
            w = (target_dte - d0) / (d1 - d0)
            return (1 - w) * v0 + w * v1
    return np.nan


def _surface_one_day(day_df: pd.DataFrame) -> dict:
    """Compute the full one-day IV surface: ATM at 30/60/90, 25d put/call at 30d, spot."""
    day_df = day_df[day_df['dte'] > 0]
    spot = _infer_spot_from_parity(day_df)

    # Per-expiration ATM / 25-delta IVs
    atm_points: list[tuple[int, float]] = []
    put25_points: list[tuple[int, float]] = []
    call25_points: list[tuple[int, float]] = []

    for exp, sub in day_df.groupby('expiration'):
        dte = int(sub['dte'].iloc[0])
        atm_points.append((dte, _atm_iv_one_expiry(sub, spot)))
        put25_points.append((dte, _delta_iv_one_expiry(sub, -0.25, 'put')))
        call25_points.append((dte, _delta_iv_one_expiry(sub,  0.25, 'call')))

    atm_30 = _interp_by_dte(atm_points, 30)
    atm_60 = _interp_by_dte(atm_points, 60)
    atm_90 = _interp_by_dte(atm_points, 90)

    put25_30  = _interp_by_dte(put25_points, 30)
    call25_30 = _interp_by_dte(call25_points, 30)

    # Fallback: if ATM 30d is missing (as on 2023-08-08), back it out from prices
    # on the closest expiry to 30 DTE using brentq Black-Scholes inversion.
    if not np.isfinite(atm_30):
        fallback = _atm_from_prices(day_df, spot, target_dte=30)
        atm_30 = fallback

    return dict(
        spot=spot,
        atm_iv_30d=atm_30,
        atm_iv_60d=atm_60,
        atm_iv_90d=atm_90,
        iv_25d_put=put25_30,
        iv_25d_call=call25_30,
    )


def _atm_from_prices(day_df: pd.DataFrame, spot: float, target_dte: int) -> float:
    """Brent-based ATM IV back-out used as a fallback when implied_volatility is NaN."""
    if not np.isfinite(spot):
        return np.nan
    dfe = day_df.dropna(subset=['mark', 'strike', 'dte'])
    if dfe.empty:
        return np.nan
    exps = dfe[['expiration', 'dte']].drop_duplicates().sort_values('dte')
    anchor = exps.iloc[(exps['dte'] - target_dte).abs().values.argsort()[0]]
    sub = dfe[dfe['expiration'] == anchor['expiration']]
    atm_strike = sub.iloc[(sub['strike'] - spot).abs().values.argsort()]['strike'].iloc[0]
    legs = sub[sub['strike'] == atm_strike]
    ivs = [_iv_from_prices(r, spot) for _, r in legs.iterrows()]
    ivs = [v for v in ivs if np.isfinite(v)]
    return float(np.mean(ivs)) if ivs else np.nan


def compute_iv_surface(chain: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
    """Apply _surface_one_day to every trading day in the chain. Cached."""
    cache = CACHE_DIR / 'iv_surface.pkl'
    chain_mtime = CHAIN_PATH.stat().st_mtime
    if use_cache and cache.exists() and cache.stat().st_mtime > chain_mtime:
        print(f'  [cache hit] iv_surface: {cache}')
        return pd.read_pickle(cache)

    print('  Computing per-day IV surface (1200+ days, expect ~3-5 minutes) ...')
    t0 = time.time()
    records = []
    groups = chain[chain['dte'] > 0].groupby('date', sort=True)
    n = groups.ngroups
    for i, (d, g) in enumerate(groups, start=1):
        rec = _surface_one_day(g)
        rec['date'] = d
        records.append(rec)
        if i % 100 == 0 or i == n:
            elapsed = time.time() - t0
            print(f'    processed {i}/{n} days  ({elapsed:.0f}s)')

    surface = pd.DataFrame(records).set_index('date').sort_index()
    surface['term_slope_30_60']   = surface['atm_iv_60d'] - surface['atm_iv_30d']
    surface['term_slope_30_90']   = surface['atm_iv_90d'] - surface['atm_iv_30d']
    surface['backwardation_flag'] = (surface['term_slope_30_90'] < 0).astype(int)
    surface['risk_reversal_25d']  = surface['iv_25d_put'] - surface['iv_25d_call']
    surface['butterfly_25d']      = 0.5 * (surface['iv_25d_put'] + surface['iv_25d_call']) - surface['atm_iv_30d']

    surface.to_pickle(cache)
    print(f'  Cached iv_surface to {cache}  ({len(surface)} days, {time.time()-t0:.0f}s)')
    return surface


# ============================================================================
# HISTORICAL-CONTEXT, RV/VRP, FLOW, PRICE
# ============================================================================

def _rank_pct(series: pd.Series, window: int) -> pd.Series:
    """Min-max rank within trailing `window` observations, scaled to [0, 1]."""
    roll_min = series.rolling(window, min_periods=max(5, window // 4)).min()
    roll_max = series.rolling(window, min_periods=max(5, window // 4)).max()
    return (series - roll_min) / (roll_max - roll_min)


def _percentile_rank(series: pd.Series, window: int) -> pd.Series:
    def _pct(x: np.ndarray) -> float:
        last = x[-1]
        return float((x <= last).sum() - 1) / max(len(x) - 1, 1)
    return series.rolling(window, min_periods=max(20, window // 4)).apply(_pct, raw=True)


def compute_historical_context(surface: pd.DataFrame) -> pd.DataFrame:
    s = surface['atm_iv_30d']
    return pd.DataFrame({
        'iv_rank_30d':         _rank_pct(s, 30),
        'iv_rank_252d':        _rank_pct(s, 252),
        'iv_percentile_252d':  _percentile_rank(s, 252),
    }, index=surface.index)


def compute_rv_vrp(spy: pd.DataFrame, surface: pd.DataFrame) -> pd.DataFrame:
    lr = spy['log_ret']
    rv_5d  = lr.rolling(5,  min_periods=3 ).std() * np.sqrt(252)
    rv_21d = lr.rolling(21, min_periods=10).std() * np.sqrt(252)
    out = pd.DataFrame({'rv_5d': rv_5d, 'rv_21d': rv_21d}, index=spy.index)
    out = out.reindex(surface.index)
    out['vrp_30d']       = surface['atm_iv_30d'] - out['rv_21d']
    out['vol_expansion'] = out['rv_5d'] / out['rv_21d']
    return out


def compute_flow(pc: pd.DataFrame, voi: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    pc = pc.set_index('date')
    voi = voi.set_index('date')
    df = pd.DataFrame(index=index)
    df['pc_ratio']        = pc['pc_ratio'].reindex(index)
    df['pc_ratio_z_20d']  = _zscore(df['pc_ratio'], 20)
    df['voi_ratio']       = voi['voi_ratio'].reindex(index)
    df['voi_ratio_z_20d'] = _zscore(df['voi_ratio'], 20)
    return df


def _zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=max(5, window // 4)).mean()
    sd = s.rolling(window, min_periods=max(5, window // 4)).std()
    return (s - mu) / sd


def compute_price(spy: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    close = spy['close']
    df = pd.DataFrame(index=spy.index)
    df['spy_close']            = close
    df['spy_ret_5d']           = close.pct_change(5)
    df['spy_ret_21d']          = close.pct_change(21)
    df['spy_dd_from_60d_high'] = close / close.rolling(60, min_periods=20).max() - 1
    df['spy_above_200d_ma']    = (close > close.rolling(200, min_periods=50).mean()).astype(int)
    return df.reindex(index)


# ============================================================================
# ORCHESTRATION
# ============================================================================

def build_features(force_rebuild: bool = False) -> pd.DataFrame:
    use_cache = not force_rebuild
    print('Phase 1 — building SPY daily feature matrix')

    chain   = load_chain(use_cache=use_cache)
    surface = compute_iv_surface(chain, use_cache=use_cache)

    pc_daily  = load_pc_ratio()
    voi_daily = load_voi_ratio(use_cache=use_cache)

    date_min = surface.index.min() - pd.Timedelta(days=SPY_START_PAD)
    date_max = surface.index.max() + pd.Timedelta(days=2)
    spy  = load_spy_prices(date_min.strftime('%Y-%m-%d'), date_max.strftime('%Y-%m-%d'),
                           use_cache=use_cache)
    vix  = fetch_vix_data(start_date=date_min.strftime('%Y-%m-%d'),
                          end_date=date_max.strftime('%Y-%m-%d'))
    vix.index = pd.to_datetime(vix.index).normalize()

    # Assemble on the surface index (the chain's trading days are the authoritative set).
    df = surface.copy()
    df = df.join(compute_historical_context(surface))
    df = df.join(compute_rv_vrp(spy, surface))
    df = df.join(compute_flow(pc_daily, voi_daily, df.index))
    df = df.join(compute_price(spy, df.index))
    df = df.join(vix[['vix_close', 'vix3m_close']].reindex(df.index))
    df['vix_minus_atm_iv_30d'] = df['vix_close'] - df['atm_iv_30d'] * 100

    ordered = [
        # base
        'spot', 'spy_close',
        # IV surface
        'atm_iv_30d', 'atm_iv_60d', 'atm_iv_90d',
        'term_slope_30_60', 'term_slope_30_90', 'backwardation_flag',
        # skew
        'iv_25d_put', 'iv_25d_call', 'risk_reversal_25d', 'butterfly_25d',
        # historical context
        'iv_rank_30d', 'iv_rank_252d', 'iv_percentile_252d',
        # rv / vrp
        'rv_5d', 'rv_21d', 'vrp_30d', 'vol_expansion',
        # flow
        'pc_ratio', 'pc_ratio_z_20d', 'voi_ratio', 'voi_ratio_z_20d',
        # price
        'spy_ret_5d', 'spy_ret_21d', 'spy_dd_from_60d_high', 'spy_above_200d_ma',
        # vix cross-check
        'vix_close', 'vix3m_close', 'vix_minus_atm_iv_30d',
    ]
    missing = [c for c in ordered if c not in df.columns]
    if missing:
        raise RuntimeError(f'Missing expected feature columns: {missing}')
    df = df[ordered]

    df.index.name = 'date'
    df.to_csv(OUT_PATH)
    print(f'\nWrote {OUT_PATH}  ({df.shape[0]} rows x {df.shape[1]} cols)')
    return df


def _summary(df: pd.DataFrame) -> None:
    print('\n=== Feature summary ===')
    print(f'Shape       : {df.shape[0]:,} rows x {df.shape[1]} cols')
    print(f'Date range  : {df.index.min().date()} -> {df.index.max().date()}')
    print('\nNaN fraction per column:')
    print((df.isna().mean() * 100).round(2).astype(str) + ' %')
    print('\nHead:')
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(df.head(3))
    print('\nTail:')
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(df.tail(3))
    print('\nDescribe (selected):')
    sel = ['atm_iv_30d','atm_iv_90d','term_slope_30_90','risk_reversal_25d',
           'iv_rank_252d','vrp_30d','pc_ratio','vix_close','vix_minus_atm_iv_30d']
    with pd.option_context('display.width', 200):
        print(df[sel].describe().round(4))


if __name__ == '__main__':
    force = '--rebuild' in sys.argv
    out = build_features(force_rebuild=force)
    _summary(out)
