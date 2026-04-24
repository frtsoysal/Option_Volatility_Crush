"""Phase 2 — Target labeling for the SPY volatility-crush strategy.

For each trading day t:
    1. Identify the ATM ~30-DTE straddle (real listed expiration nearest 30 DTE,
       strike closest to parity-inferred spot).
    2. Record the premium collected = call_mid(t) + put_mid(t).
    3. For each holding horizon N in {5, 10, 21} trading days, value the ORIGINAL
       position (same strike, same expiration) using the chain on day t+N.
    4. Compute P&L, P&L %, and a binary profitable flag.

Output: spy_strategy/data/daily_targets.csv (one row per trading day).

Inputs: loads the cached chain written by spy_features.py when available; falls
back to reading the raw CSV (with the same '-' sentinel coercion used in
Phase 0/1) if the cache is missing.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Path setup ------------------------------------------------------------------
HERE         = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
PILOT_DIR    = PROJECT_ROOT / 'option_volatility_crush.ipynb' / 'pilot_data'
DATA_DIR     = HERE / 'data'
CACHE_DIR    = DATA_DIR / 'cache'
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Configuration ---------------------------------------------------------------
CHAIN_PATH         = PILOT_DIR / 'spy_chains.csv.gz'
CHAIN_CACHE        = CACHE_DIR / 'chain.pkl'
OUT_PATH           = DATA_DIR / 'daily_targets.csv'

TARGET_ENTRY_DTE   = 30
HOLD_HORIZONS_DAYS = (5, 10, 21)

NUM_COLS_CHAIN = [
    'strike','last','mark','bid','ask','volume','open_interest',
    'implied_volatility','delta','gamma','theta','vega','rho',
    'bid_size','ask_size',
]


# ============================================================================
# LOADERS / PRICE HELPERS
# ============================================================================

def load_chain() -> pd.DataFrame:
    """Prefer the Phase-1 cache; otherwise read the raw CSV with the same coercion."""
    if CHAIN_CACHE.exists() and CHAIN_CACHE.stat().st_mtime > CHAIN_PATH.stat().st_mtime:
        print(f'  [cache hit] chain: {CHAIN_CACHE}')
        return pd.read_pickle(CHAIN_CACHE)

    print(f'  Loading {CHAIN_PATH} (no fresh cache) ...')
    chain = pd.read_csv(CHAIN_PATH, parse_dates=['date','expiration','fetch_date'], low_memory=False)
    for c in NUM_COLS_CHAIN:
        if c in chain.columns:
            chain[c] = pd.to_numeric(chain[c], errors='coerce')
    chain['dte'] = (chain['expiration'] - chain['date']).dt.days
    chain.to_pickle(CHAIN_CACHE)
    return chain


def _mid(row: pd.Series) -> float:
    """Mid = (bid+ask)/2 if both > 0; else mark; else last."""
    bid, ask = row.get('bid'), row.get('ask')
    if (isinstance(bid, (int, float)) and isinstance(ask, (int, float))
            and np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0):
        return float((bid + ask) / 2.0)
    m = row.get('mark')
    if isinstance(m, (int, float)) and np.isfinite(m) and m > 0:
        return float(m)
    l = row.get('last')
    if isinstance(l, (int, float)) and np.isfinite(l) and l > 0:
        return float(l)
    return float('nan')


def _infer_spot_from_parity(day_df: pd.DataFrame) -> float:
    """Same parity-based spot as Phase 1: strike where |call_mark - put_mark| is smallest
    on the most-listed expiration."""
    dfe = day_df[(day_df['dte'] > 0) & day_df['mark'].notna()]
    if dfe.empty:
        return float('nan')
    near_exp = dfe.groupby('expiration').size().idxmax()
    sub = dfe[dfe['expiration'] == near_exp]
    wide = sub.pivot_table(index='strike', columns='type', values='mark', aggfunc='first')
    if 'call' not in wide.columns or 'put' not in wide.columns:
        return float('nan')
    wide = wide.dropna(subset=['call','put'])
    if wide.empty:
        return float('nan')
    return float((wide['call'] - wide['put']).abs().idxmin())


# ============================================================================
# STRADDLE SELECTION
# ============================================================================

def pick_entry_straddle(day_df: pd.DataFrame) -> Optional[dict]:
    """Select the listed expiration nearest 30 DTE and the strike closest to spot.

    Returns a dict with entry_spot, entry_expiration, entry_strike, entry_dte,
    entry_call_mid, entry_put_mid, premium_collected — or None if unresolvable.
    """
    spot = _infer_spot_from_parity(day_df)
    if not np.isfinite(spot):
        return None

    dfe = day_df[day_df['dte'] > 0]
    if dfe.empty:
        return None

    exps = dfe[['expiration','dte']].drop_duplicates().sort_values('dte')
    # Nearest listed expiration to 30 DTE (ties broken by the later one so we
    # don't risk expiry during the N=21 hold).
    exps = exps.assign(d30=(exps['dte'] - TARGET_ENTRY_DTE).abs())
    exps = exps.sort_values(['d30', 'dte'], ascending=[True, False])
    if exps.empty:
        return None
    entry_exp = exps.iloc[0]['expiration']
    entry_dte = int(exps.iloc[0]['dte'])

    sub = dfe[dfe['expiration'] == entry_exp]
    # Strike closest to spot that actually has both a call and a put quoted.
    wide = sub.pivot_table(index='strike', columns='type',
                           values=['bid','ask','mark','last'], aggfunc='first')
    if wide.empty:
        return None
    has_both = wide.index.to_series().map(
        lambda k: bool(sub[(sub['strike']==k) & (sub['type']=='call')].shape[0]
                       and sub[(sub['strike']==k) & (sub['type']=='put')].shape[0])
    )
    candidates = wide.index[has_both].to_numpy()
    if len(candidates) == 0:
        return None
    entry_strike = float(candidates[np.abs(candidates - spot).argmin()])

    call_row = sub[(sub['strike']==entry_strike) & (sub['type']=='call')].iloc[0]
    put_row  = sub[(sub['strike']==entry_strike) & (sub['type']=='put' )].iloc[0]
    call_mid = _mid(call_row)
    put_mid  = _mid(put_row)
    if not (np.isfinite(call_mid) and np.isfinite(put_mid)):
        return None

    return dict(
        entry_spot         = spot,
        entry_expiration   = pd.Timestamp(entry_exp),
        entry_strike       = entry_strike,
        entry_dte          = entry_dte,
        entry_call_mid     = call_mid,
        entry_put_mid      = put_mid,
        premium_collected  = call_mid + put_mid,
    )


# ============================================================================
# EXIT VALUATION
# ============================================================================

def value_position(exit_day_df: pd.DataFrame, strike: float, expiration: pd.Timestamp) -> dict:
    """Price the original (strike, expiration) straddle using exit-day quotes.

    If the exit date falls on/after expiration, settle at intrinsic using the
    exit-day spot.  Otherwise mid the call and put on the same (strike, expiry)
    pair.  Returns dict with exit_call_mid, exit_put_mid, value_at_exit,
    exit_dte, settled_intrinsic (bool).
    """
    if exit_day_df.empty:
        return dict(exit_call_mid=np.nan, exit_put_mid=np.nan,
                    value_at_exit=np.nan, exit_dte=np.nan, settled_intrinsic=False)

    exit_date = exit_day_df['date'].iloc[0]
    dte_remaining = int((expiration - exit_date).days)

    if dte_remaining <= 0:
        spot = _infer_spot_from_parity(exit_day_df)
        call_intr = max(spot - strike, 0.0) if np.isfinite(spot) else np.nan
        put_intr  = max(strike - spot, 0.0) if np.isfinite(spot) else np.nan
        return dict(exit_call_mid=call_intr, exit_put_mid=put_intr,
                    value_at_exit=(call_intr + put_intr)
                                   if np.isfinite(call_intr) and np.isfinite(put_intr)
                                   else np.nan,
                    exit_dte=0, settled_intrinsic=True)

    sub = exit_day_df[(exit_day_df['expiration'] == expiration)
                      & (exit_day_df['strike'] == strike)]
    if sub.empty:
        return dict(exit_call_mid=np.nan, exit_put_mid=np.nan,
                    value_at_exit=np.nan, exit_dte=dte_remaining,
                    settled_intrinsic=False)

    call = sub[sub['type'] == 'call']
    put  = sub[sub['type'] == 'put']
    if call.empty or put.empty:
        return dict(exit_call_mid=np.nan, exit_put_mid=np.nan,
                    value_at_exit=np.nan, exit_dte=dte_remaining,
                    settled_intrinsic=False)

    call_mid = _mid(call.iloc[0])
    put_mid  = _mid(put.iloc[0])
    value = call_mid + put_mid if np.isfinite(call_mid) and np.isfinite(put_mid) else np.nan
    return dict(exit_call_mid=call_mid, exit_put_mid=put_mid,
                value_at_exit=value, exit_dte=dte_remaining,
                settled_intrinsic=False)


# ============================================================================
# ORCHESTRATION
# ============================================================================

def build_targets() -> pd.DataFrame:
    print('Phase 2 — building daily straddle P&L targets')
    chain = load_chain()

    # Index chain by date for O(1) lookups.
    by_date = {d: g for d, g in chain.groupby('date', sort=True)}
    trading_days = sorted(by_date.keys())
    td_index = {d: i for i, d in enumerate(trading_days)}

    print(f'  {len(trading_days)} trading days from {trading_days[0].date()} to {trading_days[-1].date()}')

    t0 = time.time()
    rows: list[dict] = []
    missing_exit_counts = {N: 0 for N in HOLD_HORIZONS_DAYS}
    settled_counts      = {N: 0 for N in HOLD_HORIZONS_DAYS}

    for i, t in enumerate(trading_days, start=1):
        entry = pick_entry_straddle(by_date[t])
        row: dict = {'date': t}
        if entry is None:
            for N in HOLD_HORIZONS_DAYS:
                for col in ('exit_date','exit_dte','exit_call_mid','exit_put_mid',
                            'value_at_exit','pnl','pnl_pct','profitable'):
                    row[f'{col}_{N}d'] = np.nan
            rows.append(row)
            continue

        row.update(entry)

        for N in HOLD_HORIZONS_DAYS:
            exit_idx = td_index[t] + N
            if exit_idx >= len(trading_days):
                ex_date = pd.NaT
                val = {'exit_call_mid': np.nan, 'exit_put_mid': np.nan,
                       'value_at_exit': np.nan, 'exit_dte': np.nan,
                       'settled_intrinsic': False}
                missing_exit_counts[N] += 1
            else:
                ex_date = trading_days[exit_idx]
                val = value_position(by_date[ex_date],
                                     strike=entry['entry_strike'],
                                     expiration=entry['entry_expiration'])
                if val['settled_intrinsic']:
                    settled_counts[N] += 1
                if not np.isfinite(val['value_at_exit']):
                    missing_exit_counts[N] += 1

            premium = entry['premium_collected']
            value   = val['value_at_exit']
            pnl     = premium - value if np.isfinite(value) else np.nan
            pnl_pct = pnl / premium if (np.isfinite(pnl) and premium > 0) else np.nan
            profitable = (1 if (np.isfinite(pnl) and pnl > 0) else
                          (0 if np.isfinite(pnl) else np.nan))

            row[f'exit_date_{N}d']     = ex_date
            row[f'exit_dte_{N}d']      = val['exit_dte']
            row[f'exit_call_mid_{N}d'] = val['exit_call_mid']
            row[f'exit_put_mid_{N}d']  = val['exit_put_mid']
            row[f'value_at_exit_{N}d'] = value
            row[f'pnl_{N}d']           = pnl
            row[f'pnl_pct_{N}d']       = pnl_pct
            row[f'profitable_{N}d']    = profitable

        rows.append(row)
        if i % 200 == 0 or i == len(trading_days):
            print(f'    processed {i}/{len(trading_days)}  ({time.time()-t0:.0f}s)')

    out = pd.DataFrame(rows).set_index('date').sort_index()

    ordered = ['entry_spot','entry_strike','entry_expiration','entry_dte',
               'entry_call_mid','entry_put_mid','premium_collected']
    for N in HOLD_HORIZONS_DAYS:
        ordered += [f'exit_date_{N}d', f'exit_dte_{N}d',
                    f'exit_call_mid_{N}d', f'exit_put_mid_{N}d',
                    f'value_at_exit_{N}d',
                    f'pnl_{N}d', f'pnl_pct_{N}d', f'profitable_{N}d']
    out = out[ordered]

    out.to_csv(OUT_PATH)
    print(f'\nWrote {OUT_PATH}  ({out.shape[0]} rows x {out.shape[1]} cols)')

    print('\n=== Exit-resolution diagnostics ===')
    for N in HOLD_HORIZONS_DAYS:
        print(f'  N={N:>2}d: end-of-series truncations={max(missing_exit_counts[N]-0,0)}, '
              f'intrinsic settlements={settled_counts[N]}')

    return out


def _summary(df: pd.DataFrame) -> None:
    print('\n=== Target label summary ===')
    for N in HOLD_HORIZONS_DAYS:
        pnl = df[f'pnl_pct_{N}d']
        prof = df[f'profitable_{N}d']
        n_valid = pnl.notna().sum()
        print(f'\nHorizon N={N}d:')
        print(f'  valid labels     : {n_valid}/{len(df)}')
        print(f'  profitable rate  : {prof.mean(skipna=True):.3%}   '
              f'(profitable={int(prof.sum()):d} / losses={int((prof==0).sum()):d})')
        print(f'  pnl_pct  mean    : {pnl.mean(skipna=True):+.4f}')
        print(f'  pnl_pct  median  : {pnl.median(skipna=True):+.4f}')
        print(f'  pnl_pct  std     : {pnl.std(skipna=True):.4f}')
        print(f'  pnl_pct  p05/p95 : {pnl.quantile(0.05):+.4f} / {pnl.quantile(0.95):+.4f}')

    print('\nHead:')
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(df.head(2).round(4))
    print('\nWorst 5 by 5d P&L %:')
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(df.nsmallest(5, 'pnl_pct_5d')[['entry_spot','entry_strike','entry_expiration',
                                              'premium_collected','value_at_exit_5d',
                                              'pnl_pct_5d']].round(4))
    print('\nBest 5 by 5d P&L %:')
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(df.nlargest(5, 'pnl_pct_5d')[['entry_spot','entry_strike','entry_expiration',
                                             'premium_collected','value_at_exit_5d',
                                             'pnl_pct_5d']].round(4))


if __name__ == '__main__':
    out = build_targets()
    _summary(out)
