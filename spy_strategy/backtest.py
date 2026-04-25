"""Phase 4 — Backtest for the SPY vol-crush short-straddle strategies.

Five strategies on Test period (2024-07-01 → 2026-04-21):
    always_short   - take every day (naive baseline)
    vrp_rule       - Phase-3 train-derived top-VRP quintile rule
    logreg         - Phase-3 calibrated LogReg with Val-picked threshold
    lgbm           - Phase-3 calibrated LGBM with Val-picked threshold
    buy_hold_spy   - long SPY, mark daily (asset-class baseline)

Trade mechanics
    Position     : short ATM 30d straddle (the same (strike, expiration) chosen
                   in Phase 2 — Phase 4 *re-prices* those trades, does not
                   re-select). Hold 5 trading days.
    Sizing       : contracts = floor($target_notional / (mid_premium × 100))
                   clamped ≥ 1.
    Concurrency  : up to 5 overlapping positions.
    Starting cap : $100K.
    Commission   : $0.65 per contract per leg, both open and close.
    Margin proxy : 20% × (100 × strike × contracts) per open position.
                   Documented as a proxy, NOT Reg-T naked-option margin.
    Slippage     : k × half-spread per leg per side, applied to the chain
                   (bid, ask) directly. See resolve_fill for the 4-tier
                   fallback (bid_ask → mark → last → skip).

Config stress tests
    slippage_k ∈ {0.5, 1.0, 2.0}         ("retail-realistic", "conservative-realistic", "stress")
    target_notional ∈ {$1K, $10K, $50K}  (sizing stress)
    drop_best_month / drop_worst_month   (single-window robustness)

Outputs
    trades_<strategy>.csv           — per-trade log incl. fill source, flags
    equity_<strategy>.csv           — daily equity, cash, deployed, margin
    backtest_summary.csv            — long-format metrics table
    backtest_stress_summary.csv     — all stress-test configs
"""
from __future__ import annotations

import sys
import pickle
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd


# Paths -----------------------------------------------------------------------
HERE            = Path(__file__).resolve().parent
PROJECT_ROOT    = HERE.parent
DATA_DIR        = HERE / 'data'
CACHE_DIR       = DATA_DIR / 'cache'

CHAIN_CACHE     = CACHE_DIR / 'chain.pkl'
SPY_CACHE       = CACHE_DIR / 'spy_prices.pkl'
TARGETS_CSV     = DATA_DIR / 'daily_targets.csv'
FEATURES_CSV    = DATA_DIR / 'spy_daily_features.csv'
PREDICTIONS_CSV = DATA_DIR / 'predictions.csv'

SUMMARY_CSV     = DATA_DIR / 'backtest_summary.csv'
STRESS_CSV      = DATA_DIR / 'backtest_stress_summary.csv'


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class BacktestConfig:
    starting_capital:   float = 100_000.0
    target_notional:    float = 10_000.0
    horizon_days:       int   = 5
    max_concurrent:     int   = 5
    margin_frac:        float = 0.20
    commission_per_leg: float = 0.65
    slippage_k:         float = 1.0
    mark_haircut:       float = 0.01
    wide_spread_thresh: float = 0.5
    price_mode:         str   = 'bid_ask'    # 'bid_ask' or 'mid'
    test_start:         str   = '2024-07-01'
    test_end:           str   = '2026-04-21'
    drop_best_month:    bool  = False
    drop_worst_month:   bool  = False

    def label(self) -> str:
        tag = f'k{self.slippage_k}_notional{int(self.target_notional)}_{self.price_mode}'
        if self.drop_best_month:  tag += '_dropbest'
        if self.drop_worst_month: tag += '_dropworst'
        return tag


STRESS_LABELS = {
    0.5: 'retail-realistic',
    1.0: 'conservative-realistic',
    2.0: 'stress',
}


# ============================================================================
# DATA LOADING + PRE-JOIN OF QUOTES
# ============================================================================

def load_predictions() -> pd.DataFrame:
    return pd.read_csv(PREDICTIONS_CSV, index_col='date', parse_dates=True)


def load_targets() -> pd.DataFrame:
    tgt = pd.read_csv(TARGETS_CSV, index_col='date', parse_dates=True)
    for c in ['entry_expiration', 'exit_date_5d']:
        if c in tgt.columns:
            tgt[c] = pd.to_datetime(tgt[c])
    return tgt


def load_chain() -> pd.DataFrame:
    return pd.read_pickle(CHAIN_CACHE)


def load_spy() -> pd.DataFrame:
    return pd.read_pickle(SPY_CACHE)


def join_bid_ask_onto_targets(targets: pd.DataFrame, chain: pd.DataFrame) -> pd.DataFrame:
    """Pre-join call/put bid/ask/mark/last for entry and exit days onto each target row.

    Keeps the simulation loop free of chain lookups.  Returns a frame with the
    same row count as targets, plus eight new quote columns per leg:
        entry_call_{bid,ask,mark,last}
        entry_put_{bid,ask,mark,last}
        exit_call_{bid,ask,mark,last}
        exit_put_{bid,ask,mark,last}
    """
    trades = targets.reset_index().copy()
    trades['entry_date'] = trades['date']

    chain_slim = chain[['date','expiration','strike','type','bid','ask','mark','last']].copy()

    def _merge_leg(df: pd.DataFrame, date_col: str, type_str: str, prefix: str) -> pd.DataFrame:
        src = chain_slim[chain_slim['type'] == type_str].drop(columns=['type']).copy()
        src = src.rename(columns={
            'date':       date_col,
            'expiration': 'entry_expiration',
            'strike':     'entry_strike',
            'bid':  f'{prefix}_bid',
            'ask':  f'{prefix}_ask',
            'mark': f'{prefix}_mark',
            'last': f'{prefix}_last',
        })
        return df.merge(src, on=[date_col, 'entry_expiration', 'entry_strike'], how='left')

    trades = _merge_leg(trades, 'entry_date',   'call', 'entry_call')
    trades = _merge_leg(trades, 'entry_date',   'put',  'entry_put')
    trades = _merge_leg(trades, 'exit_date_5d', 'call', 'exit_call')
    trades = _merge_leg(trades, 'exit_date_5d', 'put',  'exit_put')

    return trades


# ============================================================================
# FILL RESOLUTION
# ============================================================================

def resolve_fill_vec(bid, ask, mark, last, side: str, k: float, haircut: float, wide_thresh: float):
    """Vectorized fill resolver with 4-tier fallback (bid_ask → mark → last → NaN).

    Returns three parallel arrays: fill price, source tier, wide_spread flag.
    For `side='short'` the fill is towards the bid; `side='cover'` towards the ask.
    """
    bid  = np.asarray(bid,  dtype=float)
    ask  = np.asarray(ask,  dtype=float)
    mark = np.asarray(mark, dtype=float)
    last = np.asarray(last, dtype=float)

    ba_ok = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask >= bid)
    mk_ok = np.isfinite(mark) & (mark > 0)
    lt_ok = np.isfinite(last) & (last > 0)

    mid  = np.where(ba_ok, (bid + ask) * 0.5, np.nan)
    wide = np.where(ba_ok & np.isfinite(mid) & (mid > 0),
                    (ask - bid) / np.where(mid > 0, mid, 1) > wide_thresh, False)

    if side == 'short':
        ba_fill = mid - k * (mid - bid)
        mk_fill = mark * (1 - haircut)
        lt_fill = last * (1 - haircut)
    else:  # 'cover'
        ba_fill = mid + k * (ask - mid)
        mk_fill = mark * (1 + haircut)
        lt_fill = last * (1 + haircut)

    fill   = np.where(ba_ok, ba_fill,
             np.where(mk_ok, mk_fill,
             np.where(lt_ok, lt_fill, np.nan)))
    source = np.where(ba_ok, 'bid_ask',
             np.where(mk_ok, 'mark',
             np.where(lt_ok, 'last', 'none')))
    return fill, source, wide


def resolve_fill_mid(bid, ask, mark, last, haircut: float):
    """Mid-only resolution used for the reconciliation pass.  Same fallback
    tiers but both entry and exit take the true mid (no k applied).
    """
    bid  = np.asarray(bid,  dtype=float)
    ask  = np.asarray(ask,  dtype=float)
    mark = np.asarray(mark, dtype=float)
    last = np.asarray(last, dtype=float)

    ba_ok = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask >= bid)
    mk_ok = np.isfinite(mark) & (mark > 0)
    lt_ok = np.isfinite(last) & (last > 0)
    mid   = np.where(ba_ok, (bid + ask) * 0.5, np.nan)

    fill   = np.where(ba_ok, mid,
             np.where(mk_ok, mark,
             np.where(lt_ok, last, np.nan)))
    source = np.where(ba_ok, 'bid_ask',
             np.where(mk_ok, 'mark',
             np.where(lt_ok, 'last', 'none')))
    return fill, source


def compute_all_fills(trades_joined: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """Add entry/exit fill columns + source tier + wide_spread flags per leg."""
    t = trades_joined.copy()

    if cfg.price_mode == 'bid_ask':
        ec_fill, ec_src, ec_wide = resolve_fill_vec(
            t['entry_call_bid'], t['entry_call_ask'], t['entry_call_mark'], t['entry_call_last'],
            side='short', k=cfg.slippage_k, haircut=cfg.mark_haircut, wide_thresh=cfg.wide_spread_thresh,
        )
        ep_fill, ep_src, ep_wide = resolve_fill_vec(
            t['entry_put_bid'],  t['entry_put_ask'],  t['entry_put_mark'],  t['entry_put_last'],
            side='short', k=cfg.slippage_k, haircut=cfg.mark_haircut, wide_thresh=cfg.wide_spread_thresh,
        )
        xc_fill, xc_src, xc_wide = resolve_fill_vec(
            t['exit_call_bid'], t['exit_call_ask'], t['exit_call_mark'], t['exit_call_last'],
            side='cover', k=cfg.slippage_k, haircut=cfg.mark_haircut, wide_thresh=cfg.wide_spread_thresh,
        )
        xp_fill, xp_src, xp_wide = resolve_fill_vec(
            t['exit_put_bid'], t['exit_put_ask'], t['exit_put_mark'], t['exit_put_last'],
            side='cover', k=cfg.slippage_k, haircut=cfg.mark_haircut, wide_thresh=cfg.wide_spread_thresh,
        )
    else:  # 'mid' — reconciliation pass
        ec_fill, ec_src = resolve_fill_mid(t['entry_call_bid'], t['entry_call_ask'], t['entry_call_mark'], t['entry_call_last'], cfg.mark_haircut)
        ep_fill, ep_src = resolve_fill_mid(t['entry_put_bid'],  t['entry_put_ask'],  t['entry_put_mark'],  t['entry_put_last'],  cfg.mark_haircut)
        xc_fill, xc_src = resolve_fill_mid(t['exit_call_bid'],  t['exit_call_ask'],  t['exit_call_mark'],  t['exit_call_last'],  cfg.mark_haircut)
        xp_fill, xp_src = resolve_fill_mid(t['exit_put_bid'],   t['exit_put_ask'],   t['exit_put_mark'],   t['exit_put_last'],   cfg.mark_haircut)
        ec_wide = ep_wide = xc_wide = xp_wide = np.zeros(len(t), dtype=bool)

    t['entry_call_fill'], t['entry_call_src'], t['entry_call_wide'] = ec_fill, ec_src, ec_wide
    t['entry_put_fill'],  t['entry_put_src'],  t['entry_put_wide']  = ep_fill, ep_src, ep_wide
    t['exit_call_fill'],  t['exit_call_src'],  t['exit_call_wide']  = xc_fill, xc_src, xc_wide
    t['exit_put_fill'],   t['exit_put_src'],   t['exit_put_wide']   = xp_fill, xp_src, xp_wide

    # Any missing leg → trade is unexecutable
    unexec = (
        ~np.isfinite(t['entry_call_fill']) | ~np.isfinite(t['entry_put_fill']) |
        ~np.isfinite(t['exit_call_fill'])  | ~np.isfinite(t['exit_put_fill'])
    )
    t['missing_quote'] = unexec
    t['wide_spread_any'] = (t['entry_call_wide'] | t['entry_put_wide'] |
                            t['exit_call_wide']  | t['exit_put_wide'])
    t['premium_per_share'] = t['entry_call_fill'] + t['entry_put_fill']
    t['exit_value_per_share'] = t['exit_call_fill'] + t['exit_put_fill']
    t['mid_premium_per_share'] = t['premium_collected']  # Phase 2 mid-based
    return t


# ============================================================================
# SIGNAL RESOLUTION
# ============================================================================

def build_signal_matrix(targets: pd.DataFrame, predictions: pd.DataFrame,
                         vrp_threshold: float, features: pd.DataFrame,
                         test_start: str, test_end: str) -> pd.DataFrame:
    """Return a DataFrame indexed by date with one bool column per strategy."""
    mask = (targets.index >= pd.Timestamp(test_start)) & (targets.index <= pd.Timestamp(test_end))
    idx = targets.index[mask]

    sigs = pd.DataFrame(index=idx)
    sigs['always_short'] = True

    # VRP uses features (available across full test window)
    vrp_vals = features['vrp_30d'].reindex(idx)
    sigs['vrp_rule'] = (vrp_vals >= vrp_threshold).fillna(False).astype(bool).to_numpy()

    # ML preds exist only for labeled test window (may end earlier than targets)
    def _pred_series(col: str) -> np.ndarray:
        if col not in predictions.columns:
            return np.zeros(len(idx), dtype=bool)
        s = predictions[col].reindex(idx)
        return s.fillna(0).astype(bool).to_numpy()
    sigs['logreg'] = _pred_series('logreg_pred')
    sigs['lgbm']   = _pred_series('lgbm_pred')
    return sigs


# ============================================================================
# CORE SIMULATION
# ============================================================================

def simulate_straddle_strategy(strategy_name: str, signals: pd.Series,
                                trades_priced: pd.DataFrame,
                                cfg: BacktestConfig,
                                drop_months: set[pd.Period] | None = None
                                ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Simulate one strategy under one config. Returns (trades_df, daily_df, summary).

    Open logic:
      1. On each signal day where signals[date] is True and trade is executable:
         - Compute contracts from mid premium (same across configs so trade
           identity is stable).
         - Check margin head-room vs. max_concurrent and available equity.
         - Open short; record entry commission and premium received.
      2. On each exit date, close at cover fill. Realize net P&L. Free margin.
    """
    drop_months = drop_months or set()
    trades_priced = trades_priced.set_index('date').sort_index()
    trade_log: list[dict] = []
    daily_rows: list[dict] = []

    open_positions: list[dict] = []
    cash = cfg.starting_capital
    realized_pnl_cum      = 0.0
    peak_margin_util      = 0.0
    n_margin_rejects      = 0
    n_concurrency_rejects = 0
    n_missing_quote       = 0
    n_dropped_month       = 0

    all_dates = signals.index
    for d in all_dates:
        # 1) close any positions whose exit date <= d
        still_open: list[dict] = []
        for pos in open_positions:
            if pd.Timestamp(pos['exit_date']) <= d:
                # Realize: buy back at stored exit fill. Premium cash was
                # already booked at open; here we only outflow the cover cost
                # and the exit commission.
                exit_cost_cash   = pos['exit_value_ps'] * 100 * pos['contracts']
                exit_commission  = cfg.commission_per_leg * 2 * pos['contracts']
                cash            -= exit_cost_cash + exit_commission
                gross            = (pos['entry_premium_ps'] - pos['exit_value_ps']) * 100 * pos['contracts']
                total_commission = pos['entry_commission'] + exit_commission
                net              = gross - total_commission
                realized_pnl_cum += net
                pos['realized']        = True
                pos['gross_pnl']       = gross
                pos['net_pnl']         = net
                pos['exit_commission'] = exit_commission
                pos['total_commission']= total_commission
                trade_log.append(pos)
            else:
                still_open.append(pos)
        open_positions = still_open

        # 2) maybe open a new position today
        row = trades_priced.loc[d] if d in trades_priced.index else None
        signal = bool(signals.loc[d]) if d in signals.index else False
        month_p = d.to_period('M')
        opened = False
        reject_reason: Optional[str] = None

        if signal and row is not None and not row.get('missing_quote', True):
            # Month-drop gating (applies before trade-sizing)
            if month_p in drop_months:
                reject_reason = 'drop_month'
                n_dropped_month += 1
            else:
                mid_premium = float(row['mid_premium_per_share'])
                entry_prem_ps = float(row['premium_per_share'])
                if not (np.isfinite(mid_premium) and mid_premium > 0):
                    reject_reason = 'no_mid_premium'
                else:
                    contracts = max(1, int(np.floor(cfg.target_notional / (mid_premium * 100))))
                    strike    = float(row['entry_strike'])
                    # Equity-for-margin: cash minus existing open-position liabilities
                    equity_available = cash - sum(p['entry_premium_ps'] * 100 * p['contracts']
                                                  for p in open_positions)
                    # Margin proxy per the Phase-4 spec: 20 % of the $-notional
                    # position size (NOT 20 % of underlying share notional).
                    # Base math: 5 concurrent × $10K × 20 % = $10K peak. This
                    # is a loose simplification — real Reg-T naked-option
                    # margin is far stricter — but matches the user's spec.
                    margin_required = cfg.margin_frac * cfg.target_notional
                    current_margin  = sum(p['margin'] for p in open_positions)
                    # concurrency check (separate counter from margin)
                    if len(open_positions) >= cfg.max_concurrent:
                        reject_reason = 'concurrent_cap'
                        n_concurrency_rejects += 1
                    # equity-based margin check
                    elif current_margin + margin_required > equity_available:
                        reject_reason = 'margin'
                        n_margin_rejects += 1
                    else:
                        # open
                        premium_cash = entry_prem_ps * 100 * contracts
                        entry_commission = cfg.commission_per_leg * 2 * contracts
                        cash += premium_cash - entry_commission
                        pos = {
                            'strategy':          strategy_name,
                            'entry_date':        d,
                            'exit_date':         row['exit_date_5d'],
                            'entry_strike':      strike,
                            'entry_expiration':  row['entry_expiration'],
                            'contracts':         contracts,
                            'entry_premium_ps':  entry_prem_ps,
                            'mid_premium_ps':    mid_premium,
                            'exit_value_ps':     float(row['exit_value_per_share']),
                            'margin':            margin_required,
                            'premium_received_cash': premium_cash,
                            'entry_commission':  entry_commission,
                            'entry_call_src':    row['entry_call_src'],
                            'entry_put_src':     row['entry_put_src'],
                            'exit_call_src':     row['exit_call_src'],
                            'exit_put_src':      row['exit_put_src'],
                            'wide_spread_any':   bool(row['wide_spread_any']),
                            'phase2_pnl_pct_5d': float(row['pnl_pct_5d']) if pd.notna(row.get('pnl_pct_5d', np.nan)) else np.nan,
                        }
                        open_positions.append(pos)
                        opened = True
        elif signal and row is not None and row.get('missing_quote', True):
            reject_reason = 'missing_quote'
            n_missing_quote += 1

        # 3) snapshot daily state
        deployed_margin = sum(p['margin'] for p in open_positions)
        # Daily equity approximation: mark open shorts at entry premium (no
        # intra-hold MTM).  Under-reports intra-hold drawdown but terminal
        # equity is exact because every position is closed at its stored exit
        # fill.  Premium cash was added at open, so the liability we owe is
        # approximately `entry_premium × 100 × contracts` — subtract it.
        open_liability = sum(p['entry_premium_ps'] * 100 * p['contracts'] for p in open_positions)
        equity = cash - open_liability
        util = deployed_margin / max(cfg.starting_capital, 1.0)
        peak_margin_util = max(peak_margin_util, util)
        daily_rows.append({
            'date':            d,
            'cash':            cash,
            'equity':          equity,
            'deployed_margin': deployed_margin,
            'n_open':          len(open_positions),
            'realized_pnl_cum':realized_pnl_cum,
            'opened':          opened,
            'reject_reason':   reject_reason,
        })

    # Close out any still-open at the end (mark at stored exit fill, since exit
    # was already fixed in daily_targets):
    for pos in open_positions:
        gross = (pos['entry_premium_ps'] - pos['exit_value_ps']) * 100 * pos['contracts']
        commission = cfg.commission_per_leg * 2 * 2 * pos['contracts']
        pos['gross_pnl']       = gross
        pos['net_pnl']         = gross - commission
        pos['exit_commission'] = cfg.commission_per_leg * 2 * pos['contracts']
        trade_log.append(pos)

    trades_df = pd.DataFrame(trade_log)
    daily_df  = pd.DataFrame(daily_rows).set_index('date')

    summary = summarize(strategy_name, trades_df, daily_df, cfg)
    summary.update(dict(
        n_margin_rejects      = n_margin_rejects,
        n_concurrency_rejects = n_concurrency_rejects,
        n_missing_quote       = n_missing_quote,
        n_dropped_month       = n_dropped_month,
        peak_margin_util_pct  = peak_margin_util * 100,
    ))
    return trades_df, daily_df, summary


def simulate_buy_hold_spy(spy: pd.DataFrame, cfg: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Long SPY from test_start close to test_end close, $100K entered at start."""
    test_idx = spy.index[(spy.index >= pd.Timestamp(cfg.test_start)) &
                          (spy.index <= pd.Timestamp(cfg.test_end))]
    px = spy.loc[test_idx, 'close']
    initial_shares = cfg.starting_capital / px.iloc[0]
    equity = initial_shares * px
    daily = pd.DataFrame({
        'cash': 0.0, 'equity': equity,
        'deployed_margin': cfg.starting_capital,
        'n_open': 1, 'realized_pnl_cum': (equity - cfg.starting_capital),
        'opened': False, 'reject_reason': None,
    })
    # synthetic single-trade log
    trade = {
        'strategy':'buy_hold_spy', 'entry_date': equity.index[0], 'exit_date': equity.index[-1],
        'entry_strike': np.nan, 'entry_expiration': pd.NaT, 'contracts': int(initial_shares),
        'entry_premium_ps': px.iloc[0], 'mid_premium_ps': px.iloc[0],
        'exit_value_ps': px.iloc[-1],
        'gross_pnl': float((px.iloc[-1] - px.iloc[0]) * initial_shares),
        'net_pnl':   float((px.iloc[-1] - px.iloc[0]) * initial_shares),
        'entry_commission': 0.0, 'exit_commission': 0.0,
        'entry_call_src':'n/a','entry_put_src':'n/a','exit_call_src':'n/a','exit_put_src':'n/a',
        'wide_spread_any': False, 'phase2_pnl_pct_5d': np.nan,
    }
    trades_df = pd.DataFrame([trade])
    summary = summarize('buy_hold_spy', trades_df, daily, cfg)
    summary.update(dict(n_margin_rejects=0, n_missing_quote=0,
                        n_dropped_month=0, peak_margin_util_pct=100.0))
    return trades_df, daily, summary


# ============================================================================
# METRICS
# ============================================================================

def summarize(strategy: str, trades: pd.DataFrame, daily: pd.DataFrame,
              cfg: BacktestConfig) -> dict:
    if trades.empty or daily.empty:
        return dict(
            strategy=strategy, n_trades=0,
            terminal_equity=float(cfg.starting_capital),
            total_return_pct=0.0, annualized_return=np.nan,
            sharpe_non_overlap=np.nan, sortino=np.nan,
            max_dd_dollar=0.0, max_dd_pct=0.0, calmar=np.nan,
            win_rate=np.nan, profit_factor=np.nan,
            mean_trade_dollar=np.nan, best_trade_dollar=np.nan,
            worst_trade_dollar=np.nan, sum_net_pnl=0.0,
            capital_efficiency=np.nan, n_wide_spread_trades=0,
        )

    net_pnl    = trades['net_pnl'].astype(float)
    n_trades   = len(trades)
    n_wins     = int((net_pnl > 0).sum())
    n_loss     = int((net_pnl <= 0).sum())
    gross_wins = float(net_pnl[net_pnl > 0].sum())
    gross_loss = float(-net_pnl[net_pnl < 0].sum())
    profit_factor = gross_wins / gross_loss if gross_loss > 0 else np.inf

    # Equity curve (daily)
    eq       = daily['equity'].astype(float)
    terminal = float(eq.iloc[-1])
    total_ret = (terminal / cfg.starting_capital) - 1
    n_days   = max((daily.index[-1] - daily.index[0]).days, 1)
    years    = n_days / 365.25
    # Negative terminal equity produces complex numbers under fractional power;
    # report NaN (strategy is insolvent — an annualized-return number is meaningless).
    if terminal > 0 and years > 0:
        ann_ret = (terminal / cfg.starting_capital) ** (1 / years) - 1
    else:
        ann_ret = np.nan

    # Sharpe / Sortino: compute on realized trade P&L per trade, annualized
    # using non-overlapping bars. For straddles that overlap, approximate
    # trade frequency from mean entry spacing.
    if n_trades >= 2 and cfg.horizon_days > 0 and strategy != 'buy_hold_spy':
        trade_returns = (net_pnl / cfg.starting_capital).to_numpy()
        std = float(np.std(trade_returns, ddof=1))
        mean = float(np.mean(trade_returns))
        # annualize: trading-days-per-year / horizon_days
        ann_factor = np.sqrt(252.0 / cfg.horizon_days)
        sharpe_non_overlap = (mean / std) * ann_factor if std > 0 else np.nan
        neg = trade_returns[trade_returns < 0]
        downside_std = float(np.std(neg, ddof=1)) if neg.size > 1 else np.nan
        sortino = (mean / downside_std) * ann_factor if downside_std and downside_std > 0 else np.nan
    else:
        # Daily-bar Sharpe for buy_hold_spy
        daily_ret = eq.pct_change().dropna()
        if len(daily_ret) > 2 and daily_ret.std() > 0:
            sharpe_non_overlap = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))
            neg = daily_ret[daily_ret < 0]
            sortino = float(daily_ret.mean() / neg.std() * np.sqrt(252)) if len(neg) > 2 and neg.std() > 0 else np.nan
        else:
            sharpe_non_overlap = sortino = np.nan

    # Max drawdown on equity
    peak = eq.cummax()
    dd   = eq - peak
    dd_pct = dd / peak
    max_dd_dollar = float(dd.min())
    max_dd_pct    = float(dd_pct.min())
    calmar = (ann_ret / abs(max_dd_pct)) if (max_dd_pct < 0 and np.isfinite(ann_ret)) else np.nan

    win_rate = n_wins / n_trades if n_trades else np.nan

    # Capital efficiency = cumulative net P&L / Σ deployed-capital $ (daily sum)
    deployed_sum = float(daily['deployed_margin'].sum())
    cap_eff = (net_pnl.sum() / deployed_sum) if deployed_sum > 0 else np.nan

    n_wide = int(trades['wide_spread_any'].sum()) if 'wide_spread_any' in trades.columns else 0

    return dict(
        strategy            = strategy,
        n_trades            = n_trades,
        terminal_equity     = terminal,
        total_return_pct    = total_ret * 100,
        annualized_return   = ann_ret,
        sharpe_non_overlap  = sharpe_non_overlap,
        sortino             = sortino,
        max_dd_dollar       = max_dd_dollar,
        max_dd_pct          = max_dd_pct * 100,
        calmar              = calmar,
        win_rate            = win_rate,
        profit_factor       = profit_factor,
        mean_trade_dollar   = float(net_pnl.mean()),
        best_trade_dollar   = float(net_pnl.max()),
        worst_trade_dollar  = float(net_pnl.min()),
        sum_net_pnl         = float(net_pnl.sum()),
        capital_efficiency  = cap_eff,
        n_wide_spread_trades= n_wide,
    )


# ============================================================================
# ORCHESTRATION
# ============================================================================

def run_all_strategies(cfg: BacktestConfig,
                       features: pd.DataFrame,
                       predictions: pd.DataFrame,
                       trades_priced: pd.DataFrame,
                       spy: pd.DataFrame,
                       vrp_threshold: float,
                       drop_months: set[pd.Period] | None = None,
                       verbose: bool = True) -> dict:
    sigs = build_signal_matrix(pd.DataFrame(index=trades_priced['date'].unique()),
                                predictions, vrp_threshold, features,
                                cfg.test_start, cfg.test_end)
    # trades_priced index should equal the test-window; filter defensively
    t = trades_priced[(trades_priced['date'] >= pd.Timestamp(cfg.test_start)) &
                       (trades_priced['date'] <= pd.Timestamp(cfg.test_end))].copy()

    out: dict = {}
    for strat in ['always_short', 'vrp_rule', 'logreg', 'lgbm']:
        trades_df, daily_df, s = simulate_straddle_strategy(
            strat, sigs[strat], t, cfg, drop_months=drop_months,
        )
        out[strat] = dict(trades=trades_df, daily=daily_df, summary=s)
        if verbose:
            print(f'  {strat:<14s}  trades={s["n_trades"]:>3d}  terminal=${s["terminal_equity"]:>10,.0f}  '
                  f'Sharpe={s["sharpe_non_overlap"]:>+.2f}  Sortino={s.get("sortino",np.nan):>+.2f}  '
                  f'maxDD={s["max_dd_pct"]:>+.2f}%  margin_peak={s["peak_margin_util_pct"]:>5.1f}%  '
                  f'rejects(concurrency/margin/missing)={s.get("n_concurrency_rejects",0)}/'
                  f'{s["n_margin_rejects"]}/{s["n_missing_quote"]}')

    trades_df, daily_df, s = simulate_buy_hold_spy(spy, cfg)
    out['buy_hold_spy'] = dict(trades=trades_df, daily=daily_df, summary=s)
    if verbose:
        print(f'  {"buy_hold_spy":<14s}  trades={s["n_trades"]:>3d}  terminal=${s["terminal_equity"]:>10,.0f}  '
              f'Sharpe={s["sharpe_non_overlap"]:>+.2f}  maxDD={s["max_dd_pct"]:>+.2f}%')
    return out


def reconciliation_diagnostic(trades_priced_mid: pd.DataFrame) -> dict:
    """Compare Phase-4 mid-priced gross P&L vs Phase-2 pnl_pct_5d × mid_premium × 100."""
    t = trades_priced_mid.copy()
    mask = t['pnl_pct_5d'].notna() & t['premium_per_share'].notna() & t['exit_value_per_share'].notna()
    t = t[mask]
    # 1 contract economics
    phase4_gross = (t['premium_per_share'] - t['exit_value_per_share']) * 100
    phase2_gross = t['pnl_pct_5d'] * t['premium_collected'] * 100
    diff = phase4_gross - phase2_gross
    return dict(
        n              = int(len(t)),
        max_abs_diff   = float(diff.abs().max()),
        mean_abs_diff  = float(diff.abs().mean()),
        max_rel_diff   = float((diff / phase2_gross).replace([np.inf, -np.inf], np.nan).abs().max()),
    )


# ============================================================================
# ENTRYPOINT
# ============================================================================

def _load_everything():
    """Load all the input frames once."""
    features    = pd.read_csv(FEATURES_CSV, index_col='date', parse_dates=True)
    predictions = load_predictions()
    targets     = load_targets()
    chain       = load_chain()
    spy         = load_spy()

    # VRP threshold from Phase 3 — read from the pickled rule
    rule_path = DATA_DIR / 'models' / 'vrp_rule_profitable_5d.pkl'
    if rule_path.exists():
        with open(rule_path, 'rb') as f:
            rule = pickle.load(f)
        vrp_threshold = float(rule.threshold)
    else:
        vrp_threshold = float(features.loc[:'2023-12-31', 'vrp_30d'].quantile(0.80))

    trades_joined = join_bid_ask_onto_targets(targets, chain)
    return features, predictions, targets, chain, spy, vrp_threshold, trades_joined


def main():
    features, predictions, targets, chain, spy, vrp_threshold, trades_joined = _load_everything()
    print(f'VRP rule threshold (from Phase 3 pickle): {vrp_threshold:.4f}')

    # ---------- 0. Reconciliation pass (mid pricing, k=0 equivalent) ----------
    cfg_mid = BacktestConfig(price_mode='mid', slippage_k=0.0)
    trades_priced_mid = compute_all_fills(trades_joined, cfg_mid)
    recon = reconciliation_diagnostic(trades_priced_mid)
    print(f'\n=== Reconciliation (mid-priced Phase-4 vs Phase-2 labels, 1-contract $ basis) ===')
    print(f'  n={recon["n"]}  max_abs_diff=${recon["max_abs_diff"]:.6f}  '
          f'mean_abs_diff=${recon["mean_abs_diff"]:.6f}  max_rel_diff={recon["max_rel_diff"]:.2e}')

    # ---------- 1. Base config (k=1, $10K, all months) ----------
    print('\n=== Base config: k=1 ("conservative-realistic"), $10K notional, all months ===')
    cfg_base = BacktestConfig()
    trades_priced = compute_all_fills(trades_joined, cfg_base)
    out_base = run_all_strategies(cfg_base, features, predictions, trades_priced, spy, vrp_threshold)

    # Save per-strategy artifacts at base config
    for strat, blob in out_base.items():
        blob['trades'].to_csv(DATA_DIR / f'trades_{strat}.csv', index=False)
        blob['daily'].to_csv(DATA_DIR / f'equity_{strat}.csv')

    base_summary = pd.DataFrame([b['summary'] for b in out_base.values()])
    base_summary.to_csv(SUMMARY_CSV, index=False)

    # ---------- 2. Slippage stress: k=0.5, k=2 ----------
    print('\n=== Slippage stress: k=0.5 ("retail-realistic") ===')
    cfg_k05 = BacktestConfig(slippage_k=0.5)
    trades_k05 = compute_all_fills(trades_joined, cfg_k05)
    out_k05 = run_all_strategies(cfg_k05, features, predictions, trades_k05, spy, vrp_threshold)

    print('\n=== Slippage stress: k=2 ("stress: filling through book") ===')
    cfg_k2 = BacktestConfig(slippage_k=2.0)
    trades_k2 = compute_all_fills(trades_joined, cfg_k2)
    out_k2 = run_all_strategies(cfg_k2, features, predictions, trades_k2, spy, vrp_threshold)

    # ---------- 3. Notional stress: $1K, $50K ----------
    print('\n=== Notional stress: $1K ===')
    cfg_n1 = BacktestConfig(target_notional=1_000.0)
    out_n1 = run_all_strategies(cfg_n1, features, predictions, trades_priced, spy, vrp_threshold)

    print('\n=== Notional stress: $50K ===')
    cfg_n50 = BacktestConfig(target_notional=50_000.0)
    out_n50 = run_all_strategies(cfg_n50, features, predictions, trades_priced, spy, vrp_threshold)

    # ---------- 4. Drop-best / drop-worst month per strategy ----------
    print('\n=== Single-month robustness (per-strategy best/worst month dropped) ===')
    monthly_out: dict = {}
    for label in ('drop_best', 'drop_worst'):
        per_strat = {}
        for strat in ['always_short', 'vrp_rule', 'logreg', 'lgbm']:
            # first compute monthly pnl for that strategy to find the extreme month
            trades_df = out_base[strat]['trades']
            if trades_df.empty:
                continue
            monthly = trades_df.copy()
            monthly['exit_month'] = pd.to_datetime(monthly['exit_date']).dt.to_period('M')
            m = monthly.groupby('exit_month')['net_pnl'].sum()
            if m.empty:
                continue
            target_month = m.idxmax() if label == 'drop_best' else m.idxmin()
            cfg_drop = BacktestConfig(drop_best_month=(label == 'drop_best'),
                                       drop_worst_month=(label == 'drop_worst'))
            tp = compute_all_fills(trades_joined, cfg_drop)
            # Re-run only the one strategy with the dropped month
            sigs = build_signal_matrix(pd.DataFrame(index=tp['date'].unique()),
                                        predictions, vrp_threshold, features,
                                        cfg_drop.test_start, cfg_drop.test_end)
            t = tp[(tp['date'] >= pd.Timestamp(cfg_drop.test_start)) &
                   (tp['date'] <= pd.Timestamp(cfg_drop.test_end))].copy()
            drop_months = {target_month}
            td, dd, ss = simulate_straddle_strategy(strat, sigs[strat], t, cfg_drop, drop_months=drop_months)
            ss['dropped_month'] = str(target_month)
            per_strat[strat] = ss
        monthly_out[label] = per_strat

    # ---------- 5. Aggregate all stress into a single long-format table ----------
    rows = []
    def _add(config_label, out_dict):
        for strat, blob in out_dict.items():
            s = blob['summary'].copy()
            s['config'] = config_label
            rows.append(s)
    _add('base_k1_10K_allmonths',  out_base)
    _add('k0.5_10K_allmonths',     out_k05)
    _add('k2_10K_allmonths',       out_k2)
    _add('k1_1K_allmonths',        out_n1)
    _add('k1_50K_allmonths',       out_n50)
    # drop-month rows per strategy
    for label, per_strat in monthly_out.items():
        for strat, s in per_strat.items():
            s = dict(s); s['config'] = f'k1_10K_{label}'; rows.append(s)

    stress_df = pd.DataFrame(rows)
    stress_df.to_csv(STRESS_CSV, index=False)
    print(f'\nWrote {STRESS_CSV}')

    # ---------- 6. Summary line ----------
    vrp_row = [r for r in rows if r.get('strategy') == 'vrp_rule' and r.get('config') == 'base_k1_10K_allmonths']
    if vrp_row:
        print(f'\nSummary: VRP rule (base k=1, $10K) net Sharpe (non-overlap) = '
              f'{vrp_row[0]["sharpe_non_overlap"]:+.3f}')
    return dict(
        out_base=out_base, stress_df=stress_df,
        recon=recon, vrp_threshold=vrp_threshold,
        monthly_out=monthly_out,
    )


if __name__ == '__main__':
    main()
