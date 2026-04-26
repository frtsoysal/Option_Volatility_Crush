"""
Short-straddle backtest with realistic frictions.

For each event row:
  entry_premium  = call_pre_mid + put_pre_mid                       collected up front
  exit_intrinsic = max(0, abs(post_price - strike))                  paid back at expiry
  half_spread    = (call_ask-call_bid)/2 + (put_ask-put_bid)/2       pay half the bid-ask both sides
  commission     = $0.65 / contract  ×  open+close (2)  ×  call+put (2)  =  $2.60 / event
  pnl_dollars    = entry_premium - exit_intrinsic - half_spread - commission

A trade is sized to 1 contract (= 100 shares notional). Returns are normalized
to fraction of stock_price_pre so different-priced names are comparable.

Benchmarks:
  always_short    take every event
  ml_filtered     take only events where calibrated P > threshold (per model)
  vrp_only        take only when the day-of SPY VRP is positive (Joshua's signal)
  buy_hold_spy    long SPY across the same window

Outputs per strategy:
  trades_{strategy}.csv
  equity_{strategy}.csv     daily equity curve, $1.00 starting capital
  metrics.csv               headline summary
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

from . import MODELS_DIR, RESULTS_DIR
from .label_targets import temporal_split

COMMISSION_PER_CONTRACT = 0.65  # IBKR / TastyTrade ballpark
LEGS = 2                        # call + put
OPEN_AND_CLOSE = 2              # commissions paid on entry AND exit


def per_event_pnl(
    events: pd.DataFrame,
    contract_count: int = 1,
) -> pd.DataFrame:
    """
    Add per-event P&L columns to a labeled events frame.

    Required columns:
        atm_strike_pre, atm_call_mid_pre, atm_put_mid_pre,
        stock_price_pre, stock_price_post,
        atm_call_ask_pre? (optional)  atm_call_bid_pre?  same for put
    """
    df = events.copy()

    df["entry_premium"] = (df["atm_call_mid_pre"] + df["atm_put_mid_pre"]).clip(lower=0)
    df["exit_intrinsic"] = (df["stock_price_post"] - df["atm_strike_pre"]).abs().clip(lower=0)

    # If bid/ask weren't propagated to the events frame, approximate the half-spread
    # as 10% of the entry premium. 5% holds for the top ~50 names (SPY/AAPL/MSFT
    # on 30-DTE), but the bulk of SP500 is sub-$50B mid-caps where realistic
    # short-straddle slippage is 8-12%. Using 10% gives a defensible blended
    # number; for honest results compute per-event from raw chain bid/ask and
    # populate `half_spread` upstream. (Review fix #2.)
    if "half_spread" not in df.columns:
        df["half_spread"] = 0.10 * df["entry_premium"]

    df["commission"] = COMMISSION_PER_CONTRACT * LEGS * OPEN_AND_CLOSE * contract_count

    df["pnl_dollars"] = (
        df["entry_premium"] - df["exit_intrinsic"] - df["half_spread"]
    ) * 100 * contract_count - df["commission"]

    # Return normalized to underlying notional (= 100 × stock_price_pre per contract)
    notional = df["stock_price_pre"] * 100 * contract_count
    df["pnl_pct_notional"] = df["pnl_dollars"] / notional

    return df


def equity_curve(
    pnl_dollars: pd.Series,
    dates: pd.Series,
    starting_capital: float = 100_000.0,
) -> pd.DataFrame:
    """
    Daily equity curve in dollars.

    Convention: each trade is sized at 1 contract (the per_event_pnl `pnl_dollars`
    column already accounts for that). Returns are SUMMED across trades on the
    same day, then accumulated as a running dollar P&L on top of starting capital.
    NOT compounded — compounding only makes sense when each trade redeploys the
    full equity, which is not the case for fixed-contract-size short straddles.
    """
    df = pd.DataFrame({"date": pd.to_datetime(dates), "pnl": pnl_dollars}).sort_values("date")
    daily = df.groupby("date")["pnl"].sum().reset_index()
    daily["cum_pnl"] = daily["pnl"].cumsum()
    daily["equity"] = starting_capital + daily["cum_pnl"]
    return daily


def metrics(equity: pd.DataFrame, trades: pd.DataFrame) -> dict:
    """Headline metrics for one strategy. Equity is dollar-denominated."""
    if len(equity) == 0 or len(trades) == 0:
        return {"n_trades": 0}

    # Daily $ return on starting capital (not compound — fixed sizing per trade)
    daily_ret = equity["pnl"] / equity["equity"].iloc[0]
    sharpe = (
        np.sqrt(252) * daily_ret.mean() / daily_ret.std()
        if daily_ret.std() > 0 else 0.0
    )

    rolling_max = equity["equity"].cummax()
    dd = (equity["equity"] - rolling_max) / rolling_max
    max_dd = float(dd.min())

    pnls_dollars = trades["pnl_dollars"].dropna()
    pnls_pct = trades["pnl_pct_notional"].dropna()
    wins = pnls_dollars[pnls_dollars > 0]
    losses = pnls_dollars[pnls_dollars <= 0]
    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() < 0 else np.inf
    win_rate = float((pnls_dollars > 0).mean())

    starting = float(equity["equity"].iloc[0])
    ending = float(equity["equity"].iloc[-1])
    total_ret = (ending - starting) / starting

    n_days = (equity["date"].iloc[-1] - equity["date"].iloc[0]).days
    cagr = (1 + total_ret) ** (365.25 / max(n_days, 1)) - 1

    return {
        "n_trades": int(len(trades)),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(float(profit_factor), 4) if np.isfinite(profit_factor) else float("inf"),
        "avg_pnl_dollars": round(float(pnls_dollars.mean()), 2),
        "median_pnl_dollars": round(float(pnls_dollars.median()), 2),
        "avg_pnl_pct_notional": round(float(pnls_pct.mean()), 5),
        "total_pnl_dollars": round(float(pnls_dollars.sum()), 2),
        "total_return_on_100k": round(total_ret, 4),
        "cagr": round(float(cagr), 4),
        "sharpe": round(float(sharpe), 4),
        "max_drawdown": round(max_dd, 4),
    }


def run_backtests(
    events: pd.DataFrame,
    predictions: dict[str, np.ndarray] | None = None,
    threshold: float = 0.5,
    out_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    """
    Compute P&L curves for the four strategies on the test split.

    `predictions` is a dict like {'lgbm': np.array of length len(events_test)}.
    `threshold` is a single value used for ML-filtered strategies (per-model
    thresholds can be passed via predictions dict as 2-tuple (proba, threshold)).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    masks = temporal_split(events, date_col="announcement_date")
    test = events[masks["test"]].copy().reset_index(drop=True)
    print(f"Test events: {len(test)}  (date range {test['announcement_date'].min().date()} → {test['announcement_date'].max().date()})")

    test = per_event_pnl(test)

    summaries = []

    # 1. always_short — every event
    short_all = test.copy()
    eq = equity_curve(short_all["pnl_dollars"], short_all["announcement_date"])
    eq.to_csv(out_dir / "equity_always_short.csv", index=False)
    short_all.to_csv(out_dir / "trades_always_short.csv", index=False)
    summaries.append({"strategy": "always_short", **metrics(eq, short_all)})

    # 2. vrp_only — Joshua's signal: VRP > 0 means IV > realized → sell vol
    if "spy_vrp_30d" in test.columns:
        vrp = test[test["spy_vrp_30d"] > 0].copy()
        eq = equity_curve(vrp["pnl_dollars"], vrp["announcement_date"])
        eq.to_csv(out_dir / "equity_vrp_only.csv", index=False)
        vrp.to_csv(out_dir / "trades_vrp_only.csv", index=False)
        summaries.append({"strategy": "vrp_only", **metrics(eq, vrp)})

    # 3. ml_filtered — one per model
    if predictions:
        for name, proba in predictions.items():
            if len(proba) != len(test):
                print(f"  skip {name}: prediction length mismatch ({len(proba)} vs {len(test)})")
                continue
            mask = proba >= threshold
            sel = test.loc[mask].copy()
            eq = equity_curve(sel["pnl_dollars"], sel["announcement_date"])
            eq.to_csv(out_dir / f"equity_ml_{name}.csv", index=False)
            sel.to_csv(out_dir / f"trades_ml_{name}.csv", index=False)
            summaries.append({"strategy": f"ml_{name}", **metrics(eq, sel)})

    # 4. buy_hold_spy — flat $1 in SPY across the test window
    if "spy_spy_close" in test.columns:
        spy = (
            test[["announcement_date", "spy_spy_close"]]
            .dropna()
            .drop_duplicates("announcement_date")
            .sort_values("announcement_date")
        )
        if len(spy) >= 2:
            spy["ret"] = spy["spy_spy_close"].pct_change().fillna(0)
            spy["equity"] = (1.0 + spy["ret"]).cumprod()
            spy = spy.rename(columns={"announcement_date": "date"})[["date", "equity"]]
            spy.to_csv(out_dir / "equity_buy_hold_spy.csv", index=False)
            sharpe = (
                np.sqrt(252) * spy["equity"].pct_change().dropna().mean()
                / spy["equity"].pct_change().dropna().std()
                if spy["equity"].pct_change().dropna().std() > 0 else 0.0
            )
            summaries.append(
                {
                    "strategy": "buy_hold_spy",
                    "n_trades": 1,
                    "total_return": round(float(spy["equity"].iloc[-1] - 1), 4),
                    "sharpe": round(float(sharpe), 4),
                }
            )

    summary = pd.DataFrame(summaries)
    summary.to_csv(out_dir / "backtest_summary.csv", index=False)
    return summary
