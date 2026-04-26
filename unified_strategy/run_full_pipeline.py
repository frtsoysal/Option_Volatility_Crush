"""
End-to-end run of the full SP500 vol-crush pipeline on the bulk-fetched data.

Phases:
  1. Build feature matrix (Block A stock + Block B options + Block C SPY)
  2. Label (crush_profitable, crush_pnl_pct)
  3. Save 02_event_features.csv
  4. Train LR + LightGBM + XGBoost (with isotonic calibration + MCC threshold)
  5. Backtest 4 strategies on the test split
  6. Persist metrics, equity curves, trades

Designed to run unattended and write all artifacts to disk so the
presentation notebook can pick up everything from CSV/JSON.
"""

import sys
import time
from pathlib import Path

import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from unified_strategy import CACHE_DIR, RESULTS_DIR
from unified_strategy.features import (
    add_expiry_intrinsic,
    add_options_features,
    add_spy_context,
    load_all_stock_events,
    load_sp500_tickers,
)
from unified_strategy.label_targets import label_crush, split_summary, temporal_split
from unified_strategy.ml_pipeline import feature_columns, train_all
from unified_strategy.backtest import run_backtests


def main():
    t_start = time.time()

    print("=" * 60)
    print("Phase 3.2a — Build feature matrix")
    print("=" * 60)
    tickers = load_sp500_tickers()
    print(f"Tickers in SP500 list: {len(tickers)}")

    t0 = time.time()
    events = load_all_stock_events(tickers, progress=True)
    print(f"\nBlock A (stock fundamentals): {len(events):,} events × {events.shape[1]} cols  in {time.time() - t0:.1f}s")

    t0 = time.time()
    events_b = add_options_features(events, cache_dir=CACHE_DIR, progress=True)
    print(f"Block B (options metrics): {events_b.shape[1]} cols  in {time.time() - t0:.1f}s")
    print(f"  events with options data: {events_b['stock_price_pre'].notna().sum():,} / {len(events_b):,}")

    t0 = time.time()
    labeled = label_crush(events_b)
    print(f"\nLabels: crush_profitable + crush_pnl_pct  in {time.time() - t0:.1f}s")

    t0 = time.time()
    with_expiry = add_expiry_intrinsic(labeled)
    n_expiry = with_expiry["expiry_close"].notna().sum()
    print(f"Block B+ (expiry intrinsic): {n_expiry:,}/{len(with_expiry):,} events have expiry_close  in {time.time() - t0:.1f}s")

    t0 = time.time()
    final = add_spy_context(with_expiry)
    print(f"Block C (SPY context): final {final.shape[1]} cols  in {time.time() - t0:.1f}s")

    # Drop rows where we don't have everything we need to label or train
    before = len(final)
    usable = final.dropna(
        subset=["crush_profitable", "stock_price_pre", "stock_price_post", "straddle_pct_pre"]
    ).reset_index(drop=True)
    print(f"\nUsable events: {len(usable):,} / {before:,} ({100*len(usable)/before:.1f}%)")
    print(f"Crush rate: {100 * usable['crush_profitable'].mean():.2f}%")
    print()
    print("Split distribution:")
    print(split_summary(usable).to_string(index=False))

    out = HERE / "data" / "02_event_features.csv"
    out.parent.mkdir(exist_ok=True)
    usable.to_csv(out, index=False)
    print(f"\nWrote {out}: {usable.shape[0]:,} rows × {usable.shape[1]} cols")

    print()
    print("=" * 60)
    print("Phase 3.2b — Train LR + LightGBM + XGBoost (threshold for hold_to_expiry)")
    print("=" * 60)
    summary = train_all(usable, exit_mode="hold_to_expiry")
    print()
    print(summary.to_string(index=False))

    print()
    print("=" * 60)
    print("Phase 3.2c — Backtest both exit modes")
    print("=" * 60)
    masks = temporal_split(usable, date_col="announcement_date")
    test = usable[masks["test"]].reset_index(drop=True)
    print(f"Test window events: {len(test):,}")

    predictions = {}
    threshold = None
    for name in ["logreg", "lgbm", "xgb"]:
        cal_path = HERE / "models" / f"{name}_calibrated.pkl"
        if not cal_path.exists():
            continue
        cal = joblib.load(cal_path)
        cols = feature_columns(usable)
        X_test = test[cols]
        proba = cal.predict_proba(X_test)[:, 1]
        predictions[name] = proba

    if not summary.empty:
        threshold = float(summary.loc[summary["model"] == "lgbm", "threshold"].iloc[0]) \
                    if "lgbm" in summary["model"].values \
                    else float(summary.iloc[0]["threshold"])

    print("\n--- HOLD TO EXPIRY (no exit half-spread, intrinsic at expiration) ---")
    bt_hte_dir = RESULTS_DIR / "hold_to_expiry"
    bt_hte_dir.mkdir(parents=True, exist_ok=True)
    bt_hte = run_backtests(
        usable, predictions=predictions, threshold=threshold or 0.5,
        out_dir=bt_hte_dir, exit_mode="hold_to_expiry",
    )
    print(bt_hte.to_string(index=False))

    print("\n--- T+1 CLOSE (real post mids, 10% round-trip half-spread) ---")
    bt_t1_dir = RESULTS_DIR / "t_plus_1"
    bt_t1_dir.mkdir(parents=True, exist_ok=True)
    bt_t1 = run_backtests(
        usable, predictions=predictions, threshold=threshold or 0.5,
        out_dir=bt_t1_dir, exit_mode="t_plus_1",
    )
    print(bt_t1.to_string(index=False))

    # Side-by-side delta on the headline column
    print("\n--- DELTA: hold_to_expiry vs t_plus_1 ---")
    merged = bt_hte.merge(bt_t1, on="strategy", suffixes=("_hte", "_t1"))[
        ["strategy", "n_trades_hte", "win_rate_hte", "avg_pnl_dollars_hte",
         "total_pnl_dollars_hte", "sharpe_hte",
         "win_rate_t1", "avg_pnl_dollars_t1", "total_pnl_dollars_t1", "sharpe_t1"]
    ]
    print(merged.to_string(index=False))

    print()
    print("=" * 60)
    print(f"DONE in {(time.time() - t_start) / 60:.1f} min")
    print("=" * 60)
    print(f"  feature matrix:    {out}")
    print(f"  metrics_summary:   {RESULTS_DIR / 'metrics_summary.csv'}")
    print(f"  backtest_summary:  {RESULTS_DIR / 'backtest_summary.csv'}")


if __name__ == "__main__":
    main()
