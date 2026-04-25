# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 00 — Data Sanity
#
# Verify all three data sources align before committing to expensive Alpha Vantage fetches.
#
# Checks:
# 1. **SP500 ticker coverage** — every ticker in `/ML/sp500_tickers.csv` has a raw earnings file
# 2. **Event count in window** — apply the same filter as `prepare_data.py` and count events 2021-06-23 to 2026-04-21
# 3. **No-leakage assertion** — feature whitelist excludes everything in `prepare_data.py` line 66
# 4. **SPY context coverage** — `spy_strategy/data/spy_daily_features.csv` covers every event's announcement date
# 5. **SPY options chain coverage** — `pilot_data/spy_chains.csv.gz` (Joshua's bulk fetch) covers the same window
#
# This notebook makes **no** API calls. Read-only sanity check. Re-run before any fetch / training.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make `unified_strategy` importable when running from inside its own folder
if str(Path.cwd().parent) not in sys.path:
    sys.path.insert(0, str(Path.cwd().parent))

from unified_strategy import (
    SPY_FEATURES_PATH,
    SP500_TICKERS_PATH,
    ML_RAW_DIR,
    REPO_ROOT,
    WINDOW_START,
    WINDOW_END,
)
from unified_strategy.features import (
    STOCK_FEATURE_COLS,
    LEAKAGE_COLS,
    load_sp500_tickers,
    load_stock_events,
    load_all_stock_events,
    assert_no_leakage,
)
from unified_strategy.label_targets import temporal_split, split_summary

print(f"REPO_ROOT      = {REPO_ROOT}")
print(f"ML_RAW_DIR     = {ML_RAW_DIR}")
print(f"SPY_FEATURES   = {SPY_FEATURES_PATH}")
print(f"WINDOW         = {WINDOW_START}  →  {WINDOW_END}")
print(f"FEATURES (n)   = {len(STOCK_FEATURE_COLS)}")
print(f"LEAKAGE (n)    = {len(LEAKAGE_COLS)}")

# %% [markdown]
# ## 1. SP500 ticker → raw-CSV coverage

# %%
tickers = load_sp500_tickers()
print(f"SP500 list size: {len(tickers)}")

raw_files = {p.stem.replace("_earnings_with_q4", ""): p for p in ML_RAW_DIR.glob("*_earnings_with_q4.csv")}
print(f"Raw CSVs on disk: {len(raw_files)}")

missing = [t for t in tickers if t not in raw_files]
extra = [t for t in raw_files if t not in tickers]

print(f"Tickers in list but missing CSV: {len(missing)}")
if missing[:10]:
    print(f"  examples: {missing[:10]}")
print(f"CSVs on disk but not in SP500 list: {len(extra)}  (ignored — possibly delisted or non-SP500)")

# Final list to use
universe = [t for t in tickers if t in raw_files]
print(f"\nUsable universe: {len(universe)} tickers")

# %% [markdown]
# ## 2. Event count in the [2021-06-23, 2026-04-21] window
#
# Apply the same filter as `prepare_data.py:42-53`:
# - keep only quarterly rows (drop `horizon contains "fiscal year"`)
# - drop NaN `actual_eps`
# - constrain `reported_date` to our window

# %%
events = load_all_stock_events(universe, window_start=WINDOW_START, window_end=WINDOW_END)
print(f"Total events in window: {len(events):,}")
print(f"Unique tickers represented: {events['ticker'].nunique()}")
print(f"Mean events per ticker: {events.groupby('ticker').size().mean():.1f}")
print(f"Date range: {events['announcement_date'].min().date()}  →  {events['announcement_date'].max().date()}")

# %% [markdown]
# ## 3. Temporal split distribution

# %%
print(split_summary(events).to_string(index=False))

# %% [markdown]
# ## 4. No-leakage assertion
#
# The 33-feature whitelist must NOT include any of `prepare_data.py:66`:
# `actual_eps`, `eps_beat`, `eps_delta`, `elo_after`, `elo_change`, `K_adaptive`, `price_at_report`.

# %%
overlap = set(STOCK_FEATURE_COLS) & set(LEAKAGE_COLS)
assert not overlap, f"LEAKAGE: {overlap}"
print(f"✅ No overlap between {len(STOCK_FEATURE_COLS)} features and {len(LEAKAGE_COLS)} leakage columns.")

# Also verify on the actual loaded events frame
features_only = events[[c for c in STOCK_FEATURE_COLS if c in events.columns]]
assert_no_leakage(features_only)
print(f"✅ Feature matrix has {features_only.shape[1]} columns, none from leakage list.")

# Confirm leakage cols ARE present in the source frame (so we know they were dropped, not just absent)
leakage_in_raw = set(LEAKAGE_COLS) & set(events.columns)
print(f"Leakage cols present in raw events frame (kept for inspection only, will be dropped before training): {sorted(leakage_in_raw)}")

# %% [markdown]
# ## 5. SPY market regime coverage
#
# Joshua's `spy_strategy/data/spy_daily_features.csv` should cover every announcement-date-1-BD
# join key. Every event needs a SPY row to have context features.

# %%
spy = pd.read_csv(SPY_FEATURES_PATH, parse_dates=["date"])
print(f"SPY rows: {len(spy):,}")
print(f"SPY range: {spy['date'].min().date()}  →  {spy['date'].max().date()}")
print(f"SPY columns: {len(spy.columns)} → {list(spy.columns)}")

# Join key: most recent SPY trading day STRICTLY BEFORE announcement.
# `pd.merge_asof(direction="backward", allow_exact_matches=False)` finds it
# correctly — robust to US market holidays (MLK, Presidents Day, etc.) that
# `pd.tseries.offsets.BDay` doesn't know about.
events_sorted = events.sort_values("announcement_date").reset_index(drop=True)
spy_sorted = spy[["date"]].sort_values("date").rename(columns={"date": "spy_join_date"})
merged = pd.merge_asof(
    events_sorted,
    spy_sorted,
    left_on="announcement_date",
    right_on="spy_join_date",
    direction="backward",
    allow_exact_matches=False,
)

n_missing = merged["spy_join_date"].isna().sum()
print(f"\nEvents missing SPY context: {n_missing:,} / {len(events):,}  ({100*n_missing/len(events):.1f}%)")
if n_missing:
    sample_missing = merged.loc[merged["spy_join_date"].isna(), ["ticker", "announcement_date"]].head(10)
    print("Sample missing rows:")
    print(sample_missing.to_string(index=False))

# Echo back the join lag distribution so we can verify it's typically 1 day
merged["join_lag_days"] = (merged["announcement_date"] - merged["spy_join_date"]).dt.days
print("\nJoin-lag distribution (announcement_date − spy_join_date, in calendar days):")
print(merged["join_lag_days"].value_counts().sort_index().to_string())

# %% [markdown]
# ## 6. SPY chain bulk-fetch coverage
#
# `pilot_data/spy_chains.csv.gz` should also cover the full window. Quick sanity check on date count.

# %%
spy_chain_path = REPO_ROOT / "option_volatility_crush.ipynb" / "pilot_data" / "spy_chains.csv.gz"
if spy_chain_path.exists():
    # Read just the fetch_date column to avoid loading all 11M rows
    fetch_dates = pd.read_csv(spy_chain_path, usecols=["fetch_date"], parse_dates=["fetch_date"])
    unique_dates = fetch_dates["fetch_date"].drop_duplicates().sort_values()
    print(f"SPY chains: unique fetch_dates = {len(unique_dates)}")
    print(f"SPY chains range: {unique_dates.min().date()}  →  {unique_dates.max().date()}")
else:
    print(f"⚠️  {spy_chain_path} not found. Run option_volatility_crush.ipynb/fetch_spy_1yr.py")

# %% [markdown]
# ## 7. Estimated API budget for Phase 3 bulk fetch

# %%
n_events = len(events)
api_calls = n_events * 2  # pre + post
minutes_at_75rpm = api_calls / 75
hours = minutes_at_75rpm / 60
print(f"Total events: {n_events:,}")
print(f"API calls (pre+post per event): {api_calls:,}")
print(f"At 75/min premium: {minutes_at_75rpm:.0f} min  ≈  {hours:.1f} hours")
print(f"Cache footprint estimate: ~{api_calls * 0.15:.0f} MB JSON  (avg ~150KB per chain response)")

# %% [markdown]
# ## 8. Pilot-fetch subset (5 tickers × 4 quarters = 40 events)
#
# What Phase 1.4 will actually fetch first — minutes, not hours.

# %%
pilot_tickers = ["AAPL", "NVDA", "MSFT", "TSLA", "JPM"]
pilot_window = ("2024-01-01", "2024-12-31")
pilot = events[
    events["ticker"].isin(pilot_tickers)
    & events["announcement_date"].between(pilot_window[0], pilot_window[1])
]
print(f"Pilot tickers: {pilot_tickers}")
print(f"Pilot window: {pilot_window[0]} → {pilot_window[1]}")
print(f"Pilot events: {len(pilot)}")
print()
print(pilot[["ticker", "announcement_date", "fiscal_quarter_end"]].to_string(index=False))

# %% [markdown]
# ## 9. Final go/no-go gate
#
# All checks must pass before Phase 1.4 (pilot fetch) and especially before Phase 3 (overnight bulk fetch).

# %%
checks = {
    "≥ 480 SP500 tickers with raw CSVs": len(universe) >= 480,
    "≥ 5,000 events in window": len(events) >= 5_000,
    "no leakage cols in feature whitelist": not (set(STOCK_FEATURE_COLS) & set(LEAKAGE_COLS)),
    "all events have SPY context (>99%)": (n_missing / max(len(events), 1)) < 0.01,
    "pilot subset has 15-25 events": 15 <= len(pilot) <= 25,
}

print("Sanity gate:")
for k, v in checks.items():
    print(f"  {'✅' if v else '❌'}  {k}")

if all(checks.values()):
    print("\n🟢 All sanity checks passed. Proceed to 01_fetch_event_options.ipynb")
else:
    failed = [k for k, v in checks.items() if not v]
    print(f"\n🔴 {len(failed)} check(s) failed:")
    for k in failed:
        print(f"   - {k}")
