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
# # 01 — Fetch Event Option Chains
#
# For every S&P 500 earnings event in our 2021-06-23 → 2026-04-21 window we need
# the option chain on **two specific dates**:
#
# - `pre_date` = 1 trading day before the announcement → gives us the market's
#   *expected* move (`straddle_pct_pre`) and the IV going into earnings
# - `post_date` = 1 trading day after the announcement → gives us the realized
#   spot price (so we can measure `actual_move_pct`) and the IV after the crush
#
# **Source**: Alpha Vantage `HISTORICAL_OPTIONS` (premium tier, 75 req/min).
#
# **Total budget**: ~8,000 events × 2 dates ≈ **16,000 calls** ≈ 4-7 hours wall-clock
# at the actual ~0.6 calls/sec we measured (HTTP latency dominates).
#
# **Storage**: ~4.6 GB of JSON cached under `unified_strategy/cache/options/{TICKER}/{date}.json`.
# Each chain has every strike × every expiration × calls + puts (typically
# 2,000-6,000 contracts per file).

# %%
import os
import sys
from pathlib import Path

import pandas as pd

if str(Path.cwd().parent) not in sys.path:
    sys.path.insert(0, str(Path.cwd().parent))

from unified_strategy import CACHE_DIR
from unified_strategy.features import load_all_stock_events, load_sp500_tickers

# %% [markdown]
# ## 1. Build the event list
#
# Each row is one earnings announcement. The (ticker, announcement_date) pair
# determines the two API calls we make.

# %%
tickers = load_sp500_tickers()
events = load_all_stock_events(tickers)
print(f"SP500 events in window: {len(events):,}")
print(f"Unique tickers: {events['ticker'].nunique()}")
print(f"Date range: {events['announcement_date'].min().date()} → {events['announcement_date'].max().date()}")
events.head(3)

# %% [markdown]
# ## 2. Cache audit (already-fetched chains)
#
# The fetcher is idempotent — already-cached chains are skipped. Below counts
# how many JSON files we already have, broken down by ticker.

# %%
cached = {}
for d in CACHE_DIR.iterdir() if CACHE_DIR.exists() else []:
    if d.is_dir():
        cached[d.name] = len(list(d.glob("*.json")))

cached_df = pd.Series(cached).sort_values(ascending=False)
total = cached_df.sum()
print(f"Total cached chains: {total:,}")
print(f"Tickers with cache: {len(cached_df)}")
print()
print("Top 10 tickers by cache size:")
print(cached_df.head(10).to_string())

# %% [markdown]
# ## 3. The fetcher in production
#
# We use `run_bulk_fetch.py` for the actual scrape. It:
#
# 1. Detects rate-limit responses (`Information` / `Note` keys) and **does not**
#    cache them — sleeps 60s then retries indefinitely
# 2. 3× exponential backoff on transient network errors (1s → 2s → 4s)
# 3. Skips cached files instantly on re-run
# 4. Flushes a tracking CSV every 250 events for crash recovery
#
# In our actual run it survived two ISP outages with **0.18% loss** that a
# 30-second second-pass cleaned up perfectly.
#
# Run it from the shell:
#
# ```bash
# export ALPHAVANTAGE_API_KEY=...
# python3 run_bulk_fetch.py
# ```

# %% [markdown]
# ## 4. Sample a chain to confirm schema

# %%
import json

# Pick the first cached chain we can find
sample_path = next(CACHE_DIR.rglob("*.json"))
data = json.loads(sample_path.read_text())
print(f"Sample file: {sample_path.relative_to(CACHE_DIR.parent)}")
print(f"Top-level keys: {list(data.keys())}")
contracts = data.get("data", [])
print(f"Contracts in this chain: {len(contracts):,}")
print()
print("Schema of one contract:")
if contracts:
    sample = contracts[0]
    for k, v in sample.items():
        print(f"  {k}: {v}")

# %% [markdown]
# ## 5. Final coverage stats

# %%
n_events = len(events)
expected_calls = n_events * 2
cached_now = sum(cached.values())
print(f"Events:          {n_events:,}")
print(f"Calls expected:  {expected_calls:,}")
print(f"Cached:          {cached_now:,} ({100 * cached_now / expected_calls:.1f}%)")
print()
print("If coverage is 99%+, we're ready for notebook 02 (features).")
