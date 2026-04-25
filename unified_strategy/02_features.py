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
# # 02 — Features
#
# Build the unified per-event feature matrix combining three sources:
#
# | Block | Source | Count |
# |---|---|---|
# | A. Stock fundamentals | `/ML/data/raw/{TICKER}_earnings_with_q4.csv` | 33 |
# | B. Stock options | Cached Alpha Vantage HISTORICAL_OPTIONS chains | 17 (raw + derived) |
# | C. SPY market regime | `spy_strategy/data/spy_daily_features.csv` | 30 |
#
# Output: one row per earnings event, ~80 columns total, written to `data/02_event_features.csv`.

# %%
import sys, os
from pathlib import Path
import pandas as pd

if str(Path.cwd().parent) not in sys.path:
    sys.path.insert(0, str(Path.cwd().parent))

from unified_strategy import CACHE_DIR, WINDOW_END, WINDOW_START
from unified_strategy.features import (
    STOCK_FEATURE_COLS,
    OPTION_FEATURE_COLS,
    add_options_features,
    add_spy_context,
    load_all_stock_events,
    load_sp500_tickers,
)
from unified_strategy.label_targets import label_crush, split_summary, temporal_split

# Toggle: PILOT vs FULL universe
PILOT = True
PILOT_TICKERS = ["AAPL", "NVDA", "MSFT", "TSLA", "JPM"]

# %% [markdown]
# ## 1. Block A — load stock fundamentals (33 features)

# %%
if PILOT:
    tickers = PILOT_TICKERS
    print(f"PILOT mode: {tickers}")
else:
    tickers = load_sp500_tickers()
    print(f"FULL mode: {len(tickers)} SP500 tickers")

events = load_all_stock_events(tickers, window_start=WINDOW_START, window_end=WINDOW_END, progress=not PILOT)
print(f"Events loaded: {len(events):,}")
print(f"Block-A columns: {events.shape[1]}  (3 IDs + eps_beat + 33 features)")

# %% [markdown]
# ## 2. Block B — compute options features from cached chains
#
# Requires the chains already cached under `cache/options/{TICKER}/{date}.json`.
# Run notebook **01** first to populate the cache, otherwise this returns NaN rows.

# %%
events_b = add_options_features(events, cache_dir=CACHE_DIR, progress=not PILOT)
print(f"After Block B: {events_b.shape[1]} cols")
print(f"Events with options data: {events_b['stock_price_pre'].notna().sum():,} / {len(events_b):,}")

# %% [markdown]
# ## 3. Label targets (crush_profitable + crush_pnl_pct)

# %%
labeled = label_crush(events_b)
print(f"Labeled events: {labeled['crush_profitable'].notna().sum():,}")
print(f"Crush rate: {100*labeled['crush_profitable'].mean():.1f}%")
print()
print(split_summary(labeled).to_string(index=False))

# %% [markdown]
# ## 4. Block C — join SPY market regime

# %%
final = add_spy_context(labeled)
spy_cols = [c for c in final.columns if c.startswith("spy_") and c != "spy_join_date"]
print(f"SPY context columns added: {len(spy_cols)}")
print(f"Final feature matrix: {final.shape[0]:,} rows × {final.shape[1]} cols")

# %% [markdown]
# ## 5. Save

# %%
out = Path("data") / ("02_pilot_event_features.csv" if PILOT else "02_event_features.csv")
out.parent.mkdir(exist_ok=True)
final.to_csv(out, index=False)
print(f"Wrote {out}: {final.shape[0]:,} rows × {final.shape[1]} cols")

# %% [markdown]
# ## 6. Sanity peek

# %%
display_cols = [
    "ticker", "announcement_date",
    "elo_before", "elo_momentum", "eps_estimate_average", "price_change_3m_pct",
    "stock_price_pre", "stock_price_post", "straddle_pct_pre",
    "iv_avg_pre", "iv_crush_pct", "iv_term_slope",
    "spy_atm_iv_30d", "spy_vrp_30d", "spy_vix_close",
    "actual_move_pct", "crush_profitable", "crush_pnl_pct",
]
display_cols = [c for c in display_cols if c in final.columns]
print(final[display_cols].to_string(index=False))
