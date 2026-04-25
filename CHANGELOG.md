# Summary of Changes — dondadaj/Option_Volatility_Crush

**Fork of:** [frtsoysal/Option_Volatility_Crush](https://github.com/frtsoysal/Option_Volatility_Crush)
**Contributor:** Josh Vodatinsky ([@dondadaj](https://github.com/dondadaj))

---

## What Changed

The original repository contained a detailed architectural plan (README) and a conceptual Jupyter notebook explaining Black-Scholes theory and IV crush mechanics. There was no working pipeline code, no feature engineering, and no data collection infrastructure. This fork adds the first functional module: a complete market-wide volatility context pipeline that introduces regime awareness and sector-relative IV features to the strategy.

The language composition shifted from 98.1% Jupyter Notebook / 1.9% Python to 91.7% Jupyter Notebook / 8.3% Python, reflecting the move from a monolithic notebook toward a modular Python codebase.

---

## New Files Added

### `vol_crush_strategy/__init__.py`

Package initialization with clean exports. Allows importing directly:

```python
from vol_crush_strategy import add_market_wide_vol_features
```

### `vol_crush_strategy/config.py`

Centralized configuration for the entire pipeline:

- **Sector ETF mappings:** 11 GICS sectors mapped to liquid sector ETFs (Technology→XLK, Financials→XLF, Energy→XLE, etc.) and 10 sub-sector ETFs for granular IV context (Semiconductors→SMH, Banks→KBE, Biotech→XBI, Software→IGV, etc.)
- **API key management:** Reads `ALPHA_VANTAGE_API_KEY` and `FMP_API_KEY` from environment variables
- **Feature name registries:** Lists of all VIX features (7), sector IV raw features (5), and sector IV relative features (4) for downstream validation and SHAP analysis
- **Validation ranges:** Expected bounds for each feature (e.g., VIX between 5-90, percentiles between 0-1) used by the validation step
- **Tunable parameters:** VIX percentile lookback window, momentum windows, term structure regime threshold, sector IV DTE filtering bounds, IV elevation threshold

### `vol_crush_strategy/market_vol_context.py`

The core module, organized into 10 parts (A through J):

**Part A — VIX Data Fetching**
- Downloads VIX and VIX3M daily close prices from Yahoo Finance (free, no Alpha Vantage API quota consumed)
- File-based caching to `data/vol_crush/vix_cache/vix_daily.csv` to avoid re-downloading on every run
- Handles both single-ticker and multi-ticker yfinance column format variations

**Part B — VIX Feature Computation (7 features)**
- `vix_level`: raw VIX close on event date (absolute market fear level)
- `vix3m_level`: raw VIX3M close (medium-term fear baseline)
- `vix_term_structure_ratio`: VIX / VIX3M (above 1.0 = backwardation/panic, below 1.0 = contango/normal)
- `vix_term_structure_regime`: binary flag (1 = backwardation, 0 = contango)
- `vix_percentile_252d`: where current VIX sits relative to trailing 252 trading days (normalized 0-1 scale)
- `vix_change_5d`: 5-day VIX momentum (short-term fear direction)
- `vix_change_21d`: 21-day VIX trend (medium-term fear direction)

**Part C — Black-Scholes IV Inversion**
- Computes implied volatility from observed option prices using Brent's root-finding method (`scipy.optimize.brentq`)
- Handles edge cases: zero/negative prices, prices below intrinsic value, solver convergence failures (all return NaN gracefully)

**Part D — Sector ETF Option Chain Fetching**
- Fetches historical option chains from Alpha Vantage `HISTORICAL_OPTIONS` API for sector ETFs
- Rate limit handling with configurable delay between calls
- Warns on API rate limiting or error responses

**Part E — Sector ETF ATM IV Computation (4 raw features)**
- Filters option chains to valid DTE window (7-45 days to expiration)
- Identifies ATM strike (closest to ETF price)
- Computes mid-price from bid/ask (falls back to last traded price)
- `sector_atm_iv_call`: sector ETF ATM call implied volatility
- `sector_atm_iv_put`: sector ETF ATM put implied volatility
- `sector_atm_iv_avg`: average of call and put ATM IV
- `sector_straddle_pct`: sector ETF ATM straddle as percentage of ETF price

**Part F — Sector IV Pipeline with Caching**
- Orchestrates sector ETF chain fetching per earnings event
- Resolves sector labels to ETF tickers (checks sub-sector first, falls back to sector)
- File-based JSON caching per (ETF, date) pair in `data/vol_crush/sector_iv_cache/` to avoid redundant API calls
- ETF price lookups via Yahoo Finance with in-memory caching

**Part G — Relative IV Features (4 features)**
- `iv_stock_minus_sector`: stock ATM IV minus sector ETF ATM IV (positive = earnings-specific premium)
- `iv_stock_sector_ratio`: stock IV divided by sector IV (e.g., 4x means stock has 4 times the sector's vol)
- `straddle_stock_minus_sector`: straddle premium differential between stock and sector
- `iv_elevated_vs_sector`: binary flag (1 if stock/sector IV ratio exceeds 1.5)

**Part H — Missing Data Handling**
- Two strategies based on model type:
  - Tree models (LightGBM, XGBoost): forward-fill VIX, fill regime with 0 (assume contango), leave sector IV as NaN (native NaN support)
  - Linear models (Logistic Regression, Random Forest): forward-fill VIX, median-impute sector IV features

**Part I — Feature Validation**
- Sanity-checks all features against expected ranges
- Reports coverage (missing %), min/max, out-of-range counts
- Flags features with high missing rates or unexpected values

**Part J — Master Pipeline Function**
- `add_market_wide_vol_features()` runs the full 6-step pipeline in one call:
  1. Fetch VIX data
  2. Compute VIX features
  3. Compute sector IV features
  4. Compute relative IV features
  5. Handle missing data
  6. Validate

### `data/vol_crush/vix_cache/vix_daily.csv`

Cached VIX and VIX3M daily closing prices. Generated automatically by the pipeline on first run. Avoids re-downloading from Yahoo Finance on subsequent runs.

---

## Feature Summary

| Category | Feature | Description |
|----------|---------|-------------|
| VIX | `vix_level` | Raw VIX close |
| VIX | `vix3m_level` | Raw VIX3M close |
| VIX | `vix_term_structure_ratio` | VIX / VIX3M (>1 = backwardation) |
| VIX | `vix_term_structure_regime` | Binary panic flag |
| VIX | `vix_percentile_252d` | Trailing year percentile (0-1) |
| VIX | `vix_change_5d` | 5-day VIX momentum |
| VIX | `vix_change_21d` | 21-day VIX trend |
| Sector IV | `sector_atm_iv_call` | Sector ETF ATM call IV |
| Sector IV | `sector_atm_iv_put` | Sector ETF ATM put IV |
| Sector IV | `sector_atm_iv_avg` | Average ATM IV |
| Sector IV | `sector_straddle_pct` | Sector straddle as % of price |
| Relative | `iv_stock_minus_sector` | Stock IV minus sector IV |
| Relative | `iv_stock_sector_ratio` | Stock IV / sector IV |
| Relative | `straddle_stock_minus_sector` | Straddle premium differential |
| Relative | `iv_elevated_vs_sector` | Binary flag (ratio > 1.5) |

**Total: 15 new market-wide features**, bringing the projected total from ~70 to ~85 when combined with the planned stock-level features.

---

## Data Leakage Prevention

Every feature uses only data available before the earnings event:

- VIX features use the VIX close from the trading day of or immediately prior to the announcement
- Sector ETF IV is fetched for the same pre-event date
- No post-event information enters any feature computation

---

## Dependencies

- `numpy` — numerical computation
- `pandas` — data manipulation
- `scipy` — Black-Scholes IV inversion (brentq root-finding, norm CDF)
- `yfinance` — VIX/VIX3M data and ETF price lookups (free, no API key needed)
- `requests` — Alpha Vantage API calls for sector ETF option chains

---

## API Usage

| Data Source | API | Cost | Calls Needed |
|-------------|-----|------|-------------|
| VIX / VIX3M daily data | Yahoo Finance | Free | 2 downloads (cached) |
| ETF closing prices | Yahoo Finance | Free | ~100-200 unique (ETF, date) pairs (cached) |
| Sector ETF option chains | Alpha Vantage HISTORICAL_OPTIONS | Premium recommended | ~2,000-4,000 unique (ETF, date) pairs (cached) |

All API results are cached to disk. Subsequent runs consume zero API calls for previously fetched data.

---

## TODO Roadmap Progress

| Phase | Status | Notes |
|-------|--------|-------|
| Data Pipeline | **In progress** | VIX pipeline complete. Sector ETF chain fetching built. Individual stock IV fetching still needed. |
| Feature Engineering | **In progress** | 15 market-wide features built. ~20 stock-level IV features and realized vol features still needed. |
| Target Labeling | Not started | Define `crush_profitable` binary target |
| ML Pipeline | Not started | Temporal split, LightGBM/XGBoost/LR/RF, calibration |
| Backtester | Not started | Short straddle / iron condor P&L simulation |
| Metrics & Benchmarks | Not started | Sharpe, max DD, Calmar, profit factor |
| Visualization | Not started | Equity curves, SHAP, calibration plots |
| Notebook Assembly | Not started | Final presentation notebook |

---

## Next Steps

1. **`data_pipeline.py`** — Individual stock IV fetcher using Alpha Vantage HISTORICAL_OPTIONS (22K calls, requires premium tier and aggressive caching)
2. **`feature_engineering.py`** — Stock-level volatility features (realized vol estimators, IV rank/percentile, straddle pricing, historical move stats) plus merge function for all 85+ features
3. **`target_labeling.py`** — Compute `crush_profitable` (binary) and `crush_pnl_pct` (continuous) from actual vs. expected moves
