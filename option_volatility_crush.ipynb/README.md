# NVDA Volatility Crush Pilot

A single-stock pilot that tests an ML-powered option volatility crush strategy on NVDA earnings events. This is a proof-of-concept -- the dataset is intentionally small (26 events) to validate the full pipeline before expanding to the S&P 500 universe.

## The Idea

Before major events like earnings announcements, option implied volatility (IV) spikes because the market prices in uncertainty. After the event, IV drops sharply -- this is called a **volatility crush**. If the actual stock move is smaller than what the options market expected (the straddle price), selling a straddle is profitable.

The question is: **can ML predict which earnings events will have a profitable vol crush?**

```
Pre-earnings IV spike  -->  Event occurs  -->  IV collapses
                                               |
                                    Did actual move < expected move?
                                               |
                                    YES: straddle seller profits
                                    NO:  straddle seller loses
```

## What This Pilot Does

1. **Collects real historical options data** from Alpha Vantage for every NVDA earnings event (2017-2025)
2. **Computes implied volatility** from option prices and measures the IV crush magnitude
3. **Engineers 81 features** by merging existing earnings ML features (analyst estimates, revision momentum, margins) with new vol features (realized vol, IV levels, straddle pricing)
4. **Trains two ML models** (Logistic Regression + LightGBM) with temporal train/val/test splits
5. **Backtests** the strategy: ML-filtered trades vs. always-trade baseline
6. **Saves every intermediate step** to CSV so the notebook is fully reproducible without re-running API calls

## Key Findings (NVDA Pilot)

| Metric | Value |
|---|---|
| Earnings events | 26 (2017-2025) |
| Events with options data | 24 of 26 |
| Average pre-earnings IV | 97% |
| Average IV crush | 44.9% |
| Straddle as % of stock | 7.8% |
| Vol crush profitable | 88.5% of events |
| Test set (2024-2025) | 6 events, 100% win rate |

NVDA shows a strong and consistent vol crush pattern -- the actual post-earnings move is almost always smaller than what the options market implied. This makes it an ideal candidate for a short-volatility strategy.

> **Caveat:** 26 events is too small for statistically robust conclusions. This pilot validates the pipeline and shows the concept works. The real test comes when we expand to hundreds of stocks.

## Project Structure

```
option_volatility_crush.ipynb/
├── README.md                          <- This file
├── vol_crush_pilot.ipynb              <- Main notebook (13 cells)
├── vol_crush_utils.py                 <- Utility module (API, IV, metrics, theme)
├── how_to_trade_volatility_crush_option_strategy.ipynb  <- Reference notebook
├── .gitignore                         <- Excludes cache/
├── pilot_data/                        <- All saved outputs (committed to git)
│   ├── 01_nvda_earnings_events.csv    <- 26 deduped earnings events
│   ├── 02_nvda_earnings_features.csv  <- 50 existing ML features for NVDA
│   ├── 03_options_chains_raw.csv      <- 158K option contracts from Alpha Vantage
│   ├── 04_iv_straddle_metrics.csv     <- IV, straddle pricing, crush metrics
│   ├── 05_vol_features.csv            <- Realized vol, vol-of-vol, expansion ratio
│   ├── 06_full_feature_matrix.csv     <- All 81 features merged
│   ├── 07_labeled_dataset.csv         <- Dataset with crush_profitable target
│   ├── 08_train_val_test_split.csv    <- Temporal split assignments
│   ├── 09_model_predictions.csv       <- LR + LightGBM probabilities
│   ├── 10_backtest_trades.csv         <- Per-trade P&L for all strategies
│   ├── 11_backtest_equity.csv         <- Equity curves
│   ├── 12_evaluation_metrics.csv      <- MCC, AUC-PR, Brier scores
│   ├── feature_cols.csv              <- Feature column names used by models
│   └── models/
│       ├── logistic_regression.joblib <- Saved LR pipeline
│       └── lightgbm.joblib            <- Saved LightGBM pipeline
└── cache/                             <- API response cache (gitignored)
    └── NVDA/                          <- Per-date JSON files
```

## Data Pipeline

```
data/raw/NVDA.csv ──> deduplicate ──> 01_nvda_earnings_events.csv
                                              │
sp500_features.csv ──> filter NVDA ──> 02_nvda_earnings_features.csv
                                              │
Alpha Vantage API ──> cache + merge ──> 03_options_chains_raw.csv
         │                                    │
         └──> ATM options + IV ──────> 04_iv_straddle_metrics.csv
                                              │
yfinance prices ──> realized vol ──────> 05_vol_features.csv
                                              │
         merge all ────────────────────> 06_full_feature_matrix.csv
                                              │
         + target label ───────────────> 07_labeled_dataset.csv
                                              │
         temporal split ───────────────> 08_train_val_test_split.csv
                                              │
         LR + LightGBM ───────────────> 09_model_predictions.csv
                                              │
         backtest P&L ─────────────────> 10_backtest_trades.csv
                                         11_backtest_equity.csv
                                         12_evaluation_metrics.csv
```

## Methodology

### Features (81 total)

- **50 earnings features** (from existing pipeline): analyst estimates, EPS revision momentum, profit margins, price changes, beat streaks
- **16 IV/options features**: pre/post IV, IV crush %, straddle price %, ATM option prices
- **11 realized vol features**: 5d/21d/63d realized vol, vol-of-vol, expansion ratio, historical earnings move average
- **4 metadata columns**: symbol, date, year, quarter

### Target Variable

`crush_profitable = 1` if the actual post-earnings stock move (%) was smaller than the straddle-implied expected move (%). In other words, the straddle seller would have profited.

### Temporal Split

| Split | Years | Events | Purpose |
|---|---|---|---|
| Train | 2017-2022 | 17 | Model fitting |
| Validation | 2023 | 3 | Model selection |
| Test | 2024-2025 | 6 | Final evaluation |

### Models

- **Logistic Regression**: L2-regularized baseline with RobustScaler, `class_weight='balanced'`
- **LightGBM**: Gradient boosting with `is_unbalance=True`, shallow trees (`max_depth=3`, `num_leaves=8`) to prevent overfitting on small N

### Idempotent Notebook Design

Every cell follows this pattern:

```python
if output_file.exists():
    df = pd.read_csv(output_file)        # load saved
else:
    df = compute_something()              # compute fresh
    df.to_csv(output_file, index=False)   # save for next time
```

First run: fetches data, computes features, trains models (~2 min).
Second run: loads everything from CSVs (~1 sec, zero API calls).

## How to Run

```bash
cd option_volatility_crush.ipynb/
jupyter notebook vol_crush_pilot.ipynb
# Run All Cells
```

Requirements: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `scipy`, `matplotlib`, `seaborn`, `yfinance`, `python-dotenv`, `joblib`, `requests`

Alpha Vantage API key must be set in `../.env` as `ALPHA_VANTAGE_API=your_key`.

## Limitations

- **Small sample** (26 events) -- results are directional, not statistically robust
- **Single stock** (NVDA) -- no cross-sectional diversification or sector comparison
- **Simplified execution** -- uses mid-price, no slippage, no commissions
- **Risk-free rate** approximated at 5% for IV computation
- **Test set is all-positive** -- all 6 test events were profitable, so classification metrics (MCC, AUC) couldn't be computed on test

## Next Steps

1. **Expand to S&P 500** -- apply the same pipeline to ~11K earnings events across 500+ stocks for cross-sectional signals and proper statistical power
2. **Add macro events** -- use FMP API to incorporate CPI, FOMC, NFP proximity as features
3. **Hyperparameter tuning** -- walk-forward cross-validation with Optuna once N is large enough
4. **Realistic costs** -- model bid-ask spread, slippage, and commission impact on P&L
5. **Position sizing** -- implement Kelly Criterion based on model confidence
6. **Iron condor variant** -- capped-risk alternative to short straddles

## Authors

ML class project -- Polytechnic University
