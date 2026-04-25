"""
unified_strategy: pooled S&P 500 vol-crush ML pipeline.

Combines per-stock fundamentals (from /Users/ibrahimfiratsoysal/Documents/ML)
with newly-fetched per-event option chains (Alpha Vantage HISTORICAL_OPTIONS)
and SPY market regime features (from spy_strategy/). Predicts whether a
short straddle around an earnings announcement will be profitable.

Modules:
    features        feature engineering + merge across the three sources
    label_targets   crush_profitable target + temporal train/val/test masks
    fetch_event_options   bulk per-event option-chain fetcher (Alpha Vantage)
    ml_pipeline     LR + LightGBM + XGBoost trainer with calibration
    backtest        short-straddle P&L with bid-ask + commissions
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ML_REPO_ROOT = Path("/Users/ibrahimfiratsoysal/Documents/ML")

UNIFIED_DIR = REPO_ROOT / "unified_strategy"
DATA_DIR = UNIFIED_DIR / "data"
CACHE_DIR = UNIFIED_DIR / "cache" / "options"
MODELS_DIR = UNIFIED_DIR / "models"
RESULTS_DIR = UNIFIED_DIR / "results"

# Joshua's SPY daily features
SPY_FEATURES_PATH = REPO_ROOT / "spy_strategy" / "data" / "spy_daily_features.csv"

# User's per-stock raw earnings (one CSV per ticker)
ML_RAW_DIR = ML_REPO_ROOT / "data" / "raw"
SP500_TICKERS_PATH = ML_REPO_ROOT / "sp500_tickers.csv"

# vol_crush_utils lives in a directory whose name ends in .ipynb;
# Python imports break on the dot, so we add it to sys.path instead.
VOL_CRUSH_UTILS_DIR = REPO_ROOT / "option_volatility_crush.ipynb"

# Date window matching Joshua's SPY data
WINDOW_START = "2021-06-23"
WINDOW_END = "2026-04-21"

# Temporal split (per plan section A.3)
TRAIN_END = "2023-09-30"
VAL_END = "2024-09-30"
