"""
Pooled vol-crush classifier.

Three models trained on the merged 80-feature matrix:
  - Logistic Regression (baseline, calibrated)
  - LightGBM (primary)
  - XGBoost (comparison)

Pipeline per model:
  1. impute (median) + RobustScaler   [sklearn Pipeline]
  2. fit on train (announcement_date <= 2023-09-30)
  3. RandomizedSearchCV with TimeSeriesSplit(5) inside the train window,
     scoring by average_precision (AUC-PR)
  4. isotonic calibration on validation (2023-10..2024-09)
  5. threshold sweep on validation predictions, pick max MCC

Models are persisted under `unified_strategy/models/`. Metrics + threshold
under `unified_strategy/results/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer

# Keep DataFrames flowing through Pipeline steps so LightGBM/XGBoost see the
# same column names at fit and predict time (silences the feature-names warning).
set_config(transform_output="pandas")

# FrozenEstimator was added in sklearn 1.6 to replace cv='prefit' in
# CalibratedClassifierCV. Older sklearn falls back to cv='prefit'.
try:
    from sklearn.frozen import FrozenEstimator
    _HAS_FROZEN = True
except ImportError:
    _HAS_FROZEN = False
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from . import MODELS_DIR, RESULTS_DIR
from .features import LEAKAGE_COLS, OPTION_FEATURE_COLS, STOCK_FEATURE_COLS
from .label_targets import temporal_split

# Identifier and target columns that must NOT enter the feature matrix
NON_FEATURE_COLS = {
    "ticker",
    "announcement_date",
    "fiscal_quarter_end",
    "spy_join_date",
    "crush_profitable",
    "crush_pnl_pct",
    "actual_move_pct",
    # Raw price levels are excluded — they vary wildly across stocks and would
    # let the model learn ticker identity. The size-normalized straddle_pct_pre
    # carries the relevant info.
    "stock_price_pre",
    "stock_price_post",
    "atm_strike_pre",
    "atm_call_mid_pre",
    "atm_put_mid_pre",
    "straddle_price_pre",
    # NOTE: atm_open_interest_pre + atm_volume_pre are KEPT as features —
    # liquidity signals are valid pre-event predictors and not size-confounded
    # in any leakage sense. (Review fix #1.)
    *LEAKAGE_COLS,
}


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the columns of `df` that should be used as model features."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def _split_xy(
    df: pd.DataFrame, target: str = "crush_profitable"
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    cols = feature_columns(df)
    return df[cols].copy(), df[target].astype(int), cols


def build_lr_pipeline(C: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
            (
                "clf",
                LogisticRegression(
                    C=C, class_weight="balanced", solver="liblinear", max_iter=2000
                ),
            ),
        ]
    )


def build_lgbm_pipeline(**kwargs) -> Pipeline:
    if not HAS_LGBM:
        raise RuntimeError("lightgbm not installed")
    defaults = dict(
        n_estimators=500,
        num_leaves=31,
        learning_rate=0.05,
        max_depth=7,
        min_child_samples=20,
        is_unbalance=True,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    defaults.update(kwargs)
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("clf", lgb.LGBMClassifier(**defaults)),
        ]
    )


def build_xgb_pipeline(scale_pos_weight: float = 1.0, **kwargs) -> Pipeline:
    if not HAS_XGB:
        raise RuntimeError("xgboost not installed")
    defaults = dict(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )
    defaults.update(kwargs)
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("clf", xgb.XGBClassifier(**defaults)),
        ]
    )


def find_best_threshold(
    y_true: np.ndarray, y_score: np.ndarray, lo: float = 0.30, hi: float = 0.70
) -> tuple[float, float]:
    """Sweep threshold in [lo, hi] step 0.01, return (best_threshold, best_mcc)."""
    best_t, best = lo, -1.0
    for t in np.arange(lo, hi + 1e-9, 0.01):
        y_hat = (y_score >= t).astype(int)
        m = matthews_corrcoef(y_true, y_hat)
        if m > best:
            best, best_t = m, float(t)
    return best_t, best


def evaluate(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_hat = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "base_rate": float(np.mean(y_true)),
        "auc_roc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else float("nan"),
        "auc_pr": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "log_loss": float(log_loss(y_true, np.clip(y_score, 1e-9, 1 - 1e-9))),
        "mcc": float(matthews_corrcoef(y_true, y_hat)),
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def train_one(
    name: str,
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models_dir: Path = MODELS_DIR,
    results_dir: Path = RESULTS_DIR,
) -> dict:
    """Fit, calibrate on val, threshold-tune on val, evaluate on test."""
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    pipe.fit(X_train, y_train)

    # Isotonic calibration on validation predictions only
    raw_val = pipe.predict_proba(X_val)[:, 1]
    raw_test = pipe.predict_proba(X_test)[:, 1]

    if _HAS_FROZEN:
        cal = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic")
    else:
        cal = CalibratedClassifierCV(estimator=pipe, method="isotonic", cv="prefit")
    cal.fit(X_val, y_val)
    cal_val = cal.predict_proba(X_val)[:, 1]
    cal_test = cal.predict_proba(X_test)[:, 1]

    threshold, mcc_val = find_best_threshold(y_val.values, cal_val)

    metrics = {
        "name": name,
        "val_raw": evaluate(y_val.values, raw_val),
        "val_calibrated": evaluate(y_val.values, cal_val, threshold=threshold),
        "test_raw": evaluate(y_test.values, raw_test),
        "test_calibrated": evaluate(y_test.values, cal_test, threshold=threshold),
        "threshold": threshold,
    }

    joblib.dump(pipe, models_dir / f"{name}.pkl")
    joblib.dump(cal, models_dir / f"{name}_calibrated.pkl")
    return metrics


def train_all(
    df: pd.DataFrame,
    target: str = "crush_profitable",
    date_col: str = "announcement_date",
    models_dir: Path = MODELS_DIR,
    results_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    """
    Train LR + LightGBM + XGBoost on the unified frame.
    Returns a metrics DataFrame and persists models + a metrics.json.
    """
    df = df.dropna(subset=[target]).copy()
    masks = temporal_split(df, date_col=date_col)
    train, val, test = df[masks["train"]], df[masks["val"]], df[masks["test"]]
    print(f"Train: {len(train):,}   Val: {len(val):,}   Test: {len(test):,}")
    print(f"Class balance (train): {train[target].mean():.3f}")

    X_train, y_train, feature_cols = _split_xy(train, target)
    X_val = val[feature_cols].copy()
    X_test = test[feature_cols].copy()
    y_val = val[target].astype(int)
    y_test = test[target].astype(int)
    print(f"Features: {len(feature_cols)}")

    results = []

    # 1. Logistic Regression
    print("\n[1/3] Logistic Regression")
    results.append(
        train_one(
            "logreg",
            build_lr_pipeline(),
            X_train, y_train, X_val, y_val, X_test, y_test,
            models_dir, results_dir,
        )
    )

    # 2. LightGBM
    if HAS_LGBM:
        print("\n[2/3] LightGBM")
        results.append(
            train_one(
                "lgbm",
                build_lgbm_pipeline(),
                X_train, y_train, X_val, y_val, X_test, y_test,
                models_dir, results_dir,
            )
        )
    else:
        print("\n[2/3] LightGBM SKIPPED (not installed)")

    # 3. XGBoost
    if HAS_XGB:
        print("\n[3/3] XGBoost")
        spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
        results.append(
            train_one(
                "xgb",
                build_xgb_pipeline(scale_pos_weight=spw),
                X_train, y_train, X_val, y_val, X_test, y_test,
                models_dir, results_dir,
            )
        )
    else:
        print("\n[3/3] XGBoost SKIPPED (not installed)")

    # Tabular summary
    rows = []
    for r in results:
        rows.append(
            {
                "model": r["name"],
                "threshold": round(r["threshold"], 3),
                "val_auc_pr": round(r["val_calibrated"]["auc_pr"], 4),
                "val_auc_roc": round(r["val_calibrated"]["auc_roc"], 4),
                "val_mcc": round(r["val_calibrated"]["mcc"], 4),
                "val_brier": round(r["val_calibrated"]["brier"], 4),
                "test_auc_pr": round(r["test_calibrated"]["auc_pr"], 4),
                "test_auc_roc": round(r["test_calibrated"]["auc_roc"], 4),
                "test_mcc": round(r["test_calibrated"]["mcc"], 4),
                "test_brier": round(r["test_calibrated"]["brier"], 4),
            }
        )

    summary = pd.DataFrame(rows)
    (results_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
    summary.to_csv(results_dir / "metrics_summary.csv", index=False)
    print(f"\nSaved metrics → {results_dir / 'metrics.json'}")
    return summary
