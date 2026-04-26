"""
Walk-forward cross-validation for the vol-crush ML pipeline.

A single 70/15/15 temporal split tells us only one thing about
generalization: did the model that learned 2021-2023 generalize to
2024-2025? If the answer is "yes" we don't yet know whether the
strategy works **in every regime** or just got lucky on this slice.

Walk-forward CV addresses that. We slide a 6-month test window across
the entire 2021-06-23 → 2026-04-21 period, retraining the model from
scratch at each fold using only data up to that fold's start. Each test
prediction is therefore strictly out-of-sample relative to its fold's
training cutoff. Stitching the per-fold test results gives a single
4-5-year OOS track record — far more credible than a 14-month window.

Splits (default: expanding-train, 6mo val + 6mo test, sliding 6mo):

  fold | train cutoff | val window           | test window
  -----|--------------|----------------------|------------------------
   1   | 2022-09-30   | 2022-10 → 2023-03    | 2023-04 → 2023-09
   2   | 2023-03-31   | 2023-04 → 2023-09    | 2023-10 → 2024-03
   3   | 2023-09-30   | 2023-10 → 2024-03    | 2024-04 → 2024-09
   4   | 2024-03-31   | 2024-04 → 2024-09    | 2024-10 → 2025-03
   5   | 2024-09-30   | 2024-10 → 2025-03    | 2025-04 → 2025-09
   6   | 2025-03-31   | 2025-04 → 2025-09    | 2025-10 → 2026-04

Per fold we:
  1. Train LR/LightGBM/XGBoost on the train slice
  2. Calibrate probabilities (isotonic) on val
  3. Tune threshold on val to maximize $P&L under the chosen exit_mode
  4. Predict on test, score with that threshold
  5. Save per-fold metrics + per-fold test trades

After all folds: concatenate per-fold test trades into one chronological
sequence and recompute strategy-level metrics on the merged track.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import joblib
import numpy as np
import pandas as pd

from . import MODELS_DIR, RESULTS_DIR
from .backtest import metrics, equity_curve, per_event_pnl
from .label_targets import temporal_split  # noqa: F401  (kept for symmetry)
from .ml_pipeline import (
    NON_FEATURE_COLS,
    build_lgbm_pipeline,
    build_lr_pipeline,
    build_xgb_pipeline,
    feature_columns,
    find_best_threshold,
    HAS_LGBM,
    HAS_XGB,
)
from sklearn.calibration import CalibratedClassifierCV

try:
    from sklearn.frozen import FrozenEstimator
    _HAS_FROZEN = True
except ImportError:
    _HAS_FROZEN = False


@dataclass
class Fold:
    name: str
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    test_end: pd.Timestamp


def default_folds(start: str = "2022-09-30") -> list[Fold]:
    """Six 6-month-step folds covering 2022-Q4 through 2026-Q1."""
    s = pd.Timestamp(start)
    folds = []
    for i in range(6):
        train_end = s + pd.DateOffset(months=6 * i)
        val_end = train_end + pd.DateOffset(months=6)
        test_end = val_end + pd.DateOffset(months=6)
        folds.append(
            Fold(
                name=f"fold_{i + 1}",
                train_end=train_end,
                val_end=val_end,
                test_end=test_end,
            )
        )
    return folds


def split_one(df: pd.DataFrame, fold: Fold, date_col: str = "announcement_date"):
    """Return (train_df, val_df, test_df) for a single fold."""
    d = pd.to_datetime(df[date_col])
    train = df[d <= fold.train_end].copy()
    val = df[(d > fold.train_end) & (d <= fold.val_end)].copy()
    test = df[(d > fold.val_end) & (d <= fold.test_end)].copy()
    return train, val, test


def _fit_calibrated(pipe, X_train, y_train, X_val, y_val):
    pipe.fit(X_train, y_train)
    if _HAS_FROZEN:
        cal = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic")
    else:
        cal = CalibratedClassifierCV(estimator=pipe, method="isotonic", cv="prefit")
    cal.fit(X_val, y_val)
    return pipe, cal


def train_fold(
    df: pd.DataFrame,
    fold: Fold,
    exit_mode: str = "hold_to_expiry",
    target: str = "crush_profitable",
) -> dict:
    """Train all models for one fold, return per-strategy test trades + metrics."""
    train, val, test = split_one(df, fold)
    if train.empty or val.empty or test.empty:
        return {"fold": fold.name, "n_train": len(train), "n_val": len(val), "n_test": len(test)}

    # Drop nans on target
    train = train.dropna(subset=[target])
    val = val.dropna(subset=[target])
    test = test.dropna(subset=[target])

    cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X_train, y_train = train[cols], train[target].astype(int)
    X_val, y_val = val[cols], val[target].astype(int)
    X_test, y_test = test[cols], test[target].astype(int)

    val_pnl = per_event_pnl(val.copy(), exit_mode=exit_mode)["pnl_dollars"].fillna(0).values

    fold_results = {
        "fold": fold.name,
        "train_end": fold.train_end.strftime("%Y-%m-%d"),
        "val_end": fold.val_end.strftime("%Y-%m-%d"),
        "test_end": fold.test_end.strftime("%Y-%m-%d"),
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "models": {},
    }

    # Build per-fold model list (skip lgbm/xgb if not installed)
    builders = [("logreg", build_lr_pipeline())]
    if HAS_LGBM:
        builders.append(("lgbm", build_lgbm_pipeline()))
    if HAS_XGB:
        spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
        builders.append(("xgb", build_xgb_pipeline(scale_pos_weight=spw)))

    test_with_pnl = per_event_pnl(test.copy(), exit_mode=exit_mode)

    for name, pipe in builders:
        try:
            pipe, cal = _fit_calibrated(pipe, X_train, y_train, X_val, y_val)
        except Exception as e:
            fold_results["models"][name] = {"error": str(e)}
            continue
        proba_val = cal.predict_proba(X_val)[:, 1]
        proba_test = cal.predict_proba(X_test)[:, 1]
        thr, score = find_best_threshold(
            y_val.values, proba_val, pnl_dollars=val_pnl, objective="pnl"
        )
        # Test trades selected by ML
        sel_mask = proba_test >= thr
        sel = test_with_pnl[sel_mask].copy()
        sel["fold"] = fold.name
        sel["model"] = name
        sel["proba"] = proba_test[sel_mask]

        fold_results["models"][name] = {
            "threshold": float(thr),
            "val_score": float(score),
            "test_n": int(sel_mask.sum()),
            "test_win_rate": float((sel["pnl_dollars"] > 0).mean()) if len(sel) else float("nan"),
            "test_total_pnl": float(sel["pnl_dollars"].sum()),
            "test_avg_pnl": float(sel["pnl_dollars"].mean()) if len(sel) else float("nan"),
            "test_sharpe": float(np.sqrt(252) * sel["pnl_dollars"].mean() / sel["pnl_dollars"].std())
                           if len(sel) > 1 and sel["pnl_dollars"].std() > 0 else float("nan"),
            "trades": sel,  # the per-trade DataFrame, used to stitch concat track
        }

    # always-short and vrp_only for this fold
    fold_results["models"]["always_short"] = {
        "threshold": None, "test_n": len(test_with_pnl),
        "test_win_rate": float((test_with_pnl["pnl_dollars"] > 0).mean()),
        "test_total_pnl": float(test_with_pnl["pnl_dollars"].sum()),
        "test_avg_pnl": float(test_with_pnl["pnl_dollars"].mean()),
        "trades": test_with_pnl.assign(fold=fold.name, model="always_short"),
    }
    if "spy_vrp_30d" in test_with_pnl.columns:
        vrp_sel = test_with_pnl[test_with_pnl["spy_vrp_30d"] > 0].copy()
        if len(vrp_sel):
            fold_results["models"]["vrp_only"] = {
                "threshold": None, "test_n": len(vrp_sel),
                "test_win_rate": float((vrp_sel["pnl_dollars"] > 0).mean()),
                "test_total_pnl": float(vrp_sel["pnl_dollars"].sum()),
                "test_avg_pnl": float(vrp_sel["pnl_dollars"].mean()),
                "trades": vrp_sel.assign(fold=fold.name, model="vrp_only"),
            }

    return fold_results


def run_walk_forward(
    df: pd.DataFrame,
    exit_mode: str = "hold_to_expiry",
    folds: list[Fold] | None = None,
    out_dir: Path | None = None,
) -> dict:
    """Run all folds and produce per-fold + concatenated results."""
    folds = folds or default_folds()
    out_dir = (out_dir or (RESULTS_DIR / "walk_forward" / exit_mode))
    out_dir.mkdir(parents=True, exist_ok=True)

    all_fold_results = []
    print(f"Walk-forward: {len(folds)} folds, exit_mode={exit_mode}")
    print(f"Output:       {out_dir}")
    print()

    for fold in folds:
        print(f"=== {fold.name}  train≤{fold.train_end.date()}  "
              f"val→{fold.val_end.date()}  test→{fold.test_end.date()} ===")
        result = train_fold(df, fold, exit_mode=exit_mode)
        if "models" not in result:
            print(f"  skipped: empty split (n_train={result['n_train']}, n_val={result['n_val']}, n_test={result['n_test']})")
            continue

        for name, m in result["models"].items():
            if "error" in m:
                print(f"  {name}: ERROR — {m['error']}")
                continue
            thr_s = f"@{m['threshold']:.2f}" if m.get("threshold") else ""
            print(f"  {name:<14}{thr_s:<8}  n={m['test_n']:>4}  "
                  f"win={m['test_win_rate']:.2%}  avg=${m['test_avg_pnl']:>7.0f}  "
                  f"total=${m['test_total_pnl']:>10,.0f}")
        all_fold_results.append(result)
        print()

    # Stitch per-fold trades into one chronological DataFrame per model
    by_model: dict[str, list[pd.DataFrame]] = {}
    for fr in all_fold_results:
        for name, m in fr.get("models", {}).items():
            if "trades" in m and len(m["trades"]):
                by_model.setdefault(name, []).append(m["trades"])

    print("=" * 60)
    print("Concatenated walk-forward results (all folds, chronological)")
    print("=" * 60)
    summary_rows = []
    for name, frames in by_model.items():
        combined = pd.concat(frames, ignore_index=True).sort_values("announcement_date")
        eq = equity_curve(combined["pnl_dollars"], combined["announcement_date"])
        m = metrics(eq, combined)
        m["strategy"] = name
        summary_rows.append(m)
        # Persist
        combined.to_csv(out_dir / f"trades_{name}.csv", index=False)
        eq.to_csv(out_dir / f"equity_{name}.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    cols_order = ["strategy", "n_trades", "win_rate", "avg_pnl_dollars",
                  "total_pnl_dollars", "total_return_on_100k", "sharpe", "max_drawdown"]
    summary = summary[[c for c in cols_order if c in summary.columns]]
    summary.to_csv(out_dir / "walk_forward_summary.csv", index=False)
    print(summary.to_string(index=False))

    # Persist per-fold detail
    fold_details = []
    for fr in all_fold_results:
        for name, m in fr.get("models", {}).items():
            if "trades" in m:
                m_no_trades = {k: v for k, v in m.items() if k != "trades"}
                fold_details.append({
                    "fold": fr["fold"], "model": name,
                    **{k: v for k, v in fr.items() if k not in ("models",)},
                    **m_no_trades,
                })
    detail_df = pd.DataFrame(fold_details)
    detail_df.to_csv(out_dir / "walk_forward_per_fold.csv", index=False)

    return {
        "fold_results": all_fold_results,
        "summary": summary,
        "per_fold_detail": detail_df,
    }


if __name__ == "__main__":
    # Allow running this module directly:
    #   python -m unified_strategy.walk_forward
    import argparse
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-csv", default=str(Path(__file__).parent / "data" / "02_event_features.csv"))
    ap.add_argument("--exit-mode", default="hold_to_expiry",
                    choices=["t_plus_1", "hold_to_expiry"])
    args = ap.parse_args()

    df = pd.read_csv(args.features_csv, parse_dates=["announcement_date"])
    df = df.dropna(subset=["crush_profitable", "stock_price_pre", "stock_price_post"]).reset_index(drop=True)
    print(f"Loaded {len(df):,} usable events")
    run_walk_forward(df, exit_mode=args.exit_mode)
