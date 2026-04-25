"""
Target labeling and temporal train/val/test splits.

Mirrors the labeling logic from option_volatility_crush.ipynb/vol_crush_pilot.ipynb
(NVDA pilot) so the unified pipeline produces directly-comparable targets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import TRAIN_END, VAL_END


def label_crush(
    events: pd.DataFrame,
    price_pre_col: str = "stock_price_pre",
    price_post_col: str = "stock_price_post",
    straddle_pct_col: str = "straddle_pct_pre",
) -> pd.DataFrame:
    """
    Add `actual_move_pct`, `crush_profitable`, `crush_pnl_pct` to an events frame.

    crush_profitable = 1 iff abs(actual_move_pct) < straddle_pct_pre.
    Both values are in percentage units (e.g., 3.4 means 3.4%).
    """
    out = events.copy()
    out["actual_move_pct"] = (
        (out[price_post_col] / out[price_pre_col] - 1).abs() * 100
    )
    out["crush_profitable"] = (
        out["actual_move_pct"] < out[straddle_pct_col]
    ).astype(int)
    out["crush_pnl_pct"] = out[straddle_pct_col] - out["actual_move_pct"]
    return out


def temporal_split(
    events: pd.DataFrame,
    date_col: str = "announcement_date",
    train_end: str = TRAIN_END,
    val_end: str = VAL_END,
) -> dict[str, pd.Series]:
    """
    Return boolean masks for train / val / test based on announcement date.

    Default split:
      train: <= 2023-09-30
      val:   2023-10-01 .. 2024-09-30
      test:  >= 2024-10-01
    """
    dates = pd.to_datetime(events[date_col])
    return {
        "train": dates <= train_end,
        "val": (dates > train_end) & (dates <= val_end),
        "test": dates > val_end,
    }


def split_summary(events: pd.DataFrame, date_col: str = "announcement_date") -> pd.DataFrame:
    """One-line-per-split summary: rows, % of total, date range, crush rate (if labeled)."""
    masks = temporal_split(events, date_col)
    rows = []
    for name, mask in masks.items():
        sub = events[mask]
        row = {
            "split": name,
            "n_events": len(sub),
            "pct_total": round(100 * len(sub) / max(len(events), 1), 1),
            "date_min": sub[date_col].min() if not sub.empty else None,
            "date_max": sub[date_col].max() if not sub.empty else None,
        }
        if "crush_profitable" in sub.columns:
            row["crush_rate"] = (
                round(100 * sub["crush_profitable"].mean(), 1) if not sub.empty else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)
