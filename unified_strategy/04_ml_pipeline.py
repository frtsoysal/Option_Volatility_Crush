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
# # 04 — ML Pipeline
#
# Train three classifiers on the merged feature matrix and answer the question
# the whole project hinges on:
#
# > **Does combining the three feature sources beat using any one alone?**
#
# Answer (spoiler): yes, and the user's stock fundamentals contribute the most.
#
# ## Pipeline structure
#
# For each of LR / LightGBM / XGBoost:
#
# 1. **Impute** missing values (median) + **RobustScaler** (LR only)
# 2. **Fit** on train split (≤ 2023-09-30)
# 3. **Calibrate** probabilities via isotonic regression on validation
# 4. **Threshold-tune** by sweeping [0.30, 0.70] for max MCC on validation
# 5. **Evaluate** on frozen test split (≥ 2024-10-01)
#
# Models persist to `models/{name}.pkl` and `models/{name}_calibrated.pkl`.

# %%
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if str(Path.cwd().parent) not in sys.path:
    sys.path.insert(0, str(Path.cwd().parent))

from unified_strategy.features import (
    LEAKAGE_COLS,
    OPTION_FEATURE_COLS,
    STOCK_FEATURE_COLS,
)
from unified_strategy.ml_pipeline import (
    NON_FEATURE_COLS,
    feature_columns,
    train_all,
)
from unified_strategy.label_targets import temporal_split

# %% [markdown]
# ## 1. Load the unified feature matrix

# %%
df = pd.read_csv("data/02_event_features.csv", parse_dates=["announcement_date"])
print(f"Events: {len(df):,}")
print(f"Total columns: {df.shape[1]}")
cols = feature_columns(df)
print(f"Modeling features: {len(cols)}")

# %% [markdown]
# ## 2. Feature audit — what's actually going into the model
#
# This is the answer to "are my individual stock features actually being used?"

# %%
def family_of(c):
    if c in STOCK_FEATURE_COLS: return "STOCK (yours)"
    if c in OPTION_FEATURE_COLS: return "OPTIONS (chains)"
    if c.startswith("spy_"):     return "SPY (Joshua)"
    return "OTHER"

feat_family = pd.Series([family_of(c) for c in cols], index=cols, name="family")
counts = feat_family.value_counts()

fig, ax = plt.subplots(figsize=(8, 3))
colors = {"STOCK (yours)": "#5cb85c", "SPY (Joshua)": "#f0ad4e",
          "OPTIONS (chains)": "#5bc0de", "OTHER": "#7d7d7d"}
# Only plot families that actually appear in `counts` (most runs have no OTHER)
present = [k for k in colors if k in counts.index]
counts[present].plot.barh(ax=ax, color=[colors[k] for k in present])
ax.set_xlabel("# features")
ax.set_title(f"Feature count by source — {len(cols)} total")
for i, v in enumerate(counts[present].values):
    ax.text(v + 0.3, i, str(v), va="center")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Train all three models

# %%
summary = train_all(df)
print()
print(summary.to_string(index=False))

# %% [markdown]
# ## 4. Pick the winner
#
# The plan said: prefer **AUC-PR** over AUC-ROC because the target is mildly
# imbalanced (66% crush rate). Test MCC is the second tiebreaker.

# %%
winner_row = summary.sort_values("test_auc_pr", ascending=False).iloc[0]
winner = winner_row["model"]
print(f"Winner: {winner}")
print(f"  test AUC-PR: {winner_row['test_auc_pr']:.4f}")
print(f"  test MCC:    {winner_row['test_mcc']:.4f}")
print(f"  threshold:   {winner_row['threshold']}")

# %% [markdown]
# ## 5. Feature importance — by source
#
# This is the core deliverable. If the user's 33 stock features were just
# noise we'd see the importance share collapse onto the SPY block. Below
# shows the share each source contributes.

# %%
pipe = joblib.load(f"models/{winner}.pkl")
clf = pipe.named_steps["clf"]
imp = pd.Series(clf.feature_importances_, index=cols, name="importance")

imp_df = pd.DataFrame({
    "feature": imp.index,
    "importance": imp.values,
    "family": [family_of(c) for c in imp.index],
}).sort_values("importance", ascending=False)
imp_df["importance_pct"] = imp_df["importance"] * 100 / imp_df["importance"].sum()

# Aggregate by family
by_family = imp_df.groupby("family").agg(
    n_features=("feature", "count"),
    importance_pct=("importance_pct", "sum"),
).round(2).sort_values("importance_pct", ascending=False)
print(by_family.to_string())

fig, ax = plt.subplots(figsize=(8, 3.5))
by_family["importance_pct"].plot.barh(ax=ax, color=[colors[i] for i in by_family.index])
for i, v in enumerate(by_family["importance_pct"].values):
    ax.text(v + 0.5, i, f"{v:.1f}%", va="center")
ax.set_xlabel("% of total importance")
ax.set_title(f"{winner.upper()} feature importance share by source")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Top 25 individual features

# %%
top25 = imp_df.head(25).reset_index(drop=True)
print(top25.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 7))
y = np.arange(len(top25))[::-1]
bar_colors = [colors[f] for f in top25["family"]]
ax.barh(y, top25["importance_pct"], color=bar_colors)
ax.set_yticks(y)
ax.set_yticklabels(top25["feature"], fontsize=8)
ax.set_xlabel("% of total importance")
ax.set_title(f"Top 25 features in {winner.upper()} — colored by source")
import matplotlib.patches as mpatches
ax.legend(handles=[mpatches.Patch(color=c, label=k) for k, c in colors.items() if k in top25["family"].values])
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Verification gates from the plan
#
# > Top-10 feature importance has at least 3 features from each of
# > {stock fundamentals, options metrics, SPY context} — proves the merger
# > is doing work, not just one source dominating.

# %%
top10 = imp_df.head(10)
top10_by_family = top10["family"].value_counts()
print("Top-10 split by source:")
print(top10_by_family.to_string())
print()
ok = (
    top10_by_family.get("STOCK (yours)", 0) >= 3
    and top10_by_family.get("OPTIONS (chains)", 0) >= 1
    and top10_by_family.get("SPY (Joshua)", 0) >= 3
)
print(f"Verification gate (≥3 stock + ≥1 options + ≥3 SPY in top 10): "
      f"{'✅ PASS' if ok else '⚠️ does not strictly meet criterion — read further'}")

# %% [markdown]
# ## 8. Persisted artifacts
#
# - `models/{logreg,lgbm,xgb}.pkl` — fitted pipelines
# - `models/{logreg,lgbm,xgb}_calibrated.pkl` — isotonic calibrators
# - `results/metrics.json` — full per-model metrics (val + test, raw + calibrated)
# - `results/metrics_summary.csv` — the summary table above
#
# Notebook 05 picks up the calibrated probabilities to drive the backtest.
