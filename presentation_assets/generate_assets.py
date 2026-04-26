"""
Generate all charts + tables for the class presentation.

Run from repo root:
    python -m presentation_assets.generate_assets
or:
    cd unified_strategy && python ../presentation_assets/generate_assets.py
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from unified_strategy.features import OPTION_FEATURE_COLS, STOCK_FEATURE_COLS
from unified_strategy.label_targets import temporal_split
from unified_strategy.ml_pipeline import feature_columns
from unified_strategy.backtest import per_event_pnl, equity_curve

ASSETS = REPO / "presentation_assets"
ASSETS.mkdir(exist_ok=True)
DATA = REPO / "unified_strategy" / "data"
RESULTS = REPO / "unified_strategy" / "results"
SPY_FEATURES_PATH = REPO / "spy_strategy" / "data" / "spy_daily_features.csv"

# Visual style
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "stock": "#5cb85c",       # green — user's stock features
    "options": "#5bc0de",     # blue — options features
    "spy": "#f0ad4e",          # orange — Joshua's SPY features
    "ml_lgbm": "#1976d2",      # bold blue — winning model
    "ml_xgb": "#7e57c2",       # purple
    "ml_logreg": "#26a69a",    # teal
    "vrp_only": "#ef5350",     # red
    "always_short": "#9e9e9e", # gray
    "buy_hold_spy": "#212121", # near-black
    "win": "#2e7d32",
    "loss": "#c62828",
    "neutral": "#455a64",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Vol crush concept — IV time series for one ticker around earnings
# ─────────────────────────────────────────────────────────────────────────────
def vol_crush_concept():
    """Illustrative: IV ramps up before earnings, crashes after."""
    # Simulate the typical pattern. Real chains only have T-1 and T+1; we
    # generate a stylized curve consistent with academic vol-crush literature.
    days = np.arange(-15, 8)
    iv = 0.30 + 0.40 * np.exp(-(days / 7) ** 2) * (days < 1)
    iv[days >= 1] = 0.30 + 0.05 * np.exp(-((days[days >= 1] - 1) / 4))
    iv += 0.02 * np.random.RandomState(0).randn(len(days)) * 0.5
    iv = np.clip(iv, 0.18, 0.85)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(days, iv * 100, color=COLORS["spy"], linewidth=3, label="ATM Implied Volatility (%)")
    ax.fill_between(days, iv * 100, alpha=0.15, color=COLORS["spy"])
    ax.axvline(0, color=COLORS["loss"], linestyle="--", linewidth=2, alpha=0.7,
               label="Earnings announcement")
    ax.annotate("IV ramps UP\nas event approaches", xy=(-3, 65), xytext=(-13, 75),
                fontsize=11, ha="left",
                arrowprops=dict(arrowstyle="->", color="black", alpha=0.6))
    ax.annotate("IV crashes\n(the 'crush')",
                xy=(2, 38), xytext=(4, 60),
                fontsize=11, ha="left",
                arrowprops=dict(arrowstyle="->", color="black", alpha=0.6))
    ax.set_xlabel("Trading days from earnings announcement")
    ax.set_ylabel("Implied volatility (%)")
    ax.set_title("The volatility crush: IV inflates pre-earnings and collapses post-announcement")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.savefig(ASSETS / "01_vol_crush_concept.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Straddle payoff diagram
# ─────────────────────────────────────────────────────────────────────────────
def straddle_payoff():
    K = 100
    premium = 8
    spot_at_expiry = np.linspace(70, 130, 121)
    long_pl = np.maximum(spot_at_expiry - K, 0) + np.maximum(K - spot_at_expiry, 0) - premium
    short_pl = -long_pl

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(spot_at_expiry, short_pl, color=COLORS["ml_lgbm"], linewidth=3, label="Short straddle (we sell)")
    ax.plot(spot_at_expiry, long_pl, color="#bdbdbd", linewidth=2, linestyle="--",
            label="Long straddle (the buyer)")
    ax.fill_between(spot_at_expiry, short_pl, 0, where=(short_pl > 0),
                    color=COLORS["win"], alpha=0.15, label="Profit zone (small move)")
    ax.fill_between(spot_at_expiry, short_pl, 0, where=(short_pl < 0),
                    color=COLORS["loss"], alpha=0.15, label="Loss zone (big move)")

    ax.axvline(K, color="black", alpha=0.4, linestyle=":")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(K - premium, color=COLORS["loss"], alpha=0.5, linestyle="-.")
    ax.axvline(K + premium, color=COLORS["loss"], alpha=0.5, linestyle="-.")
    ax.text(K, 9.5, "Strike (K=$100)", ha="center", fontsize=9)
    ax.text(K - premium, -10.5, "Lower break-even\nK − premium", ha="center", fontsize=8.5)
    ax.text(K + premium, -10.5, "Upper break-even\nK + premium", ha="center", fontsize=8.5)
    ax.text(K, 6, "max profit\n= premium ($8)", ha="center", fontsize=10, color=COLORS["win"], weight="bold")

    ax.set_xlabel("Underlying price at expiration")
    ax.set_ylabel("Profit / loss per contract ($)")
    ax.set_title("Short-straddle P&L: profit when stock pins near strike, loss for big moves")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_ylim(-25, 12)
    fig.savefig(ASSETS / "02_straddle_payoff.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Data architecture — three sources merged
# ─────────────────────────────────────────────────────────────────────────────
def data_architecture():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axis("off")

    # Three source boxes on the left
    src_boxes = [
        ("STOCK FUNDAMENTALS\n(your /ML pipeline)\n• 33 features\n• EPS estimates, revisions\n• Custom Elo system\n• Lag-1 growth metrics\n\n8,005 SP500 events", "stock", 0.88),
        ("OPTIONS MICROSTRUCTURE\n(newly fetched)\n• 11 features\n• ATM straddle price\n• Pre/post IVs, IV crush %\n• Term-structure slope\n• Liquidity (OI, volume)\n\n16,010 chain JSONs", "options", 0.55),
        ("SPY MARKET REGIME\n(Joshua's pipeline)\n• 30 features\n• VIX, VIX3M\n• ATM IV 30/60/90d\n• VRP, IV rank\n• Skew (25-delta RR)\n\n1,212 trading days", "spy", 0.22),
    ]
    for text, ckey, y in src_boxes:
        ax.add_patch(plt.Rectangle((0.02, y - 0.13), 0.30, 0.20,
                                    facecolor=COLORS[ckey], alpha=0.85, edgecolor="black"))
        ax.text(0.17, y - 0.03, text, ha="center", va="center", fontsize=9, weight="normal")

    # Merge box in middle
    ax.add_patch(plt.Rectangle((0.43, 0.40), 0.18, 0.20,
                                facecolor=COLORS["neutral"], alpha=0.85, edgecolor="black"))
    ax.text(0.52, 0.50, "UNIFIED\nFEATURE\nMATRIX\n\n7,886 events\n× 74 features",
            ha="center", va="center", fontsize=10, color="white", weight="bold")

    # Output box on right
    ax.add_patch(plt.Rectangle((0.72, 0.40), 0.26, 0.20,
                                facecolor=COLORS["ml_lgbm"], alpha=0.9, edgecolor="black"))
    ax.text(0.85, 0.50,
            "ML PIPELINE\n\n• LR / LightGBM / XGBoost\n• Isotonic calibration\n• Walk-forward CV\n• Pooled-val threshold",
            ha="center", va="center", fontsize=9, color="white")

    # Arrows
    for _, _, y in src_boxes:
        ax.annotate("", xy=(0.43, 0.50), xytext=(0.32, y - 0.03),
                    arrowprops=dict(arrowstyle="->", lw=2, color="black"))
    ax.annotate("", xy=(0.72, 0.50), xytext=(0.61, 0.50),
                arrowprops=dict(arrowstyle="->", lw=2, color="black"))

    ax.set_title("Three data sources → 74 features → ML strategy", y=1.02)
    fig.savefig(ASSETS / "03_data_architecture.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 4. SPY market context — VIX and VRP over the 5-year window
# ─────────────────────────────────────────────────────────────────────────────
def spy_context_timeseries():
    spy = pd.read_csv(SPY_FEATURES_PATH, parse_dates=["date"])
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)

    axes[0].plot(spy["date"], spy["vix_close"], color=COLORS["loss"], linewidth=1.5)
    axes[0].fill_between(spy["date"], spy["vix_close"], alpha=0.15, color=COLORS["loss"])
    axes[0].set_ylabel("VIX")
    axes[0].set_title("SPY market regime context (Joshua's bulk fetch, 1,212 trading days)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(spy["date"], spy["vrp_30d"], color=COLORS["spy"], linewidth=1.5)
    axes[1].axhline(0, color="black", linewidth=0.7, alpha=0.6)
    axes[1].fill_between(spy["date"], spy["vrp_30d"], 0,
                          where=(spy["vrp_30d"] > 0), alpha=0.15, color=COLORS["win"], label="VRP > 0 (sell vol)")
    axes[1].fill_between(spy["date"], spy["vrp_30d"], 0,
                          where=(spy["vrp_30d"] <= 0), alpha=0.15, color=COLORS["loss"], label="VRP ≤ 0 (buy vol)")
    axes[1].set_ylabel("Variance risk premium (30d)")
    axes[1].set_xlabel("date")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.savefig(ASSETS / "04_spy_market_context.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Feature importance share by source (the "merger pays off" chart)
# ─────────────────────────────────────────────────────────────────────────────
def feature_importance_by_source():
    df = pd.read_csv(DATA / "02_event_features.csv")
    cols = feature_columns(df)
    pipe = joblib.load(REPO / "unified_strategy" / "models" / "xgb.pkl")
    imp = pd.Series(pipe.named_steps["clf"].feature_importances_, index=cols)

    def fam(c):
        if c in STOCK_FEATURE_COLS: return "STOCK\n(your work)"
        if c in OPTION_FEATURE_COLS: return "OPTIONS\n(newly fetched)"
        if c.startswith("spy_"):     return "SPY\n(Joshua's work)"
        return "OTHER"

    df_imp = pd.DataFrame({"feat": cols, "imp": imp.values, "fam": [fam(c) for c in cols]})
    by_fam = df_imp.groupby("fam")["imp"].sum() * 100 / df_imp["imp"].sum()
    by_fam = by_fam.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 3.8))
    color_map = {"STOCK\n(your work)": COLORS["stock"],
                 "OPTIONS\n(newly fetched)": COLORS["options"],
                 "SPY\n(Joshua's work)": COLORS["spy"]}
    by_fam.plot.barh(ax=ax, color=[color_map[i] for i in by_fam.index])
    for i, v in enumerate(by_fam.values):
        ax.text(v + 0.7, i, f"{v:.1f}%", va="center", fontsize=12, weight="bold")
    ax.set_xlabel("% of total feature importance (XGBoost)")
    ax.set_xlim(0, by_fam.max() * 1.20)
    ax.set_title("All three sources contribute meaningfully — your stock features lead")
    fig.savefig(ASSETS / "05_feature_importance_by_source.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Top-25 features colored by source
# ─────────────────────────────────────────────────────────────────────────────
def feature_importance_top25():
    df = pd.read_csv(DATA / "02_event_features.csv")
    cols = feature_columns(df)
    pipe = joblib.load(REPO / "unified_strategy" / "models" / "xgb.pkl")
    imp = pd.Series(pipe.named_steps["clf"].feature_importances_, index=cols)

    def fam(c):
        if c in STOCK_FEATURE_COLS: return "STOCK"
        if c in OPTION_FEATURE_COLS: return "OPTIONS"
        if c.startswith("spy_"):     return "SPY"
        return "OTHER"

    df_imp = pd.DataFrame({"feat": cols, "imp": imp.values, "fam": [fam(c) for c in cols]})
    df_imp["pct"] = df_imp["imp"] * 100 / df_imp["imp"].sum()
    top25 = df_imp.sort_values("pct", ascending=False).head(25).reset_index(drop=True)

    color_map = {"STOCK": COLORS["stock"], "OPTIONS": COLORS["options"], "SPY": COLORS["spy"]}
    fig, ax = plt.subplots(figsize=(11, 8))
    y = np.arange(len(top25))[::-1]
    ax.barh(y, top25["pct"], color=[color_map[f] for f in top25["fam"]])
    ax.set_yticks(y)
    ax.set_yticklabels(top25["feat"], fontsize=9)
    ax.set_xlabel("% of total importance")
    ax.set_title("Top 25 most important features (XGBoost) — colored by source")
    handles = [mpatches.Patch(color=color_map[k], label=k) for k in ["STOCK", "OPTIONS", "SPY"]]
    ax.legend(handles=handles, loc="lower right")
    fig.savefig(ASSETS / "06_top25_features.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Reliability diagram — model is well-calibrated
# ─────────────────────────────────────────────────────────────────────────────
def reliability_diagram():
    df = pd.read_csv(DATA / "02_event_features.csv", parse_dates=["announcement_date"])
    masks = temporal_split(df)
    val = df[masks["val"]]
    test = df[masks["test"]]
    cal = joblib.load(REPO / "unified_strategy" / "models" / "xgb_calibrated.pkl")
    cols = feature_columns(df)

    def reliability(probs, y, bins=10):
        d = pd.DataFrame({"p": probs, "y": y})
        d["bin"] = pd.cut(d["p"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
        return d.groupby("bin", observed=False).agg(
            bin_mid=("p", "mean"),
            actual_rate=("y", "mean"),
            n=("y", "count"),
        ).dropna().reset_index(drop=True)

    rv = reliability(cal.predict_proba(val[cols])[:, 1], val["crush_profitable"].values)
    rt = reliability(cal.predict_proba(test[cols])[:, 1], test["crush_profitable"].values)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="Perfect calibration")
    ax.scatter(rv["bin_mid"], rv["actual_rate"], s=rv["n"], alpha=0.6,
               color=COLORS["ml_lgbm"], label="Validation (size = n events)")
    ax.scatter(rt["bin_mid"], rt["actual_rate"], s=rt["n"], alpha=0.6,
               color=COLORS["loss"], label="Test (size = n events)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted P(crush profitable)")
    ax.set_ylabel("Actual rate (frequency of crush_profitable=1)")
    ax.set_title("Reliability diagram — XGBoost is well-calibrated\n(predicted probabilities match observed rates within 1-2pp)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.savefig(ASSETS / "07_reliability_diagram.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 8. Friction tax — gap between predicted-crush rate and money-winner rate
# ─────────────────────────────────────────────────────────────────────────────
def friction_tax():
    df = pd.read_csv(DATA / "02_event_features.csv", parse_dates=["announcement_date"])
    masks = temporal_split(df)
    test = df[masks["test"]].reset_index(drop=True)
    test = per_event_pnl(test, exit_mode="t_plus_1")

    n = len(test)
    crush_rate = test["crush_profitable"].mean()
    money_rate = (test["pnl_dollars"] > 0).mean()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(["Crush profitable\n(binary signal)", "Money-winner\n(after frictions)"],
                  [crush_rate * 100, money_rate * 100],
                  color=[COLORS["win"], COLORS["loss"]], alpha=0.85, width=0.55)
    for bar, v in zip(bars, [crush_rate * 100, money_rate * 100]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.1f}%",
                ha="center", fontsize=14, weight="bold")

    gap = (crush_rate - money_rate) * 100
    ax.annotate("", xy=(1, money_rate * 100), xytext=(1, crush_rate * 100),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    ax.text(1.07, (crush_rate + money_rate) * 50, f"Friction tax\n−{gap:.1f}pp",
            fontsize=11, color="black", weight="bold")
    ax.set_ylim(0, max(crush_rate * 100, money_rate * 100) * 1.20)
    ax.set_ylabel("Rate (%)")
    ax.set_title(f"The friction tax — {gap:.1f}pp of 'right-but-still-lost-money' trades")
    fig.savefig(ASSETS / "08_friction_tax.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 9. Per-fold P&L bars — strategy stability across walk-forward
# ─────────────────────────────────────────────────────────────────────────────
def walk_forward_per_fold():
    perf = pd.read_csv(RESULTS / "walk_forward" / "hold_to_expiry" / "walk_forward_per_fold.csv")
    # only ml strategies + always_short
    keep = ["lgbm", "xgb", "always_short"]
    perf = perf[perf["model"].isin(keep)]

    fig, ax = plt.subplots(figsize=(11, 5))
    folds = sorted(perf["fold"].unique())
    width = 0.27
    for i, m in enumerate(["lgbm", "xgb", "always_short"]):
        sub = perf[perf["model"] == m].set_index("fold").reindex(folds)
        x = np.arange(len(folds)) + i * width
        c = COLORS[f"ml_{m}"] if m != "always_short" else COLORS["always_short"]
        ax.bar(x, sub["test_total_pnl"].values / 1000, width, color=c,
               label=("ML " + m) if m != "always_short" else "always_short")

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(folds)) + width)
    ax.set_xticklabels([f"{f}\n{perf[perf['fold']==f]['train_end'].iloc[0]} → {perf[perf['fold']==f]['test_end'].iloc[0]}"
                        for f in folds], fontsize=8)
    ax.set_ylabel("Total $P&L on test ($K)")
    ax.set_title("Per-fold P&L (hold-to-expiry, fold-local thresholds)\nML beats baseline in every fold; only Folds 4 and 6 are net positive")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(ASSETS / "09_per_fold_pnl.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 10. Equity curves — calibrated walk-forward (the headline result)
# ─────────────────────────────────────────────────────────────────────────────
def equity_curves_calibrated():
    base = RESULTS / "walk_forward_calibrated" / "hold_to_expiry"
    strategies = [("lgbm", COLORS["ml_lgbm"], "ml_lgbm @ 0.83 (winner)", 3),
                  ("xgb", COLORS["ml_xgb"], "ml_xgb @ 0.75", 1.6),
                  ("logreg", COLORS["ml_logreg"], "ml_logreg @ 0.79", 1.6),
                  ("vrp_only", COLORS["vrp_only"], "vrp_only (Joshua's signal)", 1.4),
                  ("always_short", COLORS["always_short"], "always_short", 1.4)]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, color, label, lw in strategies:
        p = base / f"equity_{name}.csv"
        if not p.exists():
            continue
        eq = pd.read_csv(p, parse_dates=["date"])
        if eq.empty:
            continue
        ax.plot(eq["date"], eq["equity"] / 1000, color=color, linewidth=lw, label=label)

    ax.axhline(100, color="black", linestyle="--", alpha=0.4, label="Starting capital ($100K)")
    ax.set_xlabel("date")
    ax.set_ylabel("Equity ($K)")
    ax.set_title("Calibrated walk-forward equity curves — only ml_lgbm @ 0.83 stays above the start line")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=0.3)
    fig.savefig(ASSETS / "10_equity_curves_calibrated.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 11. Pnl distribution histogram — tail-risk visual
# ─────────────────────────────────────────────────────────────────────────────
def pnl_distribution():
    trades = pd.read_csv(RESULTS / "walk_forward_calibrated" / "hold_to_expiry" / "trades_lgbm.csv")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(trades["pnl_dollars"], bins=40, color=COLORS["ml_lgbm"], alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    mean = trades["pnl_dollars"].mean()
    median = trades["pnl_dollars"].median()
    ax.axvline(mean, color=COLORS["loss"], linewidth=2, label=f"mean ${mean:.0f}")
    ax.axvline(median, color=COLORS["win"], linewidth=2, label=f"median ${median:.0f}")
    ax.set_xlabel("Per-trade P&L ($)")
    ax.set_ylabel("Number of trades")
    ax.set_title(f"Calibrated ml_lgbm: per-trade P&L distribution (n={len(trades)} trades, hold-to-expiry)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(ASSETS / "11_pnl_distribution.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 12. Threshold sensitivity — P&L vs cutoff
# ─────────────────────────────────────────────────────────────────────────────
def threshold_sensitivity():
    df = pd.read_csv(DATA / "02_event_features.csv", parse_dates=["announcement_date"])
    masks = temporal_split(df)
    val = df[masks["val"]].reset_index(drop=True)
    val = per_event_pnl(val, exit_mode="hold_to_expiry")

    cols = feature_columns(df)
    cal = joblib.load(REPO / "unified_strategy" / "models" / "lgbm_calibrated.pkl")
    val["proba"] = cal.predict_proba(val[cols])[:, 1]

    thrs = np.arange(0.30, 0.95, 0.01)
    rows = []
    for thr in thrs:
        sel = val[val["proba"] >= thr]
        if len(sel) >= 30:
            rows.append({"thr": thr, "n": len(sel),
                         "total_pnl": sel["pnl_dollars"].sum(),
                         "avg_pnl": sel["pnl_dollars"].mean()})
    sweep = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    axes[0].plot(sweep["thr"], sweep["total_pnl"] / 1000, color=COLORS["ml_lgbm"], linewidth=2)
    axes[0].axhline(0, color="black", linewidth=0.7, alpha=0.5)
    best_thr = sweep.loc[sweep["total_pnl"].idxmax(), "thr"]
    axes[0].axvline(best_thr, color=COLORS["win"], linestyle="--",
                    label=f"Best on val: {best_thr:.2f}")
    axes[0].set_ylabel("Total $P&L on val ($K)")
    axes[0].set_title("Validation $P&L by threshold (LGBM, hold-to-expiry)")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(sweep["thr"], sweep["n"], color=COLORS["neutral"], linewidth=2)
    axes[1].axvline(best_thr, color=COLORS["win"], linestyle="--")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("# val trades")
    axes[1].grid(alpha=0.3)

    fig.savefig(ASSETS / "12_threshold_sensitivity.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 13. Methodology rigor ladder — single-split → walk-forward → calibrated
# ─────────────────────────────────────────────────────────────────────────────
def rigor_ladder():
    rungs = [
        ("Naive intrinsic exit", "+$231/trade, Sharpe 6.74", "FANTASY", COLORS["loss"]),
        ("Realistic T+1 close", "−$144/trade, Sharpe −5.46", "DISASTER", COLORS["loss"]),
        ("Hold-to-expiry, single split", "+$184/trade, Sharpe 2.67", "LUCKY", COLORS["spy"]),
        ("Plain walk-forward (per-fold thr)", "−$31/trade, Sharpe −0.65", "REGIME-DEPENDENT", COLORS["spy"]),
        ("Calibrated walk-forward (pooled thr)", "+$108/trade, Sharpe 2.36, DD −4.2%", "DEPLOYABLE", COLORS["win"]),
    ]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    y = np.arange(len(rungs))[::-1]
    for i, (label, num, verdict, color) in enumerate(rungs):
        yy = y[i]
        ax.barh(yy, 1, color=color, alpha=0.30, edgecolor=color, linewidth=2)
        ax.text(0.02, yy, label, va="center", fontsize=11, weight="bold")
        ax.text(0.55, yy, num, va="center", fontsize=10)
        ax.text(0.92, yy, verdict, va="center", fontsize=10, weight="bold",
                color=color, ha="right")

    ax.set_xlim(0, 1); ax.set_ylim(-0.6, len(rungs) - 0.4)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("Evaluation rigor ladder — each step toward honesty changed the answer")
    fig.savefig(ASSETS / "13_rigor_ladder.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 14. Train / Val / Test temporal split visualization
# ─────────────────────────────────────────────────────────────────────────────
def temporal_split_chart():
    df = pd.read_csv(DATA / "02_event_features.csv", parse_dates=["announcement_date"])
    masks = temporal_split(df)
    splits = ["train", "val", "test"]
    counts = [masks[s].sum() for s in splits]
    rates = [df[masks[s]]["crush_profitable"].mean() * 100 for s in splits]
    starts = [df[masks[s]]["announcement_date"].min().strftime("%Y-%m-%d") for s in splits]
    ends = [df[masks[s]]["announcement_date"].max().strftime("%Y-%m-%d") for s in splits]

    fig, ax = plt.subplots(figsize=(11, 3.4))
    palette = [COLORS["stock"], COLORS["spy"], COLORS["loss"]]
    for i, (s, c, r, a, b, color) in enumerate(zip(splits, counts, rates, starts, ends, palette)):
        ax.barh(0, c, left=sum(counts[:i]), color=color, alpha=0.85, edgecolor="white", linewidth=2)
        ax.text(sum(counts[:i]) + c / 2, 0,
                f"{s.upper()}\nn={c:,}\ncrush={r:.1f}%\n{a} → {b}",
                ha="center", va="center", fontsize=10, color="white", weight="bold")

    ax.set_xlim(0, sum(counts))
    ax.set_yticks([])
    ax.set_xlabel("# events (chronological)")
    ax.set_title("Single-split: temporal train/val/test (the original evaluation)")
    fig.savefig(ASSETS / "14_temporal_split.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Tables (CSV)
# ─────────────────────────────────────────────────────────────────────────────
def write_tables():
    # Final summary table
    final = pd.DataFrame([
        {"Metric": "ML model", "Value": "LightGBM"},
        {"Metric": "Exit mode", "Value": "Hold to expiration"},
        {"Metric": "Probability threshold", "Value": "0.83 (pooled-val tuned)"},
        {"Metric": "Trades over 4 years", "Value": "78 (~20/year)"},
        {"Metric": "Win rate", "Value": "61.5%"},
        {"Metric": "Avg P&L per trade", "Value": "+$108"},
        {"Metric": "Total P&L on $100K", "Value": "+$8,452"},
        {"Metric": "Sharpe ratio", "Value": "2.36"},
        {"Metric": "Max drawdown", "Value": "−4.2%"},
        {"Metric": "Calmar ratio", "Value": "≈ 1.94"},
    ])
    final.to_csv(ASSETS / "table_final_result.csv", index=False)

    # Strategy comparison
    cmp = pd.DataFrame([
        {"Strategy": "always_short", "Trades": 4853, "WinRate": "54.2%", "AvgPnL": "−$84",
         "TotalPnL": "−$408K", "Sharpe": -2.86, "MaxDD": "−418%"},
        {"Strategy": "vrp_only (Joshua's)", "Trades": 3541, "WinRate": "53.6%", "AvgPnL": "−$82",
         "TotalPnL": "−$290K", "Sharpe": -2.84, "MaxDD": "−304%"},
        {"Strategy": "ml_logreg @ 0.79", "Trades": 231, "WinRate": "55.0%", "AvgPnL": "−$7",
         "TotalPnL": "−$1.5K", "Sharpe": -0.22, "MaxDD": "−12%"},
        {"Strategy": "ml_xgb @ 0.75", "Trades": 367, "WinRate": "52.6%", "AvgPnL": "−$77",
         "TotalPnL": "−$28K", "Sharpe": -1.57, "MaxDD": "−39%"},
        {"Strategy": "ml_lgbm @ 0.83", "Trades": 78, "WinRate": "61.5%", "AvgPnL": "+$108",
         "TotalPnL": "+$8.5K", "Sharpe": 2.36, "MaxDD": "−4.2%"},
    ])
    cmp.to_csv(ASSETS / "table_strategy_comparison.csv", index=False)

    # Feature breakdown
    fb = pd.DataFrame([
        {"Source": "Stock fundamentals (your /ML pipeline)", "Count": 33, "ImportanceShare": "43.1%",
         "Examples": "EPS estimates, analyst revisions, custom Elo system, lag-1 growth metrics"},
        {"Source": "Options microstructure (newly fetched)", "Count": 11, "ImportanceShare": "19.6%",
         "Examples": "ATM straddle %, IV pre/post, IV crush %, term-structure slope, OI/volume"},
        {"Source": "SPY market regime (Joshua's pipeline)", "Count": 30, "ImportanceShare": "37.3%",
         "Examples": "VIX, VRP 30d, IV rank, term slope, 25-delta skew, P/C ratio"},
    ])
    fb.to_csv(ASSETS / "table_feature_breakdown.csv", index=False)

    # Rigor ladder table
    rl = pd.DataFrame([
        {"Step": "1. Naive intrinsic exit", "Headline": "+$231/trade, Sharpe 6.74", "Verdict": "Fantasy — assumes option decays to intrinsic instantly"},
        {"Step": "2. Realistic T+1 close", "Headline": "−$144/trade, Sharpe −5.46", "Verdict": "Disaster — pays back time value + round-trip slippage"},
        {"Step": "3. Hold-to-expiry, single-split", "Headline": "+$184/trade, Sharpe 2.67", "Verdict": "Lucky window — Fold 4 dominated"},
        {"Step": "4. Plain walk-forward (per-fold thr)", "Headline": "−$31/trade, Sharpe −0.65", "Verdict": "Regime-dependent, threshold variance hides edge"},
        {"Step": "5. Calibrated walk-forward (pooled-val thr)", "Headline": "+$108/trade, Sharpe 2.36, DD −4.2%", "Verdict": "Deployable — robust across all 6 folds"},
    ])
    rl.to_csv(ASSETS / "table_rigor_ladder.csv", index=False)


def main():
    print("Generating presentation assets…")
    vol_crush_concept();           print("  ✓ 01_vol_crush_concept.png")
    straddle_payoff();             print("  ✓ 02_straddle_payoff.png")
    data_architecture();           print("  ✓ 03_data_architecture.png")
    spy_context_timeseries();      print("  ✓ 04_spy_market_context.png")
    feature_importance_by_source();print("  ✓ 05_feature_importance_by_source.png")
    feature_importance_top25();    print("  ✓ 06_top25_features.png")
    reliability_diagram();         print("  ✓ 07_reliability_diagram.png")
    friction_tax();                print("  ✓ 08_friction_tax.png")
    walk_forward_per_fold();       print("  ✓ 09_per_fold_pnl.png")
    equity_curves_calibrated();    print("  ✓ 10_equity_curves_calibrated.png")
    pnl_distribution();            print("  ✓ 11_pnl_distribution.png")
    threshold_sensitivity();       print("  ✓ 12_threshold_sensitivity.png")
    rigor_ladder();                print("  ✓ 13_rigor_ladder.png")
    temporal_split_chart();        print("  ✓ 14_temporal_split.png")
    write_tables();                print("  ✓ 4 tables")
    print(f"\nAll assets in {ASSETS}")


if __name__ == "__main__":
    main()
