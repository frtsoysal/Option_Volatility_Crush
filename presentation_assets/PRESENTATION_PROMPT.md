# Class Presentation — Detailed Prompt for Claude Design

> **Project**: ML-Powered Earnings Volatility Crush Strategy on the S&P 500
> **Duration target**: ~15-20 slides, 10-12 min talk
> **Audience**: Quantitative finance class
> **Author**: Ibrahim Soysal (joint work with Joshua/dondadaj on the SPY component)
> **Repo**: github.com/frtsoysal/Option_Volatility_Crush

All visual assets referenced below live in `presentation_assets/`. The chart numbers
(`01_*.png`, `02_*.png`, …) are the file prefixes. The tables (`table_*.csv`) are CSVs
ready to be rendered as styled tables. Every slide has explicit asset references.

The narrative arc is:

1. Context & mechanics (slides 1-4)
2. Data architecture & three teammate streams (slides 5-8)
3. ML pipeline + target (slides 9-11)
4. The 5-step "rigor ladder" — each evaluation step changed the answer (slides 12-17)
5. Final deployable result + honest framing (slides 18-21)

---

## Global design guidelines

- **Color palette**:
  - Stock features (Ibrahim): `#5cb85c` green
  - Options features (newly fetched): `#5bc0de` light blue
  - SPY features (Joshua): `#f0ad4e` orange
  - Winning model (LightGBM): `#1976d2` bold blue
  - Loss/red: `#c62828`
  - Win/green: `#2e7d32`
  - Background neutral: `#fafafa`
- **Typography**: clean sans-serif (Inter or similar). Numbers in tabular figures.
  Headlines bold, body text regular. Avoid centered body text.
- **Layout**: prefer asymmetric layouts (image-left/text-right) over centered slides.
  One headline number per metric slide — don't crowd.
- **Whitespace generous**. Each chart should have ~30% slide area, with 1-2
  sentences of context next to it.

---

## SLIDE 1 — Title

**Layout**: Full-bleed dark background. Large title at top, subtitle below, byline
at bottom right.

**Title**: *Predicting Earnings Volatility Crush*

**Subtitle**: An ML strategy combining stock fundamentals, options microstructure, and
market-regime context across 8,005 S&P 500 events.

**Byline**: Ibrahim Soysal × Joshua Vodatinsky · Spring 2026 · Quantitative Finance

**Visual element**: Faint background overlay of `01_vol_crush_concept.png` at low
opacity (15%), or just a blurred dark gradient.

---

## SLIDE 2 — The Idea

**Headline**: *What is the volatility crush?*

**Layout**: Two-column. Left = chart. Right = text.

**Left**: `01_vol_crush_concept.png` — large, dominant.

**Right text** (3-4 short paragraphs):

> When a company is about to report earnings, the market doesn't know what's coming.
> Uncertainty has a price: option premiums **inflate** in the days before the
> announcement, peaking the day before.
>
> Once the report is out — beat, miss, or in-line — the uncertainty resolves. The
> stock moves to its new fair price, and the option premium **collapses** the next
> trading day. This collapse is the **volatility crush**.
>
> A trader who **sells** options just before the earnings announcement and closes
> the position after captures this premium decay. **If the actual stock move is
> smaller than what the option pricing implied**, the trade is profitable.

**Footer note**: "Across 8,005 S&P 500 events 2021-2026, this happens ~67% of the
time — but the question is: can we predict *which* 67%?"

---

## SLIDE 3 — What is a Straddle Option?

**Headline**: *The straddle: bet on size, not direction*

**Layout**: Chart left (large), explanation right.

**Left**: `02_straddle_payoff.png`

**Right text**:

> A **straddle** combines an at-the-money call (right to buy) and an at-the-money
> put (right to sell), both at the same strike, both expiring on the same date.
>
> **Long straddle** profits if the stock moves a lot in either direction.
> **Short straddle** (what we trade) profits when the stock pins near the strike.
>
> The two break-even points are at *strike ± premium*. Inside that band → win.
> Outside → loss, and the loss is unbounded as the stock moves further away.

**Key caption directly below the chart**: *We are systematically the seller (short
straddle) — collecting the elevated pre-earnings premium and hoping the stock
doesn't move too much.*

---

## SLIDE 4 — Why a Straddle for Vol Crush?

**Headline**: *Why straddle, specifically?*

**Layout**: Three horizontal cards, evenly spaced.

**Cards** (each ~1/3 of slide width, with icon/number at top):

1. **Direction-neutral.**
   "We don't need to predict whether the stock goes up or down — only how much
   it moves. That removes one of the hardest forecasting problems."

2. **Maximum vega exposure.**
   "ATM straddles have the highest sensitivity to implied-volatility changes
   among standard option structures. When IV crushes 50% overnight, the
   straddle pays back disproportionately."

3. **Symmetric pricing.**
   "Straddle premium ÷ stock price is the market's *expected move* —
   directly comparable to the realized move. The whole project reduces to:
   `move_pct < straddle_pct ?`"

**Footer**: A small math snippet:
```
crush_profitable  =  | actual_move_pct |  <  straddle_pct_pre
                  =  | (S₊ − S₋) / S₋ |   <  (Call_mid + Put_mid) / S₋
```

---

## SLIDE 5 — Data Sources

**Headline**: *Three data streams powering the strategy*

**Layout**: Single full-width chart.

**Visual**: `03_data_architecture.png` — full-width, dominant.

**Caption below the chart** (small, single-paragraph):

> Three pre-existing pipelines were merged. The user's individual-stock fundamentals
> repository (`/ML/scripts/with_estimates`), Joshua's SPY-level vol-crush research
> (`spy_strategy/`), and a newly-fetched per-event option-chain dataset
> (16,010 Alpha Vantage `HISTORICAL_OPTIONS` calls, 4.6 GB cached). Together they
> produce **74 features per earnings event**.

**Bottom-of-slide stat strip** (3 numbers in big typography, equal columns):

- **8,005** earnings events
- **74** features per event
- **2021-06 → 2026-04** (5 years)

---

## SLIDE 6 — Joshua's SPY Pipeline

**Headline**: *What the SPY component contributes*

**Layout**: Chart left, bullets right.

**Left**: `04_spy_market_context.png`

**Right bullets**:

- **30 features per trading day** capturing market-regime conditions
- VIX, VIX 3-month, ATM IV at 30/60/90 day horizons
- Variance Risk Premium (VRP) — the gap between implied and realized vol
- 25-delta skew (risk reversal) — fear vs greed in the option market
- IV rank and percentile over 30-day and 252-day lookbacks
- Put-call ratios and volume-to-OI ratios

> Every individual stock event picks up the SPY context as of the prior trading day
> via `merge_asof(direction="backward")`. So the model sees the macro vol regime
> at the moment we'd be entering the trade — without any look-ahead.

**Side note** (smaller, italic): *Joshua's SPY-only strategy as a standalone produced
Sharpe 1.19 on an 18-month test window. We use his features as additional context
for the per-stock model.*

---

## SLIDE 7 — Ibrahim's Stock Fundamentals Pipeline

**Headline**: *What the per-stock component contributes*

**Layout**: Two-column. Left = bulleted feature families. Right = small "diagram"
table-like visual.

**Left bullets**:

- **33 features per earnings event**, available before announcement
- **Analyst signals**: EPS estimate (avg/high/low), analyst count, revisions in last 7/30 days
- **Custom Elo rating system**: a 4-component metric (`elo_before`, `elo_decay`,
  `elo_momentum`, `elo_vol_4q`) that scores each company's track record of beating estimates
- **Price momentum**: 1-month and 3-month returns going into the print
- **Lag-1 fundamentals**: previous quarter's revenue/EPS YoY/QoQ growth, EBITDA growth,
  margins
- All labeled **strictly pre-announcement** — verified against the leakage list at
  `/ML/scripts/with_estimates/prepare_data.py:66`

**Right** (visual table or small infographic):

```
Family                        | Count
─────────────────────────────────────
Price momentum                |   4
EPS estimates                 |   8
Analyst revisions             |   4
Revenue estimates             |   4
Custom Elo system             |   4
Lag-1 historical growth       |   9
─────────────────────────────────────
Total                         |  33
```

**Footer**: *690 raw earnings CSVs across the SP500. 8,005 events fall inside our
2021-2026 window after filtering.*

---

## SLIDE 8 — How We Combined The Three Streams

**Headline**: *One row per earnings event, 74 features, three sources*

**Layout**: Single big chart + caption.

**Visual**: `05_feature_importance_by_source.png` — the "all three contribute"
result.

**Headline number above the chart** (very large):
**43% / 37% / 20%** — Stock / SPY / Options

**Caption below**:

> XGBoost feature importance breaks down nearly evenly across the three sources.
> No single block carries the model. The user's stock fundamentals lead at
> **43.1%** — including the **custom Elo system** appearing in the top 25 — but
> Joshua's SPY context (37.3%) and the newly-fetched options metrics (19.6%)
> each contribute substantively. Removing any source measurably degrades the model.

**Reference table to render alongside or below**: `table_feature_breakdown.csv`

---

## SLIDE 9 — The Top 25 Features

**Headline**: *What's actually driving predictions?*

**Layout**: Single full-width chart.

**Visual**: `06_top25_features.png` — long horizontal bar chart, full slide.

**Caption** (one sentence): *Top single feature is `dte_pre` (days to expiration of
the chosen straddle). Other top contributors mix all three sources — analyst
estimate trajectories, SPY skew/IV-rank, and per-event IV/straddle metrics.*

---

## SLIDE 10 — The Target Variable

**Headline**: *What we're actually predicting*

**Layout**: Math/code block on left, side-by-side examples on right.

**Left** (code-style block):

```python
# In every earnings event:
straddle_pct_pre = (call_mid + put_mid) / spot_pre * 100   # market's expected move
actual_move_pct = abs(spot_post / spot_pre - 1) * 100       # what really happened

crush_profitable = (actual_move_pct < straddle_pct_pre).astype(int)  # 1/0 target
crush_pnl_pct    = straddle_pct_pre - actual_move_pct                # signed P&L
```

**Right** (two example "trade cards"):

| Event | NVDA 2024-08-28 | TSLA 2024-10-23 |
|---|---|---|
| Expected move (straddle %) | 10.70% | 7.06% |
| Actual move | 7.81% | 19.54% |
| Outcome | **Crush ✓** (move ≤ priced) | **Crash ✗** (move blew through) |
| crush_pnl_pct | +2.89% | −12.48% |

**Footer note**: *Class balance: 67.1% positive (crush) on the test set — mildly
imbalanced. We score on AUC-PR rather than accuracy and tune thresholds via $P&L.*

---

## SLIDE 11 — The Train/Val/Test Split (initial approach)

**Headline**: *Single-split temporal evaluation*

**Layout**: Full-width chart.

**Visual**: `14_temporal_split.png`

**Caption**:

> Strict temporal split — no random shuffling. Train on the past, validate on
> the next year (for calibration + threshold tuning), test on the most recent
> 14 months (frozen).
>
> **Train: 3,917 events. Val: 1,774. Test: 2,195. Crush rate stays within 6pp
> across splits — no major regime drift in the simple sense.**

---

## SLIDE 12 — First Attempt: Naive Backtest

**Headline**: *Step 1 — Looked too good to be true*

**Layout**: Big number left, explanation right.

**Big number** (left, dominant):

```
Sharpe = 6.74
+$231 / trade
+$321K on $100K capital
67% win rate
```
(In the loss-red color so it reads as a warning.)

**Right text**:

> Our first backtest assumed **theoretical intrinsic exit** — that the option
> closes the day after earnings at the value `max(0, |post_spot - strike|)`.
> No time value remaining, no friction.
>
> Result looked spectacular. Sharpe 6.74 in 14 months OOS, win rate 67%. We
> almost wrote up the deck.

**Footer warning** (in red):

> **This was wrong.** An option still has 4-29 days of time decay remaining at T+1.
> Real exit prices are far from intrinsic. We built a fantasy backtest.

---

## SLIDE 13 — Reality Check: Real T+1 Exit Pricing

**Headline**: *Step 2 — Re-pricing exits at the actual market*

**Layout**: Two-column. Left = chart. Right = numbers.

**Left**: `08_friction_tax.png`

**Right** (large numbers):

```
Sharpe = −5.46
−$144 / trade
−$317K on $100K capital
52% win rate
```
(In the loss-red color.)

**Right text below numbers**:

> Once we replaced theoretical intrinsic with the **real bid/ask mid from the
> post-event chain JSON** (`atm_call_mid_post + atm_put_mid_post`), the strategy
> collapsed.
>
> Why? Even when our binary "crush" prediction was right (67% of events), only
> **52%** were money-winners after frictions. The remaining **15 percentage points**
> are events where the move was so small it cleared the binary threshold but the
> margin was thinner than the 10% half-spread we pay round-trip.

**Caption under the chart** (small):
*"Friction tax" — the gap between predicted-crush rate (binary signal) and
actual money-winner rate (after frictions).*

---

## SLIDE 14 — The Mistake & The Fix

**Headline**: *We were closing the position too early*

**Layout**: 3-step horizontal flow with arrows.

**Three boxes** (left → right):

1. **Initial assumption**
   *Close at T+1 with theoretical intrinsic*
   (Loss-red border)
   "Pays back ~50-70% of remaining time value AS A LOSS, plus pays half-spread
   on entry AND exit. Round-trip friction = ~10% of premium."

2. **Realistic correction**
   *Close at T+1 with REAL post-event mid*
   (Yellow/warning border)
   "Honest exit price, but the residual time value plus full round-trip friction
   eats almost all the vol-risk premium. Net negative across all strategies."

3. **The structural fix**
   *Hold the short straddle to its expiration date*
   (Win-green border)
   "Option naturally decays to intrinsic. No exit slippage. Friction = entry
   half-spread only (5%). This is also the natural retail play — collect premium,
   let it expire."

**Footer math**:
> `friction_round_trip ≈ 10% × premium`  →  `friction_one_side ≈ 5% × premium`
>
> A typical 10% vol risk premium gets eaten almost entirely by round-trip
> friction; one-side friction leaves edge intact.

---

## SLIDE 15 — Looked Profitable on Single Split

**Headline**: *Step 3 — Hold-to-expiry, single split*

**Layout**: Big number left, equity curve right.

**Big number** (left):

```
Sharpe = 2.67
+$184 / trade
+$18K on $100K capital
63% win rate
```
(In the win-green color.)

**Right text**:

> With realistic exit pricing AND hold-to-expiry execution, the ML-filtered
> strategy turned profitable on the test window:
>
> 99 trades over 14 months — very selective at threshold 0.73 — generated
> +$18,242 on $100K starting capital. Sharpe 2.67. We thought we had it.

**Footer** (subtle, in italic):
*This single-window result is technically valid — no leakage, real OOS test.
But one 14-month window in financial markets is not enough to claim a deployable
strategy. The next slide is what actually convinced us.*

---

## SLIDE 16 — Walk-Forward Reality Check

**Headline**: *Step 4 — Six folds across four years*

**Layout**: Chart top (per-fold P&L bars), text below.

**Top**: `09_per_fold_pnl.png`

**Text below**:

> Walk-forward CV: 6 expanding-window folds, each with 6-month val + 6-month test,
> sliding 6 months at a time. Each fold retrains models from scratch.
>
> Per-fold results vary dramatically. **Only 2 of 6 folds (Fold 4 = Q4 2024,
> Fold 6 = late 2025) are net profitable for LightGBM.** The original single-split
> result was **Fold 4 luck** — a regime-dependent profit zone, not a robust strategy.
>
> Concatenated walk-forward Sharpe = **−0.65**. The "Sharpe 2.67" we celebrated
> evaporated.

**Insight callout** (highlighted box):
> **The ML signal is consistent — it always beats no-ML. The market edge is not.**

---

## SLIDE 17 — The Calibration Fix

**Headline**: *Step 5 — Pooled-validation threshold*

**Layout**: Chart left, conceptual diagram or text right.

**Left**: `12_threshold_sensitivity.png`

**Right text** (~5 short paragraphs):

> The walk-forward issue wasn't the *model* — it was the *threshold*. Each fold
> tunes its threshold on a noisy 6-month validation slice. Across folds the
> chosen thresholds range from **0.63 to 0.71** — 8-percentage-points of variance
> on a number that determines whether 100 vs 1000 trades get taken.
>
> **The fix**: pool validation predictions across ALL six folds (5,318 OOS
> events spanning 3 years of regimes), then find ONE threshold that maximizes
> total $P&L across the pooled data.
>
> The pooled-val optimum jumps to **0.83 for LightGBM** — much higher and more
> selective than any single fold. With 36 months of validation evidence
> backing the choice, the threshold is robust.

**Bottom callout**:
> Apply this single threshold to every fold's test set. Concatenate the results
> chronologically. That's the deployable backtest.

---

## SLIDE 18 — The Final Result

**Headline**: *Calibrated walk-forward — the deployable answer*

**Layout**: Chart top, summary table below.

**Top**: `10_equity_curves_calibrated.png`

**Below** (rendered as a styled table, source: `table_strategy_comparison.csv`):

| Strategy | Trades | Win | Avg $ | Total $ | Sharpe | Max DD |
|---|---:|---:|---:|---:|---:|---:|
| always_short (no ML) | 4,853 | 54.2% | −$84 | −$408K | −2.86 | −418% |
| vrp_only (Joshua's SPY signal alone) | 3,541 | 53.6% | −$82 | −$290K | −2.84 | −304% |
| ml_logreg @ 0.79 | 231 | 55.0% | −$7 | −$1.5K | −0.22 | −12% |
| ml_xgb @ 0.75 | 367 | 52.6% | −$77 | −$28K | −1.57 | −39% |
| **ml_lgbm @ 0.83** | **78** | **61.5%** | **+$108** | **+$8.5K** | **+2.36** | **−4.2%** |

**Caption below table**:
*One model (LightGBM) at one threshold (0.83) in one exit mode (hold-to-expiry)
produces the only positive Sharpe and the only single-digit drawdown across the
4-year walk-forward period.*

---

## SLIDE 19 — Why LightGBM Won

**Headline**: *Calibration check — the model knows when it doesn't know*

**Layout**: Chart left, brief explanation right.

**Left**: `07_reliability_diagram.png`

**Right**:

> The reliability diagram shows that **LightGBM's predicted probabilities match
> observed win rates within 1-2pp at every bin**. When the model says 80% chance
> of crush, ~80% of those events actually crush. That's good calibration.
>
> Combined with the very-high pooled threshold (0.83), the strategy only trades
> the events the model is *most confident about*. That self-selection — informed
> by 33 stock fundamentals + 11 option metrics + 30 SPY context features — is
> what bridges the gap between vol risk premium and retail friction.

**P&L distribution mini-chart**: `11_pnl_distribution.png` — small inset.

---

## SLIDE 20 — The Methodology Ladder

**Headline**: *What every step taught us*

**Layout**: Single full-width chart.

**Visual**: `13_rigor_ladder.png`

**Caption below**:

> Each rung up the rigor ladder changed the headline number. The single-split
> Sharpe 2.67 was real but unreplicable. The walk-forward Sharpe −0.65 was
> the truth about the simple per-fold-threshold strategy. The calibrated
> walk-forward Sharpe 2.36 is the deployable result.
>
> **The methodology is the result.** A class deliverable that critiques itself
> is stronger than one that doesn't.

---

## SLIDE 21 — Final Result Card

**Headline**: *Deployable strategy — calibrated walk-forward across 4 years OOS*

**Layout**: Big number card on the left, explanatory bullets on the right.

**Left** (rendered as a styled "stat card", source: `table_final_result.csv`):

```
ML model              LightGBM
Exit mode             Hold to expiration
Probability threshold 0.83 (pooled-val tuned)

Trades over 4 years   78  (~20/year)
Win rate              61.5%
Avg P&L per trade     +$108
Total P&L on $100K    +$8,452
Sharpe ratio          2.36
Max drawdown          −4.2%
Calmar ratio          ≈ 1.94
```

**Right bullets**:

- Beats every baseline including Joshua's standalone SPY signal
- Only single-digit drawdown across the entire 4-year window
- **Annualized return**: ~2% on capital, ~10% on margin (Reg-T)
- Trades only on highest-confidence events — manageable for a part-time retail
  trader checking earnings calendars 5x/year
- All three feature sources contribute to the final decision

---

## SLIDE 22 — Limitations & Future Work

**Headline**: *What we know we don't know*

**Layout**: Two-column.

**Left — limitations**:

- **Survivorship bias**: SP500 ticker list is current. Companies kicked out
  during 2021-2026 (SVB, FRC, BBBY, ATVI, etc.) had catastrophic earnings moves
  we never see. Likely +3-7pp of artificial win rate.
- **Sample concentration**: 8,005 "events" cluster into ~30 earnings weeks.
  Effective sample size is smaller than it looks.
- **Half-spread approximation**: 10% round-trip is a defensible blended
  number, but per-event bid-ask from cached chains would be more honest.
- **Only one favorable model**: LightGBM works at threshold 0.83; XGBoost and
  LR don't. We don't have a fully convincing explanation.
- **Slow-moving signal**: 78 trades over 4 years = ~20/year. Hard to scale
  capital efficiently.

**Right — natural extensions**:

- Train an XGBoost regressor on `crush_pnl_pct` directly to filter the
  "right but lost money" events
- Iron condors instead of straddles to cap tail risk
- Historical SP500 constituents for survivorship correction
- Conditional alpha analysis — what's special about Folds 4 and 6?
- Walk-forward with monthly retraining for tighter regime adaptation

---

## SLIDE 23 — Conclusions

**Headline**: *Three takeaways*

**Layout**: Three cards stacked vertically with bold lead.

1. **The vol risk premium is real, but small.**
   Across 8,005 SP500 events the average vol-crush trade has a positive
   expected value of roughly 1-2% of stock notional — about the same size as
   retail bid-ask friction. Most of the time the friction wins.

2. **ML adds genuine alpha through selectivity.**
   Across all 6 walk-forward folds, the LightGBM model beats the no-ML baseline
   by ~$53/trade. The ML doesn't create new edge — it identifies which subset
   of events the existing edge actually shows up in.

3. **Methodology rigor is non-negotiable.**
   Single-split: +$184/trade (looked great). Plain walk-forward: −$31/trade
   (regime-dependent). Calibrated walk-forward: +$108/trade (genuine).
   The "right" answer is the one that holds up to the most rigorous test
   the data supports.

---

## SLIDE 24 — Q&A / Repository

**Headline**: *Questions?*

**Layout**: Centered.

- GitHub: **github.com/frtsoysal/Option_Volatility_Crush**
- Branch: `unified-merge`
- 7 modular notebooks: `unified_strategy/00_*.ipynb` through `06_*.ipynb`
- Walk-forward module: `unified_strategy/walk_forward.py`
- All charts in this deck: `presentation_assets/`

**Sub-text** (small):
*Joint work with Joshua Vodatinsky (`dondadaj`) on the SPY component.
SP500 stock fundamentals pipeline by Ibrahim Soysal. Combined ML pipeline
and walk-forward analysis by both.*

---

# Asset reference index (filename → purpose)

| File | Used in slide | Purpose |
|---|---|---|
| `01_vol_crush_concept.png` | Slides 1, 2 | Stylized IV time-series showing pre-event ramp + post-event crush |
| `02_straddle_payoff.png` | Slide 3 | Long vs short straddle P&L diagram with break-evens |
| `03_data_architecture.png` | Slide 5 | 3-source data flow into unified feature matrix |
| `04_spy_market_context.png` | Slide 6 | VIX + VRP time series 2021-2026 |
| `05_feature_importance_by_source.png` | Slide 8 | The 43/37/20 split — proves merger pays off |
| `06_top25_features.png` | Slide 9 | Top-25 feature importance bar chart, colored by source |
| `07_reliability_diagram.png` | Slide 19 | Calibration quality — predicted probs vs observed rates |
| `08_friction_tax.png` | Slide 13 | The 67% → 52% gap between binary signal and money outcome |
| `09_per_fold_pnl.png` | Slide 16 | 6-fold walk-forward P&L bars exposing regime dependency |
| `10_equity_curves_calibrated.png` | Slide 18 | Calibrated walk-forward equity curves — the headline result |
| `11_pnl_distribution.png` | Slide 19 | Per-trade P&L histogram for the winning strategy |
| `12_threshold_sensitivity.png` | Slide 17 | $P&L vs threshold curve showing the calibration optimum |
| `13_rigor_ladder.png` | Slide 20 | The 5-step methodology progression |
| `14_temporal_split.png` | Slide 11 | Single-split train/val/test visualization |
| `table_final_result.csv` | Slide 21 | Final stat card |
| `table_strategy_comparison.csv` | Slide 18 | Calibrated walk-forward strategy comparison |
| `table_feature_breakdown.csv` | Slide 8 | Three-source feature breakdown |
| `table_rigor_ladder.csv` | Slide 20 | Methodology progression as a table |

---

# Final note for Claude design

**Tone of voice**: confident but honest. The strongest part of this presentation
is that it doesn't oversell. Every previous step (intrinsic exit, single-split,
plain walk-forward) is shown as a step we took *and corrected*. That's the
methodology. The Sharpe 2.36 final number is impressive on its own merits —
no need to inflate it.

**Don't**:
- Use stock-image clipart of "stock market" candles
- Use 3D chart effects
- Center body paragraphs
- Use red/green colors on text outside of explicit win/loss contexts
- Add "AI Generated" or "Disclaimer" footers

**Do**:
- Treat numbers as the visual anchor — large, tabular, one per moment
- Use consistent color = source (green=stock, blue=options, orange=SPY) so the
  audience can pattern-match across slides
- Pair every chart with one focused caption sentence, not a paragraph
- Keep slide titles to 5-7 words max
