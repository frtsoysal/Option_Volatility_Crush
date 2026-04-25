# Project Summary — SPY Short-Straddle Volatility-Crush Strategy

## 1. What This Project Is

This project is a computer program I built to try to earn money by selling stock-market insurance — specifically, short-dated options on the S&P 500 — on days when that insurance is priced unusually expensively, and buying it back a week later after its price has typically fallen. I built it to test a well-known market theory against real historical data with realistic trading costs, and to check whether modern machine learning could beat a simple one-rule strategy in finding the best days to trade.

## 2. The Idea Behind It

When a big event is coming up — a Federal Reserve meeting, a corporate earnings report, a tariff decision — the stock market gets nervous. That nervousness pushes up the price of what are effectively insurance contracts against large moves in stock prices. Because nobody knows which way the event will break, people bid this insurance up pre-emptively. Once the event passes and the uncertainty resolves, regardless of which direction the market actually moved, the insurance is no longer needed, and its price collapses almost immediately. Traders call that collapse a "volatility crush." The strategy in this project tries to be the one selling that expensive insurance on the day before a collapse, and buying it back cheaper afterward. The same pattern exists even without obvious events — there is a persistent tendency for this kind of insurance to be priced a bit higher than the underlying market's actual jumpiness justifies, and the gap between the two is called the "variance risk premium" (a fancy phrase for "insurance tends to cost more than it is worth").

## 3. What Was Built

- **Phase 0 — Data Sanity Check**: Verified that five years of historical option-chain data were clean, consistent, and tracked the publicly-known VIX index closely enough to trust for the rest of the work.
- **Phase 1 — Feature Engineering**: Turned the raw option data into a daily table with twenty-eight numerical summary columns describing the state of the market — how expensive insurance is, how turbulent the index has actually been, how skewed pricing is toward crash protection, and so on.
- **Phase 2 — Target Labeling**: For every trading day, simulated the trade (sell the insurance, wait five days, buy it back) and recorded whether it made money — creating the learning targets for the models.
- **Phase 3 — Machine Learning Models**: Trained two predictive models against two simpler baselines, using strict out-of-sample test data, to measure whether any of them produced a reliable edge.
- **Phase 4 — Realistic Backtest**: Simulated actually running each strategy for eighteen months on unseen data, including realistic commissions, bid-ask spread costs, margin requirements, and a cap on how many positions can be open at once.
- **Phase 5 — Final Presentation**: Consolidated everything into a clean walkthrough notebook and a formal written report suitable for someone reviewing the work without running any code.

## 4. The Steps in More Detail

### Phase 0 — Data Sanity Check

For every trading day from late June 2021 through April 2026 — twelve hundred and twelve days in total — the raw option-chain data was turned into a single "implied volatility at thirty days out" number and compared to the publicly available VIX index, which is supposed to measure the same thing. If the computed numbers did not track VIX, the data would be unusable and every downstream calculation would be meaningless. They matched with a correlation of zero-point-nine-seven-four — essentially identical day-to-day. One data-quality issue surfaced in the process: on August the eighth, 2023, the upstream data provider had uploaded literal dash characters instead of volatility numbers for every option contract that day. The issue was caught, flagged, and handled with a fallback calculation. Every other day in the sample was clean.

### Phase 1 — Feature Engineering

Phase 1 compressed more than eleven million rows of individual option-contract records into a compact daily table, with each row describing a single trading day in twenty-eight numerical dimensions. The groups include "how expensive is insurance at different horizons" (implied volatility at one, two, and three months out), "how steep is the skew between crash protection and upside exposure," "how much has the index actually moved around lately" (realized volatility at five days and twenty-one days), "how large is the gap between the expected and actual movement" (the variance risk premium), daily put-to-call ratios and volume measures, and a few price-based signals like the index's rolling returns and whether it is above its two-hundred-day moving average. The April 2025 tariff shock was used as a cross-check: if the feature builder was working, the implied-volatility features should have spiked before and during that event and relaxed afterward. They did, cleanly.

### Phase 2 — Target Labeling

Phase 2 turns "here is the market on day X" into "here is whether selling insurance on day X and buying it back five trading days later would have made money." For every day in the sample, the nearest thirty-day at-the-money contracts were identified, their selling prices recorded, and five trading days later the same contracts were marked against their new prices. The resulting labels show exactly the pattern a textbook short-volatility strategy produces: about sixty-four percent of trades end up profitable, the typical (median) trade makes about five percent, but the average trade actually loses a tiny amount because occasional disasters lose more than dozens of small wins add up to. As a specific validation, consider the April 2025 tariff episode: trades opened the week before the shock hit the labels as enormous losses (the worst one lost the equivalent of one-point-seven times the premium collected), and trades opened in the days after the shock peaked hit as large wins of plus ten to plus thirty-three percent. The labeling correctly separated the two regimes.

### Phase 3 — Machine Learning Models

Phase 3 is where the machine learning lived. Two models were trained — a basic logistic regression and a modern gradient-boosted tree method called LightGBM — on about two and a half years of historical data, calibrated on six months of held-out intermediate data, and finally evaluated on eighteen months of completely unseen test data. Two simpler baselines were run for comparison: always sell (the naive benchmark) and sell only when the variance-risk-premium signal is in its top twenty percent on the training data (a one-line rule built from a single feature). The surprising finding was that the simple rule beat both models. LightGBM's internal early-stopping mechanism — which halts training when further complexity stops helping — quit after just three decision trees, compared to the hundreds or thousands a healthy fit would produce. A technique called SHAP (which attributes a model's output to individual input features) confirmed that LightGBM was drawing almost all of its predictive power from the same one feature the simple rule was using. At this training-data size, neither model could find useful patterns beyond thresholding that single signal.

### Phase 4 — Realistic Backtest

Phase 4 simulated actually running each strategy for the full eighteen-month test window with realistic trading frictions — commissions per contract, the bid-ask spread (the difference between what you can sell at and what you have to buy at), margin requirements (capital set aside against open positions), and a limit of five concurrent trades. Five strategies ran: the one-feature rule, logistic regression, LightGBM, "always sell" as a disaster baseline, and buy-and-hold S&P 500 as the asset-class baseline. The one-feature rule ended the eighteen months with a Sharpe ratio of one-point-one-nine (a standard risk-adjusted-return measure where anything above one is considered respectable for a quantitative strategy), a total return of thirty-point-nine percent, and a worst drawdown of minus seventeen-point-five percent — narrowly better than buy-and-hold's zero-point-nine-seven Sharpe and slightly smaller drawdown. Two stress-test findings are important: under a doubled-slippage assumption, the rule's Sharpe collapses from one-point-one-nine to essentially zero, meaning the edge depends heavily on execution quality; and at fifty thousand dollars per trade on a hundred-thousand-dollar account, the margin requirements bind and the strategy stops functioning.

### Phase 5 — Final Presentation

Phase 5 consolidated everything into a single end-to-end walkthrough notebook that a reader can read linearly without running any code, a formal written report in Markdown form, a "money chart" figure showing the strategy's performance against benchmarks with major market events marked, and this summary document. No new analyses or strategies were introduced. The purpose of the phase was purely to make the existing work accessible.

## 5. The Main Result

The main finding is that a one-feature rule — sell insurance on any day when the gap between implied and realized volatility is in the top one-fifth of its historical distribution — was the best-performing strategy of every version tested. In plain English: on days when the market was charging an unusually large premium for insurance relative to how much the index had actually been moving around, selling that insurance and buying it back five trading days later turned out to be reliably profitable. Over the eighteen months of out-of-sample test data, the rule produced a Sharpe ratio of one-point-one-nine on a net-of-costs basis, a total return of thirty-point-nine percent, a worst drawdown of minus seventeen-point-five percent, and a trade win-rate of sixty-five-and-a-half percent. It beat buy-and-hold S&P 500 on risk-adjusted return (Sharpe of zero-point-nine-seven) and very narrowly on total return too (thirty-one-point-eight percent). Perhaps most interestingly, because the rule only had money actually working on about a third of trading days, the profit per dollar of capital deployed per day was roughly thirty times higher than simply buying and holding the index.

## 6. What Didn't Work, and Why That Matters

The machine learning models did not add value, and that is the interesting part. LightGBM — the modern tree-based method that wins most tabular-data competitions — stopped learning after three trees. SHAP analysis showed it was leaning almost entirely on a single feature for its predictions, which is exactly what the simple rule already did. Out-of-sample, the two trained models both took trades the rule had skipped, paid for them with larger drawdowns, and in LightGBM's case ended the test window with less money than it started with. At this data size and on this problem, the simple thing worked and the complicated thing did not.

## 7. Caveats

- **The edge depends heavily on execution quality.** If the cost of buying and selling (the "slippage") were twice as bad as modeled, the strategy's Sharpe ratio would collapse from one-point-one-nine to essentially zero — flat performance.
- **The strategy does not scale up.** Sizing trades at fifty thousand dollars each on a hundred-thousand-dollar account breaks the margin model; only a small fraction of intended trades can be taken, and the strategy loses money.
- **The out-of-sample test period is eighteen months.** That is not long enough to contain a full market cycle — there was no 2008-style financial crisis and no March-2020-style pandemic shock in the test window. Short-volatility strategies are historically most dangerous in exactly those regimes; the strategy has not been tested against them.
- **The strategy was only tested on the S&P 500.** Whether the same rule generalizes to the Nasdaq-100 or the Russell 2000, let alone individual company stocks, is completely unknown.
- **The margin calculation is a simplified approximation.** Real-world regulated margin rules for short options are substantially stricter than what was modeled, which in practice would cut the number of simultaneous positions roughly in half and reshape the performance numbers.
- **Part of the edge comes from capturing the bid-ask spread.** A retail trader whose broker fills orders consistently worse than the midpoint between bid and ask would earn less than the headline numbers suggest.

## 8. The Plain-English Summary

Options are financial contracts that pay off if a stock moves more than expected, and people buy them the way they buy insurance — to protect themselves against surprises. When the market is feeling nervous about something that might happen soon, insurance gets more expensive, just like flood insurance gets more expensive before a hurricane. Once the event passes, the insurance becomes cheap again almost immediately, whether or not the disaster actually struck. I built a computer program that watches the S&P 500 options market, identifies days when this kind of insurance looks unusually expensive relative to how choppy the market has actually been lately, sells the insurance on those days, buys it back five trading days later, and pockets the difference. I tested every fashionable machine-learning technique against a simple one-line rule — "only sell on days when the insurance markup is in the top twenty percent" — and found that the simple rule was actually better. Over the eighteen months of out-of-sample testing, the strategy earned about thirty percent with a seventeen-percent worst loss, slightly beating the stock index itself on risk-adjusted return while only having money at risk about a third of the time. The risks are documented clearly: the strategy would lose its edge if trading costs were meaningfully higher, it has not been tested through a real market crisis, and scaling it up too aggressively breaks it. The most interesting result is not the profit number itself but the fact that a carefully-chosen single idea outperformed a much more elaborate machine-learning approach.

## 9. Where to Find Everything

- `REPORT.md` — the formal written report (two thousand four hundred words).
- `05_final_presentation.ipynb` — the executable end-to-end walkthrough notebook.
- `00_data_sanity.ipynb` — Phase 0 data validation.
- `03_ml_pipeline.ipynb` — Phase 3 machine-learning walkthrough.
- `04_backtest.ipynb` — Phase 4 backtest walkthrough.
- `spy_features.py`, `label_targets.py`, `ml_pipeline.py`, `backtest.py` — the Python modules behind the phases.
- `data/` — all intermediate and final data tables in CSV form, six pickled fitted models, and sixteen figures in PNG form including `money_chart.png`.
