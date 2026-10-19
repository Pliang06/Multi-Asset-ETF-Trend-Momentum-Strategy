[README.md](https://github.com/user-attachments/files/33645736/README.md)
# Quant League strategy research

This directory contains our independent strategy implementation. The
competition candidate remains long-only, uses a fixed universe of liquid
US-listed ETFs, generates signals after completed monthly bars, and invests
the portfolio fully. Qlib, vectorbt and Lean are research references; none is a
runtime dependency of the candidate.

## Research structure

- `model.py` builds the 21/63/126/252-session momentum score, skips the latest
  five sessions, filters for positive 63-session momentum and price above its
  200-session average, applies hold buffers, and builds equal, inverse-vol or
  shrinkage-covariance targets. The optimizer is constrained long-only and
  supports exposure, single-name and turnover limits with inverse-vol fallback.
- `backtest.py` applies a completed-bar signal on the next session, tracks
  daily weight drift, records one-way and two-way turnover, and charges costs.
  With OHLCV enabled it limits fills to 2.5% of reported bar volume, retains
  pending orders, estimates IBKR-style per-share fees and Lean-style
  volume-share slippage, and reports unfilled notional. Missing/zero volume
  blocks the fill.
- `run_research.py` freezes the equal-weight Top-8, inverse-volatility Top-8
  and original six-ETF baselines. It adds one-module ablations and writes
  per-rebalance rank/exit diagnostics.
- `walk_forward.py` evaluates a predeclared finite set with five years of
  training, one year of validation, six- or twelve-month tests, and a monthly
  roll. It records all candidate test windows and separately reports 8/50/100
  bps cost stress for the combined covariance-and-buffer candidate. Every
  walk-forward candidate, including frozen references, uses the same
  volume-aware execution model. Separate reports are saved for 6- and 12-month
  tests. It uses a small in-house NumPy/Pandas runner; vectorbt is not imported
  or required.
- `lean_execution_review.py` checks normal, risk-switch and missing-volume
  orders with the equations in the local Lean Volume Share model. It blocks
  fills when volume is missing, preserving the order. The Lean source license
  is Apache-2.0; no Lean source code is copied into this project.

Qlib, vectorbt and Lean projects downloaded for study are unchanged. The
vectorbt Community Edition is licensed under Apache-2.0 with Commons Clause,
so this project does not depend on or distribute vectorbt code.

## Data

The existing `data/prices.csv` is retained as the original adjusted-close
snapshot. New downloads go to `data/market_data/` and do not overwrite it:

```bash
python download_data.py --start 2010-01-01 --end 2026-10-18
```

The new snapshot contains adjusted close, raw close, raw OHLCV, and
`data_quality.json` with source, request dates, download time, hashes, observed
history, missing observations and zero-volume counts. Yahoo data is for
research only; check dates, corporate actions, symbols, prices and executions
against IBKR before competition use. No future price is filled backward.

## Run and validate

```bash
python test_strategy.py
python run_research.py
python walk_forward.py --test-months 6
python walk_forward.py --test-months 12
```

Dependencies are NumPy and Pandas only. `pytest` is optional; the deterministic
checks also run directly with Python. `walk_forward.py` uses a predeclared
one-factor candidate set rather than a full parameter Cartesian product.
The generated reports are `outputs/walk_forward_report_6m.md` and
`outputs/walk_forward_report_12m.md`.
Report every candidate and inspect the median and lower-quartile OOS Sharpe,
worst drawdown, invested-day ratio, and cash days. Do not select a strategy
from one historical competition window.

## Execution review limitations

The local Lean source defaults its `VolumeShareSlippageModel` to a 2.5%
volume limit and 10% price-impact parameter; the research simulator uses that
equation. Its commission approximation follows Lean's US equity schedule of
$0.005/share, $1 minimum, with no more than 0.5% of order value. IBKR schedules
can differ by account, routing and date, so calibrate them with the actual
competition account. A full Lean backtest has not been run in this environment:
the Lean CLI and .NET SDK are unavailable. The helper is a deterministic
scenario review, not a substitute for Lean brokerage integration or paper
fills. Next-session close in the research backtest and next-session market
fills in Lean may differ; reconcile their timestamps, price, turnover and
fees before freezing the competition strategy.

The enhanced covariance candidate is not the primary strategy unless it passes
the predeclared out-of-sample, turnover, drawdown, holding and cost gates. The
simple frozen baselines remain the comparison standard.
