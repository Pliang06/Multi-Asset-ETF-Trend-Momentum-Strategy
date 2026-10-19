# Research data snapshot

- Source: Yahoo Finance chart API (adjusted close plus raw OHLCV)
- Downloaded at UTC: 2026-10-17T18:02:21.558137+00:00
- Requested range: 2010-01-01 through 2026-10-18
- Tickers: SPY, QQQ, IWM, EFA, EEM, VNQ, XLE, XLK, XLF, XLI, XLV, XLP, XLU, XLY, XLB, TLT, IEF, SHY, HYG, LQD, GLD, SLV, DBC, UUP
- Adjusted-close rows: 4223; columns: 24
- Snapshot diagnostics: `data_quality.json`

Adjusted close is used for research returns; raw close and OHLCV are reserved for order prices, liquidity and anomaly checks. Yahoo is a research source. Validate prices, corporate actions, calendar and executions against IBKR before competition use.
