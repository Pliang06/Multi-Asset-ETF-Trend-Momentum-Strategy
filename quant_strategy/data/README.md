# Research data

`prices.csv` is the existing adjusted-close snapshot and has been retained
unchanged by this implementation. New adjusted-close and raw OHLCV downloads
go to `market_data/` so a data refresh does not silently replace that reference
snapshot. The folder also contains a trading-date calendar. `download_data.py`
writes timestamp, source, per-symbol history, missing-date counts, zero-volume
counts and SHA-256 hashes to `market_data/data_quality.json`. First observation
is only a listing-history proxy, not an independently verified legal listing
date.

The source is Yahoo Finance's chart API and is for research only. Validate all
prices, corporate actions, trading dates and execution assumptions against the
competition's IBKR account before using the strategy.
