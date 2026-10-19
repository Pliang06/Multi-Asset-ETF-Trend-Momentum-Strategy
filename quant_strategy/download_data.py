"""Download adjusted-close research data without requiring yfinance."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import ssl
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


CORE_UNIVERSE = [
    "SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ",
    "XLE", "XLK", "XLF", "XLI", "XLV", "XLP", "XLU", "XLY", "XLB",
    "TLT", "IEF", "SHY", "HYG", "LQD",
    "GLD", "SLV", "DBC", "UUP",
]


def _epoch(value: str | date) -> int:
    if isinstance(value, str):
        value = date.fromisoformat(value)
    return int(datetime(value.year, value.month, value.day, tzinfo=timezone.utc).timestamp())


def download_adjusted_close(
    tickers: list[str],
    start: str = "2010-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Return a date-indexed adjusted-close DataFrame from Yahoo's chart API.

    Yahoo's `period2` is exclusive, so one calendar day is added to the
    requested end date. A failed ticker raises an explicit error instead of
    silently creating a survivorship-biased partial universe.
    """

    if end is None:
        end = date.today().isoformat()
    end_inclusive = date.fromisoformat(end)
    period2 = end_inclusive + timedelta(days=1)
    context = ssl._create_unverified_context()
    series: dict[str, pd.Series] = {}
    errors: dict[str, str] = {}

    def fetch_one(ticker: str) -> tuple[str, pd.Series | None, str | None]:
        query = urllib.parse.urlencode(
            {
                "period1": _epoch(start),
                "period2": _epoch(period2),
                "interval": "1d",
                "events": "div,splits",
                "includeAdjustedClose": "true",
            }
        )
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ticker)}?{query}"
        request = urllib.request.Request(url, headers={"User-Agent": "quant-league-research/1.0"})
        try:
            with urllib.request.urlopen(request, context=context, timeout=30) as response:
                payload = json.load(response)
            result = (payload.get("chart") or {}).get("result")
            if not result:
                raise RuntimeError((payload.get("chart") or {}).get("error"))
            result = result[0]
            timestamps = result.get("timestamp") or []
            adjusted = (result.get("indicators") or {}).get("adjclose", [{}])[0].get("adjclose") or []
            if len(timestamps) != len(adjusted) or not timestamps:
                raise RuntimeError("missing adjusted-close observations")
            index = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).normalize()
            values = pd.Series(adjusted, index=index, name=ticker, dtype="float64")
            values = values.replace([float("inf"), float("-inf")], pd.NA).dropna()
            if values.empty:
                raise RuntimeError("empty adjusted-close series")
            return ticker, values, None
        except Exception as exc:  # noqa: BLE001 - preserve ticker-specific diagnostics
            return ticker, None, f"{type(exc).__name__}: {exc}"

    # Yahoo responses are independent; bounded concurrency keeps the download
    # fast without turning a transient response into a silent partial universe.
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(tickers))) as pool:
        for ticker, values, error in pool.map(fetch_one, tickers):
            if error:
                errors[ticker] = error
            else:
                series[ticker] = values

    if errors:
        raise RuntimeError(f"Data download failed: {errors}")
    prices = pd.concat(series.values(), axis=1).sort_index()
    prices = prices.loc[:, tickers].dropna(how="all")
    return prices


def download_ohlcv(
    tickers: list[str],
    start: str = "2010-01-01",
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Download raw OHLCV panels alongside adjusted close.

    The return value has ``open``, ``high``, ``low``, ``close`` and ``volume``
    keys. Yahoo is a research source only; the metadata records the exact
    request so the files can be audited before using account data.
    """
    if end is None:
        end = date.today().isoformat()
    period2 = date.fromisoformat(end) + timedelta(days=1)
    context = ssl._create_unverified_context()
    fields = {key: {} for key in ("open", "high", "low", "close", "volume")}
    errors: dict[str, str] = {}

    def fetch_one(ticker: str):
        query = urllib.parse.urlencode({
            "period1": _epoch(start), "period2": _epoch(period2),
            "interval": "1d", "events": "div,splits", "includeAdjustedClose": "true",
        })
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ticker)}?{query}"
        request = urllib.request.Request(url, headers={"User-Agent": "quant-league-research/1.0"})
        try:
            with urllib.request.urlopen(request, context=context, timeout=30) as response:
                payload = json.load(response)
            result = ((payload.get("chart") or {}).get("result") or [None])[0]
            if not result:
                raise RuntimeError((payload.get("chart") or {}).get("error"))
            timestamps = result.get("timestamp") or []
            quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
            if not timestamps or any(len(quote.get(name) or []) != len(timestamps) for name in fields):
                raise RuntimeError("missing OHLCV observations")
            idx = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).normalize()
            return ticker, {name: pd.Series(quote[name], index=idx, name=ticker, dtype="float64") for name in fields}, None
        except Exception as exc:  # noqa: BLE001
            return ticker, None, f"{type(exc).__name__}: {exc}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(tickers))) as pool:
        for ticker, data, error in pool.map(fetch_one, tickers):
            if error:
                errors[ticker] = error
            else:
                for name, series in data.items():
                    fields[name][ticker] = series.replace([float("inf"), float("-inf")], pd.NA)
    if errors:
        raise RuntimeError(f"Data download failed: {errors}")
    return {name: pd.concat(values.values(), axis=1).sort_index().loc[:, tickers] for name, values in fields.items()}


def main() -> None:
    default_market_dir = Path(__file__).resolve().parent / "data" / "market_data"
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--out", default=str(default_market_dir / "adjusted_close.csv"))
    parser.add_argument("--raw-close-out", default=str(default_market_dir / "raw_close.csv"))
    parser.add_argument("--volume-out", default=str(default_market_dir / "volume.csv"))
    parser.add_argument("--ohlcv-out", default=str(default_market_dir / "ohlcv.csv"))
    args = parser.parse_args()

    prices = download_adjusted_close(CORE_UNIVERSE, args.start, args.end)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(destination, index_label="Date")
    market = download_ohlcv(CORE_UNIVERSE, args.start, args.end)
    raw_close = market["close"]
    for output_path in (Path(args.raw_close_out), Path(args.volume_out), Path(args.ohlcv_out)):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_close.to_csv(Path(args.raw_close_out), index_label="Date")
    market["volume"].to_csv(Path(args.volume_out), index_label="Date")
    ohlcv = pd.concat(
        [frame.rename(columns={ticker: f"{ticker}_{name}" for ticker in CORE_UNIVERSE}) for name, frame in market.items()],
        axis=1,
    )
    ohlcv.to_csv(Path(args.ohlcv_out), index_label="Date")
    calendar_path = destination.parent / "trading_calendar.csv"
    pd.DataFrame({"Date": prices.index}).to_csv(calendar_path, index=False)
    quality = {
        "source": "Yahoo Finance chart API",
        "schema_version": "yahoo_chart_ohlcv_v1",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_start": args.start, "requested_end": args.end,
        "tickers": CORE_UNIVERSE,
        "adjusted_close_file": destination.name,
        "raw_close_file": Path(args.raw_close_out).name,
        "volume_file": Path(args.volume_out).name,
        "ohlcv_file": Path(args.ohlcv_out).name,
        "trading_calendar_file": calendar_path.name,
        "adjusted_close_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "raw_close_sha256": hashlib.sha256(Path(args.raw_close_out).read_bytes()).hexdigest(),
        "volume_sha256": hashlib.sha256(Path(args.volume_out).read_bytes()).hexdigest(),
        "ohlcv_sha256": hashlib.sha256(Path(args.ohlcv_out).read_bytes()).hexdigest(),
        "trading_calendar_sha256": hashlib.sha256(calendar_path.read_bytes()).hexdigest(),
        "first_observation": str(prices.apply(pd.Series.first_valid_index).min().date()),
        "last_observation": str(prices.apply(pd.Series.last_valid_index).max().date()),
        "valid_observations_by_ticker": {ticker: int(prices[ticker].notna().sum()) for ticker in CORE_UNIVERSE},
        "first_observation_by_ticker": {ticker: (str(prices[ticker].first_valid_index().date()) if prices[ticker].first_valid_index() is not None else None) for ticker in CORE_UNIVERSE},
        "last_observation_by_ticker": {ticker: (str(prices[ticker].last_valid_index().date()) if prices[ticker].last_valid_index() is not None else None) for ticker in CORE_UNIVERSE},
        "listing_date_note": "First valid observation in this snapshot is a history proxy, not an independently verified legal listing date.",
        "missing_by_ticker": {ticker: int(raw_close[ticker].isna().sum()) for ticker in CORE_UNIVERSE},
        "missing_dates_by_ticker": {
            ticker: [str(value.date()) for value in raw_close.index[raw_close[ticker].isna()]]
            for ticker in CORE_UNIVERSE
        },
        "zero_volume_by_ticker": {ticker: int((market["volume"][ticker].fillna(0) <= 0).sum()) for ticker in CORE_UNIVERSE},
        "missing_ohlcv_by_ticker": {
            ticker: {field: int(market[field][ticker].isna().sum()) for field in ("open", "high", "low", "close", "volume")}
            for ticker in CORE_UNIVERSE
        },
    }
    (destination.parent / "data_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    metadata = destination.parent / "README.md"
    metadata.write_text(
        "# Research data snapshot\n\n"
        f"- Source: Yahoo Finance chart API (adjusted close plus raw OHLCV)\n"
        f"- Downloaded at UTC: {quality['downloaded_at_utc']}\n"
        f"- Requested range: {args.start} through {args.end}\n"
        f"- Tickers: {', '.join(CORE_UNIVERSE)}\n"
        f"- Adjusted-close rows: {len(prices)}; columns: {len(prices.columns)}\n"
        f"- Snapshot diagnostics: `data_quality.json`\n"
        "\nAdjusted close is used for research returns; raw close and OHLCV are"
        " reserved for order prices, liquidity and anomaly checks. Yahoo is a"
        " research source. Validate prices, corporate actions, calendar and"
        " executions against IBKR before competition use.\n",
        encoding="utf-8",
    )
    print(f"saved {destination} shape={prices.shape} range={prices.index.min().date()}..{prices.index.max().date()}")


if __name__ == "__main__":
    main()
