"""Run the candidate study and write a reviewable report."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from backtest import add_drawdown, run_backtest
from download_data import CORE_UNIVERSE, download_adjusted_close
from metrics import approximate_competition_score, summarize
from model import StrategyConfig, baseline_targets, build_targets


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "prices.csv"
MARKET_DATA = ROOT / "data" / "market_data"
OUTPUT = ROOT / "outputs"


def load_prices() -> pd.DataFrame:
    market_snapshot = MARKET_DATA / "adjusted_close.csv"
    if market_snapshot.exists():
        prices = pd.read_csv(market_snapshot, index_col="Date", parse_dates=True)
        return prices.loc[:, CORE_UNIVERSE].dropna(how="all")
    if DATA_PATH.exists():
        prices = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
        return prices.loc[:, CORE_UNIVERSE].dropna(how="all")
    prices = download_adjusted_close(CORE_UNIVERSE, start="2010-01-01")
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(DATA_PATH, index_label="Date")
    return prices


def candidate_configs() -> list[tuple[str, StrategyConfig]]:
    base = StrategyConfig(frequency="monthly", top_n=8, inverse_volatility=False, max_weight=0.35, target_volatility=None)
    return [
        (
            "baseline_six_etf",
            replace(base, name="baseline_six_etf", frequency="monthly", lookbacks=(60,), signal_weights=(1.0,), skip=0, top_n=3, max_weight=1.0, target_volatility=None),
        ),
        (
            "monthly_trend_equal_top8",
            replace(base, name="monthly_trend_equal_top8"),
        ),
        (
            "monthly_trend_inverse_top8",
            replace(
                base, name="monthly_trend_inverse_top8", inverse_volatility=True,
            ),
        ),
        (
            "monthly_trend_equal_top10",
            replace(base, name="monthly_trend_equal_top10", top_n=10),
        ),
        ("monthly_hold_buffer_top8", replace(base, name="monthly_hold_buffer_top8", hold_buffer=2, max_replacements=2, min_hold_periods=1)),
        ("monthly_turnover_limited_top8", replace(base, name="monthly_turnover_limited_top8", turnover_limit=0.20)),
        ("monthly_covariance_top8", replace(base, name="monthly_covariance_top8", use_covariance_optimizer=True, max_weight=0.25, risk_on_equity_cap=0.85, risk_off_equity_cap=0.50, turnover_limit=0.30)),
        ("monthly_risk_budget_top8", replace(base, name="monthly_risk_budget_top8", risk_on_equity_cap=0.70, risk_off_equity_cap=0.40, minimum_safe_weight=0.05)),
        ("monthly_target_vol12_top8", replace(base, name="monthly_target_vol12_top8", target_volatility=0.12, minimum_safe_weight=0.05, risk_on_equity_cap=0.85, risk_off_equity_cap=0.50)),
        ("monthly_optimized_buffer_top8", replace(base, name="monthly_optimized_buffer_top8", hold_buffer=2, max_replacements=2, min_hold_periods=1, turnover_limit=0.30, use_covariance_optimizer=True, max_weight=0.25, risk_on_equity_cap=0.85, risk_off_equity_cap=0.50, target_volatility=0.12, minimum_safe_weight=0.05)),
        ("monthly_volume_execution_top8", replace(base, name="monthly_volume_execution_top8")),
        (
            "weekly_risk_managed",
            replace(base, name="weekly_risk_managed", frequency="weekly", target_volatility=0.16),
        ),
        (
            "monthly_risk_managed",
            replace(base, name="monthly_risk_managed", frequency="monthly", target_volatility=0.16),
        ),
    ]


def run_one(
    name: str,
    config: StrategyConfig,
    prices: pd.DataFrame,
    raw_close: pd.DataFrame | None = None,
    volume: pd.DataFrame | None = None,
    force_execution_model: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], pd.DataFrame]:
    if not force_execution_model and name in {"baseline_six_etf", "monthly_trend_equal_top8", "monthly_trend_inverse_top8"}:
        raw_close = None
        volume = None
    if name == "baseline_six_etf":
        tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD", "EFA"]
        targets = baseline_targets(prices[tickers], top_n=config.top_n, lookback=config.lookbacks[0])
        result = run_backtest(prices[tickers], targets, cost_bps=config.cost_bps, raw_close=raw_close[tickers] if raw_close is not None else None, volume=volume[tickers] if volume is not None else None, participation_cap=0.025 if volume is not None else None)
        diagnostics = pd.DataFrame(index=targets.index)
    else:
        targets, diagnostics = build_targets(prices, config)
        result = run_backtest(prices, targets, cost_bps=config.cost_bps, raw_close=raw_close, volume=volume, participation_cap=0.025 if volume is not None else None)
    return add_drawdown(result), targets, summarize(result), diagnostics


def rolling_windows(result: pd.DataFrame, months: int = 3) -> pd.DataFrame:
    rows = []
    # Non-overlapping quarterly windows make the comparison easier to read and
    # avoid treating every nearby start date as independent evidence.
    grouped = result["return"].resample(f"{months}ME")
    for end, returns in grouped:
        if len(returns) < 40:
            continue
        metrics = summarize(pd.DataFrame({"return": returns, "equity": (1 + returns).cumprod(), "turnover": 0.0, "rebalance": False}))
        rows.append({"window_end": end, **metrics})
    return pd.DataFrame(rows)


def competition_windows(result: pd.DataFrame) -> pd.DataFrame:
    """Replay the official calendar window across historical years."""

    rows = []
    for year in range(2010, 2026):
        start = pd.Timestamp(f"{year}-10-26")
        end = pd.Timestamp(f"{year + 1}-01-27")
        window = result.loc[start:end, "return"]
        if len(window) < 40:
            continue
        rows.append(
            {
                "competition_start": start,
                "competition_end": end,
                "days": len(window),
                "return": float((1.0 + window).prod() - 1.0),
                "sharpe": float((window.mean() / window.std(ddof=1) * (252.0 ** 0.5)) if window.std(ddof=1) else 0.0),
                "max_drawdown": float((1.0 + window).cumprod().div((1.0 + window).cumprod().cummax()).sub(1.0).min()),
                "holding_ratio_ge50": float((result.loc[start:end, "invested"] >= 0.50).mean()),
                "flat_days": int((result.loc[start:end, "invested"] <= 1e-8).sum()),
                "avg_daily_turnover": float(result.loc[start:end, "one_way_turnover"].mean()),
                "total_one_way_turnover": float(result.loc[start:end, "one_way_turnover"].sum()),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    summary: pd.DataFrame,
    windows: pd.DataFrame,
    latest: pd.DataFrame,
    cost_sensitivity: pd.DataFrame | None = None,
    competition: pd.DataFrame | None = None,
) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        frame = frame.copy()
        for column in frame.columns:
            if pd.api.types.is_float_dtype(frame[column]):
                frame[column] = frame[column].map(lambda value: f"{value:.4f}")
        headers = [str(column) for column in frame.columns]
        rows = [headers, ["---"] * len(headers)]
        rows.extend([[str(value) for value in row] for row in frame.itertuples(index=False, name=None)])
        return "\n".join("| " + " | ".join(row) + " |" for row in rows)

    preferred = summary[summary["candidate"] == "monthly_trend_equal_top8"]
    best = preferred.iloc[0] if not preferred.empty else summary.sort_values("diagnostic_score", ascending=False).iloc[0]
    lines = [
        "# Strategy research report",
        "",
        "This report is a model-selection record, not a performance promise.",
        "The official competition standardizes metrics against other teams; the diagnostic score below is only used to compare our own candidates.",
        "",
        f"Primary research candidate: **{best['candidate']}**. The primary label is fixed before the final competition window; it is not changed merely because another column has a better in-sample score.",
        "",
        "## Candidate comparison",
        "",
        markdown_table(summary),
        "",
        "## Latest target portfolio",
        "",
        markdown_table(latest),
        "",
        "## Cost sensitivity for the primary candidate",
        "",
        markdown_table(cost_sensitivity) if cost_sensitivity is not None else "Not available.",
        "",
        "## Historical competition-window replay",
        "",
        (
            f"Across {len(competition)} historical October 26 to January 27 windows, "
            f"the primary candidate had a positive return in {(competition['return'] > 0).mean():.1%} of windows; "
            f"median return was {competition['return'].median():.2%}, and the worst was {competition['return'].min():.2%}."
            if competition is not None and not competition.empty
            else "Not available."
        ),
        "",
        "## Interpretation",
        "",
        "The selected candidate is still subject to a final out-of-sample review, execution test and competition-account verification.",
        "The split-validation table is saved separately; the primary candidate was chosen because it remained strong in the pre-2023 training period and in the later test period, while no parameter was selected from the live competition window.",
        "Do not change parameters after seeing the competition-period data without recording the change and its rationale.",
    ]
    (OUTPUT / "research_report.md").write_text("\n".join(lines), encoding="utf-8")


def final_competition_target(prices: pd.DataFrame) -> tuple[pd.Series, pd.Timestamp]:
    """Return the primary candidate target from the latest completed month."""
    config = replace(
        StrategyConfig(),
        name="monthly_trend_equal_top8",
        frequency="monthly",
        top_n=8,
        inverse_volatility=False,
        max_weight=0.35,
        target_volatility=None,
    )
    targets, _ = build_targets(prices, config)
    signal_date = targets.index[-1]
    weights = targets.loc[signal_date].copy()
    weights = weights[weights > 1e-8]
    weights = weights / weights.sum()
    return weights.sort_values(ascending=False), signal_date


def split_validation(prices: pd.DataFrame) -> pd.DataFrame:
    """Evaluate fixed candidates on non-overlapping research periods."""
    base = StrategyConfig()
    configs = [
        ("top4_raw", replace(base, frequency="monthly", top_n=4, use_trend_filter=False, use_medium_momentum_filter=False, inverse_volatility=False, target_volatility=None, max_weight=0.35)),
        ("top8_trend_eq", replace(base, frequency="monthly", top_n=8, use_trend_filter=True, use_medium_momentum_filter=True, inverse_volatility=False, target_volatility=None, max_weight=0.35)),
        ("top8_trend_inv", replace(base, frequency="monthly", top_n=8, use_trend_filter=True, use_medium_momentum_filter=True, inverse_volatility=True, target_volatility=None, max_weight=0.35)),
        ("top10_trend_eq", replace(base, frequency="monthly", top_n=10, use_trend_filter=True, use_medium_momentum_filter=True, inverse_volatility=False, target_volatility=None, max_weight=0.35)),
        ("top6_trend_inv", replace(base, frequency="monthly", top_n=6, use_trend_filter=True, use_medium_momentum_filter=True, inverse_volatility=True, target_volatility=None, max_weight=0.35)),
        ("top8_raw_eq", replace(base, frequency="monthly", top_n=8, use_trend_filter=False, use_medium_momentum_filter=False, inverse_volatility=False, target_volatility=None, max_weight=0.35)),
    ]
    rows = []
    for name, config in configs:
        targets, _ = build_targets(prices, config)
        result = run_backtest(prices, targets, cost_bps=config.cost_bps)
        row = {"candidate": name}
        for label, start, end in [
            ("train", "2010-04-01", "2022-12-31"),
            ("validation", "2023-01-01", "2024-12-31"),
            ("test", "2025-01-01", None),
        ]:
            metrics = summarize(result.loc[start:end])
            row[f"{label}_return"] = metrics["total_return"]
            row[f"{label}_sharpe"] = metrics["sharpe"]
            row[f"{label}_max_drawdown"] = metrics["max_drawdown"]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    raw_path, volume_path = MARKET_DATA / "raw_close.csv", MARKET_DATA / "volume.csv"
    raw_close = pd.read_csv(raw_path, index_col="Date", parse_dates=True).reindex_like(prices) if raw_path.exists() else None
    volume = pd.read_csv(volume_path, index_col="Date", parse_dates=True).reindex_like(prices) if volume_path.exists() else None
    summary_rows = []
    window_rows = []
    latest_targets = None
    diagnostic_frames = []
    primary_result = None
    primary_targets = None
    primary_config = None
    for name, config in candidate_configs():
        result, targets, metrics, diagnostics = run_one(name, config, prices, raw_close, volume)
        row = {"candidate": name, **metrics, "diagnostic_score": approximate_competition_score(metrics)}
        summary_rows.append(row)
        windows = rolling_windows(result)
        windows.insert(0, "candidate", name)
        window_rows.append(windows)
        if name == "monthly_trend_equal_top8":
            primary_result = result
            primary_targets = targets
            primary_config = config
            latest_targets = targets.iloc[[-1]].T.reset_index().rename(columns={"index": "ticker", targets.index[-1]: "weight"})
            latest_targets = latest_targets[latest_targets["weight"] > 1e-8].sort_values("weight", ascending=False)
        result.to_csv(OUTPUT / f"{name}_daily.csv", index_label="Date")
        if not diagnostics.empty:
            diagnostics.insert(0, "candidate", name)
            diagnostic_frames.append(diagnostics.reset_index())

    summary = pd.DataFrame(summary_rows).sort_values("diagnostic_score", ascending=False)
    windows = pd.concat(window_rows, ignore_index=True)
    summary.to_csv(OUTPUT / "candidate_metrics.csv", index=False)
    windows.to_csv(OUTPUT / "rolling_windows.csv", index=False)
    if diagnostic_frames:
        pd.concat(diagnostic_frames, ignore_index=True).to_csv(OUTPUT / "rebalance_diagnostics.csv", index=False)
    if latest_targets is None:
        latest_targets = pd.DataFrame(columns=["ticker", "weight"])
    latest_targets.to_csv(OUTPUT / "latest_target.csv", index=False)
    sensitivity = None
    competition = None
    if primary_result is not None and primary_targets is not None and primary_config is not None:
        sensitivity_rows = []
        for cost in [0.0, 5.0, 8.0, 10.0, 20.0, 50.0]:
            cost_result = run_backtest(prices, primary_targets, cost_bps=cost, raw_close=raw_close, volume=volume, participation_cap=0.025 if volume is not None else None)
            cost_metrics = summarize(cost_result)
            sensitivity_rows.append({"cost_bps": cost, **cost_metrics})
        sensitivity = pd.DataFrame(sensitivity_rows)
        sensitivity.to_csv(OUTPUT / "cost_sensitivity.csv", index=False)
        competition = competition_windows(primary_result)
        competition.to_csv(OUTPUT / "competition_windows.csv", index=False)

    if competition is not None:
        competition.to_csv(OUTPUT / "historical_competition_replay.csv", index=False)
    split = split_validation(prices)
    split.to_csv(OUTPUT / "split_validation.csv", index=False)
    final_weights, final_signal_date = final_competition_target(prices)
    pd.DataFrame({"ticker": final_weights.index, "weight": final_weights.values}).to_csv(OUTPUT / "competition_candidate_target.csv", index=False)
    (OUTPUT / "competition_candidate_target.md").write_text(
        "# Competition candidate target\n\n"
        f"Signal date: {final_signal_date.date()}\n\n"
        "The weights below are a research candidate generated only from the latest fully observed month. "
        "Before submission, compare with IBKR data, check current market conditions, account permissions, liquidity and official rules. "
        "A material change must be documented and revalidated.\n\n"
        + "\n".join(f"- {ticker}: {weight:.2%}" for ticker, weight in final_weights.items())
        + "\n",
        encoding="utf-8",
    )
    write_report(summary, windows, latest_targets, sensitivity, competition)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nLatest primary candidate target:")
    print(latest_targets.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
