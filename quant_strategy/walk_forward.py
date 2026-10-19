"""Monthly rolling walk-forward evaluation with a small, predeclared grid."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from backtest import run_backtest
from download_data import CORE_UNIVERSE
from metrics import summarize
from model import StrategyConfig
from run_research import MARKET_DATA, candidate_configs, load_prices, run_one


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"


def walk_forward_configs() -> list[tuple[str, StrategyConfig]]:
    """Finite one-factor variants; no full Cartesian search is performed."""
    base = replace(
        StrategyConfig(), frequency="monthly", top_n=8, inverse_volatility=False,
        max_weight=0.25, target_volatility=None, risk_on_equity_cap=0.85,
        risk_off_equity_cap=0.50,
    )
    configs = [
        ("wf_equal8", replace(base, name="wf_equal8", max_weight=0.35, risk_on_equity_cap=0.75, risk_off_equity_cap=0.35)),
        ("wf_top6", replace(base, name="wf_top6", top_n=6)),
        ("wf_top10", replace(base, name="wf_top10", top_n=10)),
        ("wf_max20", replace(base, name="wf_max20", max_weight=0.20)),
        ("wf_max25", replace(base, name="wf_max25", max_weight=0.25)),
        ("wf_turnover20", replace(base, name="wf_turnover20", turnover_limit=0.20)),
        ("wf_turnover30", replace(base, name="wf_turnover30", turnover_limit=0.30)),
        ("wf_buffer2", replace(base, name="wf_buffer2", hold_buffer=2, max_replacements=2, min_hold_periods=1)),
        ("wf_buffer4", replace(base, name="wf_buffer4", hold_buffer=4, max_replacements=2, min_hold_periods=1)),
        ("wf_equity70_30", replace(base, name="wf_equity70_30", risk_on_equity_cap=0.70, risk_off_equity_cap=0.30)),
        ("wf_equity85_50", replace(base, name="wf_equity85_50", risk_on_equity_cap=0.85, risk_off_equity_cap=0.50)),
        ("wf_vol10", replace(base, name="wf_vol10", target_volatility=0.10, minimum_safe_weight=0.05)),
        ("wf_vol12", replace(base, name="wf_vol12", target_volatility=0.12, minimum_safe_weight=0.05)),
        ("wf_vol14", replace(base, name="wf_vol14", target_volatility=0.14, minimum_safe_weight=0.05)),
        ("wf_covariance", replace(base, name="wf_covariance", use_covariance_optimizer=True, turnover_limit=0.30, minimum_safe_weight=0.05)),
        ("wf_cov_buffer", replace(base, name="wf_cov_buffer", use_covariance_optimizer=True, turnover_limit=0.30, hold_buffer=2, max_replacements=2, target_volatility=0.12, minimum_safe_weight=0.05)),
    ]
    return configs


def _window_metrics(result: pd.DataFrame) -> dict[str, float]:
    returns = result["return"].astype("float64")
    local = pd.DataFrame(
        {
            "return": returns,
            "equity": (1.0 + returns).cumprod(),
            "turnover": result.get("turnover", 0.0),
            "rebalance": result.get("rebalance", False),
        },
        index=result.index,
    )
    metrics = summarize(local)
    metrics["invested_days_pct"] = float((result["invested"] >= 0.50).mean()) if "invested" in result else 1.0
    metrics["cash_days"] = float((result["invested"] <= 1e-8).sum()) if "invested" in result else 0.0
    pending = result["unfilled_notional"] if "unfilled_notional" in result else pd.Series(0.0, index=result.index)
    metrics["max_pending_notional"] = float(pending.max())
    return metrics


def _fold_dates(first: pd.Timestamp, last: pd.Timestamp, test_months: int):
    test_start = first + pd.DateOffset(years=6)
    while test_start <= last:
        train_start = test_start - pd.DateOffset(years=6)
        validation_start = test_start - pd.DateOffset(years=1)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)
        yield train_start, validation_start, test_start, min(test_end, last)
        test_start += pd.DateOffset(months=1)


def run_walk_forward(
    prices: pd.DataFrame,
    test_months: int = 6,
    raw_close: pd.DataFrame | None = None,
    volume: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return selected-fold, per-candidate OOS, and stress-test frames."""
    if test_months not in (6, 12):
        raise ValueError("test_months must be 6 or 12")
    base_candidates = walk_forward_configs()
    # Frozen baselines remain separate from model selection.
    baselines = [
        (name, config) for name, config in candidate_configs()
        if name in {"baseline_six_etf", "monthly_trend_equal_top8", "monthly_trend_inverse_top8"}
    ]
    all_candidates = baselines + base_candidates
    frozen_names = {"baseline_six_etf", "monthly_trend_equal_top8", "monthly_trend_inverse_top8"}
    cache: dict[str, tuple[pd.DataFrame, pd.DataFrame, StrategyConfig]] = {}
    for name, config in all_candidates:
        result, targets, _, _diagnostics = run_one(
            name, config, prices,
            raw_close,
            volume,
            force_execution_model=True,
        )
        cache[name] = (result, targets, config)

    fold_rows: list[dict[str, object]] = []
    oos_rows: list[dict[str, object]] = []
    folds = list(_fold_dates(prices.index.min(), prices.index.max(), test_months))
    for fold_number, (train_start, validation_start, test_start, test_end) in enumerate(folds, 1):
        valid_scores: list[tuple[float, str, dict[str, float], dict[str, float]]] = []
        for name, (result, _, _) in cache.items():
            train = result.loc[train_start:validation_start - pd.Timedelta(days=1)]
            validation = result.loc[validation_start:test_start - pd.Timedelta(days=1)]
            if len(train) < 800 or len(validation) < 150:
                continue
            train_m = _window_metrics(train)
            validation_m = _window_metrics(validation)
            score = (
                0.50 * min(train_m["sharpe"], validation_m["sharpe"])
                + 0.25 * (train_m["sharpe"] + validation_m["sharpe"])
                + 0.10 * (train_m["max_drawdown"] + validation_m["max_drawdown"])
            )
            if name not in frozen_names:
                valid_scores.append((score, name, train_m, validation_m))

        if not valid_scores:
            continue
        _, selected, train_m, validation_m = max(valid_scores, key=lambda item: item[0])
        selected_result = cache[selected][0].loc[test_start:test_end]
        selected_m = _window_metrics(selected_result)
        fold_rows.append(
            {
                "fold": fold_number, "train_start": train_start, "validation_start": validation_start,
                "test_start": test_start, "test_end": test_end, "selected_candidate": selected,
                "train_sharpe": train_m["sharpe"], "validation_sharpe": validation_m["sharpe"],
                **{f"test_{key}": value for key, value in selected_m.items()},
            }
        )
        for name, (result, _, _) in cache.items():
            test = result.loc[test_start:test_end]
            if len(test) < 40:
                continue
            oos_rows.append({"fold": fold_number, "candidate": name, "selected": name == selected, **_window_metrics(test)})

    # Cost stress for the predeclared combined candidate. It is fixed before
    # reviewing the test windows and reported separately from model selection.
    stress_rows: list[dict[str, object]] = []
    stress_name = "wf_cov_buffer"
    _, stress_targets, stress_config = cache[stress_name]
    for cost_bps in (8.0, 50.0, 100.0):
        stressed = run_backtest(
            prices, stress_targets, cost_bps=cost_bps,
            raw_close=raw_close, volume=volume,
            participation_cap=0.025 if volume is not None else None,
        )
        for fold_number, (_, _, test_start, test_end) in enumerate(folds, 1):
            test = stressed.loc[test_start:test_end]
            if len(test) >= 40:
                stress_rows.append({"fold": fold_number, "candidate": stress_name, "cost_bps": cost_bps, **_window_metrics(test)})
    return pd.DataFrame(fold_rows), pd.DataFrame(oos_rows), pd.DataFrame(stress_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-months", type=int, choices=(6, 12), default=6)
    args = parser.parse_args()
    prices = load_prices()
    raw_path, volume_path = MARKET_DATA / "raw_close.csv", MARKET_DATA / "volume.csv"
    raw = pd.read_csv(raw_path, index_col="Date", parse_dates=True).reindex(index=prices.index, columns=prices.columns) if raw_path.exists() else None
    volume = pd.read_csv(volume_path, index_col="Date", parse_dates=True).reindex(index=prices.index, columns=prices.columns) if volume_path.exists() else None
    folds, candidates, costs = run_walk_forward(prices, args.test_months, raw, volume)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.test_months}m"
    folds.to_csv(OUTPUT / f"walk_forward_folds_{suffix}.csv", index=False)
    candidates.to_csv(OUTPUT / f"walk_forward_candidate_oos_{suffix}.csv", index=False)
    costs.to_csv(OUTPUT / f"walk_forward_cost_stress_{suffix}.csv", index=False)
    if not candidates.empty:
        aggregate = candidates.groupby("candidate").agg(
            sharpe_median=("sharpe", "median"), sharpe_p25=("sharpe", lambda s: s.quantile(0.25)),
            worst_max_drawdown=("max_drawdown", "min"), invested_days_median=("invested_days_pct", "median"),
            cash_days_max=("cash_days", "max"),
        ).sort_values("sharpe_median", ascending=False)
        aggregate.to_csv(OUTPUT / f"walk_forward_candidate_summary_{suffix}.csv")
        stress_medians = costs.groupby("cost_bps")["sharpe"].median() if not costs.empty else pd.Series(dtype="float64")
        combined = aggregate.loc["wf_cov_buffer"] if "wf_cov_buffer" in aggregate.index else pd.Series(dtype="float64")
        simple = aggregate.loc["wf_equal8"] if "wf_equal8" in aggregate.index else pd.Series(dtype="float64")
        sharpe_pass = bool(combined.get("sharpe_median", float("nan")) >= simple.get("sharpe_median", float("inf")))
        p25_pass = bool(combined.get("sharpe_p25", float("nan")) >= 0.30)
        drawdown_pass = bool(combined.get("worst_max_drawdown", float("nan")) >= simple.get("worst_max_drawdown", float("nan")) - 0.03)
        cost50_pass = bool(stress_medians.get(50.0, float("nan")) >= 0.60)
        holding_pass = bool(combined.get("invested_days_median", 0.0) >= 0.50 and combined.get("cash_days_max", float("inf")) <= 10)
        report = [
            "# Rolling walk-forward report", "",
            f"Test length: {args.test_months} months; rolling step: one month; folds: {len(folds)}.",
            "Windows overlap and are not statistically independent. Training and validation select among the fixed one-factor candidates; test returns do not select parameters.",
            "",
            "| Candidate | Median OOS Sharpe | 25th percentile OOS Sharpe | Worst window drawdown | Median invested days |",
            "|---|---:|---:|---:|---:|",
        ]
        for name in ("wf_equal8", "wf_cov_buffer"):
            if name in aggregate.index:
                row = aggregate.loc[name]
                report.append(
                    f"| {name} | {row['sharpe_median']:.3f} | {row['sharpe_p25']:.3f} | "
                    f"{row['worst_max_drawdown']:.1%} | {row['invested_days_median']:.1%} |"
                )
        report += [
            "", "| One-way cost stress | Median OOS Sharpe for fixed covariance + buffer candidate |",
            "|---:|---:|",
        ]
        for bps in (8.0, 50.0, 100.0):
            if bps in stress_medians.index:
                report.append(f"| {bps:.0f} bps | {stress_medians[bps]:.3f} |")
        report += [
            "", "## Promotion gate", "",
            f"The covariance + buffer candidate's 25th-percentile Sharpe gate (>= 0.30): {'PASS' if p25_pass else 'FAIL'}.",
            f"Its 50 bps median Sharpe gate (>= 0.60): {'PASS' if cost50_pass else 'FAIL'}.",
            f"Its median Sharpe vs simple Top-8 gate: {'PASS' if sharpe_pass else 'FAIL'}.",
            f"Its worst-window drawdown within 3 percentage points of simple Top-8: {'PASS' if drawdown_pass else 'FAIL'}.",
            f"Its invested ratio and flat-day gate: {'PASS' if holding_pass else 'FAIL'}.",
            "Keep the frozen simple baseline as the primary candidate unless every predeclared out-of-sample, drawdown, turnover, holding and cost gate passes.",
            "",
            "These results use the current adjusted-close research snapshot. The ETF universe is fixed and has survivorship risk; historical rolling windows do not predict competition returns. Execution stress is a model until reconciled with IBKR fills.",
        ]
        report_path = OUTPUT / f"walk_forward_report_{suffix}.md"
        report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
        (OUTPUT / "walk_forward_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"walk-forward folds={len(folds)} candidate_oos_rows={len(candidates)} costs={len(costs)}")


if __name__ == "__main__":
    main()
