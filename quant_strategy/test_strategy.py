"""Deterministic checks for signal timing, portfolio constraints, and fills."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from . import model
    from .backtest import run_backtest
    from .lean_execution_review import lean_volume_share_review
    from .model import StrategyConfig, _last_observation_dates, _select_assets, _shrink_covariance, build_targets
except ImportError:
    import model
    from backtest import run_backtest
    from lean_execution_review import lean_volume_share_review
    from model import StrategyConfig, _last_observation_dates, _select_assets, _shrink_covariance, build_targets


def synthetic_prices(days: int = 900, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    columns = columns or ("SPY", "QQQ", "IWM", "EFA", "IEF", "SHY", "TLT", "GLD", "DBC", "UUP", "HYG", "LQD")
    idx = pd.bdate_range("2018-01-01", periods=days)
    rng = np.random.default_rng(7)
    common = rng.normal(0.0002, 0.008, size=(days, 1))
    shocks = common + rng.normal(0.0001, 0.009, size=(days, len(columns)))
    return pd.DataFrame(100 * np.exp(np.cumsum(shocks, axis=0)), index=idx, columns=columns)


def test_targets_are_invested_and_nonnegative():
    prices = synthetic_prices()
    targets, _ = build_targets(prices, StrategyConfig(top_n=4, frequency="monthly", target_volatility=None))
    nonempty = targets[targets.sum(axis=1) > 0]
    assert not nonempty.empty
    assert np.allclose(nonempty.sum(axis=1), 1.0)
    assert (nonempty >= 0).all().all()


def test_etf_needs_full_signal_history_before_it_is_ranked():
    prices = synthetic_prices(days=255, columns=("SPY", "IEF", "SHY", "GLD"))
    targets, _ = build_targets(prices, StrategyConfig(frequency="monthly", top_n=2, target_volatility=None))
    assert targets.sum(axis=1).eq(0.0).all()


def test_missing_signal_data_keeps_the_previous_target():
    prices = synthetic_prices(days=1_040)
    prices.loc[prices.index[-10:]] = np.nan
    targets, diagnostics = build_targets(
        prices,
        StrategyConfig(frequency="monthly", top_n=4, target_volatility=None),
    )
    # The incomplete December period is excluded altogether; no formal target
    # is created from the stale last valid prices.
    assert diagnostics.index[-1].to_period("M") == pd.Period("2021-11", freq="M")


def test_month_end_signal_excludes_incomplete_month():
    index = pd.bdate_range("2024-01-01", "2024-02-15")
    dates = _last_observation_dates(index, "monthly")
    assert list(dates) == [pd.Timestamp("2024-01-31")]


def test_signal_uses_no_future_prices():
    prices = synthetic_prices(days=950)
    cutoff = prices.index[-100]
    config = StrategyConfig(frequency="monthly", top_n=4, target_volatility=None)
    first, _ = build_targets(prices, config)
    altered = prices.copy()
    altered.loc[altered.index > cutoff] *= np.linspace(0.5, 1.8, len(altered.columns))
    second, _ = build_targets(altered, config)
    common = first.index[first.index <= cutoff].intersection(second.index)
    pd.testing.assert_frame_equal(first.loc[common], second.loc[common])


def test_turnover_reference_marks_the_old_portfolio_to_market():
    index = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"SPY": [100.0, 110.0, 121.0], "IEF": [100.0, 100.0, 100.0]}, index=index)
    marked = model._drift_weights(pd.Series({"SPY": 0.5, "IEF": 0.5}), prices, index[0], index[2])
    assert np.isclose(marked["SPY"], 1.21 / 2.21)
    assert np.isclose(marked["IEF"], 1.00 / 2.21)
    assert np.isclose(marked.sum(), 1.0)


def test_next_session_execution_and_price_drift():
    index = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.DataFrame({"SPY": [100.0, 110.0, 121.0, 121.0, 121.0], "IEF": [100.0] * 5}, index=index)
    targets = pd.DataFrame(0.0, index=[index[0], index[1]], columns=prices.columns)
    targets.loc[index[0], "SPY"] = 1.0
    targets.loc[index[1], ["SPY", "IEF"]] = 0.5
    result = run_backtest(prices, targets, cost_bps=0.0)
    assert not bool(result.loc[index[0], "rebalance"])
    assert bool(result.loc[index[1], "rebalance"])
    assert np.isclose(result.loc[index[1], "return"], 0.0)
    assert np.isclose(result.loc[index[2], "return"], 0.10)
    assert np.isclose(result.loc[index[2], "two_way_turnover"], 1.0)
    assert np.isclose(result.loc[index[2], "one_way_turnover"] * 2.0, result.loc[index[2], "two_way_turnover"])


def test_volume_limit_blocks_and_partially_fills_orders():
    index = pd.bdate_range("2024-01-01", periods=4)
    prices = pd.DataFrame({"SPY": [100.0] * 4, "IEF": [100.0] * 4}, index=index)
    targets = pd.DataFrame([[1.0, 0.0]], index=[index[0]], columns=prices.columns)
    raw = prices.copy()
    volume = pd.DataFrame(0.0, index=index, columns=prices.columns)
    volume.loc[index[2], "SPY"] = 1_000.0
    volume.loc[index[3], "SPY"] = 1_000.0
    result = run_backtest(prices, targets, cost_bps=8.0, raw_close=raw, volume=volume, participation_cap=0.025)
    assert result.loc[index[1], "invested"] == 0.0
    assert result.loc[index[1], "unfilled_notional"] >= 999_999.0
    assert result.loc[index[1], "missing_volume_orders"] == 1
    assert result.loc[index[1], "unfilled_slippage_estimate"] > 0
    assert 0.0 < result.loc[index[2], "filled_notional"] <= 2_500.0 + 1e-8
    assert result.loc[index[2], "order_pending"]
    assert result.loc[index[2], "invested"] < 0.01
    assert result.loc[index[2], "slippage_cost"] > 0.0
    absent = run_backtest(prices, targets, cost_bps=8.0, raw_close=raw, volume=None, participation_cap=0.025)
    assert absent.loc[index[1], "invested"] == 0.0
    assert absent.loc[index[1], "unfilled_slippage_estimate"] > 0


def test_cost_pressure_reduces_equity():
    index = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"SPY": [100.0] * 3, "IEF": [100.0] * 3}, index=index)
    target = pd.DataFrame([[1.0, 0.0]], index=[index[0]], columns=prices.columns)
    low = run_backtest(prices, target, cost_bps=50.0)
    high = run_backtest(prices, target, cost_bps=100.0)
    assert high["equity"].iloc[-1] < low["equity"].iloc[-1] < 1.0


def test_buffer_retains_ranked_holdings_and_limits_new_names():
    scores = pd.Series({"A": 1.0, "C": 0.9, "D": 0.8, "E": 0.6, "B": 0.1})
    config = StrategyConfig(top_n=2, hold_buffer=2, max_replacements=1, min_hold_periods=1)
    selected, replacements, _ = _select_assets(scores, list(scores.index), ["A", "B"], {"A": 1, "B": 3}, config)
    assert selected == ["A", "C"]
    assert replacements == 1


def test_covariance_shrinkage_is_positive_semidefinite():
    rng = np.random.default_rng(11)
    returns = pd.DataFrame(rng.normal(size=(30, 12)), columns=[f"A{i}" for i in range(12)])
    covariance = _shrink_covariance(returns, 0.25).to_numpy()
    assert np.linalg.eigvalsh(covariance).min() >= -1e-10


def test_target_volatility_changes_the_non_optimizer_candidate():
    prices = synthetic_prices(days=1_000)
    base = StrategyConfig(frequency="monthly", top_n=4, max_weight=0.25, target_volatility=None)
    risk = StrategyConfig(
        frequency="monthly", top_n=4, max_weight=0.25, target_volatility=0.05,
        minimum_safe_weight=0.05, risk_on_equity_cap=0.85, risk_off_equity_cap=0.50,
    )
    base_targets, _ = build_targets(prices, base)
    risk_targets, risk_diagnostics = build_targets(prices, risk)
    assert not np.allclose(base_targets.to_numpy(), risk_targets.to_numpy())
    assert risk_diagnostics["optimizer"].str.contains("target_volatility").any()


def test_optimized_weights_obey_asset_equity_and_turnover_caps():
    prices = synthetic_prices(days=1_000)
    config = StrategyConfig(
        name="test_optimizer", frequency="monthly", top_n=4, max_weight=0.25,
        use_covariance_optimizer=True, covariance_window=63, optimizer_steps=8,
        turnover_limit=0.30, target_volatility=None, minimum_safe_weight=0.05,
        risk_on_equity_cap=0.85, risk_off_equity_cap=0.50,
    )
    targets, diagnostics = build_targets(prices, config)
    nonempty = targets[targets.sum(axis=1) > 0]
    assert np.allclose(nonempty.sum(axis=1), 1.0, atol=1e-8)
    assert float(nonempty.max().max()) <= config.max_weight + 1e-8
    assert (diagnostics["equity_weight"] <= diagnostics["risk_on"].map({True: 0.85, False: 0.50}) + 1e-7).all()
    later = diagnostics[diagnostics.index > diagnostics.index[0]]
    regular = later[~later["turnover_override"].astype(bool)]
    assert (regular["one_way_turnover"] <= config.turnover_limit + 1e-7).all()
    assert diagnostics["optimizer"].notna().all()


def test_covariance_failure_uses_inverse_volatility_fallback():
    prices = synthetic_prices(days=850)
    config = StrategyConfig(frequency="monthly", top_n=4, use_covariance_optimizer=True, target_volatility=None, covariance_window=63)
    original = model._optimize_weights
    try:
        model._optimize_weights = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("forced failure"))
        _, diagnostics = build_targets(prices, config)
    finally:
        model._optimize_weights = original
    assert "inverse_volatility_fallback" in set(diagnostics["optimizer"])


def test_lean_execution_review_handles_normal_partial_and_missing_volume():
    normal = lean_volume_share_review(1_000, 100_000, 100.0)
    partial = lean_volume_share_review(2_000_000, 100_000, 100.0)
    missing = lean_volume_share_review(1_000, 0, 100.0)
    assert normal.status == "filled" and normal.filled_shares == 1_000
    assert partial.status == "partial" and partial.filled_shares == 2_500
    assert missing.status == "blocked_missing_volume" and missing.filled_shares == 0


def run_all() -> None:
    tests = [value for name, value in globals().copy().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} strategy checks passed")


if __name__ == "__main__":
    run_all()
