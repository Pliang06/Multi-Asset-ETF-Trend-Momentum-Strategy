"""Performance metrics used consistently across candidate comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_return(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return float("nan")
    growth = float((1.0 + returns).prod())
    return growth ** (252.0 / len(returns)) - 1.0 if growth > 0 else -1.0


def annualized_volatility(returns: pd.Series) -> float:
    return float(returns.dropna().std(ddof=1) * np.sqrt(252.0))


def sharpe(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    returns = returns.dropna()
    if len(returns) < 2 or returns.std(ddof=1) == 0:
        return 0.0
    excess = returns - risk_free_rate / 252.0
    return float(excess.mean() / excess.std(ddof=1) * np.sqrt(252.0))


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def stability_score(returns: pd.Series) -> float:
    """A transparent stability proxy: positive months plus downside deviation."""

    monthly = (1.0 + returns.fillna(0.0)).resample("ME").prod() - 1.0
    positive_fraction = float((monthly > 0).mean()) if len(monthly) else 0.0
    downside = returns[returns < 0].std(ddof=1) if (returns < 0).sum() > 1 else 0.0
    return float(positive_fraction - 0.25 * downside * np.sqrt(252.0))


def summarize(result: pd.DataFrame) -> dict[str, float]:
    returns = result["return"].astype("float64")
    invested = result.get("invested", pd.Series(1.0, index=result.index)).astype("float64")
    one_way = result.get("one_way_turnover", result.get("turnover", pd.Series(0.0, index=result.index)) / 2.0)
    return {
        "total_return": float(result["equity"].iloc[-1] - 1.0),
        "annualized_return": annualized_return(returns),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe": sharpe(returns),
        "max_drawdown": max_drawdown(returns),
        "stability": stability_score(returns),
        "avg_daily_turnover": float(result["turnover"].mean()),
        "rebalance_count": float(result["rebalance"].sum()),
        "positive_days": float((returns > 0).mean()),
        "holding_ratio_ge50": float((invested >= 0.50).mean()),
        "flat_days": float((invested <= 1e-8).sum()),
        "total_one_way_turnover": float(one_way.sum()),
    }


def approximate_competition_score(metrics: dict[str, float]) -> float:
    """An internal, relative score; official scoring is standardized by peers."""

    # This is intentionally only a diagnostic. It rewards return and Sharpe,
    # penalizes drawdown, and gives a modest stability term.
    return float(
        0.25 * metrics["total_return"]
        + 0.35 * metrics["sharpe"] / np.sqrt(252.0)
        + 0.25 * metrics["max_drawdown"]
        + 0.15 * metrics["stability"]
    )
