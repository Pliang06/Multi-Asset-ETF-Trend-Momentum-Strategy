"""Signal generation and constrained portfolio construction.

The module deliberately keeps the competition strategy independent from Qlib,
vectorbt and LEAN.  Their useful ideas are implemented as small, deterministic
NumPy/Pandas components so the final strategy has a small dependency surface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


EQUITY_TICKERS = {
    "SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ",
    "XLE", "XLK", "XLF", "XLI", "XLV", "XLP", "XLU", "XLY", "XLB",
}
DEFENSIVE_TICKERS = {"TLT", "IEF", "SHY", "HYG", "LQD", "GLD", "SLV", "DBC", "UUP"}
DEFENSIVE_FALLBACK = ("SHY", "IEF", "TLT", "UUP")


@dataclass(frozen=True)
class StrategyConfig:
    name: str = "monthly_trend_equal_top8"
    frequency: str = "monthly"  # weekly or monthly
    lookbacks: tuple[int, ...] = (21, 63, 126, 252)
    signal_weights: tuple[float, ...] = (0.15, 0.35, 0.30, 0.20)
    skip: int = 5
    trend_window: int = 200
    volatility_window: int = 63
    top_n: int = 8
    max_weight: float = 0.35
    use_trend_filter: bool = True
    use_medium_momentum_filter: bool = True
    inverse_volatility: bool = False
    risk_on_equity_cap: float = 0.75
    risk_off_equity_cap: float = 0.35
    target_volatility: float | None = None
    cost_bps: float = 8.0

    # Qlib-inspired turnover-aware holding controls.  Zero/None preserves the
    # original research candidate's direct top-N selection behavior.
    hold_buffer: int = 0
    max_replacements: int | None = None
    min_hold_periods: int = 0
    turnover_limit: float = 1.0  # one-way turnover, expressed as portfolio fraction

    # Covariance optimizer controls.  This is a small native optimizer rather
    # than a runtime dependency on Qlib or a commercial solver.
    use_covariance_optimizer: bool = False
    covariance_window: int = 126
    covariance_shrinkage: float = 0.25
    risk_aversion: float = 2.0
    turnover_penalty: float = 0.25
    concentration_penalty: float = 0.05
    optimizer_steps: int = 120
    # Only optimized candidates should request a mandatory safe-haven floor.
    # Keeping this at zero preserves the historical baseline definitions.
    minimum_safe_weight: float = 0.0


def _last_observation_dates(index: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    """Return actual trading dates used to make completed-period signals."""

    if frequency == "monthly":
        buckets = index.to_period("M")
    elif frequency == "weekly":
        buckets = index.to_period("W-FRI")
    else:
        raise ValueError(f"Unsupported frequency: {frequency}")
    frame = pd.Series(index, index=index)
    last_dates = frame.groupby(buckets).last().dropna()
    period_ends = pd.Series(last_dates.index, index=last_dates.index).map(
        lambda period: period.end_time.normalize()
    )
    dates = last_dates[period_ends <= index.max()].tolist()
    return pd.DatetimeIndex(dates)


def _cross_sectional_percentile(frame: pd.DataFrame) -> pd.DataFrame:
    # Keep unavailable history as NaN. Filling early-history ranks with zero
    # would admit ETFs before they have enough observations for the signal.
    return frame.rank(axis=1, pct=True, method="average")


def _capped_inverse_vol_weights(
    assets: list[str],
    volatility: pd.Series,
    max_weight: float,
) -> pd.Series:
    """Allocate inverse-volatility weights while respecting a per-asset cap."""

    if not assets:
        return pd.Series(dtype="float64")
    vol = volatility.reindex(assets).clip(lower=0.05).fillna(0.30)
    raw = 1.0 / vol
    weights = raw / raw.sum()
    fixed: dict[str, float] = {}
    active = list(assets)
    remaining = 1.0
    while active:
        active_raw = raw.reindex(active)
        proposed = active_raw / active_raw.sum() * remaining
        over = proposed[proposed > max_weight + 1e-12]
        if over.empty:
            fixed.update(proposed.to_dict())
            break
        for asset in over.index:
            fixed[asset] = max_weight
            remaining -= max_weight
            active.remove(asset)
        if remaining <= 1e-12:
            break
    result = pd.Series(fixed, dtype="float64")
    if active and remaining > 1e-12:
        result = result.add((raw.reindex(active) / raw.reindex(active).sum() * remaining), fill_value=0.0)
    return result / result.sum() if result.sum() else weights


def _normalize_with_cap(raw: pd.Series, columns: pd.Index, max_weight: float) -> pd.Series:
    """Project non-negative weights onto the simplex with an upper cap."""

    raw = raw.reindex(columns).fillna(0.0).clip(lower=0.0)
    if len(columns) * max_weight < 1.0 - 1e-10:
        raise ValueError("asset universe cannot satisfy the requested maximum weight")
    if raw.sum() <= 1e-12:
        raw = pd.Series(1.0, index=columns, dtype="float64")
    remaining = 1.0
    active = list(columns)
    result = pd.Series(0.0, index=columns, dtype="float64")
    # Iterative water filling is deterministic and does not require SciPy.
    while active and remaining > 1e-12:
        proposed_raw = raw.reindex(active)
        if proposed_raw.sum() <= 1e-12:
            proposed_raw = pd.Series(1.0, index=active, dtype="float64")
        proposed = proposed_raw / proposed_raw.sum() * remaining
        over = proposed[proposed > max_weight + 1e-12]
        if over.empty:
            result.loc[active] = proposed
            break
        for asset in over.index:
            result.loc[asset] = max_weight
            remaining -= max_weight
            active.remove(asset)
    if result.sum() <= 1e-12:
        result.loc[columns[0]] = 1.0
    return result / result.sum()


def _capped_group_weights(
    raw: pd.Series,
    assets: list[str],
    total: float,
    max_weight: float,
    lower_bounds: dict[str, float] | None = None,
) -> pd.Series:
    """Allocate a group total with box constraints using water filling."""

    lower_bounds = lower_bounds or {}
    if not assets:
        if total > 1e-10:
            raise ValueError("non-zero allocation requested for an empty asset group")
        return pd.Series(dtype="float64")
    lower = np.array([lower_bounds.get(asset, 0.0) for asset in assets], dtype="float64")
    upper = np.full(len(assets), max_weight, dtype="float64")
    if lower.sum() > total + 1e-10 or upper.sum() < total - 1e-10:
        raise ValueError("group allocation is infeasible under weight bounds")
    values = raw.reindex(assets).fillna(0.0).clip(lower=0.0).to_numpy(dtype="float64")
    if values.sum() <= 1e-12:
        values = np.ones(len(assets), dtype="float64")
    low = float(np.min(values - upper) - 1.0)
    high = float(np.max(values - lower) + 1.0)
    for _ in range(80):
        mid = (low + high) / 2.0
        proposal = np.clip(values - mid, lower, upper)
        if proposal.sum() > total:
            low = mid
        else:
            high = mid
    result = np.clip(values - (low + high) / 2.0, lower, upper)
    # Remove floating point residue without breaking bounds.
    residue = total - float(result.sum())
    if abs(residue) > 1e-10:
        room = upper - result if residue > 0 else result - lower
        for idx in np.argsort(-room):
            change = np.sign(residue) * min(abs(residue), room[idx])
            result[idx] += change
            residue -= change
            if abs(residue) <= 1e-10:
                break
    return pd.Series(result, index=assets, dtype="float64")


def _apply_equity_cap(
    weights: pd.Series,
    scores: pd.Series,
    cap: float,
    max_weight: float,
    minimum_safe_weight: float = 0.0,
) -> pd.Series:
    """Enforce equity exposure and per-asset caps.

    Defensive ETFs are added only when needed to satisfy ``cap`` or when the
    caller explicitly requests a safe-haven floor.  This is important because
    frozen historical baselines must remain comparable to their old results.
    """

    weights = weights.copy().reindex(weights.index).fillna(0.0).clip(lower=0.0)
    if weights.sum() <= 1e-12:
        return weights
    weights = weights / weights.sum()
    all_scores = scores.reindex(scores.index).fillna(-1.0)
    equity = [a for a in weights.index if a in EQUITY_TICKERS and weights.get(a, 0.0) > 0]
    defensive = [a for a in weights.index if a not in EQUITY_TICKERS and weights.get(a, 0.0) > 0]
    safe = [a for a in DEFENSIVE_FALLBACK if a in scores.index]
    safe_asset = max(safe, key=lambda a: float(all_scores.get(a, -1.0))) if safe else None
    if minimum_safe_weight > 0 and safe_asset is not None and safe_asset not in defensive:
        defensive.append(safe_asset)
        weights.loc[safe_asset] = 0.0

    # Add defensive capacity only as needed to make the exposure cap feasible.
    available_defensive = [a for a in scores.index if a not in EQUITY_TICKERS and a not in defensive]
    available_defensive.sort(key=lambda a: float(all_scores.get(a, -1.0)), reverse=True)
    min_safe = min(max(float(minimum_safe_weight), 0.0), max_weight)
    desired_equity = min(float(weights.reindex(equity).sum()), max(0.0, min(cap, 1.0 - min_safe)))
    required_defensive = 1.0 - desired_equity
    while len(defensive) * max_weight < required_defensive - 1e-10 and available_defensive:
        asset = available_defensive.pop(0)
        defensive.append(asset)
        weights.loc[asset] = 0.0
    if len(defensive) * max_weight < required_defensive - 1e-10:
        # If the available defensive names cannot carry the whole portfolio,
        # admit the highest-scoring unused equity ETF at the smallest exposure
        # needed for feasibility. This edge case matters for small synthetic
        # tests and for a temporarily incomplete data universe.
        candidates = [a for a in scores.index if a in EQUITY_TICKERS and a not in equity]
        candidates.sort(key=lambda a: float(all_scores.get(a, -1.0)), reverse=True)
        if candidates:
            equity.append(candidates[0])
            weights.loc[candidates[0]] = 0.0
            desired_equity = min(cap, max(0.0, 1.0 - len(defensive) * max_weight))
            required_defensive = 1.0 - desired_equity
        else:
            desired_equity = max(0.0, 1.0 - len(defensive) * max_weight)
            required_defensive = 1.0 - desired_equity
    if len(equity) * max_weight < desired_equity - 1e-10:
        desired_equity = len(equity) * max_weight
        required_defensive = 1.0 - desired_equity
        while len(defensive) * max_weight < required_defensive - 1e-10 and available_defensive:
            asset = available_defensive.pop(0)
            defensive.append(asset)
            weights.loc[asset] = 0.0
    equity_raw = weights.reindex(equity).fillna(0.0)
    defensive_raw = weights.reindex(defensive).fillna(0.0)
    result = pd.Series(0.0, index=weights.index.union(pd.Index(equity + defensive)), dtype="float64")
    if equity:
        result.loc[equity] = _capped_group_weights(equity_raw, equity, desired_equity, max_weight)
    lower = {safe_asset: min_safe} if safe_asset in defensive else {}
    if defensive:
        result.loc[defensive] = _capped_group_weights(
            defensive_raw, defensive, 1.0 - desired_equity, max_weight, lower
        )
    elif desired_equity < 1.0 - 1e-10:
        raise ValueError("no defensive ETF is available to satisfy the equity cap")
    result = result.clip(lower=0.0)
    if result.sum() <= 1e-12:
        raise ValueError("equity cap produced an empty portfolio")
    result = result / result.sum()
    if (result > max_weight + 1e-8).any():
        raise ValueError("equity cap allocation violates max_weight")
    equity_total = float(result.reindex(EQUITY_TICKERS).fillna(0.0).sum())
    if equity_total > cap + 1e-8:
        raise ValueError("equity cap allocation violates equity exposure cap")
    return result


def _shrink_covariance(returns: pd.DataFrame, shrinkage: float = 0.25) -> pd.DataFrame:
    """Return an annualized, positive-semidefinite shrinkage covariance."""

    clean = returns.astype("float64").replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if len(clean) < 2:
        return pd.DataFrame(np.eye(len(returns.columns)) * 0.04, index=returns.columns, columns=returns.columns)
    clean = clean.fillna(clean.median()).fillna(0.0)
    cov = clean.cov().to_numpy() * 252.0
    diag = np.diag(np.diag(cov))
    avg_var = float(np.nanmean(np.diag(cov))) if np.isfinite(cov).any() else 0.04
    target = np.eye(cov.shape[0]) * max(avg_var, 1e-8)
    alpha = float(np.clip(shrinkage, 0.0, 1.0))
    shrunk = (1.0 - alpha) * cov + alpha * target
    shrunk = (shrunk + shrunk.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(shrunk)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    psd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return pd.DataFrame(psd, index=returns.columns, columns=returns.columns)


def _defensive_base(columns: pd.Index, scores: pd.Series, max_weight: float) -> pd.Series:
    names = [a for a in DEFENSIVE_FALLBACK if a in columns]
    if not names:
        names = [a for a in columns if a in DEFENSIVE_TICKERS]
    if not names:
        return _normalize_with_cap(pd.Series(1.0, index=columns), columns, max_weight)
    raw = pd.Series(0.0, index=columns, dtype="float64")
    ranked = sorted(names, key=lambda a: float(scores.get(a, 0.0)), reverse=True)
    raw.loc[ranked] = np.linspace(1.0, 0.5, len(ranked))
    return _normalize_with_cap(raw, columns, max_weight)


def _feasible_weights(
    raw: pd.Series,
    columns: pd.Index,
    scores: pd.Series,
    max_weight: float,
    equity_cap: float,
    minimum_safe_weight: float = 0.0,
) -> pd.Series:
    weights = _normalize_with_cap(raw, columns, max_weight)
    weights = _apply_equity_cap(weights, scores, equity_cap, max_weight, minimum_safe_weight)
    return weights / weights.sum()


def _enforce_turnover(weights: pd.Series, previous: pd.Series, limit: float) -> pd.Series:
    union = weights.index.union(previous.index)
    weights = weights.reindex(union).fillna(0.0)
    previous = previous.reindex(union).fillna(0.0)
    if previous.sum() <= 1e-12:
        return weights / weights.sum()
    previous = previous / previous.sum()
    one_way = float((weights - previous).abs().sum()) / 2.0
    if one_way > limit + 1e-12 and limit < 1.0:
        weights = previous + (weights - previous) * (limit / one_way)
    return weights / weights.sum()


def _optimize_weights(
    score: pd.Series,
    covariance: pd.DataFrame,
    previous: pd.Series,
    config: StrategyConfig,
    equity_cap: float,
) -> tuple[pd.Series, str, float]:
    """Native projected-gradient optimizer with deterministic fallbacks."""

    columns = covariance.columns
    previous = previous.reindex(columns).fillna(0.0)
    previous = previous / previous.sum() if previous.sum() else pd.Series(0.0, index=columns)
    if previous.sum() <= 1e-12:
        seed = pd.Series(0.0, index=columns)
        seed.loc[score.nlargest(min(config.top_n, len(score))).index] = 1.0
        previous = _feasible_weights(seed, columns, score, config.max_weight, equity_cap, config.minimum_safe_weight)

    signal = score.reindex(columns).fillna(score.min() if len(score) else 0.0).to_numpy(dtype="float64")
    signal = signal - signal.mean()
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))
    sigma = covariance.reindex(index=columns, columns=columns).fillna(0.0).to_numpy(dtype="float64")
    sigma = (sigma + sigma.T) / 2.0
    if not np.isfinite(sigma).all():
        raise ValueError("non-finite covariance")
    eig = float(np.max(np.linalg.eigvalsh(sigma)))
    if not np.isfinite(eig) or eig <= 0:
        raise ValueError("invalid covariance spectrum")

    weights = previous.copy()
    learning_rate = 0.08 / (1.0 + config.risk_aversion * eig)
    prev_array = previous.to_numpy(dtype="float64")
    for _ in range(max(1, int(config.optimizer_steps))):
        w_array = weights.to_numpy(dtype="float64")
        gradient = -signal + 2.0 * config.risk_aversion * (sigma @ w_array)
        gradient += 2.0 * config.concentration_penalty * w_array
        if config.turnover_penalty:
            gradient += config.turnover_penalty * np.sign(w_array - prev_array)
        proposal = pd.Series(w_array - learning_rate * gradient, index=columns)
        weights = _feasible_weights(proposal, columns, score, config.max_weight, equity_cap, config.minimum_safe_weight)

    # Return a feasible point. The full previous portfolio is available only
    # to ``build_targets``; turnover is therefore enforced once there on the
    # union of old and new assets, avoiding re-normalizing a partial previous
    # vector and accidentally breaching max_weight.
    weights = _feasible_weights(weights, columns, score, config.max_weight, equity_cap, config.minimum_safe_weight)
    return weights, "covariance_projected_gradient", float(np.sqrt(weights.to_numpy() @ sigma @ weights.to_numpy()))


def _apply_target_volatility(
    weights: pd.Series,
    covariance: pd.DataFrame,
    scores: pd.Series,
    config: StrategyConfig,
    equity_cap: float,
    minimum_safe_weight: float = 0.0,
) -> pd.Series:
    if config.target_volatility is None:
        return weights
    sigma = covariance.reindex(index=weights.index, columns=weights.index).fillna(0.0).to_numpy()
    predicted = float(np.sqrt(max(weights.to_numpy() @ sigma @ weights.to_numpy(), 0.0)))
    if not np.isfinite(predicted) or predicted <= config.target_volatility:
        return weights
    defense = _defensive_base(weights.index, scores, config.max_weight)
    best = weights
    for blend in np.linspace(0.1, 1.0, 10):
        candidate = (1.0 - blend) * weights + blend * defense
        candidate = _feasible_weights(candidate, weights.index, scores, config.max_weight, equity_cap, config.minimum_safe_weight)
        candidate_sigma = covariance.reindex(index=candidate.index, columns=candidate.index).fillna(0.0).to_numpy()
        candidate_vol = float(np.sqrt(max(candidate.to_numpy() @ candidate_sigma @ candidate.to_numpy(), 0.0)))
        best = candidate
        if candidate_vol <= config.target_volatility:
            break
    return best


def _select_assets(
    row_score: pd.Series,
    eligible: list[str],
    previous_selected: list[str],
    hold_ages: dict[str, int],
    config: StrategyConfig,
) -> tuple[list[str], int, int]:
    ranked = list(row_score.index)
    if not eligible:
        eligible = ranked
    top = eligible[: config.top_n]
    if not previous_selected or config.hold_buffer <= 0:
        return top, len(set(top) - set(previous_selected)), 0

    rank = {asset: i for i, asset in enumerate(eligible)}
    keep = []
    forced_exit = 0
    for asset in previous_selected:
        age = hold_ages.get(asset, 0)
        if asset not in rank:
            forced_exit += 1
        elif rank[asset] < config.top_n + config.hold_buffer or age < config.min_hold_periods:
            keep.append(asset)
        else:
            forced_exit += 1

    # Keep the current names first, then admit at most the configured number
    # of genuinely new names.  If a trend/risk rule forces an exit, the next
    # month's portfolio can temporarily contain fewer than top_n assets; this
    # is preferable to silently violating the replacement budget.
    keep = sorted(set(keep), key=lambda asset: float(row_score.get(asset, -1.0)), reverse=True)[: config.top_n]
    newcomers = [asset for asset in top if asset not in keep and asset not in previous_selected]
    if config.max_replacements is not None:
        newcomers = newcomers[: max(0, config.max_replacements)]
    selected = keep + newcomers[: max(0, config.top_n - len(keep))]
    if not selected:
        selected = top[: config.top_n]
    if len(selected) > config.top_n:
        selected = sorted(selected, key=lambda asset: float(row_score.get(asset, -1.0)), reverse=True)[: config.top_n]
    replacements = len(set(selected) - set(previous_selected))
    return selected, replacements, forced_exit


def _drift_weights(
    weights: pd.Series,
    prices: pd.DataFrame,
    execution_date: pd.Timestamp | None,
    signal_date: pd.Timestamp,
) -> pd.Series:
    """Mark prior target weights to the signal-date close before trading."""
    current = weights.reindex(prices.columns).fillna(0.0).clip(lower=0.0)
    if current.sum() <= 1e-12 or execution_date is None or execution_date >= signal_date:
        return current / current.sum() if current.sum() else current
    first = prices.index.searchsorted(execution_date, side="right")
    last = prices.index.searchsorted(signal_date, side="right")
    if last <= first:
        return current / current.sum()
    daily_returns = prices.pct_change(fill_method=None).iloc[first:last].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    growth = (1.0 + daily_returns).prod(axis=0).reindex(prices.columns).fillna(1.0)
    marked = current * growth
    return marked / marked.sum() if marked.sum() else current / current.sum()


def build_targets(prices: pd.DataFrame, config: StrategyConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build signal-date target weights and diagnostics."""

    prices = prices.sort_index().astype("float64")
    returns = prices.pct_change(fill_method=None)
    signal_dates = _last_observation_dates(prices.index, config.frequency)
    rank_components = []
    for lookback in config.lookbacks:
        momentum = prices.shift(config.skip) / prices.shift(config.skip + lookback) - 1.0
        rank_components.append(_cross_sectional_percentile(momentum))
    score = sum(weight * component for weight, component in zip(config.signal_weights, rank_components))
    trend = prices / prices.rolling(config.trend_window).mean() - 1.0
    volatility = returns.rolling(config.volatility_window).std() * np.sqrt(252.0)
    medium_momentum = prices.shift(config.skip) / prices.shift(config.skip + 63) - 1.0
    risk_on = (trend["SPY"] > 0.0) & (medium_momentum["SPY"] > 0.0) if "SPY" in prices else pd.Series(False, index=prices.index)

    targets = pd.DataFrame(0.0, index=signal_dates, columns=prices.columns)
    diagnostics: list[dict[str, object]] = []
    previous_selected: list[str] = []
    previous_rank: dict[str, int] = {}
    hold_ages: dict[str, int] = {}
    previous_weights = pd.Series(0.0, index=prices.columns)
    previous_execution_date: pd.Timestamp | None = None
    returns_for_cov = returns
    for date in signal_dates:
        row_score = score.loc[date].dropna().sort_values(ascending=False)
        if row_score.empty:
            targets.loc[date] = previous_weights.reindex(prices.columns).fillna(0.0)
            diagnostics.append(
                {
                    "signal_date": date,
                    "risk_on": bool(risk_on.get(date, False)),
                    "selected": "",
                    "optimizer": "no_signal",
                    "equity_weight": float(previous_weights.reindex(list(EQUITY_TICKERS)).fillna(0.0).sum()),
                    "predicted_vol": float("nan"),
                    "replacements": 0,
                    "replacement_names": "",
                    "forced_exits": 0,
                    "turnover_override": False,
                    "exit_reasons": "",
                    "rank_changes": "",
                    "one_way_turnover": 0.0,
                }
            )
            continue
        prior_weights = _drift_weights(previous_weights, prices, previous_execution_date, date)
        eligible = list(row_score.index)
        if config.use_trend_filter:
            eligible = [a for a in eligible if trend.loc[date].get(a, np.nan) > 0]
        if config.use_medium_momentum_filter:
            eligible = [a for a in eligible if medium_momentum.loc[date].get(a, np.nan) > 0]
        selected, replacements, forced_exit = _select_assets(row_score, eligible, previous_selected, hold_ages, config)
        current_rank = {asset: rank + 1 for rank, asset in enumerate(row_score.index)}
        eligible_set = set(eligible)
        exit_reasons: dict[str, str] = {}
        for asset in previous_selected:
            if asset not in eligible_set:
                exit_reasons[asset] = "trend_or_momentum_filter"
            elif asset not in selected:
                exit_reasons[asset] = "rank_buffer_or_replacement_limit"
        rank_changes = {
            asset: f"{previous_rank[asset]}->{current_rank.get(asset, 'NA')}"
            for asset in previous_selected if asset in previous_rank
        }
        replacement_names = sorted(set(selected) - set(previous_selected))
        if not selected:
            selected = list(row_score.index[: min(config.top_n, len(row_score))])

        equity_cap = config.risk_on_equity_cap if bool(risk_on.get(date, False)) else config.risk_off_equity_cap
        if config.use_covariance_optimizer:
            defense_assets = [a for a in DEFENSIVE_FALLBACK if a in prices.columns and a not in selected]
            optimization_assets = list(dict.fromkeys(selected + defense_assets))
            start = max(0, prices.index.get_loc(date) - config.covariance_window + 1)
            cov_returns = returns_for_cov.iloc[start : prices.index.get_loc(date) + 1][optimization_assets]
            covariance = _shrink_covariance(cov_returns, config.covariance_shrinkage)
            opt_scores = row_score.reindex(optimization_assets).fillna(row_score.min())
            try:
                weights, optimizer_name, predicted_vol = _optimize_weights(
                    opt_scores, covariance, prior_weights.reindex(optimization_assets), config, equity_cap
                )
                weights = _apply_target_volatility(weights, covariance, opt_scores, config, equity_cap)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                optimizer_name = "inverse_volatility_fallback"
                weights = _capped_inverse_vol_weights(selected, volatility.loc[date], config.max_weight)
                weights = _apply_equity_cap(
                    weights, row_score, equity_cap, config.max_weight, config.minimum_safe_weight
                )
                predicted_vol = float("nan")
        else:
            optimizer_name = "inverse_volatility" if config.inverse_volatility else "equal_weight"
            if config.inverse_volatility:
                weights = _capped_inverse_vol_weights(selected, volatility.loc[date], config.max_weight)
            else:
                weights = pd.Series(1.0 / len(selected), index=selected, dtype="float64")
            weights = _apply_equity_cap(
                weights, row_score, equity_cap, config.max_weight, config.minimum_safe_weight
            )
            predicted_vol = float("nan")
            if config.target_volatility is not None:
                defense_assets = [a for a in DEFENSIVE_FALLBACK if a in prices.columns and a not in weights.index]
                risk_assets = list(dict.fromkeys(list(weights.index) + defense_assets))
                start = max(0, prices.index.get_loc(date) - config.covariance_window + 1)
                cov_returns = returns_for_cov.iloc[start : prices.index.get_loc(date) + 1][risk_assets]
                covariance = _shrink_covariance(cov_returns, config.covariance_shrinkage)
                risk_scores = row_score.reindex(risk_assets).fillna(row_score.min())
                weights = weights.reindex(risk_assets).fillna(0.0)
                weights = _apply_target_volatility(weights, covariance, risk_scores, config, equity_cap)
                vol_matrix = covariance.reindex(index=weights.index, columns=weights.index).fillna(0.0).to_numpy()
                predicted_vol = float(np.sqrt(max(weights.to_numpy() @ vol_matrix @ weights.to_numpy(), 0.0)))
                optimizer_name = f"{optimizer_name}_target_volatility"

        # Interpolation with a feasible previous portfolio preserves the
        # non-negativity, asset cap and equity exposure cap constraints.
        prior_equity = float(prior_weights.reindex(list(EQUITY_TICKERS)).fillna(0.0).sum())
        emergency_turnover_override = bool(
            prior_weights.sum() <= 1e-12
            or any(asset not in eligible_set for asset in previous_selected)
            or (prior_weights.sum() > 0 and prior_equity > equity_cap + 1e-8)
        )
        effective_turnover_limit = 1.0 if emergency_turnover_override else config.turnover_limit
        weights = _enforce_turnover(weights, prior_weights, effective_turnover_limit)
        # A new exposure regime may require a fast risk reduction. In that
        # case the exposure constraint takes precedence over the turnover cap.
        weights = _apply_equity_cap(
            weights, row_score, equity_cap, config.max_weight, config.minimum_safe_weight
        )
        targets.loc[date, weights.index] = weights
        held_now = set(weights[weights > 1e-8].index)
        for asset in list(hold_ages):
            if asset in held_now:
                hold_ages[asset] += 1
            else:
                del hold_ages[asset]
        for asset in held_now:
            hold_ages.setdefault(asset, 1)
        previous_selected = list(selected)
        previous_weights = targets.loc[date].copy()
        next_position = prices.index.searchsorted(date, side="right")
        previous_execution_date = prices.index[next_position] if next_position < len(prices.index) else date
        diagnostics.append(
            {
                "signal_date": date,
                "risk_on": bool(risk_on.get(date, False)),
                "selected": ",".join(weights[weights > 1e-8].sort_values(ascending=False).index),
                "equity_weight": float(weights.reindex(list(EQUITY_TICKERS)).fillna(0.0).sum()),
                "predicted_vol": predicted_vol,
                "optimizer": optimizer_name,
                "replacements": replacements,
                "replacement_names": ",".join(replacement_names),
                "forced_exits": forced_exit,
                "turnover_override": emergency_turnover_override,
                "exit_reasons": ";".join(f"{asset}:{reason}" for asset, reason in sorted(exit_reasons.items())),
                "rank_changes": ";".join(f"{asset}:{change}" for asset, change in sorted(rank_changes.items())),
                "one_way_turnover": float((weights.reindex(prices.columns).fillna(0.0) - prior_weights).abs().sum() / 2.0),
            }
        )
        previous_rank = {asset: current_rank[asset] for asset in selected if asset in current_rank}
    return targets, pd.DataFrame(diagnostics).set_index("signal_date")


def baseline_targets(prices: pd.DataFrame, top_n: int = 3, lookback: int = 60) -> pd.DataFrame:
    """Reproduce the downloaded project's simple monthly momentum baseline."""

    monthly_dates = _last_observation_dates(prices.index, "monthly")
    momentum = prices / prices.shift(lookback) - 1.0
    signals = momentum.reindex(monthly_dates)
    weights = pd.DataFrame(0.0, index=monthly_dates, columns=prices.columns)
    for date, row in signals.iterrows():
        names = row.dropna().nlargest(top_n).index
        if len(names):
            weights.loc[date, names] = 1.0 / len(names)
    return weights
