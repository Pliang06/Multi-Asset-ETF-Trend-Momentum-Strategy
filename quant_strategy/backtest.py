"""Daily portfolio simulation with close execution, drift, and order limits."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _aligned_panel(panel: pd.DataFrame | None, index: pd.Index, columns: pd.Index) -> pd.DataFrame | None:
    if panel is None:
        return None
    return panel.reindex(index=index, columns=columns).astype("float64")


def run_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    cost_bps: float = 8.0,
    initial_cost: bool = True,
    *,
    raw_close: pd.DataFrame | None = None,
    volume: pd.DataFrame | None = None,
    participation_cap: float | None = None,
    slippage_coefficient_bps: float = 1000.0,
    missing_volume_slippage_bps: float = 100.0,
    commission_per_share: float = 0.005,
    minimum_commission: float = 1.0,
    maximum_commission_rate: float = 0.005,
    initial_capital: float = 1_000_000.0,
) -> pd.DataFrame:
    """Simulate next-session close execution and daily portfolio drift.

    With no volume panel this retains the original friction model: all orders
    fill, with ``cost_bps`` charged on two-way traded notional. When volume is
    supplied, ``participation_cap`` limits each order to a share of that day's
    reported volume. Missing or zero volume blocks trading and leaves a target
    pending. A newer scheduled target replaces an older pending target.

    Slippage is a transparent research approximation proportional to squared
    participation. Commissions can be configured per share with a minimum per
    asset order; they are not a claim to reproduce every IBKR fee schedule.
    """

    if prices.empty:
        raise ValueError("prices must not be empty")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if cost_bps < 0 or (participation_cap is not None and not 0 < participation_cap <= 1):
        raise ValueError("invalid cost or participation parameter")
    volume_aware = participation_cap is not None
    prices = prices.sort_index().astype("float64")
    index, columns = prices.index, prices.columns
    close = _aligned_panel(raw_close, index, columns)
    if close is None:
        close = prices
    volumes = _aligned_panel(volume, index, columns)
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    targets = target_weights.reindex(columns=columns).fillna(0.0).sort_index()
    execution_targets: dict[pd.Timestamp, pd.Series] = {}
    for signal_date, row in targets.iterrows():
        position = index.searchsorted(signal_date, side="right")
        if position < len(index):
            desired = row.clip(lower=0.0)
            execution_targets[index[position]] = desired / desired.sum() if desired.sum() > 0 else desired

    # CASH is explicit so partial fills and blocked orders remain fully funded.
    assets = pd.Series(0.0, index=columns, dtype="float64")
    cash = 1.0
    equity = float(initial_capital)
    pending: pd.Series | None = None
    rows: list[dict[str, object]] = []
    for date in index:
        start_equity = equity
        old_assets = assets.copy()
        asset_return = float(np.dot(old_assets.to_numpy(), returns.loc[date].to_numpy()))
        # Existing holdings earn the close-to-close move before the close trade.
        grown = old_assets * (1.0 + returns.loc[date])
        cash_grown = cash
        gross_multiplier = 1.0 + asset_return
        if gross_multiplier <= 0:
            gross_multiplier = 0.0
        total = float(grown.sum() + cash_grown)
        if total > 1e-15:
            assets = grown / total
            cash = cash_grown / total
        else:
            assets = pd.Series(0.0, index=columns)
            cash = 1.0
        equity *= gross_multiplier

        scheduled = execution_targets.get(date)
        if scheduled is not None:
            pending = scheduled.reindex(columns).fillna(0.0)

        target_arrived = scheduled is not None
        rebalance_attempted = pending is not None
        two_way_turnover = 0.0
        fee_bps_cost = 0.0
        fee_cash_cost = 0.0
        slippage_cost = 0.0
        filled_notional = 0.0
        unfilled_notional = 0.0
        unfilled_slippage_estimate = 0.0
        missing_volume_orders = 0

        if pending is not None:
            initial_portfolio = float(old_assets.sum()) <= 1e-12
            desired = pending
            delta = desired - assets
            order_scale = equity
            # Sell first to release cash; buy orders cannot borrow or use
            # notional made available by unfilled sells.
            for side in (-1, 1):
                order_names = [a for a in columns if delta[a] * side > 1e-12]
                for asset in order_names:
                    requested_weight = abs(float(delta[asset]))
                    requested_notional = requested_weight * order_scale
                    if not volume_aware:
                        fill_notional = requested_notional
                        participation = 0.0
                    else:
                        px = float(close.loc[date, asset]) if pd.notna(close.loc[date, asset]) else np.nan
                        vol = (
                            float(volumes.loc[date, asset])
                            if volumes is not None and pd.notna(volumes.loc[date, asset])
                            else np.nan
                        )
                        if not np.isfinite(px) or px <= 0 or not np.isfinite(vol) or vol <= 0:
                            fill_notional = 0.0
                            participation = np.nan
                            missing_volume_orders += 1
                            unfilled_slippage_estimate += requested_notional * missing_volume_slippage_bps / 10000.0
                        else:
                            capacity = float(participation_cap) * vol * px
                            if side > 0:
                                capacity = min(capacity, cash * order_scale)
                            fill_notional = min(requested_notional, capacity)
                            participation = fill_notional / max(vol * px, 1e-12)

                    fill_weight = fill_notional / order_scale if order_scale > 0 else 0.0
                    if side < 0:
                        fill_weight = min(fill_weight, float(assets[asset]))
                        assets[asset] -= fill_weight
                        cash += fill_weight
                    else:
                        fill_weight = min(fill_weight, cash)
                        assets[asset] += fill_weight
                        cash -= fill_weight
                    filled = fill_weight * order_scale
                    filled_notional += filled
                    two_way_turnover += fill_weight
                    remaining_notional = max(0.0, requested_notional - filled)
                    unfilled_notional += remaining_notional

                    if filled > 0:
                        if not volume_aware:
                            fee_cash_cost += (filled / order_scale if order_scale > 0 else 0.0) * cost_bps / 10000.0
                        else:
                            px = float(close.loc[date, asset]) if pd.notna(close.loc[date, asset]) else np.nan
                            vol = (
                                float(volumes.loc[date, asset])
                                if volumes is not None and pd.notna(volumes.loc[date, asset])
                                else np.nan
                            )
                            shares = filled / px if np.isfinite(px) and px > 0 else 0.0
                            per_share_commission = shares * commission_per_share
                            maximum_commission = maximum_commission_rate * filled
                            if per_share_commission < minimum_commission:
                                commission = minimum_commission
                            elif per_share_commission > maximum_commission:
                                commission = maximum_commission
                            else:
                                commission = per_share_commission
                            fee_cash_cost += (commission / order_scale if order_scale > 0 else 0.0)
                            # The same bps knob supports explicit 50/100 bps
                            # cost stress runs when volume-aware execution is on.
                            fee_cash_cost += (filled / order_scale if order_scale > 0 else 0.0) * cost_bps / 10000.0
                            if np.isfinite(participation):
                                slip_rate = min(slippage_coefficient_bps / 10000.0 * participation**2, 0.10)
                            else:
                                slip_rate = missing_volume_slippage_bps / 10000.0
                            slippage_cost += (filled / order_scale if order_scale else 0.0) * slip_rate

            if not volume_aware:
                unfilled_notional = 0.0
            else:
                # If requested orders were completely filled, avoid retaining
                # a numerically tiny pending order.
                new_target = pending
                remaining = new_target - assets
                pending = new_target if (remaining.abs().sum() > 1e-8 or unfilled_notional > 1e-6) else None

            if initial_portfolio and not initial_cost:
                fee_cash_cost = 0.0
                slippage_cost = 0.0
            total_after_trade = float(assets.sum() + cash)
            if total_after_trade > 0:
                assets /= total_after_trade
                cash /= total_after_trade

            two_way_rate = two_way_turnover
            fee_bps_cost = fee_cash_cost
            if not volume_aware:
                # The legacy convention charges cost_bps times absolute weight
                # changes, which is the conventional two-way turnover here.
                total_cost_rate = two_way_rate * cost_bps / 10000.0
                equity *= max(0.0, 1.0 - total_cost_rate)
                fee_bps_cost = total_cost_rate
                slippage_cost = 0.0
            else:
                total_cost_rate = fee_cash_cost + slippage_cost
                equity *= max(0.0, 1.0 - total_cost_rate)

        portfolio_return = equity / start_equity - 1.0 if start_equity > 0 else -1.0
        one_way = two_way_turnover / 2.0
        rows.append(
            {
                "date": date,
                "return": portfolio_return,
                "gross_return": asset_return,
                "cost": fee_bps_cost + slippage_cost,
                "fee_cost": fee_bps_cost,
                "slippage_cost": slippage_cost,
                "turnover": two_way_turnover,
                "one_way_turnover": one_way,
                "two_way_turnover": two_way_turnover,
                "filled_notional": filled_notional,
                "unfilled_notional": unfilled_notional,
                "unfilled_slippage_estimate": unfilled_slippage_estimate,
                "missing_volume_orders": missing_volume_orders,
                "equity": equity / initial_capital,
                "invested": float(assets.sum()),
                "cash_weight": float(cash),
                "rebalance": bool(target_arrived),
                "order_pending": bool(pending is not None),
                "order_attempted": bool(rebalance_attempted),
            }
        )
    return pd.DataFrame(rows).set_index("date")


def add_drawdown(result: pd.DataFrame) -> pd.DataFrame:
    result = result.copy()
    result["running_max"] = result["equity"].cummax()
    result["drawdown"] = result["equity"] / result["running_max"] - 1.0
    return result
