"""Lean-compatible execution review helpers.

This module mirrors the equations in the downloaded Lean Apache-2.0 source
(``Common/Orders/Slippage/VolumeShareSlippageModel.py``) without importing
Lean. It is intentionally a review harness: a full Lean run still requires a
configured Lean data directory, brokerage model and CLI.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FillReview:
    requested_shares: float
    filled_shares: float
    unfilled_shares: float
    participation: float | None
    slippage_rate: float
    commission: float
    status: str


def lean_volume_share_review(
    requested_shares: float,
    bar_volume: float | None,
    price: float,
    *,
    volume_limit: float = 0.025,
    price_impact: float = 0.1,
    fee_per_share: float = 0.005,
    minimum_fee: float = 1.0,
) -> FillReview:
    """Review one US ETF market order using Lean's volume-share equations."""
    requested_shares = max(float(requested_shares), 0.0)
    price = float(price)
    if requested_shares == 0:
        return FillReview(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "empty")
    if bar_volume is None or bar_volume <= 0 or price <= 0:
        # Lean applies the maximum slippage percentage for zero/negative bar
        # volume on equities. We conservatively block the fill in this review
        # layer, preserving the unfilled quantity for a later bar.
        max_slip = volume_limit * volume_limit * price_impact
        return FillReview(requested_shares, 0.0, requested_shares, None, max_slip, 0.0, "blocked_missing_volume")
    capacity = volume_limit * float(bar_volume)
    filled = min(requested_shares, capacity)
    participation = filled / float(bar_volume)
    slippage_rate = min(participation, volume_limit) ** 2 * price_impact
    commission = max(minimum_fee, filled * fee_per_share) if filled else 0.0
    status = "filled" if filled >= requested_shares - 1e-12 else "partial"
    return FillReview(requested_shares, filled, requested_shares - filled, participation, slippage_rate, commission, status)


def review_scenarios() -> list[FillReview]:
    """Normal, risk-switch and illiquid examples used by the test suite."""
    return [
        lean_volume_share_review(1_000, 100_000, 100.0),
        lean_volume_share_review(2_000_000, 100_000, 100.0),
        lean_volume_share_review(1_000, 0, 100.0),
    ]


if __name__ == "__main__":
    for scenario in review_scenarios():
        print(scenario)
