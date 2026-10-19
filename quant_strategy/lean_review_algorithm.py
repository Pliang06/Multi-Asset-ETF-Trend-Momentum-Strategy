"""Minimal QuantConnect Lean adapter for brokerage and slippage checks.

Copy this file into a configured Lean Python project and pass a JSON object as
the ``targets`` parameter, for example ``{"SPY":0.5,"IEF":0.5}``. This
adapter is intentionally separate from the competition runtime.
"""

from __future__ import annotations

import json

from AlgorithmImports import *
from Orders.Slippage.VolumeShareSlippageModel import VolumeShareSlippageModel


class QuantLeagueExecutionReview(QCAlgorithm):
    def initialize(self) -> None:
        self.set_start_date(2022, 1, 1)
        self.set_end_date(2026, 10, 18)
        self.set_cash(1_000_000)
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.CASH)

        raw_targets = self.get_parameter("targets") or '{"SPY":0.5,"IEF":0.5}'
        self._targets = {str(ticker): float(weight) for ticker, weight in json.loads(raw_targets).items()}
        if any(weight < 0 for weight in self._targets.values()) or sum(self._targets.values()) > 1.000001:
            raise ValueError("targets must be long-only and sum to at most 1")

        self._symbols = {}
        for ticker in self._targets:
            security = self.add_equity(ticker, Resolution.DAILY)
            security.set_data_normalization_mode(DataNormalizationMode.RAW)
            security.set_slippage_model(VolumeShareSlippageModel(0.025, 0.1))
            self._symbols[ticker] = security.symbol

        self._rebalance_count = 0
        self.schedule.on(
            self.date_rules.month_start(next(iter(self._symbols.values()))),
            self.time_rules.after_market_open(next(iter(self._symbols.values())), 1),
            self._execute_targets,
        )

    def _execute_targets(self) -> None:
        targets = [PortfolioTarget(self._symbols[ticker], weight) for ticker, weight in self._targets.items()]
        self.set_holdings(targets, liquidate_existing_holdings=True)
        self._rebalance_count += 1
        self.log(f"EXECUTION_REVIEW rebalance={self._rebalance_count} holdings={self.portfolio.invested}")

    def on_order_event(self, order_event: OrderEvent) -> None:
        self.log(
            f"ORDER_EVENT id={order_event.order_id} status={order_event.status} "
            f"fill_qty={order_event.fill_quantity} fill_price={order_event.fill_price} "
            f"message={order_event.message}"
        )

    def on_end_of_algorithm(self) -> None:
        self.log(
            f"EXECUTION_REVIEW_COMPLETE rebalances={self._rebalance_count} "
            f"orders={self.transactions.orders_count} "
            f"final_value={self.portfolio.total_portfolio_value}"
        )
