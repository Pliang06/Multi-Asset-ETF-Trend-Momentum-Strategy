# Strategy research report

This report is a model-selection record, not a performance promise.
The official competition standardizes metrics against other teams; the diagnostic score below is only used to compare our own candidates.

Primary research candidate: **monthly_trend_equal_top8**. The primary label is fixed before the final competition window; it is not changed merely because another column has a better in-sample score.

## Candidate comparison

| candidate | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | stability | avg_daily_turnover | rebalance_count | positive_days | holding_ratio_ge50 | flat_days | total_one_way_turnover | diagnostic_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_six_etf | 4.7082 | 0.1095 | 0.1338 | 0.8439 | -0.2951 | 0.6478 | 0.0304 | 201.0000 | 0.5444 | 0.9856 | 61.0000 | 64.1411 | 1.2191 |
| monthly_trend_inverse_top8 | 2.6544 | 0.0804 | 0.1083 | 0.7686 | -0.1669 | 0.5472 | 0.0419 | 201.0000 | 0.5112 | 0.9356 | 272.0000 | 88.5451 | 0.7209 |
| monthly_turnover_limited_top8 | 2.6169 | 0.0797 | 0.1123 | 0.7399 | -0.1747 | 0.5458 | 0.0299 | 201.0000 | 0.5150 | 0.9356 | 272.0000 | 63.1233 | 0.7087 |
| monthly_trend_equal_top8 | 2.6226 | 0.0798 | 0.1160 | 0.7204 | -0.1835 | 0.5455 | 0.0401 | 201.0000 | 0.5096 | 0.9356 | 272.0000 | 84.7746 | 0.7075 |
| monthly_risk_budget_top8 | 2.5772 | 0.0790 | 0.1064 | 0.7680 | -0.1914 | 0.5569 | 0.0338 | 201.0000 | 0.5176 | 0.9356 | 272.0000 | 71.4497 | 0.6969 |
| monthly_volume_execution_top8 | 2.5579 | 0.0787 | 0.1157 | 0.7129 | -0.1845 | 0.5404 | 0.0351 | 201.0000 | 0.5086 | 0.9356 | 272.0000 | 74.1212 | 0.6901 |
| monthly_trend_equal_top10 | 2.5233 | 0.0780 | 0.1100 | 0.7385 | -0.1948 | 0.5713 | 0.0327 | 201.0000 | 0.5129 | 0.9356 | 272.0000 | 69.0315 | 0.6841 |
| monthly_hold_buffer_top8 | 2.3055 | 0.0740 | 0.1210 | 0.6503 | -0.2127 | 0.5636 | 0.0315 | 201.0000 | 0.5103 | 0.9356 | 272.0000 | 66.4189 | 0.6221 |
| monthly_risk_managed | 2.0862 | 0.0696 | 0.1109 | 0.6621 | -0.1747 | 0.5462 | 0.0350 | 201.0000 | 0.5096 | 0.9356 | 272.0000 | 73.8781 | 0.5744 |
| monthly_target_vol12_top8 | 1.6933 | 0.0609 | 0.1063 | 0.6095 | -0.1773 | 0.5619 | 0.0319 | 201.0000 | 0.5141 | 0.9356 | 272.0000 | 67.2672 | 0.4767 |
| monthly_optimized_buffer_top8 | 1.3999 | 0.0536 | 0.1011 | 0.5677 | -0.1906 | 0.5730 | 0.0304 | 201.0000 | 0.5129 | 0.9356 | 272.0000 | 64.2198 | 0.4008 |
| monthly_covariance_top8 | 1.4283 | 0.0544 | 0.1095 | 0.5388 | -0.2130 | 0.5555 | 0.0342 | 201.0000 | 0.5110 | 0.9356 | 272.0000 | 72.1793 | 0.3990 |
| weekly_risk_managed | 0.9231 | 0.0398 | 0.1074 | 0.4172 | -0.1802 | 0.5266 | 0.0869 | 875.0000 | 0.5195 | 0.9380 | 262.0000 | 183.4526 | 0.2739 |

## Latest target portfolio

| ticker | weight |
| --- | --- |
| DBC | 0.2500 |
| SPY | 0.1071 |
| QQQ | 0.1071 |
| EFA | 0.1071 |
| EEM | 0.1071 |
| XLE | 0.1071 |
| XLK | 0.1071 |
| XLV | 0.1071 |

## Cost sensitivity for the primary candidate

| cost_bps | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | stability | avg_daily_turnover | rebalance_count | positive_days | holding_ratio_ge50 | flat_days | total_one_way_turnover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | 3.0061 | 0.0863 | 0.1157 | 0.7741 | -0.1778 | 0.5553 | 0.0351 | 201.0000 | 0.5108 | 0.9356 | 272.0000 | 74.1226 |
| 5.0000 | 2.7198 | 0.0815 | 0.1157 | 0.7359 | -0.1820 | 0.5503 | 0.0351 | 201.0000 | 0.5096 | 0.9356 | 272.0000 | 74.1212 |
| 8.0000 | 2.5579 | 0.0787 | 0.1157 | 0.7129 | -0.1845 | 0.5404 | 0.0351 | 201.0000 | 0.5086 | 0.9356 | 272.0000 | 74.1212 |
| 10.0000 | 2.4548 | 0.0768 | 0.1157 | 0.6977 | -0.1862 | 0.5404 | 0.0351 | 201.0000 | 0.5082 | 0.9356 | 272.0000 | 74.1176 |
| 20.0000 | 1.9784 | 0.0673 | 0.1157 | 0.6208 | -0.1976 | 0.5305 | 0.0351 | 201.0000 | 0.5060 | 0.9356 | 272.0000 | 74.1176 |
| 50.0000 | 0.9068 | 0.0393 | 0.1164 | 0.3894 | -0.2419 | 0.5156 | 0.0351 | 201.0000 | 0.4982 | 0.9356 | 272.0000 | 74.1177 |

## Historical competition-window replay

Across 16 historical October 26 to January 27 windows, the primary candidate had a positive return in 68.8% of windows; median return was 2.75%, and the worst was -4.50%.

## Interpretation

The selected candidate is still subject to a final out-of-sample review, execution test and competition-account verification.
The split-validation table is saved separately; the primary candidate was chosen because it remained strong in the pre-2023 training period and in the later test period, while no parameter was selected from the live competition window.
Do not change parameters after seeing the competition-period data without recording the change and its rationale.