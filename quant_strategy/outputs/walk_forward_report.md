# Rolling walk-forward report

Test length: 12 months; rolling step: one month; folds: 130.
Windows overlap and are not statistically independent. Training and validation select among the fixed one-factor candidates; test returns do not select parameters.

| Candidate | Median OOS Sharpe | 25th percentile OOS Sharpe | Worst window drawdown | Median invested days |
|---|---:|---:|---:|---:|
| wf_equal8 | 0.787 | 0.110 | -18.5% | 100.0% |
| wf_cov_buffer | 0.595 | 0.043 | -19.1% | 100.0% |

| One-way cost stress | Median OOS Sharpe for fixed covariance + buffer candidate |
|---:|---:|
| 8 bps | 0.595 |
| 50 bps | 0.252 |
| 100 bps | -0.159 |

## Promotion gate

The covariance + buffer candidate's 25th-percentile Sharpe gate (>= 0.30): FAIL.
Its 50 bps median Sharpe gate (>= 0.60): FAIL.
Its median Sharpe vs simple Top-8 gate: FAIL.
Its worst-window drawdown within 3 percentage points of simple Top-8: PASS.
Its invested ratio and flat-day gate: PASS.
Keep the frozen simple baseline as the primary candidate unless every predeclared out-of-sample, drawdown, turnover, holding and cost gate passes.

These results use the current adjusted-close research snapshot. The ETF universe is fixed and has survivorship risk; historical rolling windows do not predict competition returns. Execution stress is a model until reconciled with IBKR fills.
