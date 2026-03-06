# Beta 6.1 Matrix Report (A1)

Date: 2026-03-06
Resident: HK001_jessica (only)
Seeds: 11, 22, 33
Windows: day 7-10 and full window (4-10)

## day7_10

| Seed | LR MAE | LR Improve vs Full Bundle | Bedroom Sleep MAE | Hard Gate Pass |
|---|---:|---:|---:|---:|
| 11 | 90.842 | 62.61% | 90.938 | 4/5 |
| 22 | 88.393 | 63.62% | 44.717 | 4/5 |
| 33 | 84.149 | 65.36% | 40.337 | 4/5 |

- Median LR MAE: 88.393 (improvement 63.62%)
- Mean hard-gate pass rate: 0.8000

## full_window

| Seed | LR MAE | LR Improve vs Full Bundle | Bedroom Sleep MAE | Hard Gate Pass |
|---|---:|---:|---:|---:|
| 11 | 96.543 | 60.26% | 37.879 | 16/20 |
| 22 | 116.346 | 52.11% | 56.742 | 16/20 |
| 33 | 106.675 | 56.09% | 45.416 | 16/20 |

- Median LR MAE: 106.675 (improvement 56.09%)
- Mean hard-gate pass rate: 0.8000

## Matrix Gate Evaluation
- median_lr_improvement_ge_20pct_each_window: **PASS**
- no_catastrophic_regression_gt_60pct_vs_anchor: **PASS**
- gate_pass_rate_trend_non_decreasing: **PASS**
- resident_coverage: **INSUFFICIENT**

## Decision
- **NO_GO_INCOMPLETE_COHORT**

Note: promotion remains blocked because only one resident is available in `data/raw`.
