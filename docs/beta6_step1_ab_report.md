# Beta 6.1 Step 1 A/B Report

Date: 2026-03-06
Resident: HK001_jessica | Seed: 22 | Window: day 7-10

## Variant Results

| Variant | LR MAE | Improvement vs Full Bundle | Bedroom Sleep MAE | Bedroom Δ vs A0 | Hard Gate Pass | False-Empty Rate* | Home-Empty Precision* |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 88.393 | 63.62% | 44.717 | +0.000 | 4/5 | 0.0304 | 0.2575 |
| A1 | 88.393 | 63.62% | 44.717 | +0.000 | 4/5 | 0.0304 | 0.2575 |
| A2 | 88.393 | 63.62% | 44.717 | +0.000 | 4/5 | 0.0304 | 0.2575 |

## Gate Check Against Plan

- LivingRoom MAE improve >= 20% vs full bundle: **PASS**
- Bedroom sleep MAE delta <= +2.0 minutes vs A0: **PASS**
- Hard-gate pass count non-regression: **PASS**
- Minority recall anti-collapse: **PASS**
- Occupancy-head safety non-regression: **PASS**
- Home-empty operational utility: **PASS**

## Decision

- Recommended variant: **A1**
- Overall Step-1 promotion decision: **GO**
- Operational recommendation: **CANARY_ELIGIBLE**

*Safety metrics use direct home-empty summary when present; otherwise inference fallback is used.
