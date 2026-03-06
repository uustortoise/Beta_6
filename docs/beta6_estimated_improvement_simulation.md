# Beta 6 Estimated Improvement Simulation (Counterfactual)

## What this is
A counterfactual Monte Carlo estimate (not a full retrain) of expected improvement if the planned architecture phases are implemented in sequence:
1. Activity-head loss masking for unoccupied/unknown windows.
2. Gate semantic alignment for short-window scarcity.
3. Episode-level timeline evaluation in gate path.
4. Decoder/fusion re-enable only after non-regression.

## Inputs (Jessica A/B already measured)
Window: day 7-10, seed 22

- Baseline anchor (`/tmp/beta6_sim_baseline_d7_10_seed22.json`)
  - LivingRoom MAE: 88.39
  - LivingRoom predicted occupied minutes (mean over splits): 126.83
- Full promotion bundle (`/tmp/beta6_sim_promoted_d7_10_seed22.json`)
  - LivingRoom MAE: 242.95
  - LivingRoom predicted occupied minutes (mean over splits): 622.61
- Cross-room v2 (`/tmp/beta6_sim_lr_cross_room_presence_v2_d7_10_seed22.json`)
  - LivingRoom MAE: 187.54

## Simulation model
Fitted relation from observed variants (baseline + cross-room v2 + full bundle):
- `LR_MAE ~= a + b * pred_occupied_minutes`
- fitted `a=49.13`, `b=0.2872`

Counterfactual assumptions:
- Phase 1 masking removes 35-65% of excess occupied minutes above baseline.
- Phase 4 tuned decoder/fusion removes additional 10-35% of remaining excess.
- Residual uncertainty noise on LR MAE: Gaussian `sigma=20` minutes.
- Bedroom MAE modeled as small drift around current full-bundle level (uniform -1.5 to +1.0 min).

Runs: 20,000 Monte Carlo samples.

## Estimated outcome distribution
### LivingRoom MAE (lower is better)
- P10: **111.0**
- P50: **140.5**
- P90: **170.4**
- Mean: **140.6**

### LivingRoom improvement vs full bundle (242.95)
- Improvement minutes P10/P50/P90: **72.5 / 102.4 / 132.0**
- Improvement percent P10/P50/P90: **29.9% / 42.2% / 54.3%**

### Probability targets
- `P(LR_MAE <= baseline 88.39)`: **1.1%**
- `P(LR_MAE <= 120)`: **18.7%**

### Bedroom MAE (lower is better)
- P10/P50/P90: **43.78 / 44.76 / 45.76**

## Interpretation
- The planned implementation is likely to materially improve over the current full promotion bundle.
- Median expectation is still above current baseline on Jessica, so promotion should remain shadow-first and gated.
- The highest-leverage phase remains Phase 1 (activity-head weighting fix), followed by careful decoder/fusion re-enable.

## Caveats
- This is a scenario simulation, not a direct training run of modified code.
- It depends on assumptions about how much Phase 1/4 reduce excess occupancy activation.
- Confidence should be upgraded only after true retrain A/B on at least 2-3 residents and multiple seeds.
