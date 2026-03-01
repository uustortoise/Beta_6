# Beta 6 Uncertainty Taxonomy Contract (v1)

- Date: 2026-02-25
- Status: Active
- Policy file: `backend/config/beta6_uncertainty_policy.yaml`

## 1. Canonical Classes (Mutually Exclusive)

1. `low_confidence`
2. `unknown`
3. `outside_sensed_space`

Beta 6 must never emit merged uncertainty states. If multiple states are present in one report, the gate engine emits deterministic conflict reason code `fail_uncertainty_conflict`.

## 2. Deterministic Reason-Code Mapping

1. `low_confidence` -> `fail_uncertainty_low_confidence`
2. `unknown` -> `fail_uncertainty_unknown`
3. `outside_sensed_space` -> `fail_uncertainty_outside_sensed_space`

Contract-violation mappings:
1. conflicting states/flags -> `fail_uncertainty_conflict`
2. malformed uncertainty payload -> `fail_uncertainty_invalid`

## 3. Accepted Input Shapes

1. `uncertainty_state: "<class>"`
2. `uncertainty: "<class>"`
3. `uncertainty: {state: "<class>"}`
4. boolean flags at top level:
   - `low_confidence`
   - `unknown`
   - `outside_sensed_space`
5. boolean flags under `uncertainty` mapping:
   - `uncertainty.low_confidence`
   - `uncertainty.unknown`
   - `uncertainty.outside_sensed_space`

If explicit token and flags disagree, output is `fail_uncertainty_conflict`.

## 4. Routing Targets

1. `low_confidence` -> `model_quality_review`
2. `unknown` -> `label_ambiguity_review`
3. `outside_sensed_space` -> `sensor_coverage_review`

These routing targets are encoded in `backend/config/beta6_uncertainty_policy.yaml`.
