# 2026-03-11 Beta6 Bedroom Instability Promotion Policy

## Decision

Bedroom instability is now a **blocking** promotion condition, not watch-only, when all of the following are true in saved candidate metadata:

- `promotion_time_drift_summary.risk_level == "high"`
- grouped-date stability says the candidate is unstable across date slices
- `promotion_risk_flags` includes `unstable_date_slices:bedroom`

## Rationale

This is the minimal follow-up consistent with the Bedroom-first plan and the Task 6B evidence.

- The plan/design says valid alternate-regime instability must be detected and rejected before promotion, not merely annotated for review.
- The fresh Bedroom rerun still produces a usable Dec 17 replay (`final_macro_f1 = 0.7231961200695896`), which means replay quality alone can hide the regime problem.
- The saved grouped-date summary shows the real problem clearly:
  - `worst_date = 2025-12-17`
  - `macro_f1_range = 0.2184746569435445`
  - `bedroom_normal_use` recall range `= 0.8427375318529304`
  - `risk_level = high`

That combination is strong enough to block Bedroom promotion without widening policy behavior for other rooms.

## Review-Surface Fix

`backend/scripts/run_bedroom_root_cause_matrix.py` now propagates the already-saved Bedroom review fields into each manifest variant:

- `grouped_date_stability`
- `promotion_time_drift_summary`
- `promotion_risk_flags`
- `gate_watch_reasons`
- `gate_reasons`

This makes the promotion decision readable directly from the matrix manifest.

## Fresh Verification

Fresh rerun:

- `PYTHONPATH=.:backend python3 backend/scripts/run_bedroom_root_cause_matrix.py --output-dir tmp/bedroom_root_cause_matrix_policy_20260311T225135Z --variant-name add_2025-12-05`

Fresh manifest:

- `tmp/bedroom_root_cause_matrix_policy_20260311T225135Z/manifest.json`

The new manifest now shows:

- top-level `status = gate_fail`
- variant `gate_pass = false`
- variant `gate_reasons = ["regime_instability_failed:bedroom:high_risk_unstable_date_slices:worst_date=2025-12-17"]`
- variant `gate_watch_reasons = ["grouped_date_stability_watch:bedroom:worst_date=2025-12-17:macro_range=0.218"]`
- non-null `grouped_date_stability`
- non-null `promotion_time_drift_summary`

## Scope Guardrail

This change is Bedroom-only. No broader all-room gating behavior was added in this follow-up.
