# Beta 6 Active-Learning Triage Policy (Phase 3.2)

- Date: 2026-02-26
- Status: Active
- Module: `/Users/dicksonng/DT/Development/Beta_6/backend/ml/beta6/active_learning.py`
- Runner: `/Users/dicksonng/DT/Development/Beta_6/backend/scripts/run_active_learning_triage.py`
- Config: `/Users/dicksonng/DT/Development/Beta_6/backend/config/beta6_active_learning_policy.yaml`

## 1. Queue Composition

1. Uncertainty slice (`uncertainty_fraction`): low-confidence windows.
2. Disagreement slice (`disagreement_fraction`): model vs baseline conflicts.
3. Diversity slice (`diversity_fraction`): round-robin room/class coverage.

## 2. Guardrails

1. `max_share_per_room` cap prevents single-room dominance.
2. `max_share_per_class` cap prevents single-class dominance.
3. Queue builder returns fail when input has no valid candidates.

## 3. Outputs

1. Queue CSV with `selection_reason` and uncertainty metadata.
2. JSON report with mix counts, room/class distributions, and row totals.
