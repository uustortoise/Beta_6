# 2026-03-11 Beta6 Bedroom Regime-Shift Hardening

## Scope

Task 6B applies only to `Bedroom`. The goal is not to prove an aggregate replay improvement. The goal is to make a valid alternate regime visibly unstable at promotion time so a candidate with acceptable pooled metrics does not look safer than it is.

## Code And Test Changes

- Added grouped-by-date Bedroom stability summaries in `backend/ml/training.py`.
- Added promotion-time drift summaries and persisted them into saved version metadata in `backend/ml/legacy/registry.py`.
- Added focused tests in `backend/tests/test_training.py` and `backend/tests/test_registry.py`.

## Verification

- `pytest backend/tests/test_training.py -q -k grouped_date_stability` -> `2 passed`
- `pytest backend/tests/test_registry.py -q -k grouped_date_stability` -> `1 passed`
- `pytest backend/tests/test_training.py backend/tests/test_registry.py -q` -> `150 passed`
- `PYTHONPATH=.:backend python3 backend/scripts/run_bedroom_root_cause_matrix.py --output-dir tmp/bedroom_root_cause_matrix_task6b_20260312T002900Z --variant-name add_2025-12-05` -> completed and wrote a fresh Bedroom candidate manifest and replay artifacts

## Candidate Re-Evaluation

Candidate:

- `HK0011_jessica_candidate_bedroom_rootmatrix_bedroom_root_cause_matrix_task6b_20260312T002900Z_01_add_2025-12-05`

Source pack:

- `2025-12-05`
- `2025-12-10`
- `2025-12-17`

Saved machine-readable summary:

- `tmp/bedroom_root_cause_matrix_task6b_20260312T002900Z/manifest.json`

Key holdout and replay metrics from the saved manifest:

- holdout macro-F1: `0.66987038118556`
- holdout `bedroom_normal_use` recall: `0.4990791896869245`
- Dec 17 replay final macro-F1: `0.7231961200695896`
- Dec 17 critical error families:
  - `unoccupied -> bedroom_normal_use`: `826`
  - `sleep -> unoccupied`: `292`
  - `bedroom_normal_use -> unoccupied`: `246`

The Dec 17 replay is not catastrophic, but that is not the relevant success condition for Task 6B.

## New Bedroom Stability Evidence

Grouped-by-date stability is now present in saved Bedroom metadata:

- candidate decision trace:
  - `backend/models/HK0011_jessica_candidate_bedroom_rootmatrix_bedroom_root_cause_matrix_task6b_20260312T002900Z_01_add_2025-12-05/Bedroom_v39_decision_trace.json`
- candidate version history:
  - `backend/models/HK0011_jessica_candidate_bedroom_rootmatrix_bedroom_root_cause_matrix_task6b_20260312T002900Z_01_add_2025-12-05/Bedroom_versions.json`

The new grouped-by-date evidence explicitly surfaces instability:

- `unstable_across_dates: true`
- `worst_date: 2025-12-17`
- `best_date: 2025-12-18`
- `macro_f1_range: 0.2184746569435445`
- `bedroom_normal_use` recall pooled: `0.7546598322460392`
- `bedroom_normal_use` recall range: `0.8427375318529304`
- unstable dates:
  - `2025-12-06`
  - `2025-12-11`
  - `2025-12-17`
  - `2025-12-18`
- risk reasons:
  - `macro_f1_range:0.218>0.150`
  - `bedroom_normal_use_recall_range:0.843>0.250`

The candidate also now emits explicit saved risk markers:

- `promotion_risk_flags` includes `unstable_date_slices:bedroom`
- `gate_watch_reasons` includes `grouped_date_stability_watch:bedroom:worst_date=2025-12-17:macro_range=0.218`

## Promotion-Time Drift Summary

Promotion-time drift is now saved alongside the grouped-by-date summary. The persisted summary marks the candidate as:

- `risk_level: high`
- `unstable_across_dates: true`
- `worst_date: 2025-12-17`

The drift summary also preserves the pre-sampling label-share shift by date, which makes the regime contrast visible at review time:

- `2025-12-05` source share:
  - `bedroom_normal_use: 0.3179398148148148`
  - `sleep: 0.29594907407407406`
  - `unoccupied: 0.3861111111111111`
- `2025-12-17` source share:
  - `bedroom_normal_use: 0.0699235934244038`
  - `sleep: 0.41305857837462373`
  - `unoccupied: 0.515628617735587`

This is the Branch B signal the plan asked for: the valid alternate regime is visible in the source composition, and the candidate is explicitly marked as unstable across date slices instead of looking acceptable because one pooled or one replay metric is decent.

## Task 6B Conclusion

Task 6B is complete.

The Bedroom candidate that includes the valid alternate regime still trains and can produce a non-collapsed Dec 17 replay, but the new saved artifacts now make the instability explicit:

- grouped-by-date stability marks the candidate unstable
- promotion-time drift marks the candidate high risk
- the worst saved date slice is the production-relevant `2025-12-17` replay date

That is sufficient evidence for the next decision gate. No broader all-room behavior changes were made in this task.

## Residual Caveat

Local PostgreSQL auth still failed during the rerun, so historical corrections were unavailable. The Task 6B conclusion is therefore verified within the no-corrections replay environment.
