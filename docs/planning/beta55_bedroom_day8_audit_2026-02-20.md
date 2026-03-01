# Beta 5.5 Bedroom Day-8 Audit (2026-02-20)

## Scope

- File audited: `/Users/dicksonng/DT/Development/New training files/corrected_clones/HK0011_jessica_train_8dec2025.xlsx`
- Target room: `Bedroom`
- Objective: identify high-confidence label corrections that can safely improve Day-8 hard-gate failures.

## Findings

1. Bedroom Day-8 hard-gate failures (baseline) are:
   - `occupied_f1_lt_0.550`
   - `occupied_recall_lt_0.500`
   - `recall_sleep_lt_0.400` (and for some seeds `fragmentation_score_lt_0.450`)

2. Sleep FN forensic payload from baseline seed 11 shows two GT sleep windows missed by model:
   - `2025-12-08 00:00:00` to `2025-12-08 00:25:40`
   - `2025-12-08 22:13:37` to `2025-12-08 23:59:51`

3. Contradiction scans run:
   - `unoccupied -> likely occupied` episode scan produced many low-quality candidates (not safe for automatic correction).
   - `occupied -> likely unoccupied` cross-room scan produced no high-confidence episodes under strict criteria.

## Correction Decision

- No high-confidence automatic Bedroom Day-8 label correction was applied.
- Empty correction file used for controlled rerun:
  - `/Users/dicksonng/DT/Development/Beta_5.5/docs/planning/bedroom_day8_label_corrections_2026-02-20.csv`
- Rerun result with empty correction file remained unchanged vs anchor:
  - `24/30` eligible, `45/60` full.

## Artifacts

- Candidate CSV from broad contradiction scan:
  - `/Users/dicksonng/DT/Development/Beta_5.5/docs/planning/bedroom_day8_label_audit_candidates_2026-02-20.csv`
- Empty correction CSV used for rerun:
  - `/Users/dicksonng/DT/Development/Beta_5.5/docs/planning/bedroom_day8_label_corrections_2026-02-20.csv`
- Rerun artifacts:
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/bedroom_day8_audit_rerun_20260220/ws6_rolling.json`
  - `/Users/dicksonng/DT/Development/Beta_5.5/backend/tmp/ws6_beta55_upgrade/bedroom_day8_audit_rerun_20260220/ws6_signoff.json`

