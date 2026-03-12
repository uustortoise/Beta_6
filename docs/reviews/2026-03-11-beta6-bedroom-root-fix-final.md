# 2026-03-11 Beta6 Bedroom Root-Fix Final

## Success-Criteria Check

The design / plan asked for four things:

1. prove whether the harmful Bedroom dates were bad data or valid alternate regime
2. implement the matching root fix without widening scope prematurely
3. validate Bedroom behavior with fresh evidence
4. persist enough lineage metadata that future drift is auditable

Final status:

- `1` achieved
- `2` achieved in the intended containment / control sense
- `3` achieved for promotion safety, not for full Bedroom model robustness
- `4` achieved

## Final Root-Cause Statement

### Earliest real cause

The earliest validated harmful cause is the addition of `2025-12-05` into the Bedroom source pack.

That date is not obviously corrupted. The segment audit shows dense, coherent, unflagged Bedroom blocks, especially:

- `12:00-13:00`
- `20:00-21:00`
- `13:00-14:00`
- `10:00-11:00`
- `11:00-12:00`

Those blocks are dominated by long `bedroom_normal_use` runs where the good references are mostly `unoccupied`.

So the real root cause is:

> a valid alternate Bedroom occupancy regime entered the source lineage on `2025-12-05`, and the existing Bedroom training / promotion pipeline could not absorb that regime safely.

### What was disproven

The program disproved these hypotheses:

- that the failure was primarily caused by later Bedroom sampling / threshold changes
  - the first strong divergence was already upstream at source-lineage selection
- that the harmful day was obviously bad / inconsistent data
  - the harmful slices were dense, coherent, and not flagged for missingness or sparsity
- that a cumulative interaction through later days was required before harm appeared
  - `add_2025-12-05` alone was already harmful versus anchor
- that aggregate replay gain or one acceptable replay was enough evidence of safety
  - the candidate can still post a usable Dec 17 replay while remaining unstable across date slices
- that broader all-room gating should be changed immediately
  - the evidence proved the mechanism and the policy change for Bedroom, not for all rooms

### Why Bedroom is structurally hard

Bedroom is hard here because a real but alternate regime can materially shift date-level label priors and confusion structure without looking like broken data.

In practice that means:

- pooled holdout metrics can hide slice instability
- one Dec 17 replay can look acceptable while the candidate is still unsafe
- the hard confusion family is structural:
  - `unoccupied -> bedroom_normal_use`
  - `bedroom_normal_use -> unoccupied`
  - secondarily `sleep -> unoccupied`

This is not a simple “bad workbook” problem. It is a regime-coverage and robustness problem concentrated in Bedroom.

## What Was Fixed

### Promotion / control fixes

Production is now protected by explicit Bedroom-specific controls:

- grouped-by-date Bedroom stability is persisted in saved metadata
- promotion-time Bedroom drift evidence is persisted in saved metadata
- the matrix manifest now exposes those saved risk fields directly
- Bedroom promotion is now blocked when saved metadata shows:
  - `promotion_time_drift_summary.risk_level == "high"`
  - grouped-date instability across date slices
  - `promotion_risk_flags` includes `unstable_date_slices:bedroom`

Fresh verification artifact:

- `tmp/bedroom_root_cause_matrix_policy_20260311T225135Z/manifest.json`

Fresh blocking result:

- `status = gate_fail`
- `gate_reasons = ["regime_instability_failed:bedroom:high_risk_unstable_date_slices:worst_date=2025-12-17"]`

### Observability fixes

The program also removed the provenance ambiguity that made the original forensic question hard to answer:

- exact source manifests now persist
- stable source fingerprints now persist
- per-date / per-label pre-sampling counts now persist
- all rooms now get a compact `source_lineage_review_summary`

That means future drift review no longer requires guessing which source pack trained a saved version.

### Bedroom-specific policy now in force

Bedroom remains the only room with the new instability blocking policy.

What now protects production:

- a valid alternate Bedroom regime is no longer silently promotable just because pooled metrics or one replay look decent
- the unsafe candidate is surfaced and rejected before promotion

## What Remains Open

### Why Bedroom is not fully “solved” yet

The program solved the **promotion safety** problem, not the full **model robustness** problem.

Current state:

- the pipeline can now identify and block unsafe Bedroom candidates
- it does **not** yet demonstrate a Bedroom training recipe that safely absorbs both the old anchor regime and the valid `2025-12-05` alternate regime at the same time

So the root fix is a successful control-plane / observability hardening, not a finished model-quality fix.

### Future model-improvement work still needed

Future Bedroom work would need to find a training recipe that can:

- retain acceptable holdout performance
- remain stable across date slices
- avoid the critical Bedroom confusion families
- handle the valid alternate `2025-12-05` regime without needing to exclude it

That is a future model-improvement problem, not something this root-fix program claimed to finish.

## Bedroom-Specific Boundary

What remains intentionally Bedroom-specific:

- grouped-date stability evaluation
- promotion-time drift blocking policy
- grouped-date watch reasons
- regime-instability promotion block

What was generalized safely:

- lineage visibility
- source fingerprints
- per-date class summaries
- compact room review summaries

No new all-room blocking behavior was added.

## Verification

Fresh commands run for Task 8:

- `pytest backend/tests/test_run_bedroom_root_cause_matrix.py backend/tests/test_bedroom_day_segment_audit.py backend/tests/test_training.py backend/tests/test_registry.py -q`
  - result: `161 passed`
- `python3 - <<'PY' ...`
  - read back:
    - tighten matrix manifest
    - harmful-day audit summary
    - Bedroom policy manifest
  - confirmed:
    - anchor Dec 17 macro-F1: `0.7436265817791433`
    - `add_2025-12-05` Dec 17 macro-F1: `0.7231961200695896`
    - `cumulative_through_2025-12-05` Dec 17 macro-F1: `0.6426238918375194`
    - Bedroom policy manifest `status=gate_fail`
    - Bedroom policy manifest `grouped_date_stability.unstable_across_dates=true`
    - Bedroom policy manifest `promotion_time_drift_summary.risk_level=high`
- `python3 - <<'PY' ...`
  - read back the 60-minute audit summary directly
  - confirmed standout blocks:
    - `12:00-13:00`
    - `20:00-21:00`
    - `13:00-14:00`
    - all with empty `flags`
- `PYTHONPATH=.:backend python3 - <<'PY' ...`
  - direct `LivingRoom` readback confirmed all-room lineage observability exists while `promotion_time_drift_summary` remains `room_not_targeted`

## Residual Risks

### PostgreSQL / historical-corrections caveat

Local PostgreSQL auth remained unavailable throughout the reruns, so the full program was verified in the no-corrections replay environment.

### Remaining ambiguity

There is no longer material ambiguity about the root-cause branch:

- Branch A was disproven by the bounded forensic evidence
- Branch B was supported and implemented operationally

The remaining uncertainty is narrower:

- exactly which future Bedroom training recipe can absorb both regimes safely is still unknown
- that is a future optimization / model-improvement problem, not an unresolved root-cause classification problem

## Final Decision

Decision-grade final statement:

> The Bedroom failure was caused by a real alternate regime entering source lineage on `2025-12-05`, not by obviously bad data. The implemented root fix is to make that instability explicit in saved metadata and block high-risk unstable Bedroom candidates from promotion, while generalizing only lightweight observability to other rooms. Production is better protected now, but Bedroom still needs future model-improvement work to learn both regimes robustly.
