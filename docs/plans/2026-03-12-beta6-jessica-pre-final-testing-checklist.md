# Beta6 Jessica Pre-Final Testing Checklist

## Scope

This checklist is for deep pre-final validation on branch `codex/jessica-pre-final-testing` before any final Jessica integration or release decision.

Target candidate namespace:

- `HK0011_jessica_candidate_supportfix_20260310T2312Z`

Expected room versions in the candidate:

- `Bathroom_v35`
- `Bedroom_v38`
- `Entrance_v26`
- `Kitchen_v27`
- `LivingRoom_v40`

Bedroom caveat:

- `Bedroom_v38` is still the operational anchor, but the later Bedroom root-fix work proved Bedroom is structurally unstable across valid date regimes.
- Pre-final testing must therefore treat Bedroom as `pass only if runtime behavior is acceptable and no new regression appears`; this checklist does not reinterpret Bedroom as fully solved.

## Success Criteria

The branch is ready for final decision only if all of the following are true:

1. The candidate namespace loads cleanly in the current branch with the expected room versions and two-stage wiring.
2. The Dec 17 corrected-workbook replay is internally consistent with the validated benchmark frontier for the candidate.
3. `Bathroom`, `Kitchen`, and `LivingRoom` show no new regression relative to the accepted Jessica branch evidence.
4. `Bedroom` remains operationally acceptable and does not expose a new runtime failure mode beyond the already-documented structural regime-shift risk.
5. The review artifacts are complete enough that each room can be signed off `pass`, `conditional`, or `block`.

## Preflight

Complete these before any room-level sign-off:

- [ ] Confirm branch and commit:
  - branch should be `codex/jessica-pre-final-testing`
  - record `git rev-parse --short HEAD`
- [ ] Confirm candidate namespace exists:
  - `backend/models/HK0011_jessica_candidate_supportfix_20260310T2312Z/`
- [ ] Run focused regression suite:
  - `pytest backend/tests/test_run_bedroom_root_cause_matrix.py backend/tests/test_bedroom_day_segment_audit.py backend/tests/test_training.py backend/tests/test_registry.py -q`
- [ ] Run syntax sanity:
  - `python3 -m py_compile backend/ml/pipeline.py backend/ml/training.py backend/ml/legacy/registry.py backend/scripts/run_bedroom_root_cause_matrix.py backend/scripts/bedroom_day_segment_audit.py`
- [ ] Run clean candidate load check:
  - verify `load_models_for_elder()` loads all expected rooms
  - verify `platform.two_stage_core_models` contains `Bathroom`, `Bedroom`, and `LivingRoom`
- [ ] Replay corrected `HK0011_jessica_train_17dec2025.xlsx` through the candidate namespace and save the fresh summary path
- [ ] Record the exact artifact paths reviewed for the sign-off round

## Shared Review Fields

For every room below, capture:

- expected version
- raw top-1 summary if available
- final exported summary
- low-confidence or rewrite behavior if relevant
- known historical failure family
- current verdict: `pass`, `conditional`, or `block`
- notes pointing to the reviewed artifact paths

## Bathroom

Expected state:

- version `v35`
- two-stage runtime path intact
- no alias drift or stage-model discovery regression

Mandatory checks:

- [ ] confirm `Bathroom_versions.json` reports `current_version=35`
- [ ] confirm fresh-load runtime resolves Bathroom through two-stage wiring, not a stale alias or discovered stage file
- [ ] inspect Dec 17 replay final macro-F1 against the accepted `v35` frontier
- [ ] inspect low-confidence rate versus the validated mixed benchmark
- [ ] confirm no replay evidence suggests the earlier branch-integration bug has resurfaced

Block if:

- Bathroom loads through the wrong path
- two-stage parity is broken
- final metrics materially fall below the accepted `v35` behavior without a clear artifact explanation

## Bedroom

Expected state:

- version `v38`
- still operationally acceptable for the Jessica candidate pack
- known structural issue remains: valid alternate date regimes can destabilize training and promotion

Mandatory checks:

- [ ] confirm `Bedroom_versions.json` reports `current_version=38`
- [ ] inspect Dec 17 replay final macro-F1 against the accepted support-fix candidate benchmark
- [ ] inspect low-confidence / rewrite behavior for any sign of a reintroduced confidence-runtime mismatch
- [ ] inspect final confusion shape for:
  - `unoccupied -> bedroom_normal_use`
  - `sleep -> unoccupied`
  - `bedroom_normal_use -> unoccupied`
- [ ] verify no pre-final branch change invalidates the later Bedroom control-plane safety work documented in:
  - `docs/reviews/2026-03-11-beta6-bedroom-root-fix-final.md`
  - `docs/reviews/2026-03-11-beta6-bedroom-instability-promotion-policy.md`

Decision rule:

- `pass` only if Bedroom behavior matches the accepted operational anchor with no new failure mode
- `conditional` if Bedroom is acceptable for this candidate/runtime but remains limited by the already-known regime-shift issue
- `block` if any new runtime collapse or confidence pathology appears

## Kitchen

Expected state:

- version `v27`
- unchanged from the accepted Jessica baseline

Mandatory checks:

- [ ] confirm `Kitchen_versions.json` reports `current_version=27`
- [ ] confirm Dec 17 replay remains effectively unchanged versus the accepted mixed candidate evidence
- [ ] inspect final confusion summary for any unexpected drift caused by cross-room integration

Block if:

- Kitchen version or output differs materially without an intentional reason
- cross-room integration introduced a regression that did not exist in the accepted branch evidence

## LivingRoom

Expected state:

- version `v40`
- improved over the older `v30` fallback on the Dec 17 corrected replay
- no direct-retrain collapse symptoms in the deployed candidate path

Mandatory checks:

- [ ] confirm `LivingRoom_versions.json` reports `current_version=40`
- [ ] inspect Dec 17 replay final macro-F1 against the accepted `v40` support-fix frontier
- [ ] inspect low-confidence rate; it should remain near zero in the accepted candidate path
- [ ] inspect whether `livingroom_active` recall remains stable and does not show the earlier direct-retrain collapse shape
- [ ] confirm no candidate-load issue prevents `LivingRoom` two-stage / activity-confidence runtime artifacts from loading cleanly

Block if:

- replay falls back toward the earlier `v30` or direct-retrain collapse shape
- low-confidence unexpectedly rises
- room-load/runtime artifacts are inconsistent with the accepted candidate evidence

## Final Synthesis

Do not decide from the overall macro-F1 alone. Complete the room-level sign-off table:

| Room | Expected version | Verdict | Blocking issue | Artifact notes |
| --- | --- | --- | --- | --- |
| Bathroom | v35 |  |  |  |
| Bedroom | v38 |  |  |  |
| Kitchen | v27 |  |  |  |
| LivingRoom | v40 |  |  |  |

Final release decision:

- [ ] `GO` only if all rooms are `pass` or `conditional` with understood risk and no `block`
- [ ] `NO-GO` if any room is `block`
- [ ] record the exact reason for each `conditional`

## Known Context To Preserve During Testing

- The accepted promotion-grade candidate in this branch is `HK0011_jessica_candidate_supportfix_20260310T2312Z`.
- `Bedroom` root-cause work concluded the harmful added day was `2025-12-05`, and the issue is a valid alternate regime rather than obvious bad data.
- Bedroom instability is now explicitly surfaced and blocked in the training/promotion path, but that does not by itself make mixed-regime Bedroom retrains safe.
- All-room lightweight lineage observability now exists; only Bedroom has the added instability gate.
- Historical-corrections access through local PostgreSQL remained unavailable during the Bedroom root-fix program, so that caveat should be carried into any final risk call.
