# 2026-03-11 Beta6 Room Lineage Observability

## What Now Exists For All Rooms

All rooms now persist the lightweight lineage review surface in training metadata:

- `source_manifest`
- `source_fingerprint`
- `pre_sampling_label_counts_by_date`
- `source_lineage_review_summary`

`source_lineage_review_summary` is intentionally compact and review-oriented. It exposes:

- `source_manifest_count`
- `source_dates`
- normalized `source_class_share_by_date`
- `dominant_label_by_date`

This gives promotion review and forensic follow-up a readable per-date source view without requiring the full raw trace.

## What Remains Bedroom-Specific

The following remain Bedroom-only:

- `grouped_date_stability`
- `promotion_time_drift_summary`
- Bedroom instability blocking policy
- Bedroom grouped-date watch reasons

Non-Bedroom rooms still carry those Bedroom-specific fields only as `room_not_targeted` placeholders. No all-room blocking or watch policy was added in Task 7.

## Verification

Tests:

- `pytest backend/tests/test_training.py -q -k room_lineage_observability` -> `1 passed`
- `pytest backend/tests/test_training.py backend/tests/test_registry.py -q` -> `152 passed`

Direct readback:

- `PYTHONPATH=.:backend python3 - <<'PY' ...` on `LivingRoom` returned:
  - populated `source_lineage_review_summary`
  - `promotion_time_drift_summary = {"available": False, "reason": "room_not_targeted", ...}`

That confirms the observability surface is generalized while the policy boundary stays Bedroom-specific.

## Scope Boundary

Broader all-room gating remains intentionally out of scope here because the root-cause program only proved the instability mechanism and the matching policy change for `Bedroom`. Task 7 adds observability only, so future cross-room decisions can be evidence-based instead of inferred.
