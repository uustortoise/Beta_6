# Beta6 Promotion Metadata Normalization Review

## Scope

- Fix the root cause of stale namespace metadata during room-wise promotion.
- Ensure promoted JSON artifacts are target-native on first promotion and self-heal on repeat promotion.

## Root cause

The promotion helper copied versioned artifacts verbatim from the source namespace into the target namespace.

That was correct for binary artifacts, but wrong for namespace-bound JSON artifacts such as:

- `*_decision_trace.json`
- `*_two_stage_meta.json`

Those payloads embed:

- `elder_id`
- absolute model artifact paths

So a promoted target namespace could load and replay correctly while still carrying source-namespace metadata internally.

## Permanent fix

Updated file:

- `backend/scripts/promote_room_versions_from_namespace.py`

Behavior change:

- copied JSON artifacts are now normalized to the target namespace during promotion
- duplicate JSON artifacts are compared namespace-agnostically instead of by raw file hash
- equivalent-but-stale target JSON artifacts are rewritten in place during repeat promotion
- version metadata copied into `*_versions.json` is also normalized before persistence

This keeps binary artifact handling unchanged while making metadata-bearing JSON promotion-safe.

## Regression coverage

Updated test file:

- `backend/tests/test_promote_room_versions_from_namespace.py`

Added coverage for:

- first promotion rewriting `elder_id` and artifact paths from source to target
- repeat promotion repairing a previously stale target JSON artifact instead of treating it as a conflict

Verification:

- `pytest backend/tests/test_promote_room_versions_from_namespace.py -q`
  - `2 passed`
- `pytest backend/tests/test_registry.py backend/tests/test_promote_room_versions_from_namespace.py -q`
  - `32 passed`

## Local repair verification

Re-ran promotion:

- `PYTHONPATH=backend python3 backend/scripts/promote_room_versions_from_namespace.py --backend-dir backend --source-elder-id HK0011_jessica_candidate_livingroom_fresh_20260311T023304Z --target-elder-id HK0011_jessica --room LivingRoom --version LivingRoom=52 --summary-out tmp/jessica_live_livingroom_v52_promotion_20260311.json`

Confirmed target artifacts no longer contain candidate references:

- `backend/models/HK0011_jessica/LivingRoom_decision_trace.json`
- `backend/models/HK0011_jessica/LivingRoom_two_stage_meta.json`
- `backend/models/HK0011_jessica/LivingRoom_v52_decision_trace.json`
- `backend/models/HK0011_jessica/LivingRoom_v52_two_stage_meta.json`

Verified state:

- embedded `elder_id=HK0011_jessica`
- no remaining `HK0011_jessica_candidate_livingroom_fresh_20260311T023304Z` references in those payloads

## Runtime sanity after repair

Artifact:

- `tmp/jessica_live_livingroom_v52_load_sanity_post_normalization_20260311.json`

Verified:

- all five rooms still load
- two-stage core models remain:
  - `Bathroom`
  - `Bedroom`
  - `LivingRoom`

## Conclusion

This closes the remaining promotion-path defect at the root.

Before:

- promotion was functionally correct but metadata-inconsistent

After:

- promotion is functionally correct
- promoted JSON metadata is target-native
- repeat promotion repairs stale prior copies instead of preserving drift
