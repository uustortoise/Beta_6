# Beta6 Bedroom Two-Stage Runtime Root Fix

## Scope

- Fix the permanent root bug behind the Bedroom `v40` runtime mismatch.
- Ensure two-stage runtime activation follows the training-time gate decision instead of defaulting on missing metadata.
- Repair legacy two-stage metadata that was written without the runtime gate fields.

## Root cause

The critical defect was not just the Bedroom `0.95` stage-A threshold.

The stronger root cause was this runtime metadata gap:

- training correctly computed whether a room should use two-stage runtime
- the result was stored in memory as `two_stage_result["runtime_enabled"]`
- `_write_two_stage_core_artifacts(...)` did **not** persist that field into `*_two_stage_meta.json`
- registry loading treated a missing `runtime_enabled` as `True`
- runtime therefore reactivated two-stage bundles that training had already rejected

For the Bedroom `v40` candidate this exact mismatch was visible:

- `Bedroom_v40_decision_trace.json` said `gate_source = single_stage_fallback_no_regress`
- `Bedroom_v40_decision_trace.json` said `runtime_use_two_stage = false`
- but `Bedroom_two_stage_meta.json` omitted `runtime_enabled`
- runtime then loaded the Bedroom two-stage bundle anyway

That is a correctness bug in artifact persistence and load behavior.

## Code changes

Modified:

- `backend/ml/training.py`
- `backend/ml/legacy/registry.py`
- `backend/tests/test_training.py`
- `backend/tests/test_registry.py`

Implemented fix:

- persist `runtime_enabled` into two-stage metadata
- persist `runtime_gate_source`
- persist `selected_reliable`
- persist `fail_closed`
- persist `fail_closed_reason`
- add registry self-heal for historical two-stage metadata missing `runtime_enabled`
- infer the missing runtime state from the matching decision trace
- write the repaired fields back to the versioned/latest meta JSON
- prevent latest two-stage aliases from being activated when repaired metadata says runtime is disabled

## Verification

Regression tests:

- `pytest backend/tests/test_training.py -q -k two_stage_core_artifacts_persists_runtime_gate_metadata`
- `pytest backend/tests/test_registry.py -q -k runtime_disabled_two_stage_from_decision_trace`
- `pytest backend/tests/test_training.py backend/tests/test_registry.py -q`

Result:

- `145 passed`

## Real candidate repair

Validated on the real Bedroom candidate namespace:

- `backend/models/HK0011_jessica_candidate_bedroom_sepfix_20260311T041155Z`

Load/repair artifact:

- `tmp/bedroom_two_stage_runtime_root_fix_load_check_20260311.json`

Verified state after repair:

- `Bedroom_v40_two_stage_meta.json.runtime_enabled = false`
- `Bedroom_v40_two_stage_meta.json.runtime_gate_source = single_stage_fallback_no_regress`
- latest `Bedroom_two_stage_meta.json` alias removed
- Bedroom two-stage runtime bundle no longer loads

This confirms the runtime now matches the training-time gate decision.

## Post-fix replay consequence

Bedroom-only Dec 17 runtime validation after the repair:

- `tmp/bedroom_runtime_rootfix_validation_20260311T174600Z/summary.json`
- `tmp/bedroom_runtime_rootfix_validation_20260311T174600Z/Bedroom_merged.parquet`

Post-fix Bedroom result on Dec 17:

- final macro-F1: `0.33052062399815013`
- final accuracy: `0.6024822695035461`
- raw top-1 macro-F1: `0.4201960949332568`
- low-confidence rate: `0.03368794326241135`

Pre-fix buggy runtime result from the earlier candidate replay:

- final macro-F1: `0.3931771489209137`
- final accuracy: `0.6995271867612293`

Interpretation:

- the earlier `0.3931771489209137` Bedroom candidate replay was not a valid runtime result
- it depended on a two-stage bundle that training had already marked as single-stage fallback
- after the root fix, the Bedroom candidate behaves as the training decision trace intended

## Conclusion

This fixes the permanent correctness bug at the source:

- future two-stage artifacts now persist runtime gate state explicitly
- historical artifacts missing that state are repaired from decision traces
- runtime no longer silently turns rejected two-stage candidates back on

This does **not** make Bedroom `v40` promotion-ready.

Instead, it sharpens the next conclusion:

- the prior Bedroom `v40` replay evidence was partially invalidated by the runtime metadata bug
- the Bedroom threshold-sweep artifacts remain useful as mechanism probes
- any next Bedroom promotion or retrain decision must now be based on post-fix runtime behavior, not the old buggy `v40` replay
