# Beta6 ML Snapshot Shadow Readiness Validation (2026-02-27)

## Scope
Executed the recommended shadow-readiness pass for ML Health Snapshot:
1. Live UI presence checks in `Today`, `Weekly Report`, and `Admin & Audit` sections.
2. Endpoint reconciliation across 3 residents (`HK0011_jessica`, `elder1`, `HK0099_dummy`).
3. Degradation behavior test for partial snapshot payloads.

## Environment
- Streamlit launched from repo root:
  - `PYTHONPATH=/Users/dicksonng/DT/Development/Beta_6 streamlit run backend/export_dashboard.py --server.port 8511 --server.headless true`
- Browser automation via Playwright MCP against `http://localhost:8511`.
- Reconciliation artifact:
  - `/tmp/beta6_ml_snapshot_reconcile_20260227.json`

## UI Validation Result
Status: **PASS (presence and graceful rendering)**

Verified sections:
1. `📋 Weekly Report: ML Health Snapshot` in `📊 Model Insights` tab.
2. `🩺 Today: ML Health Snapshot` in `🏠 Household Overview` tab.
3. `🛡️ Admin & Audit: ML Health Snapshot` in `⚙️ AI Configuration` tab.

Observed behavior:
1. All three sections render without page crash.
2. No-data state renders as `Safety Status: Not Available` with explanatory reason text.
3. Panel degrades gracefully (info state shown; no exception UI).

## Endpoint Reconciliation (3 Residents)

| Resident | ml-snapshot | promotion-gates | walk-forward | Notes |
|---|---|---|---|---|
| HK0011_jessica | `200 / not_available` | `200 / healthy` | `503 / error` | Snapshot has no room payload; promotion has successful recent runs. |
| elder1 | `200 / not_available` | `404 / error` | `503 / error` | Consistent no-data behavior across snapshot/promotion. |
| HK0099_dummy | `200 / not_available` | `404 / error` | `503 / error` | Consistent no-data behavior across snapshot/promotion. |

### Mismatch / Blockers Identified
1. **Mismatch:** `HK0011_jessica` shows `promotion-gates=healthy` while `ml-snapshot=not_available`.
   - Cause: snapshot currently depends on `walk_forward_gate.room_reports`; if missing, panel has no room-level metrics even when run-level promotion summary exists.
2. **Blocker:** `walk-forward` endpoint returns `503` due Beta6 leakage guard:
   - `preprocess_with_resampling(is_training=True)` is disallowed when `ENABLE_TRAIN_SPLIT_SCALING` is enabled.
   - This blocks full reconciliation of drift and fold-level metrics through current health endpoint path.

## Degradation Test Update
Added/ran a partial-payload degradation test:
- `backend/tests/test_health_server.py::test_build_ml_snapshot_report_partial_room_payload_degrades_gracefully`

Test suite result:
- `python3 -m pytest -q backend/tests/test_health_server.py`
- **8 passed**

## Readiness Assessment
Overall: **PARTIAL PASS**

Pass:
1. UI sections are implemented and render in all required placements.
2. No-data and partial-data degradation behaviors are stable.

Not yet pass:
1. Full value reconciliation against `walk-forward` endpoint (blocked by leakage-guard incompatibility in endpoint path).
2. Snapshot/promotion alignment for residents where promotion summary exists but room-level walk-forward payload is absent.

## Recommended Immediate Fixes
1. Update `build_walk_forward_report` path in `backend/health_server.py` to Beta6-compliant preprocessing (no leakage-guard violation).
2. Add fallback mapping in `build_ml_snapshot_report` when `room_reports` are absent but promotion/global gate data exists, so status is not incorrectly `not_available`.
3. Re-run this validation doc checklist after fixes and require `PASS` before merge gate.

---

## Fix Execution Update (2026-02-27, same day)

Implemented:
1. `backend/health_server.py`
   - `build_walk_forward_report(...)` now runs health preprocessing with `is_training=False` and derives labels from `activity` + loaded encoder classes when `activity_encoded` is absent.
   - `build_ml_snapshot_report(...)` now falls back to an aggregate `all_rooms` payload (watch/action-needed) when room-level walk-forward reports are missing but promotion/global signals exist.
2. `backend/tests/test_health_server.py`
   - Added regression tests for fallback mapping and walk-forward label derivation in inference-preprocessing mode.

Validation after fixes:
1. `python3 -m pytest -q backend/tests/test_health_server.py` -> **10 passed**
2. `build_ml_snapshot_report('HK0011_jessica')` now returns:
   - `status.overall = watch`
   - `status.reason_code = missing_room_reports`
   - `rooms = ['all_rooms']`
3. `build_walk_forward_report(..., room_param='bathroom', lookback_days=7)` now returns:
   - HTTP `200`
   - overall status `healthy`
   - room status `ok`

Residual known issue (unchanged by this fix):
1. One model artifact (`Entrance`) still fails deserialization in local env due Keras incompatibility; this can degrade all-room health checks when querying all rooms at once.

---

## Runtime Integration Update (2026-02-28)

Implemented:
1. `backend/health_server.py`
   - `build_ml_snapshot_report(...)` now reads runtime fallback state from `RegistryV2` (`fallback_state.json`) and applies runtime-first precedence over metadata fallback summaries.
   - Candidate rooms for fallback-state reads are assembled from room reports, metadata fallback activations, optional room filter, and discovered runtime registry room directories.
   - Aggregate `all_rooms` synthetic snapshot now escalates to `action_needed` with `reason_code=fallback_active` when any runtime fallback is active.
2. `backend/tests/test_health_server.py`
   - Added regression tests for runtime fallback precedence:
     - runtime `active=true` overrides metadata and forces `action_needed`.
     - runtime `active=false` clears stale metadata fallback activation.

Validation after runtime integration:
1. `python3 -m pytest -q backend/tests/test_health_server.py` -> **13 passed**
2. `python3 -m py_compile backend/health_server.py backend/tests/test_health_server.py` -> **pass**
3. Streamlit UI smoke (`http://localhost:8511`) via Playwright MCP:
   - `📋 Weekly Report: ML Health Snapshot` present in `📊 Model Insights`.
   - `🩺 Today: ML Health Snapshot` present in `🏠 Household Overview`.
   - `🛡️ Admin & Audit: ML Health Snapshot` present in `⚙️ AI Configuration`.
   - No browser console errors observed during panel render checks.

Operational note:
1. Direct standalone CLI invocation of `build_ml_snapshot_report` in this sandbox fails to connect to local PostgreSQL (`Operation not permitted`). Streamlit smoke and unit tests were used as the authoritative validation path for this pass.
