# Beta_6 Ops UX ML Health Snapshot: Implementation Checklist (File-by-File)

## Purpose
Turn the approved planning updates into an execution-ready checklist with:
1. sequencing
2. file-level changes
3. effort estimates
4. testing steps
5. go/no-go completion criteria

This checklist incorporates the team feedback:
1. prevent alert fatigue on `Today`
2. show active-system shadow banner
3. keep API reasons plain-language
4. make threshold provenance actionable in Admin

## Scope (This Checklist)
In scope:
1. `ML Health Snapshot` backend read endpoint and UI panel wiring
2. `Today` alert-routing rules (`Today` vs `Review Queue`)
3. shadow-mode active-system banner
4. threshold provenance links/actions in Admin panel

Out of scope:
1. model training / gate logic changes
2. threshold policy tuning
3. CRF/HMM/model architecture changes

## Recommended Delivery Order (Critical)
Implement in this order to reduce Ops confusion during pilot:
1. `Today` alert-routing rule (prevent alert fatigue)
2. Global active-system banner (prevent shadow-mode ambiguity)
3. `ml-snapshot` backend endpoint (plain-language API contract)
4. `ML Health Snapshot` panel in `Weekly Report` and `Admin & Audit`
5. Threshold provenance links/actions in Admin
6. Optional compact `Today` summary row for ML health

Reason:
1. Ops guardrails should land before adding more ML observability UI.

## Rough Effort Estimate (Engineering Only)
Assumes one engineer familiar with `export_dashboard.py` and backend health endpoints.

1. Backend endpoint (`health_server.py`): `0.5 - 1.0 day`
2. UI fetch/render (Weekly/Admin panel): `1.0 - 1.5 days`
3. `Today` alert-routing rule + queue routing: `0.5 - 1.0 day`
4. Global shadow banner across pages: `0.25 - 0.5 day`
5. Provenance links/actions in Admin panel: `0.5 day`
6. QA fixes / polish: `0.5 - 1.0 day`

Estimated total:
1. `3.25 - 5.5 engineering days`
2. QA validation: `0.5 - 1.0 day` (parallelizable)

## File-by-File Checklist

### 1) `backend/health_server.py`
Purpose:
1. Provide a read-only aggregate endpoint for Ops/Weekly/Admin UI consumption.

Changes:
1. Add `build_ml_snapshot_report(elder_id, room=None, lookback_runs=20, include_raw=False)`
2. Reuse existing builders where possible:
   - `build_walk_forward_report(...)`
   - `build_promotion_gate_report(...)`
3. Merge fields into one response shape:
   - candidate/champion macro-F1
   - transition F1
   - stability accuracy
   - effective drift threshold
   - threshold provenance metadata (when available)
4. Add route:
   - `GET /health/model/ml-snapshot?elder_id=<id>`
5. Enforce plain-language `status.reason`
6. Add stable `status.reason_code`
7. Gracefully handle missing data (`not_available`, `null` metrics, no 500)
8. Optional:
   - `include_raw=true` attaches technical payload snippets for debugging

Acceptance checks:
1. Endpoint returns 200 for residents with and without training history
2. JSON matches contract in panel spec doc
3. `status.reason` has no ML jargon by default
4. `status.reason_code` is stable for same condition

Effort:
1. `0.5 - 1.0 day`

### 2) `backend/export_dashboard.py`
Purpose:
1. Render the Ops-facing `ML Health Snapshot` panel and pilot banners.
2. Enforce `Today` alert routing to avoid alert fatigue.

Changes: data fetch layer
1. Add `fetch_ml_health_snapshot(elder_id, room=None, lookback_runs=20, include_raw=False)`
2. Use local backend health endpoint or direct helper call pattern consistent with current dashboard code
3. Cache safely with short TTL
4. Error handling:
   - fail open (panel shows `Not Available`)
   - do not block page render

Changes: UI components
1. Add reusable banner renderer:
   - `Beta 5.5 currently active (Beta 6 in Shadow)` when rollout state says shadow
2. Add reusable `ML Health Snapshot` panel renderer:
   - Ops summary row
   - technical expander
   - status badge (`Healthy/Watch/Action Needed/Not Available`)
3. Weekly Report:
   - render full panel + banner
4. Admin & Audit:
   - render full panel + provenance rows (`source`, `resolved_key`, `source_file`, `owner`)
   - render direct link/action using `editor_target` when present
5. Today page (optional compact summary row):
   - render compact ML health summary only after alert-routing changes land

Changes: alert routing (`Today` vs `Review Queue`)
1. Update `Today -> Needs Attention` item generation/filtering:
   - include only clinical/care-actionable anomalies
   - exclude routine ML low-confidence events unless care-blocking
2. Ensure excluded uncertainty items appear in `Review Queue` with plain-language reason
3. Add explicit rule comments near filter logic (short, non-ML wording)

Acceptance checks:
1. Weekly/Admin pages render panel with partial/missing data safely
2. Shadow banner appears consistently on Today/Weekly/shadow views
3. `Today` list remains clinical-actionable under expected abstain/unknown load
4. Admin provenance rows provide one-action navigation when `editor_target` exists

Effort:
1. `1.75 - 2.75 days` (including alert routing, banner, and panel rendering)

### 3) `backend/run_daily_analysis.py` (Optional / Nice-to-Have for MVP)
Purpose:
1. Improve threshold provenance fidelity for UI without inferring too much in `health_server.py`.

Changes (optional):
1. Persist resolved threshold provenance into training metadata:
   - source (`default/env/room/policy`)
   - resolved key
   - room override indicator
2. Persist reason code + plain-language reason for gate outcomes (if useful for endpoint reuse)

Acceptance checks:
1. Metadata includes provenance for sampled rooms and can be read by dashboard/health endpoint

Effort:
1. `0.25 - 0.75 day`

### 4) `backend/config/...` and Admin config editor mapping (UI integration only)
Purpose:
1. Make provenance actionable.

Changes:
1. Define/maintain a mapping from `editor_target` identifiers to Admin UI sections
2. For known threshold keys:
   - `WF_DRIFT_THRESHOLD`
   - `WF_MIN_TRANSITION_F1`
   - `WF_MIN_STABILITY_ACCURACY`
   map to stable Admin anchors
3. If room-profile YAML editor exists/is added, map room override targets too

Acceptance checks:
1. Clicking provenance action focuses the correct Admin panel section for common keys
2. Missing targets degrade gracefully (show file path only)

Effort:
1. `0.25 - 0.5 day` (mostly in `export_dashboard.py`)

## Sequenced Task Breakdown (Suggested)

### Phase A: Ops Safety Guardrails First
1. Implement `Today` alert-routing filters in `backend/export_dashboard.py`
2. Implement global shadow-mode banner renderer and wire to core pages
3. Validate with sample pilot data / mocked states

Exit criteria:
1. Ops-facing pages remain actionable and unambiguous before panel launch

### Phase B: Backend Contract
1. Implement `build_ml_snapshot_report(...)` in `backend/health_server.py`
2. Add `/health/model/ml-snapshot` route
3. Validate contract fields, plain-language `reason`, stable `reason_code`

Exit criteria:
1. Endpoint returns correct payloads for:
   - normal resident
   - sparse-data resident
   - no-data resident

### Phase C: Weekly/Admin Panel UI
1. Add dashboard fetch helper
2. Render panel in `Weekly Report`
3. Render panel in `Admin & Audit`
4. Add technical expander and threshold provenance rows

Exit criteria:
1. Panel values reconcile with existing `walk-forward` and `promotion-gates` outputs

### Phase D: Provenance Actionability + Today Compact Summary (Optional)
1. Add direct Admin links/actions for `editor_target`
2. Add compact `Today` ML health row (if desired for pilot)
3. Final copy review with Ops representative

Exit criteria:
1. Admin can navigate to threshold editor from provenance row in one action

## Testing Procedures (Execution Checklist)

### Backend API
1. `GET /health/model/ml-snapshot?elder_id=<known>` returns `200`
2. `GET /health/model/ml-snapshot?elder_id=<unknown>` returns `200` with `not_available`
3. `room=<room>` filter returns only requested room
4. `include_raw=false` omits technical payload
5. `include_raw=true` adds technical payload without breaking contract
6. `status.reason` is plain-language
7. `status.reason_code` present for all non-success and success states

### UI Functional
1. `Today` page shows only clinically actionable `Needs Attention` items
2. Routine ML uncertainty appears in `Review Queue`
3. Shadow banner visible on `Today`, `Weekly Report`, and shadow comparison view
4. Weekly panel renders candidate/champion macro-F1, transition F1, drift threshold
5. Admin panel shows threshold provenance + action links
6. Panel handles missing transition metrics without false green

### Reconciliation / Correctness
1. Panel macro-F1 values match `promotion-gates` endpoint for same resident/time window
2. Panel drift threshold matches effective walk-forward threshold (including room overrides)
3. Panel transition/stability values match latest walk-forward candidate summaries
4. Provenance source labels and keys match configured precedence

### Usability (Ops/Lead)
1. Ops can identify active care baseline (`Beta 5.5` vs `Beta 6`) in <5 seconds
2. Ops can explain what needs action from `Today` without opening technical details
3. Team lead can read exact macro-F1 / transition F1 / drift threshold from one panel

## Passing Requirements (Implementation Ready for Pilot)
1. `Today` alert-routing rule prevents routine uncertainty flooding
2. Global shadow-mode banner shows active care baseline consistently
3. `/health/model/ml-snapshot` endpoint returns contract-compliant payload
4. API returns plain-language `reason` and stable `reason_code`
5. Weekly/Admin `ML Health Snapshot` panel renders and degrades gracefully
6. Threshold provenance is visible and actionable in Admin
7. Metrics reconcile with existing `walk-forward` and `promotion-gates` sources

## Recommended Ownership (Execution)
1. UI engineer: `export_dashboard.py` (banner, panel, alert routing)
2. Backend engineer: `health_server.py` endpoint contract
3. ML/backend reviewer: metric mapping + threshold provenance validation
4. Ops representative: wording review for plain-language reasons and `Today` alerts
5. QA: cross-page functional + reconciliation checklist

## Decision Items to Confirm Before Coding
1. Should compact ML health row appear on `Today` in pilot MVP, or defer to Weekly/Admin only? (recommended: defer until alert routing is stable)
2. What is the source of truth for shadow-mode active-state banner (`rollout manager`, runtime status, or training metadata)? (recommended: runtime rollout state)
3. Which Admin sections/anchors will be used for `editor_target` links in v1?
