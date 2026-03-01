# Beta_6 Ops UX: ML Health Snapshot Panel Spec

## Purpose
Define a dedicated UX panel for showing ML quality and safety metrics that Ops and leads asked to monitor day to day:
1. `macro-F1`
2. `transition F1`
3. `walk-forward drift threshold`

The panel must be understandable by non-ML Ops while preserving technical detail for ML/backend reviewers.

## Scope
In scope:
1. UX panel layout and behavior (Ops-facing + technical details)
2. Backend response contract for a read-only aggregate endpoint
3. Data source mapping to existing Beta_6 backend reports
4. Testing and pass criteria for panel correctness

Out of scope:
1. New training logic or gate logic changes
2. Threshold policy changes
3. Model architecture changes

## Users and Placement
Primary users:
1. Ops coordinator (needs quick status)
2. Team lead (needs trend and threshold visibility)
3. ML/backend reviewer (needs raw values and threshold provenance)

Panel placement:
1. `Today` page: compact status row (read-only summary, no charts)
2. `Weekly Report`: full `ML Health Snapshot` panel with trend and technical expander
3. `Admin & Audit`: same full panel plus threshold source/provenance details
4. `Shadow dashboard` / rollout view: full panel with candidate vs champion emphasis
5. During pilot shadow mode, show under a global UI banner: `Beta 5.5 currently active (Beta 6 in Shadow)`

## UX Design (Panel)

### Panel Name
`ML Health Snapshot`

### Default (Ops) View
Show a compact summary with plain labels and traffic-light state.

Top row:
1. `Model Quality (Balanced Score)` -> latest candidate macro-F1
2. `Transition Quality` -> latest transition macro-F1
3. `Safety Drift Threshold` -> effective WF drift threshold
4. `Safety Status` -> `Healthy / Watch / Action Needed / Not Available`

Secondary row:
1. `Current Champion Score`
2. `Candidate vs Champion Delta`
3. `Stability Accuracy`
4. `Last Check Time`

Badges:
1. `Threshold Source`: `Default / Env Override / Room Override / Policy`
2. `Confidence`: `High / Medium / Low` (based on support/fold count)
3. `Fallback Active` if applicable

### Technical Details (Expander)
Show exact values and diagnostics without making them mandatory for Ops.

Fields:
1. Candidate walk-forward metrics:
   - `macro_f1_mean`
   - `accuracy_mean`
   - `transition_macro_f1_mean`
   - `stability_accuracy_mean`
2. Champion comparison:
   - `champion_macro_f1_mean`
   - `delta_vs_champion`
3. Thresholds (effective):
   - `drift_threshold`
   - `min_transition_f1`
   - `min_stability_accuracy`
   - `max_transition_low_folds`
4. Threshold provenance:
   - `source`
   - `scope` (`global` / `room`)
   - `resolved_key` (example: `WF_MIN_TRANSITION_F1_BEDROOM`)
   - `source_file` (config/policy file path when applicable)
   - `editor_target` (Admin UI editor section ID / anchor)
   - `owner` (team/role responsible for change approval)
5. Support / reliability context:
   - `fold_count`
   - `transition_supported_folds`
   - `candidate_low_folds`
   - `candidate_low_transition_folds`
6. Gate result:
   - `pass`
   - `reasons`

### Status Logic (UI)
Use a single status badge with clear precedence.

Status rules:
1. `Not Available` if no recent walk-forward or promotion-gate data exists
2. `Action Needed` if:
   - fallback active for the room/resident, or
   - candidate failed gate and production impact exists, or
   - effective metrics are below thresholds with sufficient support
3. `Watch` if:
   - metrics are near threshold (configurable warning band), or
   - transition support is low / confidence low, or
   - data is stale
4. `Healthy` if:
   - latest metrics meet effective thresholds and no fallback/divergence alert is active

Recommended warning bands (UI-only, not gate logic):
1. `macro-F1`: within `0.03` of threshold
2. `transition F1`: within `0.05` of threshold
3. data stale: no new check in `>24h`

## Data Sources (Existing Beta_6)
Use existing backend reports first to avoid duplicating training logic.

Primary source A (promotion comparison):
1. `backend/health_server.py` -> `build_promotion_gate_report(...)`
2. Provides room-level latest candidate/champion macro-F1 and deltas

Primary source B (walk-forward health):
1. `backend/health_server.py` -> `build_walk_forward_report(...)`
2. Provides room summaries and current drift threshold config

Secondary source (fallback / serving state if displayed):
1. Existing rollout/fallback status sources used by dashboard (`training_history` metadata and/or runtime status helper)

Threshold source provenance:
1. Base defaults from env in `run_daily_analysis.py`
2. Room-specific overrides resolved in `run_daily_analysis.py`
3. Macro-F1 release thresholds from `backend/config/release_gates.json` (global rollout gate schedule)

## Proposed Backend Contract

### Endpoint
`GET /health/model/ml-snapshot?elder_id=<id>`

Optional query params:
1. `room=<room_key>` (filter to one room)
2. `lookback_runs=<n>` (default `20`)
3. `include_raw=true|false` (default `false`)

Notes:
1. Read-only aggregation endpoint
2. No DB schema changes required for MVP if derived from existing reports
3. Must return partial data gracefully (do not 500 on missing metric subsets)

### Response Shape (JSON)
```json
{
  "elder_id": "HK0011_jessica",
  "generated_at": "2026-02-26T12:34:56Z",
  "status": {
    "overall": "watch",
    "reason": "Model is uncertain about Bedroom activity sequence",
    "reason_code": "transition_near_threshold",
    "data_freshness_hours": 6.2
  },
  "thresholds": {
    "global_release_macro_f1_threshold": {
      "value": 0.65,
      "source": "policy",
      "policy_file": "backend/config/release_gates.json",
      "effective_day_bucket": "day_10_21"
    }
  },
  "rooms": [
    {
      "room": "bedroom",
      "status": "watch",
      "last_check_time": "2026-02-26T09:12:00Z",
      "fallback_active": false,
      "metrics": {
        "candidate_macro_f1_mean": 0.71,
        "champion_macro_f1_mean": 0.74,
        "delta_vs_champion_macro_f1": -0.03,
        "candidate_transition_macro_f1_mean": 0.82,
        "candidate_stability_accuracy_mean": 0.995,
        "candidate_accuracy_mean": 0.96
      },
      "thresholds": {
        "drift_threshold": {
          "value": 0.60,
          "source": "env_default",
          "resolved_key": "WF_DRIFT_THRESHOLD",
          "source_file": "backend/run_daily_analysis.py",
          "editor_target": "admin.thresholds.walk_forward",
          "owner": "ML/Ops"
        },
        "min_transition_f1": {
          "value": 0.80,
          "source": "room_override",
          "resolved_key": "WF_MIN_TRANSITION_F1_BEDROOM",
          "source_file": "backend/config/beta6_room_capability_gate_profiles.yaml",
          "editor_target": "admin.thresholds.room_overrides.bedroom",
          "owner": "ML/Ops"
        },
        "min_stability_accuracy": {
          "value": 0.99,
          "source": "env_default",
          "resolved_key": "WF_MIN_STABILITY_ACCURACY",
          "source_file": "backend/run_daily_analysis.py",
          "editor_target": "admin.thresholds.walk_forward",
          "owner": "ML/Ops"
        }
      },
      "support": {
        "fold_count": 5,
        "transition_supported_folds": 4,
        "candidate_low_folds": 0,
        "candidate_low_transition_folds": 1
      },
      "gate": {
        "pass": true,
        "reasons": []
      }
    }
  ],
  "raw": null
}
```

### Response Rules
1. `status.overall` values: `healthy`, `watch`, `action_needed`, `not_available`
2. `rooms[]` can be empty if no reports exist
3. Numeric fields are `null` when unavailable (do not coerce to `0.0`)
4. `thresholds.*.source` enum:
   - `policy`
   - `env_default`
   - `env_override`
   - `room_override`
   - `unknown`
5. `raw` only populated when `include_raw=true`
6. `status.reason` must be plain-language, Ops-readable text (avoid ML jargon by default)
7. `status.reason_code` carries a stable machine-readable category for frontend analytics/tests
8. Technical wording belongs in `raw` (or another technical field only when `include_raw=true`)

## Implementation Touchpoints (Planned)
Files to update for MVP implementation:
1. `backend/health_server.py`
   - add `build_ml_snapshot_report(elder_id, room=None, lookback_runs=20, include_raw=False)`
   - add route `/health/model/ml-snapshot`
2. `backend/export_dashboard.py`
   - add fetch helper `fetch_ml_health_snapshot(...)`
   - render `ML Health Snapshot` panel in `Weekly Report` and `Admin & Audit`
   - render compact summary card/row in `Today`
   - render threshold provenance entries as direct Admin links/actions when `editor_target` is present
3. `backend/run_daily_analysis.py` (optional, only if provenance metadata is missing)
   - expose threshold source details in persisted metadata to avoid UI-side inference

## Testing Procedures

### Functional
1. Call endpoint for resident with recent training data and verify room metrics populate
2. Call endpoint for resident with no training history and verify `not_available` status
3. Verify `room` filter returns only requested room
4. Verify `include_raw=false` omits large payloads
5. Verify panel renders with partial data (for example missing transition F1)
6. Verify API returns plain-language `status.reason` and stable `status.reason_code` for representative conditions

### Correctness (Metric Mapping)
1. Candidate/champion macro-F1 in panel matches `promotion-gates` health report values
2. Drift threshold in panel matches effective `walk-forward` report config/room override
3. Transition F1/stability accuracy match latest walk-forward candidate summary in training metadata
4. Threshold source label matches actual precedence resolution (`room > env > default/policy`)
5. Threshold provenance fields (`source_file`, `editor_target`) resolve to the expected Admin section/file

### UX / Ops Acceptance
1. Ops can identify `Healthy / Watch / Action Needed` without opening technical details
2. Ops can click into technical details and read exact macro-F1 / transition F1 / thresholds
3. Missing data state is explicit (`Not Available`) and not shown as healthy
4. Admin can navigate from threshold provenance to the relevant config editor section in one action

### Performance
1. Endpoint p95 response under 500 ms for `lookback_runs<=20` on normal local DB
2. UI refresh does not block page render if endpoint fails (panel degrades gracefully)

## Passing Requirements (MVP)
1. `ML Health Snapshot` visible in `Weekly Report` and `Admin & Audit`
2. Displays current:
   - candidate macro-F1
   - champion macro-F1
   - transition F1
   - drift threshold
   - threshold source
3. Status badge logic works for `healthy`, `watch`, `action_needed`, and `not_available`
4. Panel values reconcile with existing `promotion-gates` and `walk-forward` health endpoints
5. No regression to existing dashboard sections when endpoint is unavailable
6. API returns plain-language `status.reason` plus stable `reason_code` for all supported status states

## Open Decisions for Team
1. Should `macro-F1` threshold shown in this panel use rollout schedule threshold, room drift threshold, or both (recommended: show both with labels)?
2. Should `Today` page show one aggregate status only, or top 3 rooms by risk (recommended: aggregate + top 1 reason)?
3. Should fallback state be sourced from runtime status endpoint or inferred from latest training metadata (recommended: runtime status when available)?
4. Should a technical explanation field be exposed outside `raw`, or remain gated behind `include_raw=true` (recommended: keep technical wording inside `raw`)?
