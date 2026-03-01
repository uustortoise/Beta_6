# Beta 6 Timeline Hard-Gate Policy (v1)

- Date: 2026-02-25
- Status: Active
- Policy file: `backend/config/beta6_timeline_hard_gates.yaml`

## 1. Primary Gate Rule

Timeline quality is a hard gate. A room must fail when either:
1. `duration_mae_minutes` exceeds room profile threshold.
2. `fragmentation_rate` exceeds room profile threshold.

This gate applies even when F1-style metrics are acceptable.

## 2. Deterministic Reason Codes

1. missing/invalid timeline metrics -> `fail_timeline_metrics_missing`
2. MAE regression -> `fail_timeline_mae`
3. fragmentation regression -> `fail_timeline_fragmentation`

## 3. Threshold Source

Thresholds are sourced from room capability profiles:
1. `max_timeline_mae_minutes`
2. `max_fragmentation_rate`

Reference file:
`backend/config/beta6_room_capability_gate_profiles.yaml`
