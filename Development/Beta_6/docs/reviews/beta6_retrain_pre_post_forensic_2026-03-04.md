# Beta6 Retrain Pre/Post Forensic (2026-03-04)

## Scope
- Objective: Execute recommended structural changes and run a full retrain for `HK0011_jessica`, then compare to prior baseline run.
- Baseline run: `beta6_daily_HK0011_jessica_20260303T143003Z`
- Post-change run: `beta6_daily_HK0011_jessica_20260303T223326Z`
- Log source: `backend/logs/pipeline.log`
- Artifact source: `backend/tmp/beta6_gate_artifacts/HK0011_jessica/`

## Code Changes Applied
- Commit: `3eabf06` on branch `codex/pilot-bootstrap-gates`
- Files:
  - `backend/ml/training.py`
  - `backend/ml/legacy/prediction.py`
  - `backend/ml/policy_config.py`
  - `backend/tests/test_policy_config.py`
  - `backend/tests/test_prediction.py`

## Verification
- Command: `cd backend && pytest tests/test_policy_config.py tests/test_prediction.py -q`
- Result: `30 passed, 3 warnings`

## Pre/Post Room Metrics
| Room | Baseline Acc | Baseline Macro-F1 | Baseline Gate | Post Acc | Post Macro-F1 | Post Gate | Delta F1 |
|---|---:|---:|---|---:|---:|---|---:|
| Bathroom | 0.3664 | 0.6030 | PASS | 0.3664 | 0.6621 | PASS | +0.0591 |
| Bedroom | 0.4541 | 0.1800 | FAIL | 0.5131 | 0.1410 | FAIL | -0.0390 |
| Entrance | 0.9027 | 0.4960 | PASS | 0.9027 | 0.4960 | PASS | +0.0000 |
| Kitchen | 0.7925 | 0.7750 | PASS | 0.7925 | 0.7747 | PASS | -0.0003 |
| LivingRoom | 0.5606 | 0.7240 | PASS | 0.5606 | 0.3806 | FAIL | -0.3434 |

## Gate Outcome Delta
- Baseline failed rooms: `bedroom`
- Post-change failed rooms: `bedroom`, `livingroom`

### Baseline blocking reasons
- Bedroom:
  - `room_threshold_failed:bedroom:f1=0.180<required=0.200`
  - `lane_b_gate_failed:bedroom:recall_sleep_duration`

### Post-change blocking reasons
- Bedroom:
  - `room_threshold_failed:bedroom:f1=0.141<required=0.200`
  - `lane_b_gate_failed:bedroom:recall_sleep_duration`
- LivingRoom:
  - `no_regress_failed:livingroom:drop=0.343>max_drop=0.050`

## Forensic Findings

### 1) Bedroom remains structurally blocked by Lane-B sleep-duration recall
Evidence:
- Decision trace (`Bedroom_v12_decision_trace.json`) shows:
  - `metrics.lane_b_gate.derived_gate_metrics.event_recalls.sleep_duration = 0.2815`
  - Tier-1 threshold is `0.5` -> hard fail.
- Final gate reasons are still:
  - `room_threshold_failed:bedroom:f1=0.141<required=0.200`
  - `lane_b_gate_failed:bedroom:recall_sleep_duration`

Observation:
- Continuity bridge is active and very aggressive (thousands of converted windows logged), but did not raise event-level sleep-duration recall enough for gate pass.

### 2) LivingRoom regression introduced in this post-change run
Evidence:
- Baseline LivingRoom macro-F1 `0.724` -> post `0.381`.
- Gate failed on no-regression policy:
  - `drop=0.343 > max_drop=0.050`.
- Lane-B livingroom event recall itself is good in post run (`~0.81`), so fail is not lane-B; it is no-regression against champion.

### 3) Gate-aligned checkpoint scoring is still not truly gate-aligned for Bedroom
Evidence:
- Bedroom checkpoint summary in decision trace shows high lane-B proxy recall (`~0.899`), but final lane-B gate recall is `0.2815`.
- This indicates proxy scoring at checkpoint selection is not matching final gate behavior after full post-processing/gating path.

## Immediate Recommended Next Run (Executable)
1. Disable bedroom continuity bridge for the next A/B run to remove over-correction risk.
2. Neutralize lane-B proxy weight in checkpoint scoring until it is aligned with final gate metrics.
3. Re-run retrain and compare against this post run.

Suggested env for next run:
- `ENABLE_BEDROOM_SLEEP_CONTINUITY=false`
- `GATE_ALIGNED_LANE_B_WEIGHT=0.0`
- `GATE_ALIGNED_LANE_B_FLOOR_PENALTY=0.0`

## Artifacts for Team Review
- Post evaluation report:
  - `backend/tmp/beta6_gate_artifacts/HK0011_jessica/beta6_daily_HK0011_jessica_20260303T223326Z/beta6_daily_HK0011_jessica_20260303T223326Z_evaluation_report.json`
- Post rejection artifact:
  - `backend/tmp/beta6_gate_artifacts/HK0011_jessica/beta6_daily_HK0011_jessica_20260303T223326Z/beta6_daily_HK0011_jessica_20260303T223326Z_rejection_artifact.json`
- Baseline evaluation report:
  - `backend/tmp/beta6_gate_artifacts/HK0011_jessica/beta6_daily_HK0011_jessica_20260303T143003Z/beta6_daily_HK0011_jessica_20260303T143003Z_evaluation_report.json`
