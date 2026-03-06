# Beta 6.1 Revised Step 1-4 Plan (Team Discussion Draft)

Date: 2026-03-05

Input incorporated:
- Team re-baseline review: `/Users/dicksonng/Desktop/beta6_1_plan_review.md`
- Existing architecture plan: `/Users/dicksonng/DT/Development/Beta_6/docs/beta6_architecture_improvement_plan_ab.md`
- Jessica A/B evidence artifacts in `/tmp`

## Executive Decision
Keep the original Step 1-4 sequence, but insert a mandatory "WS0.5" inside Step 1.

Reason:
- Current A/B shows promoted variants underperform baseline on Jessica.
- Without fixing activity-head gradient contamination first, shadow soak is likely to produce no-go regardless of decoder/fusion sophistication.

## Step 1 (WS0 + WS0.5): Stabilize Tests + Fix Base Model Signal
Target window: 2026-03-06 to 2026-03-07

### Step 1A (WS0): Stabilization and test-contract repair
Scope:
- Fix regressions in `/Users/dicksonng/DT/Development/Beta_6/backend/tests/test_run_daily_analysis_thresholds.py`.
- Ensure runtime-preflight early-exit behavior is explicitly covered and not silently changing pass/fail semantics.

Definition of done:
- Target test module passes locally.
- Runtime-preflight behavior is deterministic in tests for enabled/disabled runtime flags.

### Step 1B (WS0.5): Base model fix before feature promotion
Scope:
- Enable timeline multitask only for experiment runs (not production default).
- In `/Users/dicksonng/DT/Development/Beta_6/backend/ml/training.py` around activity sample weighting, mask both `unoccupied` and `unknown` windows for `activity_logits` loss.
- Add optional floor via env: `TIMELINE_ACTIVITY_UNOCCUPIED_WEIGHT_FLOOR` with default `0.0`, fallback candidate `0.05`.

Implementation notes:
- Preserve full-window gradients for occupancy head.
- Do not change runtime decoder flags in this step.

A/B variants for Step 1B:
- A0: current code.
- A1: masked activity loss, floor `0.0`.
- A2: masked activity loss, floor `0.05`.

Primary metrics:
- LivingRoom MAE (minutes).
- Bedroom sleep duration MAE (minutes).
- LivingRoom occupied precision/recall/F1.
- Hard-gate pass count.

Go criteria to move to Step 2:
- A1 or A2 improves LivingRoom MAE vs current full bundle by at least 20%.
- Bedroom MAE non-regression (<= +2.0 min).
- No occupancy-head collapse (occupied recall drop <= 0.02 absolute on core rooms).

No-go fallback:
- If A1 fails and A2 passes, keep floor `0.05`.
- If both fail, stop and investigate feature/label quality before Step 2.

## Step 2 (WS1 + Gate alignment subset): Segment De-Authority with Safety Rail
Target window: 2026-03-08 to 2026-03-09

### Step 2A: T4 heuristic de-authority behind flags
Scope:
- In `/Users/dicksonng/DT/Development/Beta_6/backend/utils/segment_utils.py`, isolate corrective sleep merge behavior behind explicit feature flag.
- Keep structural segmentation path intact.

Reasoned caveat:
- Heuristic removal can temporarily worsen timelines if base logits still weak.
- Therefore Step 2 is blocked on Step 1 success.

### Step 2B: timeline-mode gate alignment for short windows
Scope:
- Use pilot evidence profile for short-window runs (`pilot_stage_b`) via `/Users/dicksonng/DT/Development/Beta_6/backend/ml/policy_config.py` policy path and env.
- Keep low-support outcomes as watch/not-evaluated behavior; do not convert scarcity to hard fail.

Recommended threshold policy for <=14 days:
- `min_validation_class_support`: 8.
- `min_recall_support`: 8.
- Tier gate minimum support for Lane-B checks: 10.

Go criteria:
- Gate failures attributable to quality regressions, not pure support scarcity.
- No increase in safety-critical collapse fails.

## Step 3 (WS2 + WS3): Episode-Level Evaluation + Rollback Alignment
Target window: 2026-03-10 to 2026-03-12

### Step 3A: Episode-level metrics in Beta6 decision path
Scope:
- Wire episode metrics into Beta6 evaluation payload and gate engine.
- Files:
- `/Users/dicksonng/DT/Development/Beta_6/backend/ml/beta6/evaluation/evaluation_engine.py`
- `/Users/dicksonng/DT/Development/Beta_6/backend/ml/beta6/registry/gate_engine.py`
- reuse `/Users/dicksonng/DT/Development/Beta_6/backend/ml/timeline_metrics.py`

Required metrics:
- Duration MAE.
- Boundary precision/recall/F1.
- Fragmentation ratio.
- Episode count ratio (predicted episodes / true episodes).

### Step 3B: Rollback policy alignment and drill
Scope:
- Keep 2-consecutive-day rollback trigger rule.
- Validate rollback effect is not only "decoder off" but actual quality recovery to pre-Beta6 behavior.
- Run rollback drill proving fallback path improves or at least arrests adverse metrics.

Go criteria:
- Episode metrics visible in gate artifacts and used in run decision context.
- Rollback drill validated end-to-end with artifact evidence.

## Step 4 (WS4): Shadow Soak and Controlled Promotion Readiness
Target window: 2026-03-13 to 2026-03-20

### Shadow configuration
- Keep production defaults conservative.
- Enable runtime decoders only in shadow/canary cohorts after Step 1-3 pass.
- Start HMM first; CRF only under explicit canary allow flag.

### Runtime/soak checks
Daily tracked metrics:
- false-empty rate.
- unexplained divergence rate.
- fragmentation rate.
- unknown/abstain rates.
- LivingRoom precision-recall balance.

Rollback triggers (unchanged intent):
- false-empty > 5% for 2 consecutive days.
- unexplained divergence > 5% for 2 consecutive days.
- fragmentation >= 11% for 2 consecutive days.
- unknown >= 16% or abstain >= 18% for 2 consecutive days.

Promotion readiness criteria:
- Non-regression vs baseline on core rooms for the soak window.
- No hard-gate deterioration trend.
- No unresolved critical rollback alerts.

## A/B Reporting Pack (required for team signoff)
Produce one pack per step:
- Configuration diff.
- Seed/day matrix.
- KPI deltas vs anchor baseline.
- Gate outcomes (hard/warn/not-evaluated).
- Decision recommendation (go/no-go) with explicit reason codes.

Minimum matrix:
- Seeds: 11, 22, 33.
- Residents: Jessica + at least 1 additional resident.
- Window: day 7-10 and full available window where runtime permits.

## Roles and ownership (recommended)
- ML training changes: model pipeline owner.
- Gate policy tuning: MLOps + safety reviewer.
- Runtime rollout/rollback drill: platform/runtime owner.
- Final promotion decision: senior engineer + domain safety reviewer.

## Discussion Questions for Team
1. Should Step 1 default floor be `0.0` with auto-fallback to `0.05`, or start at `0.05` directly?
2. For <=14-day runs, do we lock `pilot_stage_b` globally or per-room override?
3. Should episode count ratio be blocking or watch-only in first rollout?
4. What is the minimum additional resident set for Step 4 go/no-go confidence?
