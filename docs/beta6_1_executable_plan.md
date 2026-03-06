# Beta 6.1 Executable Plan

Date: 2026-03-05
Audience: ML + MLOps + Runtime + QA
Scope: Convert revised Step 1-4 into an execution runbook with explicit tasks, file changes, tests, and pass gates.

## 0. Control Rules
- No runtime promotion until Step 1 and Step 2 pass.
- All gating decisions are metric-driven; no manual override for hard failures.
- Every task must attach artifact evidence under `/tmp` or `backend/models_beta6_registry_v2`.

## 1. Baseline Targets and Guardrails
Primary resident regression anchor (already measured): `HK001_jessica`, day `7-10`, seed `22`.

Current reference metrics:
- Baseline anchor LR MAE: `88.39`
- Full promoted bundle LR MAE: `242.95`
- Cross-room v2 LR MAE: `187.54`
- Semantic v3 LR MAE: `1007.77`

Global pass policy for this plan:
- Must improve over full promoted bundle by >= 20% on LivingRoom MAE.
- Must not regress Bedroom sleep MAE by more than +2.0 minutes.
- Must not reduce hard-gate pass count.
- Must keep rollback path validated and runnable.

---

## 2. Execution Sequence

## Step 1 (WS0 + WS0.5) - Stabilize + Base-Signal Fix

### Task S1-01: Repair runtime preflight test contracts
Description:
- Fix existing regressions in threshold/runtime-preflight tests before model edits.

Files to change:
- `/Users/dicksonng/DT/Development/Beta_6/backend/tests/test_run_daily_analysis_thresholds.py`
- `/Users/dicksonng/DT/Development/Beta_6/backend/run_daily_analysis.py` (only if behavior mismatch requires code update)

Testing procedure:
```bash
cd /Users/dicksonng/DT/Development/Beta_6/backend
pytest -q tests/test_run_daily_analysis_thresholds.py
```

Pass requirements:
- Test module exits 0.
- Runtime-preflight early-exit scenarios explicitly covered.

---

### Task S1-02: Activity-head loss mask (unoccupied + unknown)
Description:
- Prevent activity head from learning unoccupied majority shortcut.
- Apply mask to `activity_logits` sample weights for `unoccupied` and `unknown` labels.
- Add optional floor `TIMELINE_ACTIVITY_UNOCCUPIED_WEIGHT_FLOOR` (default `0.0`, fallback candidate `0.05`).
- Clarify class-weight interaction when mask is active to avoid stale class-prior calibration.

Files to change:
- `/Users/dicksonng/DT/Development/Beta_6/backend/ml/training.py` (around sample_weight construction for timeline multitask)
- `/Users/dicksonng/DT/Development/Beta_6/backend/tests/test_training.py` (add tests for masked weighting)
- `/Users/dicksonng/DT/Development/Beta_6/backend/.env.example` (document mask controls and timeline activation flags)

Implementation details:
- In timeline multitask path, derive class IDs for `unoccupied` and `unknown` from room label encoder.
- Build mask where occupied/activity labels keep full weight.
- Apply floor to masked windows: `effective_mask = mask + (1-mask)*floor`.
- When mask is active, recompute `compute_class_weight('balanced')` on occupied-only subset for activity-head weighting.
- Keep a fallback mode (`TIMELINE_ACTIVITY_REWEIGHT_MODE=global`) to reuse current all-class weights if occupied-only support is too small.
- Keep occupancy/boundary heads unchanged.
- Persist diagnostics in metrics:
  - `activity_loss_mask_coverage`
  - `activity_loss_mask_floor`
  - `activity_class_weight_mode`
  - `activity_class_weight_support`
  - `activity_weight_mean_occupied`
  - `activity_weight_mean_masked`

Activation prerequisites (before A/B):
- Set `ENABLE_TIMELINE_MULTITASK=true`.
- Set `TIMELINE_NATIVE_ROOMS=bedroom,livingroom,bathroom,kitchen`.

Testing procedure:
```bash
cd /Users/dicksonng/DT/Development/Beta_6/backend
pytest -q tests/test_training.py -k "timeline_multitask or timeline_native or lane_b_event_gates"
```

Pass requirements:
- New/updated unit tests pass.
- No failure in existing timeline multitask tests.

---

### Task S1-03: Controlled A/B for Step 1 variants
Description:
- Measure actual effect of masking before any downstream rollout.

Variants:
- A0: current (no mask).
- A1: mask + floor 0.0.
- A2: mask + floor 0.05.

Files to change:
- Optional runner profile yaml for repeatable variants:
  - `/tmp/beta6_step1_ab_profiles.yaml` (generated at run time)
- If needed for explicit env wiring in runner:
  - `/Users/dicksonng/DT/Development/Beta_6/backend/scripts/run_event_first_variant_backtest.py` (only if missing env passthrough)

Testing/simulation procedure (Jessica first):
```bash
# Baseline anchor already exists; rerun for reproducibility if needed
/Users/dicksonng/DT/Development/Beta_6/backend/venv/bin/python \
  /Users/dicksonng/DT/Development/Beta_6/backend/scripts/run_event_first_variant_backtest.py \
  --profiles-yaml /Users/dicksonng/DT/Development/Beta_6/backend/config/event_first_matrix_profiles.yaml \
  --variant anchor_top2_frag_v3 \
  --data-dir /Users/dicksonng/DT/Development/Beta_6/data/raw \
  --elder-id HK001_jessica --seed 22 --min-day 7 --max-day 10 \
  --output /tmp/beta6_s1_a0_seed22.json
```

For A1/A2, run same pipeline with the new mask/floor configuration.

Pass requirements (Step 1 gate):
- LivingRoom MAE improved vs full bundle by >= 20%.
- Bedroom MAE delta <= +2.0 minutes vs baseline anchor.
- Hard-gate pass count non-regression.
- Minority recall anti-collapse check:
  - For each core room, at least 2 occupied minority labels with support >= 5 must have recall > 0.0.
  - `sleep` recall must remain > 0.0 when `sleep` support >= 5.
- Occupancy-head safety non-regression:
  - `home_empty_false_empty_rate` must not increase by more than +0.01 vs A0 and must stay <= 0.05.
  - `home_empty_precision` must not drop by more than 0.02 vs A0.
- If A1 fails and A2 passes, select A2.

Evidence artifact:
- `/tmp/beta6_step1_ab_summary.json`
- `/Users/dicksonng/DT/Development/Beta_6/docs/beta6_step1_ab_report.md`

---

## Step 2 (WS1 + gate alignment subset) - Segment De-Authority and Support Semantics

### Task S2-01: Feature-flag corrective segment heuristics
Description:
- De-authority sleep corrective merges while preserving structural segmentation.

Files to change:
- `/Users/dicksonng/DT/Development/Beta_6/backend/utils/segment_utils.py`
- `/Users/dicksonng/DT/Development/Beta_6/backend/process_data.py` (wire flag usage if required)
- `/Users/dicksonng/DT/Development/Beta_6/backend/tests/test_process_data_arbitration.py` and/or add dedicated segment-utils tests

Testing procedure:
```bash
cd /Users/dicksonng/DT/Development/Beta_6/backend
pytest -q tests/test_process_data_arbitration.py
pytest -q tests/test_training.py -k "lane_b_event_gates"
```

Pass requirements:
- Flag-off behavior is backward compatible.
- Flag-on path does not increase contradiction artifacts or hard failures on Jessica A/B.

---

### Task S2-02: Timeline-mode evidence profile and support floors
Description:
- Apply short-window support thresholds without turning scarcity into hard fail.
- Add uncertainty visibility for low-support recall so reviewers can interpret pass confidence.

Files to change:
- `/Users/dicksonng/DT/Development/Beta_6/backend/ml/policy_config.py`
- `/Users/dicksonng/DT/Development/Beta_6/backend/ml/training.py` (if additional override logic needed)
- `/Users/dicksonng/DT/Development/Beta_6/backend/ml/event_gates.py`
- Optional config update:
  - `/Users/dicksonng/DT/Development/Beta_6/backend/config/release_gates.json`

Target settings for <=14-day windows:
- `min_validation_class_support`: 8
- `min_recall_support`: 8
- Event gate tier minimum support: 10
- Include diagnostics per recall gate:
  - `recall_confidence_interval_low`
  - `recall_confidence_interval_high`
  - `recall_confidence_interval_width`

Testing procedure:
```bash
cd /Users/dicksonng/DT/Development/Beta_6/backend
pytest -q tests/test_training.py -k "lane_b_event_gates or bootstrap"
pytest -q tests/test_run_daily_analysis_thresholds.py
```

Pass requirements:
- Low-support conditions produce watch/not-evaluated semantics (not hard block) unless collapse criteria are hit.
- Safety-critical collapse blocking remains intact.

Evidence artifact:
- `/tmp/beta6_step2_gate_semantics_report.json`

---

## Step 3 (WS2 + WS3) - Episode Metrics + Rollback Alignment

### Task S3-01: Add episode-level metrics to Beta6 evaluation path
Description:
- Extend Beta6 report payload to include episode/timeline quality metrics used by gating context.

Files to change:
- `/Users/dicksonng/DT/Development/Beta_6/backend/ml/beta6/evaluation/evaluation_engine.py`
- `/Users/dicksonng/DT/Development/Beta_6/backend/ml/beta6/registry/gate_engine.py`
- Reuse functions from:
  - `/Users/dicksonng/DT/Development/Beta_6/backend/ml/timeline_metrics.py`
  - `/Users/dicksonng/DT/Development/Beta_6/backend/ml/event_labels.py`

Metrics to include:
- `duration_mae_minutes`
- `fragmentation_rate`
- `boundary_precision`, `boundary_recall`, `boundary_f1`
- `episode_count_ratio` (pred/true)

Testing procedure:
```bash
cd /Users/dicksonng/DT/Development/Beta_6/backend
pytest -q tests/test_beta6_shadow_compare.py
pytest -q tests/test_beta6_orchestrator.py
pytest -q tests/test_run_daily_analysis_thresholds.py
```

Pass requirements:
- Metrics present in room evaluation payload for runs with timeline data.
- Gate engine consumes timeline metrics without schema break.
- `episode_count_ratio` is watch-only in Step 3 and does not block promotion yet.

---

### Task S3-02: Rollback policy drill with quality verification
Description:
- Validate rollback is effective on quality, not only feature-flag toggling.
- Rollback target path is explicit: pre-Beta6 flat-softmax runtime path.

Files to change:
- `/Users/dicksonng/DT/Development/Beta_6/backend/run_daily_analysis.py` (if drill hooks/telemetry required)
- `/Users/dicksonng/DT/Development/Beta_6/backend/ml/beta6/serving/runtime_preflight.py` (only if policy checks need extension)
- Test updates in:
  - `/Users/dicksonng/DT/Development/Beta_6/backend/tests/test_beta6_runtime_preflight.py`
  - `/Users/dicksonng/DT/Development/Beta_6/backend/tests/test_prediction_beta6_runtime_hook_parity.py`

Testing procedure:
```bash
cd /Users/dicksonng/DT/Development/Beta_6/backend
pytest -q tests/test_beta6_runtime_preflight.py
pytest -q tests/test_prediction_beta6_runtime_hook_parity.py
```

Pass requirements:
- Rollback trigger path validated end-to-end.
- Rollback configuration is explicit and verified:
  - `ENABLE_TIMELINE_MULTITASK=false`
  - `ENABLE_BETA6_HMM_RUNTIME=false`
  - `ENABLE_BETA6_CRF_RUNTIME=false`
- Fallback quality check passes:
  - post-rollback LivingRoom MAE <= baseline anchor MAE + 2.0 minutes.

Evidence artifact:
- `/tmp/beta6_step3_rollback_drill_report.json`

---

## Step 4 (WS4) - Shadow Soak and Promotion Readiness

### Task S4-01: Shadow soak execution (HMM first, CRF canary only)
Description:
- Run shadow cohort for 7-14 days with daily health snapshots.
- Explicitly enable HMM runtime for soak cohort, keep CRF canary-only.

Files touched (configuration/runtime only):
- `/Users/dicksonng/DT/Development/Beta_6/backend/.env`
- `/Users/dicksonng/DT/Development/Beta_6/backend/config/release_gates.json`
- Runtime policy artifacts under registry v2.

Required runtime flags for soak cohort:
- `ENABLE_BETA6_HMM_RUNTIME=true`
- `ENABLE_BETA6_CRF_RUNTIME=false`
- `BETA6_RUNTIME_ALLOW_CRF_CANARY=true` (canary scope only)

Daily checks:
- false-empty rate
- unexplained divergence
- fragmentation
- unknown/abstain rates
- LivingRoom precision-recall balance

Rollback triggers (2 consecutive days):
- false-empty > 5%
- unexplained divergence > 5%
- fragmentation >= 11%
- unknown >= 16% or abstain >= 18%

Testing procedure during soak:
```bash
cd /Users/dicksonng/DT/Development/Beta_6/backend
pytest -q tests/test_run_daily_analysis_thresholds.py -k "beta6_authority or runtime"
```

Pass requirements for promotion-ready state:
- No unresolved critical rollback alerts.
- Hard-gate pass non-regression across soak window.
- Core room MAE/F1 non-regression vs approved baseline.

Evidence artifacts:
- Phase 6 shadow compare report in training history metadata.
- Signed run decision artifacts in registry v2.

---

## 3. Cross-Resident/Seed Matrix (Mandatory Before Promotion)
Minimum matrix:
- Residents: Jessica + at least 1 additional resident.
- Seeds: 11, 22, 33.
- Windows: day 7-10 and full available window.

Suggested execution command family:
```bash
/Users/dicksonng/DT/Development/Beta_6/backend/venv/bin/python \
  /Users/dicksonng/DT/Development/Beta_6/backend/scripts/run_event_first_matrix.py \
  --profiles-yaml <profile.yaml> \
  --profile <profile_name> \
  --data-dir /Users/dicksonng/DT/Development/Beta_6/data/raw \
  --elder-id <elder_id> \
  --output-dir /tmp/<run_id> \
  --min-day 7 --max-day 10
```

Pass requirements:
- Median improvement vs full promoted bundle on LR MAE >= 20%.
- No resident with catastrophic regression (LR MAE > +60% vs baseline anchor) unless gated and explained.
- Gate pass-rate trend non-decreasing.

---

## 4. Required Deliverables Per Step
Each step must produce:
- `change_summary.md` (what changed + why)
- `test_results.txt` (commands + pass/fail)
- `ab_metrics.json` (numeric deltas)
- `go_no_go.md` (decision with reason codes)

Storage location:
- `/Users/dicksonng/DT/Development/Beta_6/docs/execution_artifacts/<step_id>/`

---

## 5. Team Discussion Decision Points
1. Default floor for masked windows: `0.0` vs `0.05`.
2. Whether short-window evidence profile (`pilot_stage_b`) is global or room-specific.
3. Threshold policy for promoting `episode_count_ratio` from watch-only to blocking.
4. Minimum additional resident set beyond Jessica for Step 4 signoff.

---

## 6. Initial Assignment Proposal
- ML owner: S1-02, S1-03, S3-01
- MLOps owner: S2-02, S4-01
- Runtime owner: S3-02
- QA owner: S1-01 + all pass/fail artifact verification
