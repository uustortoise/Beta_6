# Beta6 Activity Confidence Architecture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace raw-softmax runtime gating for Beta6 timeline-native rooms with a persisted activity-confidence calibrator and stability-aware threshold selection, then verify the Jessica Dec 17 benchmark gap narrows.

**Architecture:** Training will learn a per-room top-1 correctness calibrator from prediction geometry and persist it as a registry artifact. Runtime will derive an `activity_acceptance_score` from that artifact and use that same score for both class-thresholding and Beta6 unknown-abstain routing. Threshold learning will move into this calibrated score space and add a density-based fallback to avoid decision cliffs.

**Tech Stack:** Python, NumPy, scikit-learn, TensorFlow/Keras, pytest, Beta6 registry/runtime bridge

---

### Task 1: Add failing tests for the new score-space contract

**Files:**
- Create: `backend/tests/test_beta6_activity_confidence.py`
- Modify: `backend/tests/test_prediction.py`
- Modify: `backend/tests/test_registry.py`
- Modify: `backend/tests/test_training.py`

**Step 1: Write the failing tests**

Add tests that assert:

- the calibrator can score top-1 predictions into `[0, 1]`
- threshold selection avoids a dense-band cliff and records a stability fallback
- `PredictionPipeline.run_prediction()` uses the acceptance score for both class-thresholding and Beta6 unknown routing when a calibrator artifact is present
- registry persists and reloads the calibrator alias/versioned artifact
- training passes the new artifact into registry persistence

**Step 2: Run tests to verify they fail**

Run:

```bash
pytest backend/tests/test_beta6_activity_confidence.py backend/tests/test_prediction.py backend/tests/test_registry.py backend/tests/test_training.py -q
```

Expected:

- new tests fail because the calibrator module and registry/runtime integration do not exist yet

**Step 3: Keep failures minimal and specific**

Make sure the red phase fails on missing symbols or wrong legacy behavior, not on fixture bugs.

**Step 4: Commit checkpoint**

```bash
git add backend/tests/test_beta6_activity_confidence.py backend/tests/test_prediction.py backend/tests/test_registry.py backend/tests/test_training.py
git commit -m "test: codify beta6 activity confidence contract"
```

### Task 2: Implement the activity-confidence artifact and scoring module

**Files:**
- Create: `backend/ml/beta6/serving/activity_confidence.py`
- Modify: `backend/ml/beta6/serving/__init__.py` if needed
- Test: `backend/tests/test_beta6_activity_confidence.py`

**Step 1: Write the minimal implementation**

Implement:

- feature extraction from probability arrays
- logistic calibrator fit for top-1 correctness
- JSON-serializable artifact payload
- runtime scoring from persisted coefficients
- helper for density-aware threshold search in acceptance-score space

**Step 2: Run focused tests**

Run:

```bash
pytest backend/tests/test_beta6_activity_confidence.py -q
```

Expected:

- new module tests pass

**Step 3: Refactor only if needed**

Keep the API small and explicit; do not couple it to unrelated Beta6 orchestration code.

**Step 4: Commit checkpoint**

```bash
git add backend/ml/beta6/serving/activity_confidence.py backend/tests/test_beta6_activity_confidence.py
git commit -m "feat: add beta6 activity confidence calibrator"
```

### Task 3: Integrate calibrated score space into training

**Files:**
- Modify: `backend/ml/training.py`
- Modify: `backend/ml/policy_config.py`
- Modify: `backend/config/beta6_policy_defaults.yaml`
- Modify: `backend/ml/beta6/beta6_schema.py`
- Modify: `backend/tests/test_training.py`
- Modify: `backend/tests/test_beta6_config_schema.py`
- Modify: `backend/tests/test_beta6_policy_config_schemas.py`

**Step 1: Train and persist calibrator outputs**

Extend training to:

- fit the activity-confidence calibrator on the calibration split for timeline-native rooms
- compute thresholds in acceptance-score space for predicted-class subsets
- attach stability debug metadata to `metrics`
- pass the calibrator artifact into registry save calls

**Step 2: Add any necessary policy defaults**

Add only the calibration knobs needed for stable threshold search, such as:

- density window
- max dense-band share
- minimum fit support

**Step 3: Run focused tests**

Run:

```bash
pytest backend/tests/test_training.py backend/tests/test_beta6_config_schema.py backend/tests/test_beta6_policy_config_schemas.py -q
```

Expected:

- training and config tests pass

**Step 4: Commit checkpoint**

```bash
git add backend/ml/training.py backend/ml/policy_config.py backend/config/beta6_policy_defaults.yaml backend/ml/beta6/beta6_schema.py backend/tests/test_training.py backend/tests/test_beta6_config_schema.py backend/tests/test_beta6_policy_config_schemas.py
git commit -m "feat: train beta6 thresholds in calibrated acceptance space"
```

### Task 4: Persist and load calibrator artifacts in the registry

**Files:**
- Modify: `backend/ml/legacy/registry.py`
- Modify: `backend/tests/test_registry.py`

**Step 1: Add calibrator artifact suffix handling**

Support `_activity_confidence_calibrator.json` in:

- save
- alias sync
- rollback
- cleanup
- load into platform state

**Step 2: Run focused tests**

Run:

```bash
pytest backend/tests/test_registry.py -q
```

Expected:

- registry tests pass with the new artifact lifecycle

**Step 3: Commit checkpoint**

```bash
git add backend/ml/legacy/registry.py backend/tests/test_registry.py
git commit -m "feat: persist beta6 activity confidence artifacts"
```

### Task 5: Switch runtime thresholding and unknown routing to the unified score space

**Files:**
- Modify: `backend/ml/legacy/prediction.py`
- Modify: `backend/ml/beta6/serving/prediction.py`
- Modify: `backend/ml/beta6/serving/runtime_hooks.py`
- Modify: `backend/tests/test_prediction.py`
- Modify: `backend/tests/test_beta6_unknown_calibration.py`
- Modify: `backend/tests/test_prediction_beta6_runtime_hook_parity.py`

**Step 1: Thread acceptance scores through runtime**

Implement:

- acceptance score computation inside `PredictionPipeline.run_prediction()`
- thresholding based on acceptance score when a room artifact exists
- Beta6 unknown path support for caller-supplied confidence scores
- output columns exposing acceptance score and source

**Step 2: Preserve compatibility**

If no calibrator artifact exists for a room, keep legacy raw-confidence behavior unchanged.

**Step 3: Run focused tests**

Run:

```bash
pytest backend/tests/test_prediction.py backend/tests/test_beta6_unknown_calibration.py backend/tests/test_prediction_beta6_runtime_hook_parity.py backend/tests/test_pipeline_beta6_phase4_runtime_bridge.py -q
```

Expected:

- runtime tests pass

**Step 4: Commit checkpoint**

```bash
git add backend/ml/legacy/prediction.py backend/ml/beta6/serving/prediction.py backend/ml/beta6/serving/runtime_hooks.py backend/tests/test_prediction.py backend/tests/test_beta6_unknown_calibration.py backend/tests/test_prediction_beta6_runtime_hook_parity.py
git commit -m "feat: unify beta6 runtime confidence score space"
```

### Task 6: Run full targeted verification

**Files:**
- No code changes required unless failures reveal a defect

**Step 1: Run the affected automated suite**

Run:

```bash
pytest \
  backend/tests/test_beta6_activity_confidence.py \
  backend/tests/test_beta6_unknown_calibration.py \
  backend/tests/test_prediction_beta6_runtime_hook_parity.py \
  backend/tests/test_pipeline_beta6_phase4_runtime_bridge.py \
  backend/tests/test_prediction.py \
  backend/tests/test_registry.py \
  backend/tests/test_training.py \
  backend/tests/test_beta6_config_schema.py \
  backend/tests/test_beta6_policy_config_schemas.py -q
```

Expected:

- all targeted tests pass

**Step 2: Record verification evidence**

Capture the exact command and pass/fail summary in the handoff notes.

### Task 7: Re-benchmark Jessica on Dec 17 with the architecture fix

**Files:**
- Create: `tmp/jessica_17dec_eval_activity_confidence/` outputs
- Create: `docs/reviews/2026-03-10-beta6-activity-confidence-jessica-benchmark.md`

**Step 1: Retrain on existing Dec 4-10 Jessica pack with the new code**

Use the deterministic retrain or existing Jessica retrain flow that produces versioned artifacts without adding Dec 17 yet.

**Step 2: Re-run Dec 17 inference**

Keep evaluation mode aligned with the forensic benchmark:

- clone unlabeled copy for inference
- raw Beta6 prediction
- no golden-sample post-correction
- merge predictions back by timestamp

**Step 3: Produce comparison report**

Report:

- raw top-1 accuracy / macro-F1
- final exported accuracy / macro-F1
- per-room deltas for LivingRoom and Bedroom
- raw-vs-final gap before and after the architecture change
- calibrated score distribution evidence around LivingRoom `unoccupied` and Bedroom `sleep`

### Task 8: Controlled retrain with Dec 17 added

**Files:**
- Update or create controlled retrain artifacts under `backend/models/HK0011_jessica/`
- Update: `backend/models/HK0011_jessica/_last_training_run.json`
- Update: `docs/reviews/2026-03-10-beta6-activity-confidence-jessica-benchmark.md`

**Step 1: Add corrected Dec 17 to the Dec 4-10 pack**

Run one controlled retrain only after the architecture-only benchmark is complete.

**Step 2: Re-run Dec 17 benchmark**

Measure whether the architecture fix plus the additional corrected day improves:

- final exported LivingRoom macro-F1
- Bedroom `sleep` stability
- raw-vs-final gap

**Step 3: Summarize residual risk**

Call out remaining separate issues for:

- Bathroom hardening
- Entrance `out`
- LivingRoom label quality debt

Plan complete and saved to `docs/plans/2026-03-10-beta6-activity-confidence-architecture-implementation.md`. Proceeding in this session per your end-to-end instruction.
