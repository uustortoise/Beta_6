# RFC: Beta 6 ML Platform Greenfield Architecture (Post Beta 5.5 Validation)

- Status: Proposed
- Date: February 16, 2026
- Owner: Senior ML Engineer
- Scope: End-to-end retraining, evaluation, promotion, registry, and serving contracts for Beta 6
- Prerequisite: Beta 5.5 validation and stabilization complete

## 1. Decision Summary

For Beta 6, we should build an artifact-first, contract-driven ML platform as a **modular monolith** (single deployable codebase, strict module boundaries). We should not begin with distributed event-sourcing services.

Core decisions:

1. Separate training from promotion by design:
   - Training writes candidate artifacts only.
   - Promotion happens only after room-level and run-level gates pass.
2. Make every step consume and produce immutable artifacts.
3. Use append-only decision events + atomic champion pointers in the registry.
4. Define strict typed contracts for run spec, features, candidates, evaluation reports, and promotion decisions.
5. Enforce determinism and idempotency with run fingerprints.

## 2. Beta 5.5 Exit Gate (Start Condition for Beta 6)

Beta 6 implementation starts only after Beta 5.5 passes all of the following in production-like runs:

1. Registry: 0 destructive regressions across 5 consecutive retrains.
2. Leakage: 0 known temporal leakage paths in active training path.
3. Gating: 0 promotions from statistically invalid candidates.
4. Reproducibility: deterministic rerun with same `(data_fingerprint, policy_hash, code_version)` is a no-op with identical decision outcome.
5. Observability: rejection reason artifacts and run metadata are complete for all failed rooms.

If any condition fails, Beta 6 start is blocked until the gap is closed.

## 3. Non-Negotiable Architecture Principles

1. Fail-closed by default: contract violations block promotion.
2. Immutable artifacts: no in-place mutation of candidate evidence.
3. Promotion authority is centralized: one gatekeeper path controls champion updates.
4. Determinism is verified, not assumed.
5. Every decision is explainable from persisted artifacts within 5 minutes.

## 4. Target System Architecture

### 4.1 Modules

1. `ingest_contracts`
   - Validates input schema, timestamps, label integrity, duplicate semantics.
2. `feature_store`
   - Materializes versioned `FeatureSnapshot` from validated raw data.
3. `training_fabric`
   - Produces room-level `CandidateModelArtifact` from snapshot + train spec.
4. `evaluation_engine`
   - Produces `EvaluationReport` with walk-forward and support diagnostics.
5. `gate_engine`
   - Produces room decisions and run-level decision.
6. `registry_v2`
   - Stores append-only events and atomically updates champion pointers.
7. `serving_runtime`
   - Loads champion bundles only; no candidate artifacts in serving path.
8. `observability`
   - Structured logs, metrics counters, decision trace indexing.

### 4.2 Control Flow

1. Validate raw run input.
2. Materialize features (or fetch cached snapshot by fingerprint).
3. Train room candidates.
4. Evaluate each candidate.
5. Apply room-level gates.
6. Apply run-level gates.
7. Promote winning candidates atomically.
8. Publish run summary artifact.

## 5. Contract-First Interfaces

### 5.1 Run Spec

```yaml
run_spec_version: "v1"
run_id: "2026-02-16T15:00:00Z_elder_HK001"
elder_id: "HK001"
mode: "auto_aggregate"

data:
  manifest_paths:
    - "/abs/path/to/file1.csv"
    - "/abs/path/to/file2.csv"
  time_zone: "Asia/Hong_Kong"
  max_ffill_gap_seconds: 60
  duplicate_resolution_policy: "majority_vote_latest_tiebreak"

features:
  sequence_window_seconds: 600
  stride_seconds: 10
  feature_version: "feature_schema_v3"

training:
  architecture_family: "transformer"
  random_seed: 42
  profile: "production"
  optimizer: "adam"
  learning_rate: 1.0e-4
  epochs: 20

evaluation:
  walk_forward:
    lookback_days: 90
    min_train_days: 7
    valid_days: 1
    step_days: 1
    max_folds: 30

gating:
  room_policy_ref: "release_policy_v5"
  run_policy_ref: "global_gate_policy_v3"
```

### 5.2 Service Protocols (Python)

```python
class FeatureStore(Protocol):
    def materialize(self, run_spec: RunSpec) -> FeatureSnapshotRef: ...

class TrainingFabric(Protocol):
    def train_room(self, snapshot: FeatureSnapshotRef, room: str, run_spec: RunSpec) -> CandidateRef: ...

class EvaluationEngine(Protocol):
    def evaluate(self, candidate: CandidateRef, snapshot: FeatureSnapshotRef, run_spec: RunSpec) -> EvaluationReport: ...

class GateEngine(Protocol):
    def decide_room(self, report: EvaluationReport, run_spec: RunSpec) -> RoomDecision: ...
    def decide_run(self, room_decisions: list[RoomDecision], run_spec: RunSpec) -> RunDecision: ...

class RegistryV2(Protocol):
    def record_candidate(self, candidate: CandidateRef, report: EvaluationReport, room_decision: RoomDecision) -> None: ...
    def finalize_run(self, run_decision: RunDecision, room_decisions: list[RoomDecision]) -> PromotionResult: ...
```

## 6. Artifact and Storage Contracts

### 6.1 Directory Layout

```text
artifacts/
  runs/{run_id}/
    run_spec.yaml
    run_summary.json
    run_decision.json
  features/{elder_id}/{feature_fingerprint}/
    snapshot.parquet
    snapshot_meta.json
  models/{elder_id}/{room}/candidates/v{version}/
    model.keras
    scaler.pkl
    label_encoder.pkl
    thresholds.json
    candidate_meta.json
  registry/{elder_id}/{room}/
    events.jsonl
    champion_pointer.json
```

### 6.2 Required Metadata

Every candidate metadata must include:

1. `candidate_id`
2. `run_id`
3. `room`
4. `feature_fingerprint`
5. `run_spec_hash`
6. `policy_hash`
7. `code_version`
8. `dependency_lock_hash`
9. `training_metrics`
10. `gate_status`

## 7. Registry Event Model and Atomicity

### 7.1 Event Types

1. `candidate_saved`
2. `room_gate_passed`
3. `room_gate_failed`
4. `run_gate_passed`
5. `run_gate_failed`
6. `candidate_promoted`
7. `promotion_rollback`
8. `champion_deactivated`

### 7.2 Atomic Promotion Rule

Promotion transaction for a room:

1. Append `candidate_promoted` event with target version.
2. Write new `champion_pointer.json` to temp path.
3. `fsync` and atomic rename into final pointer path.
4. Verify pointer references existing bundle.
5. Emit `promotion_committed` metric.

If any step fails, promotion is marked failed and previous champion pointer remains active.

## 8. Gating Model (Room and Run)

### 8.1 Room-Level Gates

1. Data viability gate
2. Statistical support gate
3. Threshold quality gate
4. No-regress gate against prior champion
5. Calibration support gate

### 8.2 Run-Level Gates

1. Decision trace completeness gate
2. Walk-forward aggregate gate
3. Backbone alignment gate (when shared backbone mode enabled)
4. Global macro-F1 gate

A room candidate can only be promoted when:

1. Room gate = pass
2. Run gate = pass
3. Registry commit = success

## 9. Determinism and Idempotency Contract

### 9.1 Run Fingerprint

```text
run_fingerprint = sha256(
  data_fingerprint +
  run_spec_hash +
  policy_hash +
  code_version +
  dependency_lock_hash
)
```

### 9.2 Behavior

1. Same fingerprint + non-failed prior outcome => no-op retrain.
2. Same fingerprint + failed prior outcome => allow rerun only with explicit override.
3. Different fingerprint => full run required.

## 10. Testing Strategy

### 10.1 Unit Tests

1. Contract validators (schema, timestamp monotonicity, duplicate handling)
2. Feature materialization determinism
3. Gate reason correctness and precedence
4. Registry atomic pointer updates

### 10.2 Integration Tests

1. End-to-end single resident multi-room run
2. Candidate-only save then gated promotion
3. Rollback from failed run-level gate
4. Serving loads only champion pointers

### 10.3 Chaos and Fault Injection

1. Interrupt write during promotion pointer update
2. Corrupt one candidate artifact mid-run
3. Drop decision trace file before run finalize

Expected: fail closed, preserve prior champions.

## 11. 8-Week Delivery Plan (After Beta 5.5 Validation)

### Phase 0 (Week 0): Beta 5.5 Validation Sign-Off

1. Confirm Beta 5.5 exit gate completion.
2. Freeze Beta 6 contracts (`RunSpec v1`, artifact schemas, gate schema).
3. Publish ownership map and module boundaries.

Exit criterion: signed architecture contract and test harness baseline.

### Phase 1 (Week 1): Contracts and Skeleton

1. Create `beta6/contracts` package with typed schemas.
2. Add validation library and fixtures.
3. Add orchestration state machine skeleton.

Exit criterion: contract unit tests pass in CI.

### Phase 2 (Week 2): Feature Store

1. Implement deterministic feature materialization.
2. Add snapshot fingerprinting and cache reuse.
3. Persist `snapshot_meta.json`.

Exit criterion: repeated materialization yields identical fingerprint and outputs.

### Phase 3 (Week 3): Training Fabric

1. Implement candidate training API with room isolation.
2. Write candidate bundles and metadata.
3. Add deterministic training controls.

Exit criterion: candidate bundles generated for all rooms with valid contracts.

### Phase 4 (Week 4): Evaluation and Gates

1. Integrate walk-forward evaluator.
2. Implement room/run gate engines.
3. Persist structured gate reason artifacts.

Exit criterion: gate outcomes reproducible and explainable from artifacts.

### Phase 5 (Week 5): Registry V2 and Serving Loader

1. Implement append-only registry events.
2. Implement atomic champion pointers.
3. Implement serving loader for pointer-based bundles.

Exit criterion: promotion and rollback chaos tests pass.

### Phase 6 (Week 6): Dual-Run Shadow

1. Run Beta 6 in shadow mode alongside Beta 5.5.
2. Compare promotion decisions and metrics.
3. Track disagreement reasons.

Exit criterion: <=5% unexplained decision divergence for 7 consecutive days.

### Phase 7 (Week 7): Controlled Cutover

1. Enable Beta 6 promotion for limited resident cohort.
2. Keep Beta 5.5 as immediate fallback.
3. Monitor promotion stability and rejection patterns.

Exit criterion: zero destructive promotion incidents during cohort rollout.

### Phase 8 (Week 8): Full Cutover and Cleanup

1. Expand Beta 6 to all residents.
2. Keep rollback tooling active.
3. Deprecate Beta 5.5 promotion paths after stabilization window.

Exit criterion: 14 consecutive days stable operation and complete audit traces.

## 12. Cutover and Fallback Plan

1. Keep Beta 5.5 champion pointers immutable during shadow.
2. Beta 6 cutover uses feature flag per resident cohort.
3. On critical incident, revert serving to Beta 5.5 pointers within one deployment step.
4. Preserve all Beta 6 events/artifacts for forensic review.

## 13. Risks and Mitigations

1. Risk: Contract churn slows implementation.
   - Mitigation: freeze `RunSpec v1` in Week 0 and version changes explicitly.
2. Risk: Dual-run compute overhead.
   - Mitigation: shadow only selected residents in Week 6.
3. Risk: Hidden nondeterminism in TF stack.
   - Mitigation: deterministic seed controls + parity tests + tolerance policy.
4. Risk: Registry complexity creeps back.
   - Mitigation: strict pointer-only serving and event append discipline.

## 14. Immediate Actions (When Beta 5.5 Testing Completes)

1. Open Beta 6 implementation branch and bootstrap `beta6/contracts`.
2. Stand up a minimal end-to-end dry run for one resident and one room.
3. Start Week 1 contract test suite before model code migration.
