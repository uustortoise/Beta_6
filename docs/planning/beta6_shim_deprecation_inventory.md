# Beta6 Shim Deprecation Inventory

- Date: 2026-02-27
- Scope: Root-level `backend/ml/beta6/*.py` shim modules used for migration compatibility
- Hard removal target: **2026-04-30 (end of Phase 5)**

## Shim Modules

1. `backend/ml/beta6/active_learning.py`
2. `backend/ml/beta6/beta6_trainer.py`
3. `backend/ml/beta6/calibration.py`
4. `backend/ml/beta6/capability_profiles.py`
5. `backend/ml/beta6/data_manifest.py`
6. `backend/ml/beta6/evaluation_engine.py`
7. `backend/ml/beta6/feature_fingerprint.py`
8. `backend/ml/beta6/feature_store.py`
9. `backend/ml/beta6/fine_tune_safe_classes.py`
10. `backend/ml/beta6/gate_engine.py`
11. `backend/ml/beta6/head_factory.py`
12. `backend/ml/beta6/intake_gate.py`
13. `backend/ml/beta6/intake_precheck.py`
14. `backend/ml/beta6/prediction.py`
15. `backend/ml/beta6/registry_events.py`
16. `backend/ml/beta6/registry_v2.py`
17. `backend/ml/beta6/rejection_artifact.py`
18. `backend/ml/beta6/representation_eval.py`
19. `backend/ml/beta6/runtime_eval_parity.py`
20. `backend/ml/beta6/self_supervised_pretrain.py`
21. `backend/ml/beta6/slo_observability.py`
22. `backend/ml/beta6/timeline_hard_gates.py`

## Policy

1. New production code must import canonical subpackage modules (`ml.beta6.data.*`, `ml.beta6.training.*`, `ml.beta6.evaluation.*`, `ml.beta6.serving.*`, `ml.beta6.registry.*`, `ml.beta6.gates.*`, `ml.beta6.sequence.*`).
2. Shim paths are temporary compatibility bridges only and are blocked for new non-test imports by CI guard.
3. Remaining temporary shim consumers must be migrated before Phase 5 exit.

