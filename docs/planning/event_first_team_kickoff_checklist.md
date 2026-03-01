# Event-First Team Kickoff Checklist

## Source of Truth
Primary execution plan:
`/Users/dicksonng/DT/Development/Beta_5.5/docs/planning/event_first_cnn_transformer_execution_plan.md`

This checklist is a short operational extraction for day-1 kickoff.

## Global Rules (All Lanes)
1. Production target remains **CNN+Transformer**.
2. No promotion from RandomForest scaffold path.
3. Leakage control is mandatory for all rolling-split evidence.
4. Hard safety gates must pass on all seeds and at least 5/6 splits.
5. Every PR must include:
   - tests,
   - artifact evidence (where applicable),
   - exact run command in PR notes.

## PR Order and Dependencies
1. `PR-A1` -> `PR-A2`  
2. `PR-B1` depends on `PR-A1`  
3. `PR-B2` and `PR-B3` depend on `PR-B1`  
4. `PR-C1` depends on `PR-A1`  
5. `PR-C2` depends on `PR-C1`, `PR-B2`, `PR-B3`  
6. `PR-D1` depends on `PR-C2`  
7. `PR-D2` depends on `PR-D1`  
8. `PR-D3` depends on `PR-D2`

## Lane A (Contract + Compatibility)

### Start tasks
1. Create schema + v1 registry + loader.
2. Add migration validator + CI checks.

### Files
1. `backend/config/adl_event_registry.v1.yaml`
2. `backend/config/schemas/adl_event_registry.schema.json`
3. `backend/ml/adl_registry.py`
4. `backend/ml/ci_adl_registry_validator.py`
5. `backend/tests/test_adl_event_registry_schema.py`
6. `backend/tests/test_adl_registry_loader.py`
7. `backend/tests/test_adl_registry_migrations.py`

### Commands
```bash
cd /Users/dicksonng/DT/Development/Beta_5.5/backend
pytest -q tests/test_adl_event_registry_schema.py tests/test_adl_registry_loader.py
pytest -q tests/test_adl_registry_migrations.py tests/test_ci_gate_validator.py
```

### Done criteria
1. Unknown labels map safely.
2. Alias collisions are blocked.
3. Breaking taxonomy changes fail CI without migration map.

## Lane B (Event + Gates + Home Empty)

### Start tasks
1. Finalize event compiler/decoder/derived events.
2. Implement event KPI and tiered gate checks.
3. Implement home-empty fusion and household safety gate.

### Files
1. `backend/ml/event_labels.py`
2. `backend/ml/event_decoder.py`
3. `backend/ml/derived_events.py`
4. `backend/ml/event_metrics.py`
5. `backend/ml/event_gates.py`
6. `backend/ml/home_empty_fusion.py`
7. `backend/tests/test_event_labels.py`
8. `backend/tests/test_event_decoder.py`
9. `backend/tests/test_derived_events.py`
10. `backend/tests/test_event_metrics.py`
11. `backend/tests/test_event_gates.py`
12. `backend/tests/test_home_empty_fusion.py`

### Commands
```bash
cd /Users/dicksonng/DT/Development/Beta_5.5/backend
pytest -q tests/test_event_labels.py tests/test_event_decoder.py tests/test_derived_events.py
pytest -q tests/test_event_metrics.py tests/test_event_gates.py tests/test_home_empty_fusion.py
```

### Done criteria
1. Deterministic event outputs.
2. Unknown-budget checks enforced.
3. Home-empty false-empty protection validated by tests.

## Lane C (CNN+Transformer + Shadow Integration)

### Start tasks
1. Add occupancy/activity head output contract in CNN+Transformer path.
2. Add per-split calibration flow (`train -> calib -> test`).
3. Wire event-first shadow mode into pipeline.

### Files
1. `backend/ml/training.py`
2. `backend/ml/transformer_head_ab.py`
3. `backend/ml/transformer_backbone.py` (if needed)
4. `backend/ml/pipeline.py`
5. `backend/ml/unified_training.py`
6. `backend/ml/policy_config.py`
7. `backend/tests/test_training.py`
8. `backend/tests/test_temporal_split.py`
9. `backend/tests/test_unified_training_path.py`
10. `backend/tests/test_pipeline_integration.py`
11. `backend/tests/test_event_first_shadow_pipeline.py`

### Commands
```bash
cd /Users/dicksonng/DT/Development/Beta_5.5/backend
pytest -q tests/test_training.py tests/test_policy_config.py tests/test_temporal_split.py
pytest -q tests/test_unified_training_path.py tests/test_pipeline_integration.py tests/test_event_first_shadow_pipeline.py
```

### Done criteria
1. Existing entrypoints unchanged when shadow disabled.
2. Event-first artifacts emitted when shadow enabled.
3. Calibration metrics (ECE/Brier) available in run artifacts.

## Lane D (Backtest + Validation + Signoff + Canary)

### Start tasks
1. Run 3-seed rolling backtest and emit D1 artifact.
2. Run controlled validation and emit D2 signoff pack.
3. Run 3-7 day canary shadow and decide D3.

### Files
1. `backend/scripts/run_event_first_backtest.py`
2. `backend/ml/validation_run.py`
3. `backend/logs/eval/hk0011_event_first_rolling_dec4_to10_3seed.json`
4. `backend/logs/eval/hk0011_event_first_vs_legacy_signoff.json`
5. `backend/logs/eval/hk0011_event_registry_change_impact.json`

### Commands
```bash
cd /Users/dicksonng/DT/Development/Beta_5.5/backend
python3 scripts/run_event_first_backtest.py --data-dir "/Users/dicksonng/DT/Development/New training files" --elder-id HK0011_jessica --min-day 4 --max-day 10 --seed 11 --output backend/logs/eval/hk0011_event_first_seed11_dec4_to10.json
python3 scripts/run_event_first_backtest.py --data-dir "/Users/dicksonng/DT/Development/New training files" --elder-id HK0011_jessica --min-day 4 --max-day 10 --seed 22 --output backend/logs/eval/hk0011_event_first_seed22_dec4_to10.json
python3 scripts/run_event_first_backtest.py --data-dir "/Users/dicksonng/DT/Development/New training files" --elder-id HK0011_jessica --min-day 4 --max-day 10 --seed 33 --output backend/logs/eval/hk0011_event_first_seed33_dec4_to10.json
python3 -m ml.validation_run start --elder-id HK0011_jessica --duration-days 7
python3 -m ml.validation_run finalize --elder-id HK0011_jessica --run-id <run_id>
```

### Done criteria
1. D1 includes reproducibility + leakage checklist + calibration summary.
2. D2 has explicit `gate_decision` and `failed_reasons`.
3. D3 canary shows no new critical failures before enabling `event_first_enabled=true`.

## Hard Safety Gates (Quick Reference)
1. Home-empty precision >= 0.95
2. Home-empty false-empty rate <= 0.05
3. Unknown-rate caps:
   - global <= 0.15
   - per room <= 0.20
4. No critical collapse (recall <= 0.02 at support >= 30)
5. Tiered minimum recall at support >= 30:
   - Tier-1 >= 0.50
   - Tier-2 >= 0.35
   - Tier-3 >= 0.20

## Daily Sync Template (15 min)
1. Yesterday done:
2. Today target:
3. Blockers:
4. Dependency handoff needed:
5. Risk to timeline:

## De-scope Rule
If `PR-C2` slips by >1 day:
1. Freeze UI scope and non-critical enhancements.
2. Prioritize leakage validity, calibration correctness, D1 evidence integrity.
